# Spec for the deep-learning part (v2). Read CODE_SPEC.md first; this file ADDS to it and wins on conflicts.

## What changed
Model lineup is now: Dummy, TF-IDF+LR, TF-IDF+DecisionTree, TF-IDF+MLP (sklearn), BM25+LR (all classical, run locally by the
orchestrator), plus the deep part below which is meant to run on a GPU SERVER (shared RTX A6000s) via `run_deep.py`.
SVM and HistGradientBoosting are gone. Torch MLP head is gone (heads use sklearn).

## Safety rules (updated)
- Same as CODE_SPEC.md, EXCEPT: torch and transformers are now installed (CPU build) in `.venv`. You may run ONLY tiny smoke
  tests: a randomly initialised tiny BERT (`BertConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
  intermediate_size=64, vocab_size=<tiny>)`) with a tiny hand-made vocab, 16-32 synthetic rows, 1 epoch, CPU, finishing in
  seconds. NEVER download a pretrained model or tokenizer, never use real data in a model run, never call
  `SentenceTransformer(...)`/`from_pretrained(<hub name>)` in anything you execute. All commands wrapped as
  `timeout 120 taskset -c 0-1 nice -n 19 <cmd>`.
- Everything you write must also work on a GPU box: choose `device = "cuda" if torch.cuda.is_available() else "cpu"` only in
  the runner, pass `device` into the helpers.

## Presentability
Same as CODE_SPEC.md: short plain functions, WHY-comments, the course's training-loop style (Session 5: `opt.zero_grad()`
first in the loop, `model.train()` / `model.eval()` + `torch.no_grad()`, DataLoader with a seeded generator, raw logits into
the loss). No Trainer API, no accelerate, no peft. Standard Hugging Face `AutoTokenizer` / `AutoModelForSequenceClassification`.

## Files you own
1. `modules/deep_helper.py`: KEEP `embed_texts` exactly as is. DELETE `make_mlp`, `_train_epoch`, `_val_pr_auc`, `train_mlp`,
   `predict_scores`, `fine_tune_note` and the now unused imports/`_PATIENCE`. Update `tests/test_deep_helper.py` accordingly
   (it only compiles / checks signatures; embed_texts is never executed with a real model; you MAY test its cache-hit branch
   by pre-writing tiny `_emb.npy` / `_ids.npy` files into a temp dir, which never imports sentence_transformers).
2. `modules/finetune_helper.py` (NEW), public API:
   - `load_model_and_tokenizer(model_name="distilbert-base-uncased", num_labels=2)` -> `(model, tokenizer)` (hub download happens
     here; lazy import inside).
   - `encode_texts(tokenizer, texts, max_len=256)` -> dict of CPU tensors `input_ids`, `attention_mask` (truncation + padding to
     max_len; one batch call; no per-row Python loops over tokens).
   - `make_loader(enc, labels, batch_size, shuffle, seed)` -> DataLoader over a TensorDataset; when `shuffle` use
     `torch.Generator().manual_seed(seed)`; never shuffle validation/test.
   - `class_weights(y_train, mode="sqrt")` -> tensor `[1.0, w]` where `w = sqrt(n_neg/n_pos)` for `"sqrt"`, `n_neg/n_pos` for
     `"balanced"`, `1.0` for `"none"`. Comment why (4.8% positives; threshold is chosen on validation anyway).
   - `predict_proba(model, loader, device)` -> numpy array of P(fraud) (softmax of logits, column 1), eval mode, no_grad,
     autocast bf16 on cuda only.
   - `fine_tune(model, train_loader, val_loader, y_val, device, epochs=3, lr=2e-5, weight_decay=0.01, class_weight=None, seed=0)`
     -> `(model, history)`: AdamW, `CrossEntropyLoss(weight=class_weight)` on raw logits, linear warm-up/decay via
     `get_linear_schedule_with_warmup` (10% warm-up), grad-norm clip 1.0, bf16 autocast only when device is cuda, evaluate
     validation PR-AUC (`sklearn.metrics.average_precision_score`) after every epoch, keep the best weights (copy the state_dict
     to CPU), stop when no improvement for 2 epochs, reload the best weights at the end. `history` = list of dicts
     (epoch, train_loss, val_pr_auc, seconds). Call `seed_everything(seed)` from `modules.data_helper` first.
3. `run_deep.py` (NEW, at `code/run_deep.py`, argparse CLI). Run from the `code/` directory.
   `python run_deep.py --stage {embed,heads,bert,all} --seeds 0 1 2 3 4 --out outputs/deep
    [--embed-model sentence-transformers/all-MiniLM-L6-v2] [--bert-model distilbert-base-uncased]
    [--epochs 3] [--batch-size 16] [--max-len 256] [--lr 2e-5] [--device auto] [--data ../data/fake_job_postings.csv]`
   Common start: `df = dh.clean(dh.load_raw(path))`, `df = ft.add_engineered_features(df)` (gives the joined `text` column),
   splits per seed from `splits/seed{N}.npz` via `dh.load_splits(path, df)` (if the file is missing: `dh.make_splits(df, seed)`
   then `dh.save_splits(path, splits, df)`), print the sizes and class balance of each split.
   - `embed`: `deep_helper.embed_texts(df["text"].tolist(), df["job_id"].to_numpy(), model_name=..., cache_dir="features",
     device=...)`; prints shape and time. Embeddings are shared by all seeds.
   - `heads` (CPU only, cheap, must also run on a laptop once the embeddings file exists; it must NOT import torch):
     for each seed fit two heads on the train rows only: `emb_logreg` = `LogisticRegression(class_weight="balanced",
     max_iter=2000)` with C chosen from [0.1, 1, 10] by validation PR-AUC, and `emb_mlp` = sklearn `MLPClassifier(
     hidden_layer_sizes=(128,), early_stopping=True, n_iter_no_change=5, max_iter=100, random_state=seed)` (no class_weight
     support; the threshold is chosen on validation). Use `ev.pick_threshold` on validation, then `ev.evaluate` on val and test.
   - `bert`: for each seed: tokenise train/val/test once, `make_loader`, `fine_tune`, `predict_proba` on val and test, pick the
     threshold on validation, evaluate, save results with model name `bert_<slug of --bert-model>`; also save
     `history` into the JSON under key `history`. Skip a (model, seed) whose result files already exist (resume).
   - Print what is running (device, GPU name and free memory if cuda, batch size, seq len) before the first step, and
     `nvidia-smi`-free: use `torch.cuda.mem_get_info()` for free memory.
   - Results contract (below) for every (model, seed).

## Results contract (shared with the local classical runner)
Directory `<out>/`; per (model, seed) two files `<model>_seed<N>.json` and `<model>_seed<N>.npz`.
JSON keys: `model`, `group` ("deep"), `seed`, `thr` (float chosen on VALIDATION), `val` and `test` (each the dict returned by
`eval_helper.evaluate(y, score, thr)`), `train_pr_auc` (float or null), `best_params` (JSON-able dict), `seconds` (float),
`notes` (str), optional `history`.
NPZ keys: `y_val`, `score_val`, `y_test`, `score_test`, row order identical to `splits["val"]` and `splits["test"]`.
Write JSON with `json.dump(..., indent=2)`; cast numpy scalars to Python floats/ints first.
The test split is evaluated ONCE per (model, seed), after the threshold was fixed on validation.

## Tests you write
`tests/test_finetune_helper.py` (plain-assert script, tiny random BERT as described above): `class_weights` known answers;
`encode_texts` shapes; `make_loader` reproducible order for a fixed seed; `fine_tune` runs 1-2 epochs on 32 synthetic rows
whose label depends on a keyword, history has the right keys, best weights restored; `predict_proba` returns values in [0, 1]
with the right length. `tests/test_run_deep.py`: only checks that `python run_deep.py --help` works and that the results
writer produces files with the contract's keys (call the writer function directly with fake numbers).

## Final report
Files written, deviations, exact commands with output, and what is NOT verified (anything that needs a real pretrained model,
a GPU, or real data).
