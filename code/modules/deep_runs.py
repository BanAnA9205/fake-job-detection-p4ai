"""Deep runs on the fake job postings (called from notebook 03): frozen MiniLM embeddings with
sklearn heads, and fine-tuning of a pretrained encoder.

The heavy libraries (torch, transformers, sentence-transformers) are imported inside the
functions that need them, so this module imports on a laptop without them.
Results land in out_dir as <model>_seed<N>.json and <model>_seed<N>.npz.
"""

import json
import os
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.utils.class_weight import compute_class_weight

from modules import data_helper as dh
from modules import deep_helper
from modules import eval_helper as ev
from modules import explain_helper as eh
from modules import finetune_helper as fth
from modules import results_helper as rh
from modules.classical_runs import get_splits

EMBED_CACHE = "features"
LOGREG_CS = [0.1, 1.0, 10.0]
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BERT_MODEL = "distilbert-base-uncased"
LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def print_splits(splits, y, seed):
    """Print size and class balance of every split, as the course requires."""
    print(f"seed {seed}:")
    for name in ("train", "val", "test"):
        idx = splits[name]
        print(f"  {name:5s} n={len(idx):5d} positives={int(y[idx].sum()):4d} "
              f"rate={y[idx].mean():.4f}")


def load_embeddings(embed_model, job_ids, cache_dir=EMBED_CACHE):
    """Load the cached embeddings for these rows; never imports sentence-transformers."""
    slug = embed_model.replace("/", "_")
    emb_path = os.path.join(cache_dir, slug + "_emb.npy")
    ids_path = os.path.join(cache_dir, slug + "_ids.npy")

    # The cache must exist...
    if not (os.path.exists(emb_path) and os.path.exists(ids_path)):
        raise FileNotFoundError(f"no embeddings for {embed_model} in {cache_dir}/; run the embedding step first")

    # ...and must have been built for exactly these rows, in this order.
    assert np.array_equal(np.load(ids_path, allow_pickle=True), np.asarray(job_ids)), (
        "cached embeddings belong to different rows; rerun the embedding step"
    )
    return np.load(emb_path)


def evaluate_scores(y, splits, score_val, score_test):
    """Fix the threshold on validation, then evaluate validation and test at that threshold."""
    # Threshold from validation only; then both splits are scored with that one number.
    thr = ev.pick_threshold(y[splits["val"]], score_val)
    val_metrics = ev.evaluate(y[splits["val"]], score_val, thr)
    test_metrics = ev.evaluate(y[splits["test"]], score_test, thr)
    return thr, val_metrics, test_metrics


def run_embed(df, device, embed_model=EMBED_MODEL):
    """Build or reuse the embedding cache shared by all split seeds."""
    start = time.time()
    emb = deep_helper.embed_texts(
        df["text"].tolist(), df["job_id"].to_numpy(),
        model_name=embed_model, cache_dir=EMBED_CACHE, device=device,
    )
    print(f"embeddings: {emb.shape} ready in {time.time() - start:.1f}s")


def fit_logreg_head(Xtr, ytr, Xva, yva, seed):
    """Logistic regression on embeddings, with C picked by validation PR-AUC."""
    best_auc, best_model, best_params = -np.inf, None, {}
    for C in LOGREG_CS:
        # Fit on train for this C, score validation PR-AUC.
        model = LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=seed)
        model.fit(Xtr, ytr)
        auc = float(average_precision_score(yva, ev.score_of(model, Xva)))

        # Keep the best C so far.
        if auc > best_auc:
            best_auc, best_model, best_params = auc, model, {"C": C}
    return best_model, best_params


def save_head_result(out_dir, name, seed, model, params, X, y, splits, start):
    """Score a fitted head, evaluate it, and write the result files."""
    tr, va, te = splits["train"], splits["val"], splits["test"]

    # Scores on validation and test, then metrics at the validation-chosen threshold.
    score_val = ev.score_of(model, X[va])
    score_test = ev.score_of(model, X[te])
    thr, val_metrics, test_metrics = evaluate_scores(y, splits, score_val, score_test)

    # Save the shared json + npz pair and report validation and test PR-AUC.
    rh.save_result(
        out_dir, name, "deep", seed, thr, val_metrics, test_metrics, y[va], score_val,
        y[te], score_test,
        train_pr_auc=average_precision_score(y[tr], ev.score_of(model, X[tr])),
        best_params=params, seconds=time.time() - start,
        notes="frozen embeddings with a sklearn head; threshold on validation")
    print(f"{name} seed {seed}: val PR-AUC {val_metrics['pr_auc']:.3f}, "
          f"test PR-AUC {test_metrics['pr_auc']:.3f}")


def run_heads(df, seeds, out_dir, embed_model=EMBED_MODEL):
    """Fit the logistic-regression head on cached embeddings (CPU only, no torch import)."""
    y = df[dh.TARGET].to_numpy()
    emb = load_embeddings(embed_model, df["job_id"].to_numpy())
    print(f"embeddings: {emb.shape}")
    for seed in seeds:
        # The saved split for this seed.
        splits = get_splits(df, seed)
        print_splits(splits, y, seed)
        tr, va = splits["train"], splits["val"]

        # Logistic regression, the same classifier as TF-IDF logreg, so the pair compares the
        # text representation only (skipped when its result files exist).
        if rh.result_exists(out_dir, "emb_logreg", seed):
            print(f"skip emb_logreg seed {seed}: result files already exist")
        else:
            start = time.time()
            model, params = fit_logreg_head(emb[tr], y[tr], emb[va], y[va], seed)
            save_head_result(out_dir, "emb_logreg", seed, model, params, emb, y, splits, start)


def run_bert(df, seeds, out_dir, device, bert_model=BERT_MODEL, epochs=3, batch_size=16,
             max_len=256, lr=2e-5, log_every=100, explain_seed=0):
    """Fine-tune the encoder per seed; tokenise once per seed and evaluate test once."""
    y = df[dh.TARGET].to_numpy()
    name = "bert_" + bert_model.replace("/", "_")
    for seed in seeds:
        # Resume support: skip seeds that already have result files.
        if rh.result_exists(out_dir, name, seed):
            print(f"skip {name} seed {seed}: result files already exist")
            continue

        # The saved split for this seed.
        splits = get_splits(df, seed)
        print_splits(splits, y, seed)
        tr, va, te = splits["train"], splits["val"], splits["test"]

        # Fresh pretrained model per seed, and the texts tokenised once per split.
        start = time.time()
        print(f"loading {bert_model} (downloads on the first run)...", flush=True)
        dh.seed_everything(seed)  # the new classification head is randomly initialised here
        model, tokenizer = fth.load_model_and_tokenizer(bert_model)
        print("tokenising...", flush=True)
        enc_tr = fth.encode_texts(tokenizer, df["text"].iloc[tr].tolist(), max_len)
        enc_va = fth.encode_texts(tokenizer, df["text"].iloc[va].tolist(), max_len)
        enc_te = fth.encode_texts(tokenizer, df["text"].iloc[te].tolist(), max_len)

        # Loaders: only the training loader shuffles.
        train_loader = fth.make_loader(enc_tr, y[tr], batch_size, True, seed)
        val_loader = fth.make_loader(enc_va, y[va], batch_size, False, seed)
        test_loader = fth.make_loader(enc_te, y[te], batch_size, False, seed)

        # Class weights: sklearn's "balanced" weights, square-rooted so the loss is only
        # mildly tilted towards fraud (the threshold is tuned on validation anyway).
        balanced = compute_class_weight("balanced", classes=np.array([0, 1]), y=y[tr])

        # Fine-tune; the best epoch on validation PR-AUC is restored.
        print("fine-tuning...", flush=True)
        model, history = fth.fine_tune(
            model, train_loader, val_loader, y[va], device, epochs=epochs, lr=lr,
            class_weight=np.sqrt(balanced), seed=seed, log_every=log_every
        )

        # Scores, then metrics at the validation-chosen threshold (test is scored once).
        score_val = fth.predict_proba(model, val_loader, device)
        score_test = fth.predict_proba(model, test_loader, device)
        thr, val_metrics, test_metrics = evaluate_scores(y, splits, score_val, score_test)
        # A separate ordered loader keeps the train scores aligned with y_train.
        train_eval_loader = fth.make_loader(enc_tr, y[tr], batch_size, False, seed)
        score_train = fth.predict_proba(model, train_eval_loader, device)

        seconds = time.time() - start  # training and scoring only, not the explanation below

        # Model explanation on one seed only: the fine-tuned weights are not saved, so it is done
        # here, before the results are saved (a crash here then reruns this seed instead of skipping it).
        if seed == explain_seed:
            explain_bert(df.iloc[te], y[te], score_test, thr, model, tokenizer, device, max_len,
                         os.path.join(out_dir, "explain", f"{name}_seed{seed}.json"))

        # Save the shared json + npz pair (with the per-epoch history) and report.
        rh.save_result(
            out_dir, name, "deep", seed, thr, val_metrics, test_metrics, y[va], score_val,
            y[te], score_test,
            train_pr_auc=average_precision_score(y[tr], score_train),
            best_params={"model_name": bert_model, "lr": lr, "epochs": epochs,
                         "batch_size": batch_size, "max_len": max_len,
                         "class_weight": "sqrt"},
            seconds=seconds,
            notes="fine-tuned encoder with sqrt class weights; best epoch on validation",
            history=history)
        print(f"{name} seed {seed}: val PR-AUC {val_metrics['pr_auc']:.3f}, "
              f"test PR-AUC {test_metrics['pr_auc']:.3f}, {time.time() - start:.0f}s")


def explain_bert(test_df, y_test, score_test, thr, model, tokenizer, device, max_len, path):
    """Integrated Gradients for 10 test ads: 4 caught frauds, 3 missed frauds, 3 false alarms."""
    order = np.argsort(-score_test)  # most fraud-like first
    fraud = [i for i in order if y_test[i] == 1]
    real = [i for i in order if y_test[i] == 0]
    picks = ([("caught fraud", i) for i in fraud[:4]]
             + [("missed fraud", i) for i in fraud[::-1][:3]]
             + [("false alarm", i) for i in real[:3]])

    rows = []
    for kind, i in picks:
        tokens, attr = eh.bert_token_attributions(model, tokenizer, test_df["text"].iloc[i],
                                                  device, max_len=max_len)
        rows.append({"kind": kind, "job_id": int(test_df["job_id"].iloc[i]),
                     "label": int(y_test[i]), "score": float(score_test[i]),
                     "flagged": bool(score_test[i] >= thr),
                     "tokens": tokens, "attributions": attr.tolist()})
    os.makedirs(os.path.dirname(path), exist_ok=True)  # own folder: result globs must not see it
    with open(path, "w") as f:
        json.dump(rows, f)
    print(f"explanations for {len(rows)} test ads saved to {path}")


def run_zero_shot(df, seeds, out_dir, device, llm_model=LLM_MODEL, batch_size=16, max_tokens=512):
    """Score every ad once with a zero-shot LLM (cached), then evaluate on each seed's split."""
    y = df[dh.TARGET].to_numpy()
    name = "zeroshot_" + llm_model.split("/")[-1]
    cache = os.path.join(EMBED_CACHE, name + "_scores.npz")

    # Nothing is trained, so the scores do not depend on the split: compute them once.
    if os.path.exists(cache):
        z = np.load(cache)
        assert np.array_equal(z["ids"], df["job_id"].to_numpy()), "cached scores belong to other rows"
        scores, seconds = z["scores"], float(z["seconds"])
    else:
        start = time.time()
        scores = deep_helper.zero_shot_scores(df["text"].tolist(), llm_model, device,
                                              batch_size=batch_size, max_tokens=max_tokens)
        seconds = time.time() - start
        os.makedirs(EMBED_CACHE, exist_ok=True)
        np.savez(cache, ids=df["job_id"].to_numpy(), scores=scores, seconds=seconds)

    # Per seed: threshold on validation, test scored once, same files as every other model.
    for seed in seeds:
        if rh.result_exists(out_dir, name, seed):
            print(f"skip {name} seed {seed}: result files already exist")
            continue
        splits = get_splits(df, seed)
        tr, va, te = splits["train"], splits["val"], splits["test"]
        thr, val_metrics, test_metrics = evaluate_scores(y, splits, scores[va], scores[te])
        rh.save_result(
            out_dir, name, "deep", seed, thr, val_metrics, test_metrics, y[va], scores[va],
            y[te], scores[te], train_pr_auc=average_precision_score(y[tr], scores[tr]),
            best_params={"model_name": llm_model, "max_tokens": max_tokens},
            seconds=seconds, notes="zero-shot Yes/No prompt, no training; threshold on validation")
        print(f"{name} seed {seed}: val PR-AUC {val_metrics['pr_auc']:.3f}, "
              f"test PR-AUC {test_metrics['pr_auc']:.3f}")
