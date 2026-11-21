"""Deep run on the fake job postings (called from notebook 03): fine-tuning DistilBERT and
explaining it with Integrated Gradients.

The heavy libraries (torch, transformers, captum) are imported inside the functions that need
them, so this module imports on a laptop without them.
Results land in out_dir as <model>_seed<N>.json and <model>_seed<N>.npz.
"""

import json
import os
import time

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.utils.class_weight import compute_class_weight

from modules import data_helper as dh
from modules import eval_helper as ev
from modules import explain_helper as eh
from modules import finetune_helper as fth
from modules import results_helper as rh
from modules.classical_runs import get_splits

BERT_MODEL = "distilbert-base-uncased"


def print_splits(splits, y, seed):
    """Print size and class balance of every split, as the course requires."""
    print(f"seed {seed}:")
    for name in ("train", "val", "test"):
        idx = splits[name]
        print(f"  {name:5s} n={len(idx):5d} positives={int(y[idx].sum()):4d} "
              f"rate={y[idx].mean():.4f}")


def evaluate_scores(y, splits, score_val, score_test):
    """Fix the threshold on validation, then evaluate validation and test at that threshold."""
    # Threshold from validation only; then both splits are scored with that one number.
    thr = ev.pick_threshold(y[splits["val"]], score_val)
    val_metrics = ev.evaluate(y[splits["val"]], score_val, thr)
    test_metrics = ev.evaluate(y[splits["test"]], score_test, thr)
    return thr, val_metrics, test_metrics


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
