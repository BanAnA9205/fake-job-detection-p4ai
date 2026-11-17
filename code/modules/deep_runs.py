"""Deep runs on the fake job postings (called from notebook 03): frozen MiniLM embeddings with
sklearn heads, and fine-tuning of a pretrained encoder.

The heavy libraries (torch, transformers, sentence-transformers) are imported inside the
functions that need them, so this module imports on a laptop without them.
Results land in out_dir as <model>_seed<N>.json and <model>_seed<N>.npz.
"""

import os
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_class_weight

from modules import data_helper as dh
from modules import deep_helper
from modules import eval_helper as ev
from modules import finetune_helper as fth
from modules import results_helper as rh
from modules.classical_runs import get_splits

EMBED_CACHE = "features"
LOGREG_CS = [0.1, 1.0, 10.0]
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BERT_MODEL = "distilbert-base-uncased"


def print_splits(splits, y, seed):
    """Print size and class balance of every split, as the course requires."""
    print(f"seed {seed}:")
    for name in ("train", "val", "test"):
        idx = splits[name]
        print(f"  {name:5s} n={len(idx):5d} positives={int(y[idx].sum()):4d} "
              f"rate={y[idx].mean():.4f}")


def load_embeddings(embed_model, job_ids, texts, cache_dir=EMBED_CACHE):
    """Load embeddings only if the ordered text, ids and encoder settings match."""
    return deep_helper.load_cached_embeddings(
        texts, job_ids, model_name=embed_model, cache_dir=cache_dir,
    )


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


def fit_logreg_head(Xtr, ytr, seed, groups=None):
    """Select C with CV on TRAIN only; leave validation for threshold selection.

    Group-aware CV is used when group IDs are provided (as in the real runs).
    This prevents using the same validation labels to tune C and tune threshold.
    """
    if groups is None:
        splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    else:
        splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    best_auc, best_model, best_params = -np.inf, None, {}
    for C in LOGREG_CS:
        model = LogisticRegression(C=C, class_weight="balanced",
                                   max_iter=2000, random_state=seed)
        auc = float(cross_val_score(
            model, Xtr, ytr, groups=groups, cv=splitter,
            scoring="average_precision", error_score="raise",
        ).mean())
        if auc > best_auc:
            best_auc, best_model = auc, model
            best_params = {"C": C, "train_cv_ap": auc, "train_cv_folds": 3}
    best_model.fit(Xtr, ytr)
    return best_model, best_params


def fit_mlp_head(Xtr, ytr, seed):
    """MLP on embeddings; stop on training loss, without an ungrouped holdout."""
    model = MLPClassifier(hidden_layer_sizes=(128,), early_stopping=False, n_iter_no_change=5,
                          max_iter=100, random_state=seed)
    model.fit(Xtr, ytr)
    return model, {"hidden_layer_sizes": [128]}


def save_head_result(out_dir, name, seed, model, params, X, y, splits, start,
                     run_signature=None):
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
        notes="frozen embeddings; training-only hyperparameter/early-stop; threshold on validation",
        run_signature=run_signature)
    print(f"{name} seed {seed}: val PR-AUC {val_metrics['pr_auc']:.3f}, "
          f"test PR-AUC {test_metrics['pr_auc']:.3f}")


def run_heads(df, seeds, out_dir, embed_model=EMBED_MODEL):
    """Fit the two sklearn heads on cached embeddings (CPU only, no torch import)."""
    y = df[dh.TARGET].to_numpy()
    emb = load_embeddings(embed_model, df["job_id"].to_numpy(), df["text"].tolist())
    print(f"embeddings: {emb.shape}")
    data_hash = rh.dataset_fingerprint(df)
    sources = (dh.__file__, deep_helper.__file__, ev.__file__, rh.__file__, __file__)
    for seed in seeds:
        # The saved split for this seed.
        splits = get_splits(df, seed)
        print_splits(splits, y, seed)
        tr, va = splits["train"], splits["val"]

        # Head 1: logistic regression (skipped when its result files exist).
        lr_signature = rh.experiment_signature(data_hash, {
            "family": "embedding", "head": "emb_logreg", "seed": seed,
            "encoder": embed_model, "C_candidates": LOGREG_CS,
            "cv": "3-fold grouped train", "max_seq_length": 256,
        }, sources)
        if rh.result_exists(out_dir, "emb_logreg", seed, expected_signature=lr_signature):
            print(f"skip emb_logreg seed {seed}: result files already exist")
        else:
            start = time.time()
            model, params = fit_logreg_head(
                emb[tr], y[tr], seed, groups=df["group"].to_numpy()[tr])
            save_head_result(out_dir, "emb_logreg", seed, model, params, emb, y,
                             splits, start, run_signature=lr_signature)

        # Head 2: small MLP.
        mlp_signature = rh.experiment_signature(data_hash, {
            "family": "embedding", "head": "emb_mlp", "seed": seed,
            "encoder": embed_model, "max_seq_length": 256,
            "hidden_layer_sizes": [128], "early_stopping": False,
        }, sources)
        if rh.result_exists(out_dir, "emb_mlp", seed, expected_signature=mlp_signature):
            print(f"skip emb_mlp seed {seed}: result files already exist")
        else:
            start = time.time()
            model, params = fit_mlp_head(emb[tr], y[tr], seed)
            save_head_result(out_dir, "emb_mlp", seed, model, params, emb, y,
                             splits, start, run_signature=mlp_signature)


def run_bert(df, seeds, out_dir, device, bert_model=BERT_MODEL, epochs=3, batch_size=16,
             max_len=256, lr=2e-5, log_every=100):
    """Fine-tune the encoder per seed; tokenise once per seed and evaluate test once."""
    y = df[dh.TARGET].to_numpy()
    name = "bert_" + bert_model.replace("/", "_")
    data_hash = rh.dataset_fingerprint(df)
    sources = (dh.__file__, ev.__file__, rh.__file__, fth.__file__, __file__)
    for seed in seeds:
        signature = rh.experiment_signature(data_hash, {
            "family": "fine_tune", "seed": seed, "encoder": bert_model,
            "epochs": epochs, "batch_size": batch_size, "max_len": max_len,
            "lr": lr, "class_weight": "sqrt", "optimizer": "AdamW",
        }, sources)
        # Never accept results from mismatched code, data or training settings.
        if rh.result_exists(out_dir, name, seed, expected_signature=signature):
            print(f"skip {name} seed {seed}: result files already exist")
            continue

        # The saved split for this seed.
        splits = get_splits(df, seed)
        print_splits(splits, y, seed)
        tr, va, te = splits["train"], splits["val"], splits["test"]

        # Fresh pretrained model per seed, and the texts tokenised once per split.
        start = time.time()
        print(f"loading {bert_model} (downloads on the first run)...", flush=True)
        dh.seed_everything(seed)  # also seeds the newly initialized classification head
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

        # Save the shared json + npz pair (with the per-epoch history) and report.
        rh.save_result(
            out_dir, name, "deep", seed, thr, val_metrics, test_metrics, y[va], score_val,
            y[te], score_test,
            train_pr_auc=average_precision_score(y[tr], score_train),
            best_params={"model_name": bert_model, "lr": lr, "epochs": epochs,
                         "batch_size": batch_size, "max_len": max_len,
                         "class_weight": "sqrt"},
            seconds=time.time() - start,
            notes="fine-tuned encoder with sqrt class weights; best epoch on validation",
            history=history, run_signature=signature)
        print(f"{name} seed {seed}: val PR-AUC {val_metrics['pr_auc']:.3f}, "
              f"test PR-AUC {test_metrics['pr_auc']:.3f}, {time.time() - start:.0f}s")
