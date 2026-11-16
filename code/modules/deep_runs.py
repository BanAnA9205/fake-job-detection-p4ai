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
from sklearn.neural_network import MLPClassifier

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


def resolve_device(name):
    """"auto" means cuda when it exists, so the same command works on laptop and GPU box."""
    if name != "auto":
        return name
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def print_run_info(device, batch_size, max_len):
    """Report the device and batch settings before any heavy work starts."""
    print(f"device: {device}")
    if str(device).startswith("cuda"):
        import torch

        free, total = torch.cuda.mem_get_info()
        print(f"gpu: {torch.cuda.get_device_name(0)}; free {free / 1e9:.1f} of {total / 1e9:.1f} GB")
    print(f"batch size: {batch_size}; max sequence length: {max_len}")


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
    if not (os.path.exists(emb_path) and os.path.exists(ids_path)):
        raise FileNotFoundError(f"no embeddings for {embed_model} in {cache_dir}/; run --stage embed first")
    assert np.array_equal(np.load(ids_path, allow_pickle=True), np.asarray(job_ids)), (
        "cached embeddings belong to different rows; rerun --stage embed"
    )
    return np.load(emb_path)


def evaluate_scores(y, splits, score_val, score_test):
    """Fix the threshold on validation, then evaluate validation and test at that threshold."""
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
        model = LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=seed)
        model.fit(Xtr, ytr)
        auc = float(average_precision_score(yva, ev.score_of(model, Xva)))
        if auc > best_auc:
            best_auc, best_model, best_params = auc, model, {"C": C}
    return best_model, best_params


def fit_mlp_head(Xtr, ytr, seed):
    """sklearn MLP on embeddings; early_stopping holds out its own slice of train."""
    model = MLPClassifier(hidden_layer_sizes=(128,), early_stopping=True, n_iter_no_change=5,
                          max_iter=100, random_state=seed)
    model.fit(Xtr, ytr)
    return model, {"hidden_layer_sizes": [128]}


def save_head_result(out_dir, name, seed, model, params, X, y, splits, start):
    """Score a fitted head, evaluate it, and write the result files."""
    tr, va, te = splits["train"], splits["val"], splits["test"]
    score_val = ev.score_of(model, X[va])
    score_test = ev.score_of(model, X[te])
    thr, val_metrics, test_metrics = evaluate_scores(y, splits, score_val, score_test)
    rh.save_result(
        out_dir, name, "deep", seed, thr, val_metrics, test_metrics, y[va], score_val,
        y[te], score_test,
        train_pr_auc=average_precision_score(y[tr], ev.score_of(model, X[tr])),
        best_params=params, seconds=time.time() - start,
        notes="frozen embeddings with a sklearn head; threshold on validation")
    print(f"{name} seed {seed}: val PR-AUC {val_metrics['pr_auc']:.3f}, "
          f"test PR-AUC {test_metrics['pr_auc']:.3f}")


def run_heads(df, seeds, out_dir, embed_model=EMBED_MODEL):
    """Fit the two sklearn heads on cached embeddings (CPU only, no torch import)."""
    y = df[dh.TARGET].to_numpy()
    emb = load_embeddings(embed_model, df["job_id"].to_numpy())
    print(f"embeddings: {emb.shape}")
    for seed in seeds:
        splits = get_splits(df, seed)
        print_splits(splits, y, seed)
        tr, va = splits["train"], splits["val"]
        if rh.result_exists(out_dir, "emb_logreg", seed):
            print(f"skip emb_logreg seed {seed}: result files already exist")
        else:
            start = time.time()
            model, params = fit_logreg_head(emb[tr], y[tr], emb[va], y[va], seed)
            save_head_result(out_dir, "emb_logreg", seed, model, params, emb, y, splits, start)
        if rh.result_exists(out_dir, "emb_mlp", seed):
            print(f"skip emb_mlp seed {seed}: result files already exist")
        else:
            start = time.time()
            model, params = fit_mlp_head(emb[tr], y[tr], seed)
            save_head_result(out_dir, "emb_mlp", seed, model, params, emb, y, splits, start)


def run_bert(df, seeds, out_dir, device, bert_model=BERT_MODEL, epochs=3, batch_size=16,
             max_len=256, lr=2e-5):
    """Fine-tune the encoder per seed; tokenise once per seed and evaluate test once."""
    y = df[dh.TARGET].to_numpy()
    name = "bert_" + bert_model.replace("/", "_")
    for seed in seeds:
        if rh.result_exists(out_dir, name, seed):
            print(f"skip {name} seed {seed}: result files already exist")
            continue
        splits = get_splits(df, seed)
        print_splits(splits, y, seed)
        tr, va, te = splits["train"], splits["val"], splits["test"]
        start = time.time()
        model, tokenizer = fth.load_model_and_tokenizer(bert_model)
        enc_tr = fth.encode_texts(tokenizer, df["text"].iloc[tr].tolist(), max_len)
        enc_va = fth.encode_texts(tokenizer, df["text"].iloc[va].tolist(), max_len)
        enc_te = fth.encode_texts(tokenizer, df["text"].iloc[te].tolist(), max_len)
        train_loader = fth.make_loader(enc_tr, y[tr], batch_size, True, seed)
        val_loader = fth.make_loader(enc_va, y[va], batch_size, False, seed)
        test_loader = fth.make_loader(enc_te, y[te], batch_size, False, seed)
        model, history = fth.fine_tune(
            model, train_loader, val_loader, y[va], device, epochs=epochs, lr=lr,
            class_weight=fth.class_weights(y[tr], mode="sqrt"), seed=seed,
        )
        score_val = fth.predict_proba(model, val_loader, device)
        score_test = fth.predict_proba(model, test_loader, device)
        thr, val_metrics, test_metrics = evaluate_scores(y, splits, score_val, score_test)
        # A separate ordered loader keeps the train scores aligned with y_train.
        train_eval_loader = fth.make_loader(enc_tr, y[tr], batch_size, False, seed)
        score_train = fth.predict_proba(model, train_eval_loader, device)
        rh.save_result(
            out_dir, name, "deep", seed, thr, val_metrics, test_metrics, y[va], score_val,
            y[te], score_test,
            train_pr_auc=average_precision_score(y[tr], score_train),
            best_params={"model_name": bert_model, "lr": lr, "epochs": epochs,
                         "batch_size": batch_size, "max_len": max_len,
                         "class_weight": "sqrt"},
            seconds=time.time() - start,
            notes="fine-tuned encoder with sqrt class weights; best epoch on validation",
            history=history)
        print(f"{name} seed {seed}: val PR-AUC {val_metrics['pr_auc']:.3f}, "
              f"test PR-AUC {test_metrics['pr_auc']:.3f}, {time.time() - start:.0f}s")
