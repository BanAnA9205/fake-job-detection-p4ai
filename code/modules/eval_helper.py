"""Evaluation helpers: scores, metrics, threshold choice, tables and plots."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)


def score_of(model, X):
    """Positive-class score: predict_proba if available, else decision_function."""
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1])
    return np.asarray(model.decision_function(X))


def metrics_at_threshold(y, score, thr):
    """Precision, recall, F1 and the confusion counts at one threshold."""
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    assert y.shape == score.shape, "y and score must have the same shape"
    pred = (score >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def evaluate(y, score, thr):
    """Thresholded metrics plus PR-AUC and ROC-AUC.

    thr is required: probability scores and SVM decision scores need
    different thresholds, so the caller must state one.
    """
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    out = metrics_at_threshold(y, score, thr)
    out["pr_auc"] = float(average_precision_score(y, score))
    out["roc_auc"] = float(roc_auc_score(y, score))
    return out


def pick_threshold(y_val, score_val, strategy="max_f1", min_recall=None):
    """Pick a threshold on VALIDATION scores only (default: maximize F1).

    If min_recall is given, only thresholds that reach that recall qualify.
    """
    if strategy != "max_f1":
        raise ValueError(f"unknown strategy: {strategy}")
    y = np.asarray(y_val).astype(int)
    s = np.asarray(score_val, dtype=float)
    assert y.shape == s.shape, "y_val and score_val must have the same shape"
    best_thr, best_f1 = None, -1.0
    # Predictions only change at the observed score values, so those are the
    # only candidate thresholds. On ties >= keeps the HIGHER threshold, which
    # flags fewer rows at the same F1.
    for thr in np.unique(s):
        m = metrics_at_threshold(y, s, thr)
        if min_recall is not None and m["recall"] < min_recall:
            continue
        if m["f1"] >= best_f1:
            best_f1, best_thr = m["f1"], float(thr)
    if best_thr is None:
        raise ValueError("no threshold reaches the requested min_recall")
    return best_thr


def sweep_thresholds(y, score, n=50):
    """Precision, recall and flagged count at n evenly spaced thresholds."""
    y = np.asarray(y).astype(int)
    s = np.asarray(score, dtype=float)
    assert y.shape == s.shape, "y and score must have the same shape"
    rows = []
    for thr in np.linspace(float(s.min()), float(s.max()), n):
        m = metrics_at_threshold(y, s, thr)
        rows.append(
            {
                "threshold": float(thr),
                "precision": m["precision"],
                "recall": m["recall"],
                "flagged": m["tp"] + m["fp"],
            }
        )
    return pd.DataFrame(rows)


def compare_table(results):
    """One row per model comparing train and test metrics.

    results maps model name to {"train": <evaluate dict>, "test": <evaluate dict>}.
    gap is train PR-AUC minus test PR-AUC, i.e. the generalization gap.
    """
    rows = []
    for name, res in results.items():
        tr, te = res["train"], res["test"]
        rows.append(
            {
                "model": name,
                "train_pr_auc": tr["pr_auc"],
                "test_pr_auc": te["pr_auc"],
                "train_f1": tr["f1"],
                "test_f1": te["f1"],
                "train_recall": tr["recall"],
                "test_recall": te["recall"],
                "train_precision": tr["precision"],
                "test_precision": te["precision"],
                "gap": tr["pr_auc"] - te["pr_auc"],
            }
        )
    return pd.DataFrame(rows)


def aggregate_seeds(rows):
    """Mean and std of every numeric metric per model across split seeds.

    rows is a list of dicts, each with a "model" key plus numeric metrics,
    for example compare_table(...).to_dict("records") from one seed.
    """
    df = pd.DataFrame(rows)
    assert "model" in df.columns, "each row needs a 'model' key"
    num_cols = [
        c for c in df.columns if c not in ("model", "seed")
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    agg = df.groupby("model")[num_cols].agg(["mean", "std"])
    # Flatten the two-level columns to "<metric>_mean" / "<metric>_std".
    agg.columns = [f"{c}_{stat}" for c, stat in agg.columns]
    return agg.reset_index()


def plot_confusions(cm_by_name, class_names, n_cols=3, title="Confusion matrices"):
    """Grid of annotated confusion-matrix heatmaps, one per model. Returns the Figure."""
    names = list(cm_by_name)
    assert names, "cm_by_name is empty"
    n_rows = int(np.ceil(len(names) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), squeeze=False
    )
    for ax, name in zip(axes.ravel(), names):
        cm = np.asarray(cm_by_name[name])
        ax.imshow(cm, cmap="Blues")
        ax.set_title(name)
        ax.set_xticks(range(len(class_names)), labels=class_names)
        ax.set_yticks(range(len(class_names)), labels=class_names)
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                )
    for ax in axes.ravel()[len(names):]:
        ax.set_visible(False)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_pr_curves(curves_by_name, pos_rate=None, title="Precision-recall curves"):
    """Overlay precision-recall curves. Returns the Figure, never shows it.

    curves_by_name maps a model name to (precision, recall) arrays,
    as returned by sklearn's precision_recall_curve.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (prec, rec) in curves_by_name.items():
        ax.step(rec, prec, where="post", label=name)
    if pos_rate is not None:
        # A random classifier stays flat at the positive rate, so that is the
        # baseline every curve should beat.
        ax.axhline(pos_rate, linestyle="--", color="grey", label="baseline")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(-0.02, 1.02)
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    return fig


def group_breakdown(y, pred, groups):
    """Count, positives, recall and precision for each value of a grouping column."""
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    g = pd.Series(groups).reset_index(drop=True)
    assert len(g) == len(y) == len(pred), "y, pred and groups must align"
    rows = []
    for value, count in g.value_counts(dropna=False).sort_index().items():
        if pd.isna(value):
            mask = g.isna().to_numpy()
        else:
            mask = (g == value).to_numpy()
        yy, pp = y[mask], pred[mask]
        pos = int(yy.sum())
        tp = int(((pp == 1) & (yy == 1)).sum())
        pred_pos = int(pp.sum())
        rows.append(
            {
                "group": value,
                "count": int(count),
                "positives": pos,
                "recall": tp / pos if pos else np.nan,
                "precision": tp / pred_pos if pred_pos else np.nan,
            }
        )
    return pd.DataFrame(rows)
