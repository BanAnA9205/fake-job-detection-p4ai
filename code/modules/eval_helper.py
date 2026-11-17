"""Evaluation helpers: scores, metrics at a threshold, threshold choice and a comparison table."""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def score_of(model, X):
    """Positive-class score: predict_proba if available, else decision_function."""
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1])
    return np.asarray(model.decision_function(X))


def _validated_binary_scores(y, score):
    """Reject malformed evaluation inputs rather than silently reporting misleading metrics."""
    y = np.asarray(y)
    score = np.asarray(score)
    if y.ndim != 1 or score.ndim != 1 or len(y) == 0 or len(y) != len(score):
        raise ValueError("y and score must be nonempty, equally sized 1-D arrays")
    if not np.isin(y, [0, 1]).all() or np.unique(y).size != 2:
        raise ValueError("binary evaluation requires both labels 0 and 1")
    if not np.issubdtype(score.dtype, np.number) or not np.isfinite(score).all():
        raise ValueError("scores must be finite numeric values")
    return y.astype(int), score.astype(float)


def evaluate(y, score, thr):
    """Thresholded metrics plus PR-AUC and ROC-AUC.

    thr is required: probability scores and SVM decision scores need
    different thresholds, so the caller must state one.
    """
    y, score = _validated_binary_scores(y, score)
    if not np.isfinite(thr):
        raise ValueError("classification threshold must be finite")

    # Hard predictions at this threshold and their confusion counts.
    pred = (score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    # zero_division=0: a model that flags nothing gets precision 0 instead of a warning.
    return {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        # Threshold-free ranking metrics.
        "pr_auc": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
    }


def pick_threshold(y_val, score_val):
    """Threshold that maximises F1, chosen on VALIDATION scores only.

    On ties the HIGHER threshold wins, which flags fewer rows at the same F1.
    """
    y, s = _validated_binary_scores(y_val, score_val)

    # Precision and recall at every distinct score. The last point (precision 1,
    # recall 0) has no threshold, so it is dropped.
    precision, recall, thresholds = precision_recall_curve(y, s)
    precision, recall = precision[:-1], recall[:-1]

    # F1 at each threshold (0 where precision and recall are both 0).
    denom = np.where(precision + recall > 0, precision + recall, 1.0)
    f1 = 2 * precision * recall / denom

    # Thresholds come in ascending order, so the last maximum is the highest threshold.
    best = len(f1) - 1 - int(np.argmax(f1[::-1]))
    return float(thresholds[best])


def threshold_for_model(model, y_val, score_val):
    """Keep DummyClassifier's prescribed predictions; tune real models on validation.

    Optimizing F1 for a constant-score dummy would turn a majority-class
    (all-real) predictor into an all-fraud predictor whenever the threshold is
    equal to its constant score. 0.5 preserves both most_frequent and stratified
    dummy predictions, whose class probabilities are zero or one.
    """
    if isinstance(model, DummyClassifier):
        return 0.5
    return pick_threshold(y_val, score_val)


def compare_table(results):
    """One row per model comparing train and test metrics.

    results maps model name to {"train": <evaluate dict>, "test": <evaluate dict>}.
    gap is train PR-AUC minus test PR-AUC, i.e. the generalization gap.
    """
    # One row per model; "gap" is the generalisation gap in PR-AUC.
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
