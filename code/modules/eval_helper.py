"""Shared scoring, threshold selection, and imbalance-aware evaluation."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)


def score_of(model, X):
    """Positive-class score: predict_proba if available, else decision_function."""
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1])
    return np.asarray(model.decision_function(X))


def evaluate(y, score, thr):
    """Thresholded metrics plus average precision.

    The threshold is explicit because probability and SVM decision scores use different scales.
    """
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    pred = (score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "pr_auc": float(average_precision_score(y, score)),
    }


def pick_threshold(y_val, score_val):
    """Maximize F1 with integer ratios, breaking exact ties toward fewer alarms."""
    y = np.asarray(y_val).astype(int)
    s = np.asarray(score_val, dtype=float)
    _, _, thresholds = precision_recall_curve(y, s)
    positives = int(y.sum())
    best_num, best_den, best_alerts = 0, 1, len(y) + 1
    best_threshold = float(thresholds[0])
    for threshold in thresholds:
        predicted = s >= threshold
        tp = int(np.sum(predicted & (y == 1)))
        fp = int(np.sum(predicted & (y == 0)))
        fn = positives - tp
        alerts = tp + fp
        denominator = 2 * tp + fp + fn
        if denominator == 0:
            continue
        # Compare exact integer ratios instead of floating-point F1 values.
        if 2 * tp * best_den > best_num * denominator or (
            2 * tp * best_den == best_num * denominator and alerts < best_alerts
        ):
            best_num, best_den = 2 * tp, denominator
            best_alerts, best_threshold = alerts, float(threshold)
    return best_threshold
