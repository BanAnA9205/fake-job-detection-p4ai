"""Tests for modules/eval_helper.py on small synthetic data.

Run with: .venv/bin/python tests/test_eval_helper.py
(or pytest if available). All asserts are plain; no fixtures needed.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import matplotlib

matplotlib.use("Agg")  # tests run headless, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.svm import LinearSVC

from modules.eval_helper import (
    aggregate_seeds,
    compare_table,
    evaluate,
    group_breakdown,
    metrics_at_threshold,
    pick_threshold,
    plot_confusions,
    plot_pr_curves,
    score_of,
    sweep_thresholds,
)


def test_score_of_proba_and_decision():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    y = (X[:, 0] > 0).astype(int)
    assert y.sum() > 0 and y.sum() < 40, "both classes needed"
    lr = LogisticRegression().fit(X, y)
    svc = LinearSVC(dual="auto").fit(X, y)
    s1, s2 = score_of(lr, X), score_of(svc, X)
    assert s1.shape == (40,) and s2.shape == (40,)
    # Higher feature0 means class 1, so scores must rise with feature0.
    assert np.corrcoef(s2, X[:, 0])[0, 1] > 0.5


def test_metrics_at_threshold():
    y = [1, 0, 1, 0]
    score = [0.9, 0.8, 0.2, 0.1]
    m = metrics_at_threshold(y, score, thr=0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    assert np.isclose(m["precision"], 0.5)
    assert np.isclose(m["recall"], 0.5)
    assert np.isclose(m["f1"], 0.5)
    perfect = metrics_at_threshold([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1], thr=0.5)
    assert (perfect["tp"], perfect["fp"]) == (2, 0)
    assert np.isclose(perfect["f1"], 1.0)


def test_evaluate():
    y = [0, 0, 1, 1]
    score = [0.1, 0.2, 0.8, 0.9]
    ev = evaluate(y, score, thr=0.5)
    assert np.isclose(ev["pr_auc"], 1.0)
    assert np.isclose(ev["roc_auc"], 1.0)
    assert np.isclose(ev["f1"], 1.0)
    for key in ("precision", "recall", "f1", "tp", "fp", "fn", "tn"):
        assert key in ev
    # Random-ish scores: PR-AUC should sit near the positive rate.
    y2 = [1, 0, 1, 0, 1, 0, 1, 0]
    s2 = [0.5, 0.4, 0.6, 0.3, 0.2, 0.7, 0.1, 0.8]
    assert 0.0 <= evaluate(y2, s2, thr=0.5)["pr_auc"] <= 1.0


def test_pick_threshold():
    y = [1, 1, 0, 0]
    s = [0.9, 0.8, 0.2, 0.1]
    thr = pick_threshold(y, s)
    assert np.isclose(metrics_at_threshold(y, s, thr)["f1"], 1.0)
    # 3 positives, one of them buried at the bottom of the ranking: the plain
    # max-F1 threshold misses it, min_recall=1.0 must force a lower threshold.
    y2 = [1, 1, 1, 0, 0, 0, 0]
    s2 = [0.9, 0.85, 0.1, 0.8, 0.7, 0.6, 0.5]
    thr_plain = pick_threshold(y2, s2)
    assert np.isclose(metrics_at_threshold(y2, s2, thr_plain)["f1"], 0.8)
    thr_forced = pick_threshold(y2, s2, min_recall=1.0)
    assert np.isclose(metrics_at_threshold(y2, s2, thr_forced)["recall"], 1.0)
    try:
        pick_threshold(y2, s2, min_recall=1.5)
        raised = False
    except ValueError:
        raised = True
    assert raised, "impossible min_recall must raise"


def test_sweep_thresholds():
    y = [1, 0, 1, 0]
    s = [0.9, 0.8, 0.2, 0.1]
    tab = sweep_thresholds(y, s, n=5)
    assert list(tab.columns) == ["threshold", "precision", "recall", "flagged"]
    assert len(tab) == 5
    # Recall and flagged count can only shrink as the threshold rises.
    assert (np.diff(tab["recall"]) <= 1e-12).all()
    assert (np.diff(tab["flagged"]) <= 0).all()
    assert (tab["precision"].between(0, 1)).all()


def test_compare_table():
    tr = evaluate([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1], thr=0.5)
    te = evaluate([1, 0], [0.6, 0.4], thr=0.5)
    tab = compare_table({"m1": {"train": tr, "test": te}})
    assert list(tab["model"]) == ["m1"]
    assert np.isclose(tab["gap"].iloc[0], tr["pr_auc"] - te["pr_auc"])
    for col in ("train_f1", "test_f1", "train_recall", "test_precision"):
        assert col in tab.columns


def test_aggregate_seeds():
    rows = [
        {"model": "a", "f1": 0.5, "recall": 0.6},
        {"model": "a", "f1": 0.7, "recall": 0.8},
        {"model": "b", "f1": 0.2, "recall": 0.1},
    ]
    agg = aggregate_seeds(rows)
    a = agg[agg["model"] == "a"].iloc[0]
    assert np.isclose(a["f1_mean"], 0.6)
    assert np.isclose(a["f1_std"], np.sqrt(0.02))  # sample std, ddof=1
    b = agg[agg["model"] == "b"].iloc[0]
    assert np.isclose(b["recall_mean"], 0.1)
    assert np.isnan(b["recall_std"])  # single seed: std undefined
    assert {"f1_mean", "f1_std", "recall_mean", "recall_std"} <= set(agg.columns)


def test_plot_confusions():
    fig = plot_confusions(
        {"m1": np.array([[10, 2], [3, 5]]), "m2": np.array([[8, 4], [1, 7]])},
        class_names=["real", "fraud"],
    )
    assert isinstance(fig, plt.Figure)
    visible = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible) == 2
    plt.close(fig)


def test_plot_pr_curves():
    y = [1, 1, 0, 0, 1, 0]
    s = [0.9, 0.7, 0.3, 0.1, 0.5, 0.2]
    prec, rec, _ = precision_recall_curve(y, s)
    fig = plot_pr_curves({"m1": (prec, rec)}, pos_rate=0.5)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 1
    assert len(fig.axes[0].lines) >= 2  # curve + baseline
    plt.close(fig)


def test_group_breakdown():
    y = [1, 0, 1, 1, 0]
    pred = [1, 1, 0, 1, 0]
    groups = pd.Series(["x", "x", "y", "z", "z"])
    tab = group_breakdown(y, pred, groups)
    tab = tab.set_index("group")
    assert set(tab.index) == {"x", "y", "z"}
    x = tab.loc["x"]
    assert x["count"] == 2 and x["positives"] == 1
    assert np.isclose(x["recall"], 1.0) and np.isclose(x["precision"], 0.5)
    yrow = tab.loc["y"]
    assert yrow["recall"] == 0.0 and np.isnan(yrow["precision"])  # nothing predicted
    zrow = tab.loc["z"]
    assert np.isclose(zrow["recall"], 1.0) and np.isclose(zrow["precision"], 1.0)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(tests)} tests passed")
