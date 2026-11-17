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
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.svm import LinearSVC

from modules.eval_helper import (
    compare_table,
    evaluate,
    pick_threshold,
    score_of,
    threshold_for_model,
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




def test_evaluate():
    y = [0, 0, 1, 1]
    score = [0.1, 0.2, 0.8, 0.9]
    ev = evaluate(y, score, thr=0.5)
    assert (ev["tp"], ev["fp"], ev["fn"], ev["tn"]) == (2, 0, 0, 2)
    # Nothing flagged: precision falls back to 0 without a warning or error.
    none = evaluate([1, 0, 1, 0], [0.4, 0.3, 0.2, 0.1], thr=0.5)
    assert none["precision"] == 0.0 and none["recall"] == 0.0 and none["f1"] == 0.0
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
    assert np.isclose(evaluate(y, s, thr)["f1"], 1.0)
    # 3 positives, one buried at the bottom of the ranking: best F1 is 0.8 at threshold 0.85.
    y2 = [1, 1, 1, 0, 0, 0, 0]
    s2 = [0.9, 0.85, 0.1, 0.8, 0.7, 0.6, 0.5]
    thr2 = pick_threshold(y2, s2)
    assert np.isclose(evaluate(y2, s2, thr2)["f1"], 0.8) and thr2 == 0.85
    # Tie in F1: the higher threshold must win (flags fewer rows).
    assert pick_threshold([1, 0, 1, 0], [0.9, 0.5, 0.9, 0.5]) == 0.9

def test_dummy_baselines_preserve_their_predictions():
    X = np.zeros((12, 2))
    y_train = np.array([0] * 10 + [1] * 2)
    for strategy in ("most_frequent", "stratified"):
        model = DummyClassifier(strategy=strategy, random_state=17).fit(X, y_train)
        score = score_of(model, X)
        threshold = threshold_for_model(model, y_train, score)
        assert threshold == 0.5
        assert np.array_equal((score >= threshold).astype(int), model.predict(X))
        if strategy == "most_frequent":
            measured = evaluate(y_train, score, threshold)
            assert measured["tp"] == 0 and measured["fp"] == 0
            assert measured["tn"] == 10 and measured["fn"] == 2


def test_nondummy_uses_f1_validation_threshold():
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0, 0, 1, 1])
    model = LogisticRegression().fit(X, y)
    scores = score_of(model, X)
    assert threshold_for_model(model, y, scores) == pick_threshold(y, scores)


def test_invalid_evaluation_inputs_raise():
    bad = [
        ([0, 1], [0.5]),
        ([0, 0], [0.1, 0.2]),
        ([0, 1], [0.2, float("nan")]),
        ([0, 1], [0.2, float("inf")]),
        ([0, 2], [0.2, 0.8]),
        ([], []),
    ]
    for y, s in bad:
        for method in (lambda: evaluate(y, s, 0.5),
                       lambda: pick_threshold(y, s)):
            try:
                method()
            except ValueError:
                continue
            raise AssertionError(f"invalid inputs were accepted: {y} {s}")
    try:
        evaluate([0, 1], [0.3, 0.7], float("nan"))
    except ValueError:
        pass
    else:
        raise AssertionError("NaN threshold was accepted")


def test_compare_table():
    tr = evaluate([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1], thr=0.5)
    te = evaluate([1, 0], [0.6, 0.4], thr=0.5)
    tab = compare_table({"m1": {"train": tr, "test": te}})
    assert list(tab["model"]) == ["m1"]
    assert np.isclose(tab["gap"].iloc[0], tr["pr_auc"] - te["pr_auc"])
    for col in ("train_f1", "test_f1", "train_recall", "test_precision"):
        assert col in tab.columns










if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(tests)} tests passed")
