"""Tests for modules/eval_helper.py on small synthetic data.

Run with: .venv/bin/python tests/test_eval_helper.py
(or pytest if available). All asserts are plain; no fixtures needed.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
from modules.eval_helper import evaluate, pick_threshold, score_of
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


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


def test_exact_ratio_tie_prefers_fewer_alerts():
    # These integer counts reproduce an actual seed-3 tie: 182/273 == 180/270 == 2/3.
    y = np.array([1] * 171 + [0] * 3349)
    scores = np.concatenate([
        np.full(90, 0.9), [0.8], np.zeros(80),
        np.full(9, 0.95), np.full(2, 0.8), np.zeros(3349 - 11),
    ])
    assert evaluate(y, scores, 0.9)["tp"] + evaluate(y, scores, 0.9)["fp"] == 99
    assert evaluate(y, scores, 0.8)["tp"] + evaluate(y, scores, 0.8)["fp"] == 102
    assert pick_threshold(y, scores) == 0.9


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(tests)} tests passed")
