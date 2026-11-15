"""Tests for modules/explain_helper.py on small synthetic data.

Run with: .venv/bin/python tests/test_explain_helper.py
(or pytest if available). Fits tiny models on ~60 synthetic rows only.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from modules.explain_helper import permutation_table, show_errors, top_linear_terms


def _make_data(n=60, seed=0):
    """Tiny synthetic job-posting table; only 'description' carries signal."""
    rng = np.random.default_rng(seed)
    fraud_words = ["earn", "cash", "weekly", "bonus"]
    real_words = ["team", "research", "engineer", "support"]
    rows = []
    for i in range(n):
        y = int(i % 3 == 0)  # 20 positives out of 60
        vocab = fraud_words if y else real_words
        rows.append(
            {
                "title": ("urgent money" if y else "role") + f" {i}",
                "description": " ".join(rng.choice(vocab, size=8)),
                "cat": rng.choice(["a", "b", "c"]),
                "y": y,
            }
        )
    return pd.DataFrame(rows)


def _make_pipe():
    pre = ColumnTransformer(
        [
            ("txt", TfidfVectorizer(), "description"),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["cat"]),
        ]
    )
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=300))])


def test_top_linear_terms():
    df = _make_data()
    pipe = _make_pipe().fit(df[["description", "cat"]], df["y"])
    tab = top_linear_terms(pipe, k=3)
    assert list(tab.columns) == ["term", "coef", "direction"]
    assert len(tab) == 6
    pos, neg = tab[tab["direction"] == "positive"], tab[tab["direction"] == "negative"]
    assert len(pos) == 3 and len(neg) == 3
    assert (pos["coef"] > 0).all()
    assert (neg["coef"] < 0).all()
    # Fraud vocabulary must dominate the positive side.
    assert pos["term"].str.contains("txt__").all()
    assert pos["term"].str.contains("earn|cash|weekly|bonus").any()
    # k larger than the vocabulary is capped per side, so each direction
    # returns all n terms: 2n rows in total.
    big = top_linear_terms(pipe, k=100)
    n_feat = pipe.named_steps["pre"].get_feature_names_out().shape[0]
    assert len(big) == 2 * n_feat


def test_top_linear_terms_rejects_nonlinear():
    from sklearn.ensemble import HistGradientBoostingClassifier

    df = _make_data(n=30)
    df["n"] = np.arange(len(df), dtype=float)
    pre = ColumnTransformer([("num", "passthrough", ["n"])])
    pipe = Pipeline([("pre", pre), ("clf", HistGradientBoostingClassifier())])
    pipe.fit(df[["n"]], df["y"])
    try:
        top_linear_terms(pipe, k=2)
        raised = False
    except TypeError:
        raised = True
    assert raised


def test_permutation_table():
    df = _make_data()
    X = df[["description", "cat"]]
    pipe = _make_pipe().fit(X, df["y"])
    tab = permutation_table(pipe, X, df["y"], n_repeats=2, seed=0)
    assert set(tab.columns) == {"col", "importance_mean", "importance_std"}
    assert set(tab["col"]) == {"description", "cat"}
    # The text column carries all signal, the noise column none.
    desc = tab[tab["col"] == "description"].iloc[0]
    cat = tab[tab["col"] == "cat"].iloc[0]
    assert desc["importance_mean"] > cat["importance_mean"]
    assert desc["importance_mean"] > 0
    # Column filter returns the subset only.
    only_cat = permutation_table(pipe, X, df["y"], n_repeats=2, cols=["cat"])
    assert set(only_cat["col"]) == {"cat"}
    # Sorted descending by importance.
    assert (np.diff(tab["importance_mean"].to_numpy()) <= 1e-12).all()


def test_show_errors():
    df = _make_data(n=12)
    # Hand-made scores: rows 0-4 are labelled 0 but score high (false
    # positives), rows 5-9 are labelled 1 but score low (false negatives).
    y_true = [0] * 5 + [1] * 5 + [0, 1]
    score = [0.9, 0.85, 0.8, 0.75, 0.7, 0.3, 0.25, 0.2, 0.15, 0.1, 0.6, 0.95]
    out = show_errors(df, y_true, score, thr=0.5, k=3)
    assert set(out.columns) == {
        "error_type", "score", "y_true", "title", "description",
    }
    fp = out[out["error_type"] == "false_positive"]
    fn = out[out["error_type"] == "false_negative"]
    assert len(fp) == 3 and len(fn) == 3
    # Most confident first: FPs by descending score, FNs by ascending score.
    assert np.all(np.diff(fp["score"]) < 0)
    assert np.all(np.diff(fn["score"]) > 0)
    assert (fp["y_true"] == 0).all() and (fn["y_true"] == 1).all()
    assert (out["description"].str.len() <= 200).all()
    assert out["title"].notna().all()
    # k larger than the number of errors returns all of them:
    # 6 false positives (rows 0-4 and the tail row labelled 0) plus 5 FNs.
    all_errs = show_errors(df, y_true, score, thr=0.5, k=50)
    assert len(all_errs) == 11


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(tests)} tests passed")
