"""Tests for modules/explain_helper.py on small synthetic data.

Run with: .venv/bin/python tests/test_explain_helper.py
(or pytest if available). Fits tiny models on ~60 synthetic rows only.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
from modules.explain_helper import top_linear_terms
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


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


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(tests)} tests passed")
