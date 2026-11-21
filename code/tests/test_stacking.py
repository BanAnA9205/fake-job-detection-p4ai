"""Tests for modules.stacking (CPU only, no model is retrained).

Run: .venv/bin/python tests/test_stacking.py
"""

import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
from sklearn.metrics import average_precision_score as ap

from modules import data_helper as dh, features as ft, results_helper as rh, stacking


def _fake_scores(y, informative, rng):
    """Probability-like scores that separate the classes only on the rows marked informative."""
    z = rng.normal(size=len(y)) + np.where(informative, 3.0 * y, 0.0)
    return 1 / (1 + np.exp(-z))


def test_logit_is_finite_at_0_and_1():
    out = stacking._logit([0.0, 0.5, 1.0])
    assert np.isfinite(out).all() and out[1] == 0 and np.isclose(out[0], -out[2])


def test_stack_beats_both_inputs_when_they_know_different_rows():
    """Known answer: A sees fraud only in the first half, B only in the second; together they see all."""
    rng = np.random.default_rng(0)
    n = 4000
    y = (rng.random(n) < 0.1).astype(int)
    half = np.arange(n) < n // 2
    a = {"score_val": _fake_scores(y, half, rng)}
    b = {"score_val": _fake_scores(y, ~half, rng)}
    groups = np.arange(n) // 2  # pairs of rows share a group

    stacker, oof, thr = stacking.stack_seed(a, b, y, groups, seed=0)
    assert ap(y, oof) > max(ap(y, a["score_val"]), ap(y, b["score_val"])) + 0.1
    assert (stacker.coef_[0] > 0).all(), "both inputs must get a positive weight"
    assert 0 < thr < 1


def test_run_stack_on_the_real_split_with_fake_runs():
    """End to end on the saved seed-0 split: files written, test scored, resume skips."""
    df = ft.add_engineered_features(dh.clean(dh.load_raw("../data/fake_job_postings.csv")))
    sp = dh.load_splits("splits/seed0.npz", df)
    y = df["fraudulent"].to_numpy()
    rng = np.random.default_rng(1)
    run_dir = tempfile.mkdtemp()
    for name, k in (("bert_x", 0), ("logreg_bm25", 1)):
        s = _fake_scores(y, np.arange(len(y)) % 2 == k, rng)
        rh.save_result(run_dir, name, "fake", 0, 0.5, {}, {}, y[sp["val"]], s[sp["val"]],
                       y[sp["test"]], s[sp["test"]])
    out = tempfile.mkdtemp()
    stacking.run_stack(df, [0], (run_dir, "bert_x"), (run_dir, "logreg_bm25"), out_dir=out)
    res = rh.load_results(out)
    assert list(res["model"]) == ["stack_bert_logreg_bm25"] and res.loc[0, "group"] == "stack"
    assert res.loc[0, "test_pr_auc"] > 0.5
    before = os.path.getmtime(os.path.join(out, "stack_bert_logreg_bm25_seed0.json"))
    stacking.run_stack(df, [0], (run_dir, "bert_x"), (run_dir, "logreg_bm25"), out_dir=out)
    assert os.path.getmtime(os.path.join(out, "stack_bert_logreg_bm25_seed0.json")) == before


if __name__ == "__main__":
    test_logit_is_finite_at_0_and_1()
    test_stack_beats_both_inputs_when_they_know_different_rows()
    test_run_stack_on_the_real_split_with_fake_runs()
    print("all stacking tests passed")
