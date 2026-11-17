"""Tests for modules.classical_runs and modules.deep_runs (no torch, no model downloads).

Run: .venv/bin/python tests/test_runs.py
"""

import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np

from modules import classical_runs, data_helper as dh, deep_runs, features as ft
from modules import results_helper as rh


def test_deep_runs_imports_without_torch():
    assert "torch" not in sys.modules, "deep_runs must not import torch at import time"


def test_heads_on_random_embeddings():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 8)).astype("float32")
    y = (X[:, 0] + 0.3 * rng.normal(size=300) > 1.0).astype(int)
    model, params = deep_runs.fit_logreg_head(X[:200], y[:200], X[200:], y[200:], seed=0)
    assert params["C"] in deep_runs.LOGREG_CS
    assert model.predict_proba(X[200:]).shape == (100, 2)
    mlp, _ = deep_runs.fit_mlp_head(X[:200], y[:200], seed=0)
    assert mlp.predict_proba(X[200:]).shape == (100, 2)


def test_classical_run_with_dummy_only_and_resume():
    df = ft.add_engineered_features(dh.clean(dh.load_raw("../data/fake_job_postings.csv")))
    out = tempfile.mkdtemp()
    classical_runs.run_classical(df, [0], out_dir=out, models=["dummy_majority"])
    assert rh.result_exists(out, "dummy_majority", 0)
    before = os.path.getmtime(os.path.join(out, "dummy_majority_seed0.json"))
    classical_runs.run_classical(df, [0], out_dir=out, models=["dummy_majority"])  # must skip
    assert os.path.getmtime(os.path.join(out, "dummy_majority_seed0.json")) == before
    res = rh.load_results(out)
    assert len(res) == 1 and 0.04 < res.loc[0, "test_pr_auc"] < 0.06  # base rate
    assert res.loc[0, "test_recall"] == 0.0
    assert res.loc[0, "test_f1"] == 0.0  # majority baseline predicts all jobs real


if __name__ == "__main__":
    test_deep_runs_imports_without_torch()
    test_heads_on_random_embeddings()
    test_classical_run_with_dummy_only_and_resume()
    print("all run tests passed")
