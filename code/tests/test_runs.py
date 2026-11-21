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


def test_classical_run_with_dummy_only_and_resume():
    df = ft.add_engineered_features(dh.load_processed())
    out = tempfile.mkdtemp()
    classical_runs.run_classical(df, [0], out_dir=out, models=["dummy_majority"])
    assert rh.result_exists(out, "dummy_majority", 0)
    before = os.path.getmtime(os.path.join(out, "dummy_majority_seed0.json"))
    classical_runs.run_classical(df, [0], out_dir=out, models=["dummy_majority"])  # must skip
    assert os.path.getmtime(os.path.join(out, "dummy_majority_seed0.json")) == before
    res = rh.load_results(out)
    assert len(res) == 1 and 0.04 < res.loc[0, "test_pr_auc"] < 0.06  # base rate, 4.8% positives


def test_explain_bert_picks_the_right_ads():
    """4 highest-scored frauds, 3 lowest-scored frauds, 3 highest-scored real ads."""
    import json
    import pandas as pd

    y = np.array([1] * 10 + [0] * 20)
    score = np.r_[np.linspace(0.1, 1.0, 10), np.linspace(0.0, 0.9, 20)]
    test_df = pd.DataFrame({"text": [f"ad {i}" for i in range(30)], "job_id": np.arange(30)})
    real = deep_runs.eh.bert_token_attributions
    deep_runs.eh.bert_token_attributions = lambda *a, **k: (["[CLS]", "ad"], np.array([0.0, 1.0]))
    try:
        path = os.path.join(tempfile.mkdtemp(), "explain.json")
        deep_runs.explain_bert(test_df, y, score, 0.5, None, None, "cpu", 256, path)
    finally:
        deep_runs.eh.bert_token_attributions = real
    with open(path) as f:
        rows = json.load(f)
    ids = {k: [r["job_id"] for r in rows if r["kind"] == k] for k in ("caught fraud", "missed fraud", "false alarm")}
    assert ids == {"caught fraud": [9, 8, 7, 6], "missed fraud": [0, 1, 2], "false alarm": [29, 28, 27]}
    assert rows[0]["flagged"] and not rows[4]["flagged"]


if __name__ == "__main__":
    test_deep_runs_imports_without_torch()
    test_classical_run_with_dummy_only_and_resume()
    test_explain_bert_picks_the_right_ads()
    print("all run tests passed")
