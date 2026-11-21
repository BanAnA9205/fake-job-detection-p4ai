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


def test_zero_shot_run_with_fake_scorer():
    """Scores are computed once, cached, and evaluated per seed; a fake scorer replaces the LLM."""
    df = ft.add_engineered_features(dh.load_processed())
    y = df["fraudulent"].to_numpy()
    calls = []

    def fake_scores(texts, model_name, device, batch_size=16, max_tokens=512):
        calls.append(len(texts))
        return (y * 0.5 + np.random.default_rng(0).random(len(y)) * 0.6).astype("float32")

    real, real_cache = deep_runs.deep_helper.zero_shot_scores, deep_runs.EMBED_CACHE
    deep_runs.deep_helper.zero_shot_scores = fake_scores
    deep_runs.EMBED_CACHE = tempfile.mkdtemp()
    try:
        out = tempfile.mkdtemp()
        deep_runs.run_zero_shot(df, [0, 1], out, device="cpu", llm_model="org/fake-llm")
        deep_runs.run_zero_shot(df, [0, 1], tempfile.mkdtemp(), device="cpu", llm_model="org/fake-llm")
        assert calls == [len(df)], "the LLM must score every ad exactly once (second run uses the cache)"
        # Same rows, different text: the cache must not be reused.
        changed = df.assign(text=df["text"] + " edited")
        deep_runs.run_zero_shot(changed, [0], tempfile.mkdtemp(), device="cpu", llm_model="org/fake-llm")
        assert calls == [len(df), len(df)], "changed text must be scored again"
    finally:
        deep_runs.deep_helper.zero_shot_scores, deep_runs.EMBED_CACHE = real, real_cache
    res = rh.load_results(out)
    assert sorted(res["seed"]) == [0, 1] and set(res["model"]) == {"zeroshot_fake-llm"}
    assert (res["test_pr_auc"] > 0.5).all()  # the fake scores separate the classes well


def test_load_embeddings_rejects_other_text():
    """A cache built from other text (same rows) must be refused, not silently reused."""
    cache = tempfile.mkdtemp()
    np.save(os.path.join(cache, "m_emb.npy"), np.zeros((2, 3), dtype=np.float32))
    np.save(os.path.join(cache, "m_ids.npy"), np.array([1, 2]))
    with open(os.path.join(cache, "m_text.txt"), "w") as f:
        f.write(deep_runs.deep_helper.text_fingerprint(["old a", "old b"]))
    assert deep_runs.load_embeddings("m", [1, 2], ["old a", "old b"], cache).shape == (2, 3)
    try:
        deep_runs.load_embeddings("m", [1, 2], ["new a", "new b"], cache)
    except AssertionError:
        return
    raise AssertionError("embeddings built from other text were reused")


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
    test_heads_on_random_embeddings()
    test_classical_run_with_dummy_only_and_resume()
    test_zero_shot_run_with_fake_scorer()
    test_load_embeddings_rejects_other_text()
    test_explain_bert_picks_the_right_ads()
    print("all run tests passed")
