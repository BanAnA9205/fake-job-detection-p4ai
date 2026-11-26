"""Tests for modules.results_helper: round trip of the shared results format."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pytest
from modules.eval_helper import evaluate, pick_threshold
from modules.results_helper import (
    assert_result_lineage,
    load_results,
    load_scores,
    result_exists,
    save_result,
    verify_results,
)

METRICS = {"precision": 0.5, "recall": 0.25, "f1": 1 / 3, "tp": 1, "fp": 1, "fn": 3, "tn": 5,
           "pr_auc": 0.4}


def test_round_trip_and_resume_flag():
    d = tempfile.mkdtemp()
    assert not result_exists(d, "m", 3)
    save_result(d, "m", "classical", 3, np.float64(0.5), METRICS, METRICS,
                [0, 1], [0.1, 0.9], [1, 0], [np.float32(0.8), 0.2],
                train_pr_auc=np.float64(0.99), best_params={"hidden": (64,)}, seconds=1.5)
    assert result_exists(d, "m", 3) and not result_exists(d, "m", 4)
    df = load_results(d)
    assert len(df) == 1 and df.loc[0, "seed"] == 3 and df.loc[0, "test_recall"] == 0.25
    s = load_scores(d, "m", 3)
    assert np.allclose(s["score_test"], [0.8, 0.2]) and s["y_val"].tolist() == [0, 1]


def test_load_results_empty_folder_and_two_folders():
    a, b = tempfile.mkdtemp(), tempfile.mkdtemp()
    assert load_results(a).empty
    save_result(a, "x", "classical", 0, 0.5, METRICS, METRICS, [0], [0.1], [0], [0.1])
    save_result(b, "y", "deep", 0, 0.5, METRICS, METRICS, [0], [0.1], [0], [0.1])
    assert sorted(load_results(a, b)["model"]) == ["x", "y"]


def test_lineage_round_trip_and_stale_resume_rejection():
    d = tempfile.mkdtemp()
    save_result(
        d, "m", "classical", 3, 0.5, METRICS, METRICS,
        [0, 1], [0.1, 0.9], [1, 0], [0.8, 0.2],
        data_fingerprint="cleaned-v1", val_indices=[5, 1], test_indices=[4, 0],
    )
    expected = {"data_fingerprint": "cleaned-v1"}
    scores = load_scores(d, "m", 3)
    assert scores["val_indices"].tolist() == [5, 1]
    assert scores["test_indices"].tolist() == [4, 0]
    assert_result_lineage(expected, scores, "cleaned-v1", [5, 1], [4, 0])
    assert result_exists(d, "m", 3, "cleaned-v1", [5, 1], [4, 0])

    # Different text/model rows can retain the same labels and row order, so labels alone are insufficient.
    with pytest.raises(ValueError, match="different cleaned data"):
        result_exists(d, "m", 3, "cleaned-v2", [5, 1], [4, 0])
    with pytest.raises(ValueError, match="test_indices"):
        result_exists(d, "m", 3, "cleaned-v1", [5, 1], [0, 4])
    with pytest.raises(ValueError, match="different cleaned data"):
        assert_result_lineage({}, scores, "cleaned-v1", [5, 1], [4, 0])


def _verified_dir(tamper=None):
    """Two seeds of one model whose saved metrics, threshold and lineage are all consistent."""
    d = tempfile.mkdtemp()
    y = np.array([0, 1, 0, 1, 0, 1, 0, 0])
    splits = {0: {"val": np.array([0, 1, 2, 3]), "test": np.array([4, 5, 6, 7])},
              1: {"val": np.array([4, 5, 6, 7]), "test": np.array([0, 1, 2, 3])}}
    for seed, split in splits.items():
        y_val, y_test = y[split["val"]], y[split["test"]]
        s_val, s_test = np.array([0.1, 0.9, 0.3, 0.6]), np.array([0.2, 0.8, 0.4, 0.1])
        thr = pick_threshold(y_val, s_val) + (0.01 if tamper == "threshold" else 0)
        test = evaluate(y_test, s_test, thr)
        if tamper == "metric":
            test = dict(test, f1=0.123)
        save_result(d, "m", "classical", seed, thr, evaluate(y_val, s_val, thr), test,
                    y_val, s_val, y_test, s_test, data_fingerprint="fp",
                    val_indices=split["val"], test_indices=split["test"])
    return d, y, splits


def test_verify_results_accepts_consistent_runs():
    d, y, splits = _verified_dir()
    res = verify_results({d: ("m",)}, y, splits, "fp")
    assert len(res) == 2 and set(res["seed"]) == {0, 1}


def test_verify_results_rejects_stale_metrics_thresholds_and_inventory():
    for tamper, message in (("metric", "recomputed"), ("threshold", "threshold")):
        d, y, splits = _verified_dir(tamper)
        with pytest.raises(ValueError, match=message):
            verify_results({d: ("m",)}, y, splits, "fp")
    d, y, splits = _verified_dir()
    with pytest.raises(ValueError, match="inventory"):
        verify_results({d: ("m", "missing_model")}, y, splits, "fp")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(tests)} tests passed")
