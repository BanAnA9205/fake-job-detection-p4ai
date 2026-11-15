"""Tests for modules.results_helper: round trip of the shared results format."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from modules.results_helper import load_results, load_scores, result_exists, save_result

METRICS = {"precision": 0.5, "recall": 0.25, "f1": 1 / 3, "tp": 1, "fp": 1, "fn": 3, "tn": 5,
           "pr_auc": 0.4, "roc_auc": 0.7}


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


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(tests)} tests passed")
