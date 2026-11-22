"""Tests for modules.qc: the checks and the grouping that keeps related ads in one split.

Run: .venv/bin/python tests/test_qc.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules import qc
from modules.data_helper import _check_splits


def _frame(descriptions, companies, labels=None):
    n = len(descriptions)
    return pd.DataFrame({"description": descriptions, "company_profile": companies,
                         "fraudulent": labels if labels is not None else [0] * n})


def test_group_ids_links_exact_company_but_not_near_copies():
    long = "we build data pipelines in python and sql for the analytics platform team daily"
    df = _frame(
        ["Alpha beta gamma", "alpha  BETA gamma",          # 0, 1: same description up to case/space
         "delta epsilon zeta", "eta theta iota",           # 2, 3: same company
         long, long + " now",                              # 4, 5: near-copies
         "kappa lambda mu", "nu xi omicron"],              # 6, 7: unrelated, no company
        [None, None, "Acme Ltd", "acme  ltd", None, None, None, None])
    g = qc.group_ids(df)
    assert g[0] == g[1] and g[2] == g[3] and g[4] != g[5]
    assert len({g[0], g[2], g[4], g[6], g[7]}) == 5, "unrelated ads must stay apart"


def test_group_ids_chains_and_ignores_empty_company():
    df = _frame(["one two three", "four five six", "four five six", "seven eight nine"],
                ["Acme", "Acme", None, None])
    g = qc.group_ids(df)
    assert g[0] == g[1] == g[2], "company links 0-1, description links 1-2"
    assert g[3] != g[0], "a missing company profile must not link ads"


def test_split_check_rejects_a_shared_group():
    groups = np.array(["a", "a", "b", "c"])
    y = np.array([0, 1, 0, 1])
    good = {"train": np.array([0, 1]), "val": np.array([2]), "test": np.array([3])}
    bad = {"train": np.array([0, 2]), "val": np.array([1]), "test": np.array([3])}
    try:
        _check_splits(bad, y, groups)
    except (AssertionError, ValueError) as e:
        assert "groups appear in both" in str(e)
    else:
        raise AssertionError("a group in two splits must be rejected")
    try:
        _check_splits(good, y, groups)
    except ValueError:
        pass  # four rows cannot be balanced; only the group check matters here


def test_single_value_columns_includes_all_missing_and_all_one():
    frame = pd.DataFrame({
        "job_id": [1, 2, 3],
        "all_missing": [None, None, None],
        "all_one": [1, 1, 1],
        "one_and_missing": [1, None, 1],
        "rare": [0, 0, 1],
        "fraudulent": [0, 0, 1],
    })
    assert qc.single_value_columns(frame) == ["all_missing", "all_one"]
    assert "one_and_missing" not in qc.single_value_columns(frame), "missingness remains a signal"


def test_group_ids_links_original_contact_but_not_fuzzy_text():
    df = _frame(
        ["alpha beta gamma", "alpha beta gamma now", "delta epsilon zeta"],
        [None, None, None],
    )
    contacts = pd.Series([
        ("URL:abc", "EMAIL:shared"),
        (),
        ("PHONE:123", "EMAIL:shared"),
    ])
    groups = qc.group_ids(df, contacts)
    assert groups[0] != groups[1], "fuzzy near-copies are not merged"
    assert groups[0] == groups[2], "an original contact hash links related ads"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)} qc tests passed")
