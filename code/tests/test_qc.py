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


def test_group_ids_links_exact_company_and_near_copies():
    long = "we build data pipelines in python and sql for the analytics platform team daily"
    df = _frame(
        ["Alpha beta gamma", "alpha  BETA gamma",          # 0, 1: same description up to case/space
         "delta epsilon zeta", "eta theta iota",           # 2, 3: same company
         long, long + " now",                              # 4, 5: near-copies
         "kappa lambda mu", "nu xi omicron"],              # 6, 7: unrelated, no company
        [None, None, "Acme Ltd", "acme  ltd", None, None, None, None])
    g = qc.group_ids(df)
    assert g[0] == g[1] and g[2] == g[3] and g[4] == g[5]
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


def test_near_duplicate_clusters_without_shared_words():
    assert qc.near_duplicate_clusters(pd.Series(["alpha", "beta", "gamma"])).tolist() == [0, 1, 2]


def test_assert_clean_catches_leftovers():
    good = pd.DataFrame({"job_id": [1, 2], "group": ["a", "b"], "description": ["fine", "also fine"]})
    qc.assert_clean(good, ["description"])
    for bad_text in (["#NAME?", "fine"], ["see #URL_ab12#", "fine"], ["same", "same"]):
        bad = good.assign(description=bad_text)
        try:
            qc.assert_clean(bad, ["description"])
        except AssertionError:
            continue
        raise AssertionError(f"assert_clean missed {bad_text}")


def test_checks():
    df = pd.DataFrame({"requirements": ["#NAME?", None, "python", "."],
                       "fraudulent": [1, 0, 0, 1]})
    dm = qc.disguised_missing(df, ["requirements"])
    assert dm.loc["requirements", "NaN"] == 1 and dm.loc["requirements", "junk"] == 2
    text = pd.Series(["see #URL_ab12# and #URL_cd34#", "#EMAIL_3a1e#PHONE_ad7b##", "plain"])
    pc = qc.placeholder_counts(text)
    assert pc.loc["URL", "ads"] == 1 and pc.loc["URL", "distinct hashes"] == 2
    assert pc.loc["PHONE", "ads"] == 1 and pc.loc["EMAIL", "ads"] == 1
    assert qc.glued_boundaries(pd.Series(["one place.We go", "fine text"])).tolist() == [1, 0]
    assert qc.group_purity(["a", "a", "b", "b", "c"], [1, 1, 0, 1, 0]) == {
        "groups with 2+ ads": 2, "all fraud": 1, "all real": 0, "mixed": 1}
    splits = {"train": np.array([0, 1]), "val": np.array([2]), "test": np.array([3, 4])}
    ov = qc.split_overlap(splits, {"company": ["A", "", "B", "A", ""]}, np.array([0, 0, 0, 1, 0]))
    assert ov.loc["company", "test ads"] == 1 and ov.loc["company", "test frauds"] == 1


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)} qc tests passed")
