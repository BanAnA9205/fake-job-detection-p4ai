"""Plain-assert tests for modules/data_helper.py (pytest optional).

Run: .venv/bin/python tests/test_data_helper.py
"""

import os
import random
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.data_helper import (
    BIN_COLS,
    CAT_COLS,
    TEXT_COLS,
    audit,
    clean,
    load_raw,
    load_splits,
    make_splits,
    save_splits,
    seed_everything,
)


def make_frame(n_groups=20, per_group=10, pos_per_group=2):
    """Small fake dataset: each description group has the same positive rate."""
    rows = []
    job_id = 0
    for g in range(n_groups):
        for i in range(per_group):
            rows.append({
                "job_id": job_id,
                "title": f"Engineer {g}-{i}",
                "location": "US, NY" if i % 2 else "GB, London",
                "department": None,
                "salary_range": "50000-60000" if i % 3 else None,
                "company_profile": "we hire people" if i else None,
                "description": f"Group {g} builds software.",
                "requirements": "python",
                "benefits": "none",
                "telecommuting": i % 2,
                "has_company_logo": 1,
                "has_questions": 0,
                "employment_type": "Full-time",
                "required_experience": "Mid-Senior level",
                "required_education": "Bachelor's Degree",
                "industry": "Information Technology",
                "function": "Engineering",
                "fraudulent": int(i < pos_per_group),
            })
            job_id += 1
    return pd.DataFrame(rows)


def expect_error(exc, fn):
    """Fail the test when fn does not raise exc."""
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__}")


def test_constants():
    assert TEXT_COLS == ["title", "company_profile", "description", "requirements", "benefits"]
    assert len(CAT_COLS) == 5 and len(BIN_COLS) == 3
    assert not set(TEXT_COLS) & set(CAT_COLS)
    assert not set(TEXT_COLS + CAT_COLS) & set(BIN_COLS)


def test_load_raw():
    frame = make_frame()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "raw.csv")
        frame.to_csv(path, index=False)
        loaded = load_raw(path)
        assert loaded.shape == (200, 18)
        assert loaded["job_id"].is_unique
        assert sorted(loaded["fraudulent"].unique()) == [0, 1]

        bad_target = frame.copy()
        bad_target.loc[0, "fraudulent"] = 2
        bad_target.to_csv(path, index=False)
        expect_error(AssertionError, lambda: load_raw(path))

        bad_id = frame.copy()
        bad_id.loc[1, "job_id"] = bad_id.loc[0, "job_id"]
        bad_id.to_csv(path, index=False)
        expect_error(AssertionError, lambda: load_raw(path))

        frame.drop(columns=["benefits"]).to_csv(path, index=False)
        expect_error(AssertionError, lambda: load_raw(path))


def test_audit():
    frame = make_frame()
    duplicate = frame.loc[[0]].copy()
    duplicate["job_id"] = 9000
    frame = pd.concat([frame, duplicate], ignore_index=True)
    info = audit(frame)
    assert info["shape"] == [201, 18]
    assert info["class_counts"] == {0: 160, 1: 41}
    assert abs(info["positive_rate"] - 41 / 201) < 1e-5
    assert info["n_duplicate_rows"] == 1
    assert info["n_conflicting_duplicate_groups"] == 0
    assert info["n_rows_shared_description"] == 201
    assert info["missing_rate"]["department"] == 1.0
    assert info["missing_rate"]["title"] == 0.0
    assert len(info["missing_rate"]) == 18

    pair = frame.loc[[5, 5]].copy()
    pair["job_id"] = [9001, 9002]
    pair["fraudulent"] = [0, 1]
    conflicted = pd.concat([frame, pair], ignore_index=True)
    info = audit(conflicted)
    assert info["n_conflicting_duplicate_groups"] == 1
    assert info["class_counts"] == {0: 161, 1: 42}


def test_clean_drops_duplicates_and_hashes_groups():
    frame = make_frame()
    duplicate = frame.loc[[0]].copy()
    duplicate["job_id"] = 9000
    frame = pd.concat([frame, duplicate], ignore_index=True)
    cleaned = clean(frame)
    assert len(cleaned) == 200
    assert list(cleaned.index) == list(range(200))
    assert "group" in cleaned.columns
    assert cleaned["group"].nunique() == 20
    assert cleaned["group"].notna().all()
    assert cleaned["department"].isna().all()  # clean does not impute
    assert cleaned["employment_type"].tolist() == ["Full-time"] * 200
    again = clean(frame)
    assert again["group"].tolist() == cleaned["group"].tolist()


def test_clean_normalises_whitespace_and_case():
    frame = make_frame(n_groups=2, per_group=5, pos_per_group=1)
    frame.loc[0, "title"] = "First"
    frame.loc[1, "title"] = "Second"
    frame.loc[0, "description"] = "  Hello   World "
    frame.loc[1, "description"] = "hello world"
    out = clean(frame)
    assert out.loc[0, "group"] == out.loc[1, "group"]
    assert out.loc[2, "group"] != out.loc[0, "group"]


def test_clean_handles_missing_description():
    frame = make_frame(n_groups=1, per_group=3, pos_per_group=1)
    frame.loc[0, "description"] = np.nan
    frame.loc[1, "description"] = ""
    frame.loc[2, "description"] = "   "
    out = clean(frame)
    assert out["group"].notna().all()
    assert out.loc[0, "group"] == out.loc[1, "group"] == out.loc[2, "group"]


def test_clean_refuses_conflicting_duplicates():
    frame = make_frame()
    pair = frame.loc[[5, 5]].copy()
    pair["job_id"] = [9001, 9002]
    pair["fraudulent"] = [0, 1]
    conflicted = pd.concat([frame, pair], ignore_index=True)
    expect_error(AssertionError, lambda: clean(conflicted))


def test_make_splits():
    cleaned = clean(make_frame())
    splits = make_splits(cleaned, seed=0)
    assert set(splits) == {"train", "val", "test"}
    stacked = np.concatenate(list(splits.values()))
    assert len(stacked) == 200
    assert np.array_equal(np.sort(stacked), np.arange(200))
    assert abs(len(splits["test"]) - 40) <= 10
    assert abs(len(splits["val"]) - 40) <= 10
    groups = cleaned["group"].to_numpy()
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not set(groups[splits[a]]) & set(groups[splits[b]])
    y = cleaned["fraudulent"].to_numpy()
    for name, idx in splits.items():
        assert abs(y[idx].mean() - y.mean()) <= 0.01
    same = make_splits(cleaned, seed=0)
    assert all(np.array_equal(splits[k], same[k]) for k in splits)
    other = make_splits(cleaned, seed=1)
    assert not np.array_equal(splits["test"], other["test"])


def test_make_splits_rejects_tiny_data():
    small = clean(make_frame(n_groups=3, per_group=4, pos_per_group=1))
    expect_error(ValueError, lambda: make_splits(small))


def test_make_splits_rejects_unbalanced_data():
    frame = make_frame(n_groups=10, per_group=10, pos_per_group=2)
    frame["fraudulent"] = 0
    frame.loc[frame.index[:10], "fraudulent"] = 1  # every positive sits in one group
    expect_error(ValueError, lambda: make_splits(clean(frame)))


def test_save_load_splits_roundtrip():
    cleaned = clean(make_frame())
    splits = make_splits(cleaned, seed=5)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "splits.npz")
        save_splits(path, splits, cleaned)
        loaded = load_splits(path, cleaned)
        # Splits saved for one frame must be refused for a frame with a row removed.
        changed = cleaned.iloc[1:].reset_index(drop=True)
        expect_error(AssertionError, lambda: load_splits(path, changed))
    assert all(np.array_equal(splits[k], loaded[k]) for k in ("train", "val", "test"))


def test_seed_everything():
    seed_everything(123)
    first = (random.random(), float(np.random.rand()))
    seed_everything(123)
    second = (random.random(), float(np.random.rand()))
    assert first == second
    assert os.environ["PYTHONHASHSEED"] == "123"
    seed_everything(7)
    assert os.environ["PYTHONHASHSEED"] == "7"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} tests passed")
