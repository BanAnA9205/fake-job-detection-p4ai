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
    clean,
    load_processed,
    load_raw,
    load_splits,
    make_splits,
    save_splits,
    seed_everything,
)


def make_frame(n_groups=20, per_group=10, pos_per_group=2):
    """Small fake dataset: each group (own company, own description words) has the same positive rate."""
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
                "company_profile": f"company{g}x hires{g}x people{g}x" if i else None,
                "description": f"team{g}x builds{g}x software{g}x.",
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
    assert "department" not in cleaned  # all-missing fields are removed during QC
    assert "has_company_logo" not in cleaned  # all-one fields are removed too
    assert "has_questions" not in cleaned
    assert "employment_type" not in cleaned  # a uniform category is removed as well
    assert cleaned["telecommuting"].nunique() == 2  # variable flags are retained
    again = clean(frame)
    assert again["group"].tolist() == cleaned["group"].tolist()


def test_clean_groups_descriptions_that_differ_only_in_case_and_spacing():
    frame = make_frame(n_groups=2, per_group=5, pos_per_group=1)
    # Row 0 (group 0, no company) and row 5 (group 1, no company) get the same description.
    frame.loc[0, "description"] = "  Hello   World "
    frame.loc[5, "description"] = "hello world"
    out = clean(frame)
    assert out.loc[0, "group"] == out.loc[5, "group"]


def test_clean_fixes_text():
    frame = make_frame(n_groups=1, per_group=3, pos_per_group=1)
    frame.loc[0, "requirements"] = "#NAME?"
    frame.loc[1, "description"] = "Mail #EMAIL_3a1e#PHONE_ad7b## today.We reply"
    frame.loc[2, "description"] = "Corporate overviewAker uses JavaScript"
    out = clean(frame)
    assert pd.isna(out.loc[0, "requirements"])
    assert out.loc[1, "description"] == "Mail #EMAIL#PHONE## today.We reply"
    assert out.loc[2, "description"] == "Corporate overviewAker uses JavaScript"


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
    for idx in splits.values():
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


def test_save_load_processed_roundtrip():
    cleaned = clean(make_frame())
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "sub", "clean.csv.gz")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cleaned.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
        back = load_processed(path)
        # Same values; None and NaN both mean missing.
        same = lambda d: d.astype(object).where(d.notna(), "<missing>").astype(str)
        pd.testing.assert_frame_equal(same(back), same(cleaned))
        assert "department" not in back
        expect_error(AssertionError, lambda: load_processed(os.path.join(tmp, "missing.csv.gz")))


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


def test_split_fingerprint_rejects_label_or_content_changes():
    cleaned = clean(make_frame())
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "splits.npz")
        save_splits(path, {"train": np.array([0]), "val": np.array([1]), "test": np.array([2])}, cleaned)
        changed = cleaned.copy()
        changed.loc[0, "fraudulent"] = 1 - changed.loc[0, "fraudulent"]
        expect_error(AssertionError, lambda path=path, changed=changed: load_splits(path, changed))
        changed = cleaned.copy()
        changed.loc[0, "requirements"] = "changed text"
        expect_error(AssertionError, lambda path=path, changed=changed: load_splits(path, changed))


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
