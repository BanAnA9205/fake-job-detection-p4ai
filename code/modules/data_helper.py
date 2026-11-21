"""Loading, auditing, cleaning and splitting the fake job postings data."""

import hashlib
import os
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from modules import qc

TARGET = "fraudulent"

TEXT_COLS = ["title", "company_profile", "description", "requirements", "benefits"]
CAT_COLS = ["employment_type", "required_experience", "required_education", "industry", "function"]
BIN_COLS = ["telecommuting", "has_company_logo", "has_questions"]

BALANCE_TOL = 0.01  # allowed drift of the positive rate between splits

# The cleaned data (with its split groups), written once by notebook 01 and read by every later step.
PROCESSED = "../data/processed/emscad_clean.csv.gz"


def load_raw(path):
    """Read the CSV and check the schema the rest of the code relies on."""
    df = pd.read_csv(path)

    # Schema: column count, the target and id exist, every column we use is present.
    assert df.shape[1] == 18, f"expected 18 columns, got {df.shape[1]}"
    assert TARGET in df.columns and "job_id" in df.columns, "missing target or job_id"
    missing = [c for c in TEXT_COLS + CAT_COLS + BIN_COLS if c not in df.columns]
    assert not missing, f"missing columns: {missing}"

    # Values: ids are unique and the label is binary.
    assert df["job_id"].is_unique, "job_id is not unique"
    assert set(df[TARGET].unique()) <= {0, 1}, f"{TARGET} must only contain 0 and 1"
    return df


def _content_cols(df):
    """Columns that describe a posting: everything except the row id and the label."""
    return [c for c in df.columns if c not in ("job_id", TARGET)]


def _normalise_text(text):
    """Lowercase, trim and collapse whitespace so cosmetic edits compare equal."""
    if pd.isna(text):
        return ""
    return " ".join(str(text).split()).lower()


def _count_duplicate_conflicts(df):
    """Count duplicate blocks (same content apart from job_id) that disagree on the label."""
    cols = _content_cols(df)
    blocks = df[df.duplicated(subset=cols, keep=False)]
    if blocks.empty:
        return 0
    labels = blocks.groupby(cols, dropna=False)[TARGET].nunique()
    return int((labels > 1).sum())


def audit(df):
    """Summarise shape, class balance, missingness and duplicate structure."""
    cols = _content_cols(df)
    y = df[TARGET]
    desc_counts = df["description"].map(_normalise_text).value_counts()
    return {
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "class_counts": {0: int((y == 0).sum()), 1: int((y == 1).sum())},
        "positive_rate": round(float(y.mean()), 6),
        "missing_rate": {c: round(float(df[c].isna().mean()), 6) for c in df.columns},
        "n_duplicate_rows": int(df.duplicated(subset=cols, keep="first").sum()),
        "n_rows_shared_description": int(desc_counts[desc_counts > 1].sum()),
        "n_conflicting_duplicate_groups": _count_duplicate_conflicts(df),
    }


def _fix_text(s):
    """Junk cells become missing, hashed placeholders become generic, glued boundaries get a space."""
    s = s.where(~s.astype(str).str.strip().isin(qc.JUNK_VALUES))
    s = s.str.replace(qc.PLACEHOLDER, r"#\1", regex=True)         # "#URL_3f9a..#" -> "#URL#"
    s = s.str.replace(qc.GLUE_SENTENCE, r"\1\2 \3", regex=True)
    return s.str.replace(qc.GLUE_WORD, r"\1 \2", regex=True)


def clean(df):
    """Fix the text, drop exact duplicate postings, check labels agree, and tag split groups.

    Text: Excel errors and filler become missing, each hashed link/email/phone placeholder becomes
    a generic one (a model should learn "has a phone number", not one number), and sentences glued
    together by removed HTML are separated. Groups: see qc.group_ids.
    """
    out = df.copy()
    for col in TEXT_COLS + ["department"]:
        out[col] = _fix_text(out[col])

    # Duplicates with conflicting labels would make "keep the first" hide a real label error.
    conflicts = _count_duplicate_conflicts(out)
    assert conflicts == 0, f"{conflicts} duplicate blocks disagree on {TARGET}; review the data"

    # Drop exact duplicates (ignoring job_id), then tag each row with its split group.
    out = out.drop_duplicates(subset=_content_cols(out), keep="first").reset_index(drop=True)
    out["group"] = qc.group_ids(out)
    return out


def save_processed(df, path=PROCESSED):
    """Write the cleaned frame (gzip CSV; mtime=0 keeps the file identical across runs)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})


def load_processed(path=PROCESSED):
    """Read the cleaned frame written by notebook 01."""
    assert os.path.exists(path), f"{path} not found: run notebook 01 first"
    # Text columns are read as text even when a column happens to be entirely empty.
    text = TEXT_COLS + CAT_COLS + ["location", "department", "salary_range", "group"]
    df = pd.read_csv(path, dtype={c: "str" for c in text})
    assert "group" in df.columns, f"{path} has no 'group' column: it is not the cleaned data"
    return df


def _check_splits(splits, y, groups):
    """Assert splits cover every row once, share no group and stay close in balance."""
    names = list(splits)

    # Coverage: every row appears in exactly one split.
    stacked = np.concatenate([splits[n] for n in names])
    assert np.array_equal(np.sort(stacked), np.arange(len(y))), "splits must cover every row once"

    # No leakage: no group (shared description, company or near-copy) sits in two splits.
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = set(groups[splits[a]]) & set(groups[splits[b]])
            assert not shared, f"{len(shared)} groups appear in both {a} and {b}"

    # Balance: each split keeps roughly the overall positive rate.
    overall = float(y.mean())
    for name in names:
        rate = float(y[splits[name]].mean())
        if abs(rate - overall) > BALANCE_TOL:
            raise ValueError(
                f"positive rate in {name} is {rate:.3f} vs {overall:.3f} overall; "
                "the data is too small or too clumped for a balanced grouped split"
            )


def make_splits(df, seed=0, val_size=0.2, test_size=0.2):
    """Grouped, stratified train/val/test split returned as positional row indices."""
    assert "group" in df.columns, "run clean() first: df has no 'group' column"
    assert 0 < val_size < 1 and 0 < test_size < 1 and val_size + test_size < 1
    y = df[TARGET].to_numpy()
    groups = df["group"].to_numpy()
    n = len(df)

    # Step 1: cut the test split off first (1 fold out of 1/test_size folds).
    # StratifiedGroupKFold keeps a whole group in one split and balances the label.
    n_test_folds = max(2, round(1.0 / test_size))
    outer = StratifiedGroupKFold(n_splits=n_test_folds, shuffle=True, random_state=seed)
    trainval, test = next(outer.split(np.zeros((n, 1)), y, groups))

    # Step 2: split the rest into train and validation, so val_size is a share of all rows.
    n_val_folds = max(2, round(1.0 / (val_size / (1.0 - test_size))))
    inner = StratifiedGroupKFold(n_splits=n_val_folds, shuffle=True, random_state=seed + 1)
    tr, va = next(inner.split(np.zeros((len(trainval), 1)), y[trainval], groups[trainval]))

    # Step 3: back to positional indices of the full frame, then verify the result.
    splits = {
        "train": np.sort(trainval[tr]).astype(int),
        "val": np.sort(trainval[va]).astype(int),
        "test": np.sort(test).astype(int),
    }
    _check_splits(splits, y, groups)
    return splits


def save_splits(path, splits, df):
    """Save split indices plus a fingerprint of the cleaned data, so any machine can reuse them."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, train=splits["train"], val=splits["val"], test=splits["test"],
             fingerprint=_fingerprint(df))


def load_splits(path, df):
    """Load saved indices and check they belong to this exact cleaned data frame."""
    z = np.load(path)

    # Refuse indices made for a different cleaned frame (rows dropped, added or reordered).
    assert str(z["fingerprint"]) == _fingerprint(df), "splits were made for different data; rebuild them"
    return {k: z[k] for k in ("train", "val", "test")}


def _fingerprint(df):
    """Short hash of the ordered job_ids and their groups.

    It changes if rows are dropped, added or reordered, or if the grouping changes, so splits made
    under an older grouping are refused instead of silently reused.
    """
    ids = ",".join(f"{j}:{g}" for j, g in zip(df["job_id"], df["group"]))
    return hashlib.sha256(ids.encode("utf-8")).hexdigest()[:16]


def seed_everything(seed):
    """Seed Python and numpy, plus torch when it happens to be installed."""
    # Python, numpy and hash-based ordering.
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Torch (CPU and every GPU) only when it is installed.
    try:
        import torch  # lazy: torch is optional and must never slow down a normal import
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
