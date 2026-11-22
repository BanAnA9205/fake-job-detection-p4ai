"""Loading, cleaning, and grouped splitting for the fake job postings data."""

import hashlib
import os
import random

import numpy as np
import pandas as pd
from modules import qc
from sklearn.model_selection import StratifiedGroupKFold

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


def clean(df):
    """Repair clear defects, remove exact duplicates and single-value fields, and tag groups."""
    out = df.copy()
    contacts = qc.contact_keys(out, TEXT_COLS)
    for col in TEXT_COLS + ["department"]:
        text = out[col].where(~out[col].astype(str).str.strip().isin(qc.JUNK_VALUES))
        out[col] = text.str.replace(qc.PLACEHOLDER, r"#\1", regex=True)

    content_cols = _content_cols(out)
    blocks = out[out.duplicated(subset=content_cols, keep=False)]
    conflicts = 0 if blocks.empty else int(
        (blocks.groupby(content_cols, dropna=False)[TARGET].nunique() > 1).sum()
    )
    assert conflicts == 0, f"{conflicts} duplicate blocks disagree on {TARGET}; review the data"

    out = out.drop_duplicates(subset=content_cols, keep="first")
    contacts = contacts.loc[out.index].reset_index(drop=True)
    out = out.reset_index(drop=True)
    out["group"] = qc.group_ids(out, contacts)
    return out.drop(columns=qc.single_value_columns(out))


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

    # No leakage: no group (shared exact text or original contact identity) crosses splits.
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
    """Hash split-relevant raw content so stale indices are rejected after data edits."""
    raw_cols = TEXT_COLS + CAT_COLS + BIN_COLS + [
        "location", "department", "salary_range", "job_id", "group", TARGET,
    ]
    cols = [c for c in raw_cols if c in df]
    values = pd.util.hash_pandas_object(df[cols], index=False).to_numpy().tobytes()
    return hashlib.sha256(",".join(cols).encode() + values).hexdigest()[:16]


def data_fingerprint(df):
    """Return the cleaned-content fingerprint saved in split files and result lineage."""
    return _fingerprint(df)


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
