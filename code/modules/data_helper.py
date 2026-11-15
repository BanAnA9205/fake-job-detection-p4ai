"""Loading, auditing, cleaning and splitting the fake job postings data."""

import hashlib
import json
import os
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

TARGET = "fraudulent"

TEXT_COLS = ["title", "company_profile", "description", "requirements", "benefits"]
CAT_COLS = ["employment_type", "required_experience", "required_education", "industry", "function"]
BIN_COLS = ["telecommuting", "has_company_logo", "has_questions"]

BALANCE_TOL = 0.01  # allowed drift of the positive rate between splits


def load_raw(path):
    """Read the CSV and check the schema the rest of the code relies on."""
    df = pd.read_csv(path)
    assert df.shape[1] == 18, f"expected 18 columns, got {df.shape[1]}"
    assert TARGET in df.columns and "job_id" in df.columns, "missing target or job_id"
    missing = [c for c in TEXT_COLS + CAT_COLS + BIN_COLS if c not in df.columns]
    assert not missing, f"missing columns: {missing}"
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


def _desc_hash(text):
    """Stable short hash of the normalised description, used as the split group id."""
    return hashlib.sha256(_normalise_text(text).encode("utf-8")).hexdigest()[:16]


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


def clean(df):
    """Drop exact duplicate postings, check labels agree, and tag description groups."""
    cols = _content_cols(df)
    conflicts = _count_duplicate_conflicts(df)
    # Keeping the first row of a block with two labels would hide a real label error.
    assert conflicts == 0, f"{conflicts} duplicate blocks disagree on {TARGET}; review the data"
    out = df.drop_duplicates(subset=cols, keep="first").copy()
    out["group"] = out["description"].map(_desc_hash)
    out = out.reset_index(drop=True)
    assert not out.duplicated(subset=cols).any(), "duplicates survived the drop"
    assert out["group"].notna().all(), "group hash failed"
    return out


def _require_groups(groups, n_splits):
    """Fail early when there are too few description groups to fill the folds."""
    found = len(np.unique(groups))
    if found < n_splits:
        raise ValueError(f"need at least {n_splits} distinct description groups, found {found}")


def _check_splits(splits, y, groups):
    """Assert splits cover every row once, share no group and stay close in balance."""
    names = list(splits)
    stacked = np.concatenate([splits[n] for n in names])
    assert np.array_equal(np.sort(stacked), np.arange(len(y))), "splits must cover every row once"
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = set(groups[splits[a]]) & set(groups[splits[b]])
            assert not shared, f"{len(shared)} description groups appear in both {a} and {b}"
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
    n_test_folds = max(2, round(1.0 / test_size))
    _require_groups(groups, n_test_folds)
    # StratifiedGroupKFold keeps a whole description in one split and balances the label.
    outer = StratifiedGroupKFold(n_splits=n_test_folds, shuffle=True, random_state=seed)
    trainval, test = next(outer.split(np.zeros((n, 1)), y, groups))
    n_val_folds = max(2, round(1.0 / (val_size / (1.0 - test_size))))
    _require_groups(groups[trainval], n_val_folds)
    inner = StratifiedGroupKFold(n_splits=n_val_folds, shuffle=True, random_state=seed + 1)
    tr, va = next(inner.split(np.zeros((len(trainval), 1)), y[trainval], groups[trainval]))
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
    assert str(z["fingerprint"]) == _fingerprint(df), "splits were made for different data; rebuild them"
    return {k: z[k] for k in ("train", "val", "test")}


def _fingerprint(df):
    """Short hash of the ordered job_ids; changes if rows are dropped, added or reordered."""
    ids = ",".join(map(str, df["job_id"].tolist()))
    return hashlib.sha256(ids.encode("utf-8")).hexdigest()[:16]


def save_metadata(path, df, splits, seed):
    """Write split sizes, positives and column lists to a JSON file."""
    y = df[TARGET].to_numpy()
    meta = {
        "seed": int(seed),
        "n_rows": int(len(df)),
        "splits": {
            name: {
                "size": int(len(idx)),
                "positives": int(y[idx].sum()),
                "positive_rate": round(float(y[idx].mean()), 6),
            }
            for name, idx in splits.items()
        },
        "columns": {
            "all": list(df.columns),
            "text": TEXT_COLS,
            "categorical": CAT_COLS,
            "binary": BIN_COLS,
        },
    }
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_metadata(path):
    """Read back the JSON written by save_metadata."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def seed_everything(seed):
    """Seed Python and numpy, plus torch when it happens to be installed."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch  # lazy: torch is optional and must never slow down a normal import
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
