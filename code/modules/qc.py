"""Data-quality checks and exact related-ad grouping for EMSCAD."""

import re

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

# Values that look populated but contain no useful information.
JUNK_VALUES = {"#NAME?", "#VALUE!", "#REF!", ".", "-", "N/A", "n/a", "NA", "none", "None"}
PLACEHOLDER = r"#(URL|EMAIL|PHONE)_([0-9a-fA-F]+)"


def _norm(s: pd.Series) -> pd.Series:
    """Lowercase, trim, and collapse whitespace for exact comparisons."""
    return s.fillna("").astype(str).str.lower().str.split().str.join(" ")


def single_value_columns(df: pd.DataFrame) -> list[str]:
    """Return non-key columns containing at most one distinct value, including missing."""
    protected = {"job_id", "fraudulent", "group"}
    return [
        c for c in df.columns
        if c not in protected and df[c].nunique(dropna=False) <= 1
    ]


def contact_keys(df: pd.DataFrame, text_cols: list[str],
                 kinds: tuple[str, ...] = ("EMAIL", "PHONE")) -> pd.Series:
    """Return original contact identities per row before placeholders are genericized.

    URLs default to exclusion: shared websites link huge, heterogeneous collections of ads,
    unlike direct email/phone contact details. Consequently, split guarantees are explicitly
    limited to exact descriptions, company profiles, email addresses, and phone numbers.
    """
    text = df[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
    keys = text.map(lambda value: tuple(sorted({
        f"{kind}:{digest.lower()}"
        for kind, digest in re.findall(PLACEHOLDER, value)
        if kind in kinds
    })))
    keys.index = df.index
    return keys


def _key_edges(key: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Create a chain of graph edges for rows sharing one exact key."""
    rows, cols = [], []
    for idx in key.groupby(key).indices.values():
        rows.extend(idx[:-1])
        cols.extend(idx[1:])
    return np.asarray(rows, dtype=int), np.asarray(cols, dtype=int)


def group_ids(df: pd.DataFrame, contacts: pd.Series | None = None) -> pd.Series:
    """Group rows sharing a normalized description, company profile, or original contact.

    Empty normalized descriptions share the empty-description key. Empty company profiles instead
    receive distinct synthetic keys so missing company data do not collapse unrelated ads. Relations
    chain through shared identities. No fuzzy text similarity or record removal
    is involved.
    """
    n = len(df)
    description = _norm(df["description"])
    company = _norm(df["company_profile"])
    company = company.where(company != "", pd.Series(
        [f"_none{i}" for i in range(n)], index=df.index
    ))

    edges = [_key_edges(description), _key_edges(company)]
    if contacts is not None:
        assert len(contacts) == n and contacts.index.equals(df.index)
        long = [(pos, key) for pos, values in enumerate(contacts) for key in values]
        if long:
            frame = pd.DataFrame(long, columns=["row", "contact"])
            contact_rows, contact_cols = [], []
            row_ids = frame["row"].to_numpy()
            for idx in frame.groupby("contact").indices.values():
                linked = row_ids[idx]
                contact_rows.extend(linked[:-1])
                contact_cols.extend(linked[1:])
            edges.append((np.asarray(contact_rows, dtype=int), np.asarray(contact_cols, dtype=int)))

    rows = np.concatenate([edge[0] for edge in edges])
    cols = np.concatenate([edge[1] for edge in edges])
    graph = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    labels = connected_components(graph, directed=False)[1]
    return pd.Series([f"g{label}" for label in labels], index=df.index)
