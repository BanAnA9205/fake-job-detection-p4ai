"""Data quality checks for EMSCAD, and the grouping that keeps related ads in one split.

The check functions only read the data and return counts or tables; notebook 01 shows their
results. `group_ids` is used by data_helper.clean to build the split groups.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Cell values that look filled but carry no content (Excel errors and filler).
JUNK_VALUES = {"#NAME?", "#VALUE!", "#REF!", ".", "-", "N/A", "n/a", "NA", "none", "None"}

# The dataset hides each real link, email and phone number behind a hashed placeholder. Some are
# nested ("#EMAIL_3a1e#PHONE_ad7b##": a phone number inside an email), so the closing "#" is not
# part of the pattern; replacing "#KIND_hash" with "#KIND" handles both forms.
PLACEHOLDER = r"#(URL|EMAIL|PHONE)_[0-9a-fA-F]+"

# Boundaries where removed HTML glued two sentences or two words together.
GLUE_SENTENCE = r"([a-z])([.!?:;])([A-Z])"   # "place.We"     -> "place. We"
GLUE_WORD = r"([a-z]{2})([A-Z][a-z])"         # "overviewAker" -> "overview Aker"

NEAR_DUP_COSINE = 0.9  # descriptions at least this similar count as copies of one template


def _norm(s):
    """Lowercase, trim and collapse whitespace."""
    return s.fillna("").astype(str).str.lower().str.split().str.join(" ")


# ---------------------------------------------------------------------------
# Checks (read-only)
# ---------------------------------------------------------------------------

def disguised_missing(df, cols):
    """Per column: true NaN, junk cells that only look filled, and the share missing by class."""
    y = df["fraudulent"]
    rows = []
    for c in cols:
        nan = df[c].isna()
        junk = (~nan) & df[c].astype(str).str.strip().isin(JUNK_VALUES)
        miss = nan | junk
        rows.append({"column": c, "NaN": int(nan.sum()), "junk": int(junk.sum()),
                     "missing % real": round(100 * miss[y == 0].mean(), 1),
                     "missing % fraud": round(100 * miss[y == 1].mean(), 1)})
    return pd.DataFrame(rows).set_index("column")


def placeholder_counts(text):
    """Ads containing each kind of placeholder, and how many distinct hashes each kind uses."""
    rows = {}
    for kind in ("URL", "EMAIL", "PHONE"):
        hashes = text.str.findall(f"#{kind}_[0-9a-fA-F]+")
        rows[kind] = {"ads": int((hashes.str.len() > 0).sum()),
                      "distinct hashes": len({h for hs in hashes for h in hs})}
    return pd.DataFrame(rows).T


def glued_boundaries(text):
    """Per ad, how many sentence or word boundaries were glued together by removed HTML."""
    return text.str.count(GLUE_SENTENCE) + text.str.count(GLUE_WORD)


def group_purity(groups, y):
    """Groups with 2+ ads: how many are all fraud, all real, or mixed."""
    g = pd.DataFrame({"group": groups, "y": y}).groupby("group")["y"].agg(["size", "mean"])
    multi = g[g["size"] > 1]
    return {"groups with 2+ ads": len(multi), "all fraud": int((multi["mean"] == 1).sum()),
            "all real": int((multi["mean"] == 0).sum()),
            "mixed": int(((multi["mean"] > 0) & (multi["mean"] < 1)).sum())}


def split_overlap(splits, keys, y):
    """For each key (company, description cluster, ...): test ads whose key value also occurs in train.

    keys maps a name to one value per row; empty strings mean "no value" and never count as shared.
    """
    rows = {}
    for name, key in keys.items():
        key = np.asarray(key, dtype=object)
        seen = set(key[splits["train"]]) - {""}
        hit = np.array([k in seen for k in key[splits["test"]]])
        rows[name] = {"test ads": int(hit.sum()),
                      "test frauds": int(y[splits["test"]][hit].sum())}
    return pd.DataFrame(rows).T


def assert_clean(df, text_cols):
    """Fail loudly if cleaning left junk cells, hashed placeholders, duplicates or missing groups."""
    junk = sum(int(df[c].astype(str).str.strip().isin(JUNK_VALUES).sum()) for c in text_cols)
    hashed = sum(int(df[c].str.count(PLACEHOLDER).fillna(0).sum()) for c in text_cols)
    dups = int(df.drop(columns=["job_id", "group"]).duplicated().sum())
    assert junk == 0, f"{junk} junk cells left"
    assert hashed == 0, f"{hashed} hashed placeholders left"
    assert dups == 0, f"{dups} duplicate ads left"
    assert df["group"].notna().all(), "rows without a group"


# ---------------------------------------------------------------------------
# Grouping used for the splits
# ---------------------------------------------------------------------------

def near_duplicate_clusters(texts, threshold=NEAR_DUP_COSINE):
    """Cluster id per text: texts linked by TF-IDF cosine >= threshold, directly or through others."""
    try:
        X = TfidfVectorizer(min_df=2, sublinear_tf=True).fit_transform(texts)
    except ValueError:  # no word occurs twice, so no two texts can be near-copies
        return np.arange(len(texts))
    graph = NearestNeighbors(radius=1 - threshold, metric="cosine",
                             algorithm="brute").fit(X).radius_neighbors_graph(X)
    return connected_components(graph, directed=False)[1]


def _key_edges(key):
    """Graph edges joining the rows that share a key value (a chain through each value's rows)."""
    key = pd.Series(np.asarray(key, dtype=object))
    rows, cols = [], []
    for idx in key.groupby(key).indices.values():
        rows.extend(idx[:-1])
        cols.extend(idx[1:])
    return np.asarray(rows, dtype=int), np.asarray(cols, dtype=int)


def group_ids(df, threshold=NEAR_DUP_COSINE):
    """One split group per set of related ads.

    Ads are related when they share the exact description (including an empty one), the same
    non-empty company profile, or a near-duplicate description cluster; relations chain.
    """
    n = len(df)
    desc, company = _norm(df["description"]), _norm(df["company_profile"])
    company = company.where(company != "", pd.Series([f"_none{i}" for i in range(n)], index=df.index))
    keys = [desc.to_numpy(), company.to_numpy(), near_duplicate_clusters(desc, threshold)]

    edges = [_key_edges(k) for k in keys]
    rows = np.concatenate([e[0] for e in edges])
    cols = np.concatenate([e[1] for e in edges])
    graph = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    labels = connected_components(graph, directed=False)[1]
    return pd.Series([f"g{k}" for k in labels], index=df.index)
