"""Explanation helpers: linear coefficients, permutation importance, error reading."""

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.tree import export_text


def top_linear_terms(pipe, k=20):
    """Top k positive and top k negative terms of the fitted linear model.

    pipe must be a fitted Pipeline with steps named "pre" (ColumnTransformer)
    and "clf" (a linear estimator that exposes coef_).
    """
    pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        raise TypeError("top_linear_terms needs a linear estimator with coef_")

    # Feature names from the fitted preprocessor, aligned with the coefficients.
    names = np.asarray(pre.get_feature_names_out())
    # Binary linear models store coef_ as (n_features,) or (1, n_features).
    coefs = np.asarray(clf.coef_, dtype=float).ravel()
    assert len(names) == len(coefs), "feature names and coefficients misaligned"

    # Rank the coefficients and take both ends: most fraud-like and most real-like terms.
    order = np.argsort(coefs)  # ascending: most negative first
    neg_idx = order[:k]
    pos_idx = order[::-1][:k]  # descending: most positive first

    pos = pd.DataFrame(
        {"term": names[pos_idx], "coef": coefs[pos_idx], "direction": "positive"}
    )
    neg = pd.DataFrame(
        {"term": names[neg_idx], "coef": coefs[neg_idx], "direction": "negative"}
    )

    return pd.concat([pos, neg], ignore_index=True)


def permutation_table(
    pipe, X, y, scoring="average_precision", n_repeats=5, seed=0, n_jobs=1, cols=None
):
    """Permutation importance on the RAW input columns of the dataframe.

    Permutation importance shuffles one raw input column at a time and measures
    how much the score drops; the bigger the drop, the more the model relied on
    that column. Shuffling raw (not transformed) columns keeps the result easy
    to explain.
    """
    n_jobs = max(1, min(int(n_jobs), 4))  # safety cap from the project spec

    # Shuffle each raw column n_repeats times and record the score drop.
    r = permutation_importance(
        pipe, X, y, scoring=scoring, n_repeats=n_repeats,
        random_state=seed, n_jobs=n_jobs,
    )

    # Tabulate by column name.
    if hasattr(X, "columns"):
        col_names = list(X.columns)
    else:
        col_names = [f"x{i}" for i in range(r.importances.shape[0])]
    out = pd.DataFrame(
        {
            "col": col_names,
            "importance_mean": r.importances_mean,
            "importance_std": r.importances_std,
        }
    )

    # We always permute every column (the pipeline needs them all) and only
    # filter the table afterwards when the caller asks for a subset.
    if cols is not None:
        out = out[out["col"].isin(cols)]
    return out.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def show_errors(df, y_true, score, thr, k=10):
    """k most confident false positives and false negatives, with short previews.

    False positives are ranked by highest score (model was most sure they were
    fraud), false negatives by lowest score (model was most sure they were not).
    """
    assert {"title", "description"} <= set(df.columns), "df needs title and description"
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype=float)
    assert len(df) == len(y) == len(s), "df, y_true and score must align"

    # Find the wrong calls at this threshold.
    pred = s >= thr
    fp_idx = np.where((pred == 1) & (y == 0))[0]
    fn_idx = np.where((pred == 0) & (y == 1))[0]

    # Keep the k most confident mistakes of each kind.
    fp_idx = fp_idx[np.argsort(-s[fp_idx])][:k]
    fn_idx = fn_idx[np.argsort(s[fn_idx])][:k]

    # One table row per mistake, with a text preview.
    rows = []
    for kind, idx in (("false_positive", fp_idx), ("false_negative", fn_idx)):
        for i in idx:
            row = df.iloc[i]
            rows.append(
                {
                    "error_type": kind,
                    "score": float(s[i]),
                    "y_true": int(y[i]),
                    "title": str(row["title"]),
                    # First 200 chars is enough to read the error in a slide.
                    "description": str(row["description"])[:200],
                }
            )
    return pd.DataFrame(rows)


def top_tree_features(pipe, k=20):
    """The k features a fitted decision tree relies on most (impurity importance)."""
    pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        raise TypeError("top_tree_features needs a tree estimator with feature_importances_")
    names = np.asarray(pre.get_feature_names_out())
    imp = np.asarray(clf.feature_importances_, dtype=float)
    assert len(names) == len(imp), "feature names and importances misaligned"

    # Top k features by impurity importance.
    order = np.argsort(-imp)[:k]
    return pd.DataFrame({"term": names[order], "importance": imp[order]})


def tree_rules(pipe, max_depth=3):
    """The top levels of a fitted decision tree as readable if/else text."""
    pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
    return export_text(clf, feature_names=list(pre.get_feature_names_out()),
                       max_depth=max_depth)
