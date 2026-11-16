"""Explanation helpers: linear coefficients, tree importances and rules, error reading."""

import numpy as np
import pandas as pd
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
