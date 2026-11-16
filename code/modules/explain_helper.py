"""Explanation helpers: linear coefficients, tree importances and rules, error reading."""

import numpy as np
import pandas as pd
from sklearn.tree import export_text


def top_linear_terms(pipe, k=20):
    """Top k positive and top k negative terms of the fitted linear model.

    pipe must be a fitted Pipeline with steps named "pre" (ColumnTransformer)
    and "clf" (a linear estimator that exposes coef_).
    """
    # One coefficient per feature name, sorted from most real-like to most fraud-like.
    coefs = pd.Series(np.ravel(pipe.named_steps["clf"].coef_),
                      index=pipe.named_steps["pre"].get_feature_names_out()).sort_values()

    # Both ends: the k most fraud-like and the k most real-like terms.
    ends = {"positive": coefs[::-1].head(k), "negative": coefs.head(k)}
    return pd.concat([pd.DataFrame({"term": c.index, "coef": c.values, "direction": d})
                      for d, c in ends.items()], ignore_index=True)


def show_errors(df, y_true, score, thr, k=10):
    """k most confident false positives and false negatives, with short previews.

    False positives are ranked by highest score (model was most sure they were
    fraud), false negatives by lowest score (model was most sure they were not).
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype=float)

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
    imp = pd.Series(pipe.named_steps["clf"].feature_importances_,
                    index=pipe.named_steps["pre"].get_feature_names_out())

    # Top k features by impurity importance.
    top = imp.nlargest(k)
    return pd.DataFrame({"term": top.index, "importance": top.values})


def tree_rules(pipe, max_depth=3):
    """The top levels of a fitted decision tree as readable if/else text."""
    pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
    return export_text(clf, feature_names=list(pre.get_feature_names_out()),
                       max_depth=max_depth)
