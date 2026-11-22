"""Shared coefficient extraction for linear text models."""

import numpy as np
import pandas as pd


def top_linear_terms(pipe, k=20):
    """Return the k most positive and k most negative fitted linear terms."""
    coefs = pd.Series(
        np.ravel(pipe.named_steps["clf"].coef_),
        index=pipe.named_steps["pre"].get_feature_names_out(),
    ).sort_values()
    ends = {"positive": coefs[::-1].head(k), "negative": coefs.head(k)}
    return pd.concat([
        pd.DataFrame({"term": values.index, "coef": values.values, "direction": direction})
        for direction, values in ends.items()
    ], ignore_index=True)
