"""Baselines, candidate models, tuning grids and the grouped CV search helper."""

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from modules.features import (BINARY_COLS, NOMINAL_COLS, NUMERIC_COLS,
                              ORDINAL_COLS, make_preprocessor, make_tfidf)

# Number of dense text components given to the tree model.
TEXT_SVD_COMPONENTS = 100


def get_baselines(seed=0):
    """The two dumb reference models that every real model must beat."""
    return {
        "dummy_majority": DummyClassifier(strategy="most_frequent", random_state=seed),
        "dummy_stratified": DummyClassifier(strategy="stratified", random_state=seed),
    }


def _hist_gb_pipeline(seed):
    """Dense pipeline for the tree model: compressed text plus metadata columns."""
    meta_cols = NOMINAL_COLS + ORDINAL_COLS + NUMERIC_COLS + BINARY_COLS
    text_branch = Pipeline([
        ("tfidf", make_tfidf()),
        # TruncatedSVD is LSA: 20000 sparse TF-IDF columns become 100 dense ones,
        # because HistGradientBoostingClassifier cannot take sparse input.
        ("svd", TruncatedSVD(n_components=TEXT_SVD_COMPONENTS, random_state=seed)),
    ])
    pre = ColumnTransformer(
        [("meta", make_preprocessor("meta"), meta_cols),
         ("text_svd", text_branch, "text")],
        sparse_threshold=0,
    )
    clf = HistGradientBoostingClassifier(class_weight="balanced", random_state=seed)
    return Pipeline([("pre", pre), ("clf", clf)])


def get_models(seed=0, kind_pre="tfidf"):
    """Three candidate pipelines: two sparse linear models and one dense tree model.

    kind_pre picks the preprocessor for the linear models; hist_gb always uses
    its own dense text plus metadata block.
    """
    logreg = Pipeline([
        ("pre", make_preprocessor(kind_pre)),
        ("clf", LogisticRegression(class_weight="balanced", solver="liblinear",
                                   max_iter=1000, random_state=seed)),
    ])
    svm = Pipeline([
        ("pre", make_preprocessor(kind_pre)),
        ("clf", LinearSVC(class_weight="balanced", dual="auto", max_iter=5000,
                          random_state=seed)),
    ])
    return {"logreg": logreg, "linear_svm": svm, "hist_gb": _hist_gb_pipeline(seed)}


# Small grids for tune(); names follow the Pipeline structure (step__param).
# The pre__text__... keys assume the default kind_pre="tfidf".
PARAM_GRIDS = {
    "logreg": {
        "clf__C": [0.1, 1, 10],
        "pre__text__max_features": [10000, 30000],
        "pre__text__ngram_range": [(1, 1), (1, 2)],
    },
    "linear_svm": {
        "clf__C": [0.01, 0.1, 1],
        "pre__text__max_features": [10000, 30000],
        "pre__text__ngram_range": [(1, 1), (1, 2)],
    },
    "hist_gb": {
        "clf__learning_rate": [0.05, 0.1],
        "clf__max_leaf_nodes": [15, 31, 63],
        "pre__text_svd__tfidf__max_features": [10000, 30000],
    },
}


def tune(pipe, grid, X, y, groups, n_iter=8, cv=5, scoring="average_precision",
         seed=0, n_jobs=1):
    """Random search on the training rows only; returns the fitted search object."""
    # StratifiedGroupKFold keeps the class ratio per fold and keeps rows with the
    # same description (a duplicate group) inside one fold, so CV does not leak.
    splitter = StratifiedGroupKFold(n_splits=cv, shuffle=True, random_state=seed)
    search = RandomizedSearchCV(
        pipe,
        param_distributions=grid,
        n_iter=n_iter,
        scoring=scoring,
        cv=splitter,
        random_state=seed,
        n_jobs=n_jobs,
        refit=True,
        error_score="raise",
    )
    search.fit(X, y, groups=groups)
    return search
