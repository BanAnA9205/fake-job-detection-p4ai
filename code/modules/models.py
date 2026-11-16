"""Baselines, candidate models, tuning grids and the grouped CV search helper."""

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from modules.features import make_preprocessor


def get_baselines(seed=0):
    """The two dumb reference models that every real model must beat."""
    return {
        "dummy_majority": DummyClassifier(strategy="most_frequent", random_state=seed),
        "dummy_stratified": DummyClassifier(strategy="stratified", random_state=seed),
    }


def _logreg(seed):
    # class_weight="balanced" makes a missed fraud cost about 20x a false alarm (4.8% positives).
    return LogisticRegression(class_weight="balanced", solver="liblinear",
                              max_iter=1000, random_state=seed)


def get_models(seed=0):
    """Four candidate pipelines. Each is preprocessing ("pre") followed by a classifier ("clf").

    logreg and logreg_bm25 differ only in how words are weighted, so the pair isolates
    TF-IDF versus BM25. tree and mlp use the TF-IDF features.
    """
    return {
        "logreg": Pipeline([("pre", make_preprocessor("tfidf")), ("clf", _logreg(seed))]),
        "logreg_bm25": Pipeline([("pre", make_preprocessor("bm25")), ("clf", _logreg(seed))]),
        "tree": Pipeline([
            ("pre", make_preprocessor("tfidf")),
            ("clf", DecisionTreeClassifier(class_weight="balanced", min_samples_leaf=5,
                                           random_state=seed)),
        ]),
        "mlp": Pipeline([
            ("pre", make_preprocessor("tfidf")),
            # MLPClassifier has no class_weight; the decision threshold is chosen on
            # validation later, which handles the imbalance for ranking-based scores.
            ("clf", MLPClassifier(hidden_layer_sizes=(128,), early_stopping=True,
                                  n_iter_no_change=5, max_iter=100, random_state=seed)),
        ]),
    }


# Small grids for tune(); names follow the Pipeline structure ("step__part__param").
# In the logreg_bm25 grid the text pipeline has one more level: pre__text__bm25__k1.
PARAM_GRIDS = {
    "logreg": {
        "clf__C": [0.1, 1, 10],
        "pre__text__max_features": [10000, 30000],
        "pre__text__ngram_range": [(1, 1), (1, 2)],
    },
    "logreg_bm25": {
        "clf__C": [0.1, 1, 10],
        "pre__text__bm25__k1": [1.2, 1.5, 2.0],
        "pre__text__bm25__b": [0.5, 0.75, 1.0],
    },
    "tree": {
        "clf__max_depth": [10, 20, 40],
        "clf__min_samples_leaf": [2, 5, 20],
    },
    "mlp": {
        "clf__hidden_layer_sizes": [(64,), (128,), (256,)],
        "clf__alpha": [1e-4, 1e-3, 1e-2],
    },
}


def tune(pipe, grid, X, y, groups, n_iter=8, cv=5, scoring="average_precision",
         seed=0, n_jobs=1):
    """Random search on the training rows only; returns the fitted search object."""
    # CV folds: keep the class ratio per fold and keep rows with the same description
    # (a duplicate group) inside one fold, so CV does not leak.
    splitter = StratifiedGroupKFold(n_splits=cv, shuffle=True, random_state=seed)
    search = RandomizedSearchCV(
        pipe,
        param_distributions=grid,
        n_iter=n_iter,
        scoring=scoring,
        cv=splitter,
        random_state=seed,
        n_jobs=n_jobs,
        error_score="raise",
    )

    # Fit on the training rows; groups tells the splitter which rows share a description.
    search.fit(X, y, groups=groups)
    return search
