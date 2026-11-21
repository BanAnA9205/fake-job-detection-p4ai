"""Baselines, candidate models, tuning grids and the grouped CV search helper."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.svm import OneClassSVM

from modules.features import make_preprocessor, make_tfidf


def get_baselines(seed=0):
    """The dumb reference model that every real model must beat."""
    return {"dummy_majority": DummyClassifier(strategy="most_frequent", random_state=seed)}


class OneClassDetector(ClassifierMixin, BaseEstimator):
    """Anomaly detection as a fraud score: fit on real ads only, score = how unusual an ad is.

    The labels are used only to keep the real ads; the detector never sees a fraud example.
    """

    def __init__(self, detector):
        self.detector = detector

    def fit(self, X, y):
        # Learn what a normal (real) ad looks like from the real training ads only.
        self.detector_ = clone(self.detector).fit(X[np.asarray(y) == 0])
        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, X):
        # OneClassSVM gives high scores to normal points, so flip the sign: high = unusual = fraud.
        return -self.detector_.decision_function(X)


def _logreg(seed):
    # class_weight="balanced" makes a missed fraud cost about 20x a false alarm (4.8% positives).
    return LogisticRegression(class_weight="balanced", solver="liblinear",
                              max_iter=1000, random_state=seed)


def get_models(seed=0):
    """The two models: preprocessing ("pre") followed by a classifier ("clf").

    logreg uses the full TF-IDF feature matrix; ocsvm uses the TF-IDF text only.
    """
    return {
        "logreg": Pipeline([("pre", make_preprocessor()), ("clf", _logreg(seed))]),
        # Unsupervised, text only: in the full feature matrix four numeric columns make up
        # about half of each row's length and would dominate the distances an RBF kernel uses.
        # The kernel also needs short dense input, so TF-IDF is reduced with PCA ("arpack"
        # works on the sparse matrix directly) and each row is rescaled to unit length.
        "ocsvm": Pipeline([
            ("pre", ColumnTransformer([("text", make_tfidf(), "text")])),
            ("clf", OneClassDetector(Pipeline([
                ("pca", PCA(n_components=100, svd_solver="arpack", random_state=seed)),
                ("norm", Normalizer()),
                # Defaults = the best setting of a grouped-CV map on the seed-0 training rows.
                ("svm", OneClassSVM(kernel="rbf", nu=0.1, gamma=30.0)),
            ]))),
        ]),
    }


# Small grids for tune(); names follow the Pipeline structure ("step__part__param").
PARAM_GRIDS = {
    "logreg": {
        "clf__C": [0.1, 1, 10],
        "pre__text__max_features": [10000, 30000],
        "pre__text__ngram_range": [(1, 1), (1, 2)],
    },
    # gamma sets how local the RBF kernel is (rows have unit length, so squared distances lie
    # between 0 and 4). A seed-0 map of the CV score showed that gamma decides almost everything:
    # broad kernels (gamma <= 1) rank frauds as more typical than real ads, the score peaks
    # around gamma = 30 and falls again beyond it, and nu (roughly the share of training ads
    # allowed to look unusual) barely matters.
    "ocsvm": {
        "clf__detector__svm__nu": [0.05, 0.2],
        "clf__detector__svm__gamma": [3.0, 10.0, 30.0, 100.0, 300.0],
    },
}

# Models whose whole grid is searched instead of n_iter random candidates.
FULL_GRID = {"ocsvm"}


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
