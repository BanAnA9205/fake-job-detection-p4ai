"""Shared preprocessing, classical estimators, and compact tuning grids."""

import numpy as np
from modules.features import BINARY_COLS, NOMINAL_COLS, NUMERIC_COLS
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, OneHotEncoder, StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.tree import DecisionTreeClassifier


def make_tfidf(max_features: int = 20000, ngram_range: tuple[int, int] = (1, 2),
               min_df: int = 3) -> TfidfVectorizer:
    """Return the shared word TF-IDF configuration."""
    return TfidfVectorizer(
        sublinear_tf=True,
        strip_accents="unicode",
        stop_words="english",
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
    )


def make_preprocessor(max_features: int = 20000, ngram_range: tuple[int, int] = (1, 2),
                      min_df: int = 3, structured: bool = True) -> ColumnTransformer:
    """Build the feature matrix without pruning rare categories or binary flags."""
    transformers = [("text", make_tfidf(max_features, ngram_range, min_df), "text")]
    if structured:
        categorical = Pipeline([
            ("missing", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.extend([
            ("nominal", categorical, NOMINAL_COLS),
            ("numeric", StandardScaler(with_mean=False), NUMERIC_COLS),
            ("binary", "passthrough", BINARY_COLS),
        ])
    return ColumnTransformer(transformers)


class OneClassDetector(ClassifierMixin, BaseEstimator):
    """Fit a complete anomaly pipeline on known-real rows and score unusualness as fraud."""

    def __init__(self, detector):
        self.detector = detector

    def fit(self, X, y):
        """Fit every detector stage without fraudulent training rows."""
        self.detector_ = clone(self.detector).fit(X[np.asarray(y) == 0])
        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, X):
        """Flip OneClassSVM's score so larger values indicate greater fraud unusualness."""
        return -self.detector_.decision_function(X)


def _logreg(seed: int) -> LogisticRegression:
    return LogisticRegression(
        class_weight="balanced", solver="liblinear", max_iter=1000, random_state=seed
    )


def get_models(seed: int = 0) -> dict[str, object]:
    """Return the two LR feature variants, decision tree, and one-class detector."""
    return {
        "logreg_text": Pipeline([
            ("pre", make_preprocessor(structured=False)),
            ("clf", _logreg(seed)),
        ]),
        "logreg": Pipeline([
            ("pre", make_preprocessor()),
            ("clf", _logreg(seed)),
        ]),
        "tree": Pipeline([
            ("pre", make_preprocessor()),
            ("clf", DecisionTreeClassifier(class_weight="balanced", random_state=seed)),
        ]),
        # PCA reduces sparse TF-IDF before the RBF kernel; all stages see real rows only.
        "ocsvm": OneClassDetector(Pipeline([
            ("pre", ColumnTransformer([("text", make_tfidf(), "text")])),
            ("pca", PCA(n_components=100, svd_solver="arpack", random_state=seed)),
            ("norm", Normalizer()),
            ("svm", OneClassSVM(kernel="rbf", nu=0.1, gamma="scale")),
        ])),
    }


PARAM_GRIDS = {
    "logreg_text": {"clf__C": [0.1, 1, 10]},
    "logreg": {"clf__C": [0.1, 1, 10]},
    "tree": {
        "clf__max_depth": [5, 10, None],
        "clf__min_samples_leaf": [1, 10],
    },
    "ocsvm": {
        "detector__svm__nu": [0.05, 0.1],
        "detector__svm__gamma": [1.0, 10.0, 30.0],
    },
}
