"""Tests for modules.models on a handful of hand-written synthetic rows.

Run with pytest, or directly: .venv/bin/python tests/test_models.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules import models as models_mod
from modules.features import add_engineered_features

# Ten repeated templates, four spam-like and six normal. Repeating each text six
# times keeps every token above TF-IDF min_df=3 and makes one duplicate "group".
TEMPLATES = [
    ("Data entry clerk", "Operations", 1,
     "Earn money fast from home, no experience needed, wire transfer only."),
    ("Payment processor", "Finance", 1,
     "Work from home and earn 500 per week, send your bank details today."),
    ("Mystery shopper", "Sales", 1,
     "No experience required, make money online, western union payments only."),
    ("Customer service agent", "Support", 1,
     "Urgent hiring, earn cash daily from home, contact us at hr@fastcash.example."),
    ("Software engineer", "Engineering", 0,
     "Build python data pipelines with sql and docker for the platform team."),
    ("Accountant", "Finance", 0,
     "Prepare monthly finance reports in excel for the audit team."),
    ("Marketing assistant", "Marketing", 0,
     "Run email campaigns and track sales leads in the crm."),
    ("Lecturer", "Education", 0,
     "Teach undergraduate statistics and supervise student projects."),
    ("Maintenance technician", "Operations", 0,
     "Repair and maintain production machines during the night shift."),
    ("Clinical coordinator", "Research", 0,
     "Coordinate clinical trial documentation with the research nurses."),
]


def _toy_df(n_per=6):
    """Small deterministic frame with the real schema: 60 rows, 24 positives."""
    rows = []
    job_id = 0
    for t, (title, department, label, description) in enumerate(TEMPLATES):
        for k in range(n_per):
            industry = ["IT", "Sales"][t % 2]
            rows.append({
                "job_id": job_id,
                "title": title,
                "location": ["US, NY", "GB, London"][k % 2],
                "department": department,
                "salary_range": "50000-60000" if k % 2 == 0 else np.nan,
                "company_profile": f"We are a growing company in the {industry} sector.",
                "description": description,
                "requirements": "See the job description for details.",
                "benefits": "Health insurance and paid leave.",
                "telecommuting": k % 2,
                "has_company_logo": 1,
                "has_questions": int(k % 3 == 0),
                "employment_type": ["Full-time", "Part-time"][k % 2],
                "required_experience": ["Entry level", "Mid-Senior level"][k % 2],
                "required_education": ["Bachelor's Degree",
                                       "High School or equivalent"][k % 2],
                "industry": industry,
                "function": ["Engineering", "Marketing"][t % 2],
                "fraudulent": label,
                "group": f"g{t}",
            })
            job_id += 1
    return pd.DataFrame(rows)


def _xy():
    """Return the toy frame and its engineered feature matrix."""
    df = _toy_df()
    X = add_engineered_features(df).drop(columns=["fraudulent"])
    return df, X


def test_baselines():
    df, X = _xy()
    y = df["fraudulent"].to_numpy()
    baselines = models_mod.get_baselines(seed=0)
    assert set(baselines) == {"dummy_majority", "dummy_stratified"}
    predictions = {}
    for name, clf in baselines.items():
        clf.fit(X, y)
        predictions[name] = clf.predict(X)
    assert predictions["dummy_majority"].sum() == 0, "majority class here is 0"
    assert 0 < predictions["dummy_stratified"].sum() < len(df)


def test_models_are_pipelines():
    models = models_mod.get_models(seed=0)
    assert set(models) == {"logreg", "linear_svm", "hist_gb"}
    for name, pipe in models.items():
        assert isinstance(pipe, Pipeline), name
        assert [step[0] for step in pipe.steps] == ["pre", "clf"], name
    pre = models["hist_gb"].named_steps["pre"]
    assert [name for name, _, _ in pre.transformers] == ["meta", "text_svd"]
    text_branch = pre.transformers[1][1]
    assert text_branch.named_steps["svd"].n_components == models_mod.TEXT_SVD_COMPONENTS


def test_param_grids_match_pipeline_params():
    models = models_mod.get_models(seed=0)
    assert set(models_mod.PARAM_GRIDS) == set(models)
    for name, grid in models_mod.PARAM_GRIDS.items():
        params = models[name].get_params()
        for key, values in grid.items():
            assert key in params, f"{name}: {key} is not a real pipeline parameter"
            assert len(values) >= 2, f"{name}: {key} needs at least two choices"


def test_logreg_fits_and_predicts():
    df, X = _xy()
    y = df["fraudulent"].to_numpy()
    pipe = models_mod.get_models(seed=0)["logreg"]
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (len(df), 2)
    assert np.isfinite(proba).all()
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert set(pipe.predict(X)) <= {0, 1}


def test_linear_svm_fits_and_scores():
    df, X = _xy()
    y = df["fraudulent"].to_numpy()
    pipe = models_mod.get_models(seed=0)["linear_svm"]
    pipe.fit(X, y)
    scores = pipe.decision_function(X)
    assert scores.shape == (len(df),)
    assert np.isfinite(scores).all()


def test_hist_gb_fits_on_dense_text():
    df, X = _xy()
    y = df["fraudulent"].to_numpy()
    old = models_mod.TEXT_SVD_COMPONENTS
    # Tiny text needs fewer components; the real run keeps the module default.
    models_mod.TEXT_SVD_COMPONENTS = 4
    try:
        pipe = models_mod.get_models(seed=0)["hist_gb"]
    finally:
        models_mod.TEXT_SVD_COMPONENTS = old
    pre = pipe.named_steps["pre"]
    Xt = pre.fit_transform(X)
    assert not sparse.issparse(Xt), "hist_gb needs dense input"
    assert Xt.shape[0] == len(df)
    assert Xt.shape[1] > 4
    assert not np.isnan(Xt).any()
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (len(df), 2)
    assert np.isfinite(proba).all()


def test_tune_returns_fitted_search():
    df, X = _xy()
    y = df["fraudulent"].to_numpy()
    grid = {"clf__C": [0.1, 1.0], "pre__text__ngram_range": [(1, 1)]}
    # 10 groups in 4 folds means at most 3 groups per test fold, so the 4 spam
    # groups and 6 normal groups keep both classes in every training fold.
    search = models_mod.tune(
        models_mod.get_models(seed=0)["logreg"], grid, X, y,
        groups=df["group"].to_numpy(), n_iter=2, cv=4, scoring="average_precision",
        seed=0,
    )
    assert isinstance(search.cv, StratifiedGroupKFold)
    assert set(search.best_params_) == set(grid)
    assert 0.0 <= search.best_score_ <= 1.0
    assert search.best_estimator_ is not None


def main():
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print("ok", test.__name__)
    print(f"{len(tests)} models checks passed")


if __name__ == "__main__":
    main()
