"""Tests for modules.models on a handful of hand-written synthetic rows.

Run with pytest, or directly: .venv/bin/python tests/test_models.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
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
    for name, m in models_mod.get_baselines().items():
        m.fit(X, df["fraudulent"])
        assert m.predict(X).shape == (len(df),), name


def test_models_are_named_pipelines():
    models = models_mod.get_models()
    assert set(models) == {"logreg", "logreg_bm25", "mlp", "cnb", "ocsvm"}
    for name, pipe in models.items():
        assert isinstance(pipe, Pipeline), name
        assert [step for step, _ in pipe.steps] == ["pre", "clf"], name


def test_param_grids_match_pipeline_params():
    models = models_mod.get_models()
    assert set(models_mod.PARAM_GRIDS) == set(models)
    for name, grid in models_mod.PARAM_GRIDS.items():
        valid = set(models[name].get_params())
        for key in grid:
            assert key in valid, f"{name}: {key} is not a parameter of the pipeline"


def test_every_model_fits_and_scores():
    """Each pipeline learns the toy spam/normal split and ranks spam above normal."""
    df, X = _xy()
    y = df["fraudulent"].to_numpy()
    # Toy-sized overrides: keep every word, and let the MLP train on all 60 rows
    # (its early stopping would hold out only 6 rows, too few to learn from).
    small = {"pre__text__min_df": 1, "clf__early_stopping": False, "clf__max_iter": 300}
    for name, pipe in models_mod.get_models().items():
        if name == "ocsvm":
            continue  # unsupervised, no predict_proba: tested on its own below
        pipe.set_params(**{k: v for k, v in small.items() if k in pipe.get_params()})
        pipe.fit(X, y)
        score = pipe.predict_proba(X)[:, 1]
        assert score.shape == (len(df),), name
        assert score[y == 1].mean() > score[y == 0].mean() + 0.2, name


def test_one_class_detector_scores_outliers_higher():
    """Fit on label-0 rows only; points far from them must get the higher (fraud) score."""
    from sklearn.svm import OneClassSVM

    rng = np.random.default_rng(0)
    normal = rng.normal(0, 1, size=(200, 2))
    outliers = rng.normal(6, 1, size=(20, 2))
    X = np.vstack([normal, outliers])
    y = np.r_[np.zeros(200, int), np.ones(20, int)]
    det = models_mod.OneClassDetector(OneClassSVM(nu=0.05, gamma="scale")).fit(X, y)
    assert det.detector_.fit_status_ == 0
    assert det.detector_.support_vectors_.max() < 5, "fraud rows must not be used for fitting"
    score = det.decision_function(X)
    assert score[y == 1].min() > np.median(score[y == 0])


def test_tune_returns_fitted_search():
    df, X = _xy()
    y, groups = df["fraudulent"].to_numpy(), df["group"].to_numpy()
    pipe = models_mod.get_models()["logreg"]
    pipe.set_params(pre__text__min_df=1)
    grid = {"clf__C": [0.1, 1.0]}
    search = models_mod.tune(pipe, grid, X, y, groups, n_iter=2, cv=3)
    assert search.best_params_["clf__C"] in (0.1, 1.0)
    assert hasattr(search.best_estimator_, "predict_proba")


def main():
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print("ok", test.__name__)
    print(f"{len(tests)} models checks passed")


if __name__ == "__main__":
    main()
