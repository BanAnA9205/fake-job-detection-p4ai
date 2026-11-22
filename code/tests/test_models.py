"""Small known-answer tests for all classical model families."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules import models as models_mod
from modules.features import add_engineered_features

TEMPLATES = [
    ("Data entry clerk", "Operations", 1, "Earn money fast from home, wire transfer only."),
    ("Payment processor", "Finance", 1, "Work from home and earn 500 per week."),
    ("Mystery shopper", "Sales", 1, "No experience required, western union payments only."),
    ("Customer service agent", "Support", 1, "Urgent hiring, earn cash daily from home."),
    ("Software engineer", "Engineering", 0, "Build python data pipelines with sql and docker."),
    ("Accountant", "Finance", 0, "Prepare monthly finance reports in excel for the audit team."),
    ("Marketing assistant", "Marketing", 0, "Run email campaigns and track sales leads."),
    ("Lecturer", "Education", 0, "Teach undergraduate statistics and supervise student projects."),
    ("Maintenance technician", "Operations", 0, "Repair and maintain factory machines."),
    ("Clinical coordinator", "Research", 0, "Coordinate clinical trial documentation."),
]


def _toy_df(n_per=6):
    rows = []
    job_id = 0
    for template, (title, department, label, description) in enumerate(TEMPLATES):
        for k in range(n_per):
            industry = ["IT", "Sales"][template % 2]
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
                "required_education": ["Bachelor's Degree", "High School or equivalent"][k % 2],
                "industry": industry,
                "function": ["Engineering", "Marketing"][template % 2],
                "fraudulent": label,
                "group": f"g{template}",
            })
            job_id += 1
    return pd.DataFrame(rows)


def _xy():
    df = _toy_df()
    return df, add_engineered_features(df)


def test_models_are_expected_estimators():
    models = models_mod.get_models()
    assert set(models) == {"logreg_text", "logreg", "tree", "ocsvm"}
    for name in ("logreg_text", "logreg", "tree"):
        assert isinstance(models[name], Pipeline)
        assert [step for step, _ in models[name].steps] == ["pre", "clf"]
    assert isinstance(models["ocsvm"], models_mod.OneClassDetector)


def test_param_grids_match_pipeline_params():
    models = models_mod.get_models()
    assert set(models_mod.PARAM_GRIDS) == set(models)
    for name, grid in models_mod.PARAM_GRIDS.items():
        assert all(key in models[name].get_params() for key in grid)


def test_supervised_models_fit_and_score():
    df, X = _xy()
    y = df["fraudulent"].to_numpy()
    for name in ("logreg_text", "logreg", "tree"):
        pipe = models_mod.get_models()[name]
        pipe.set_params(pre__text__min_df=1)
        pipe.fit(X, y)
        score = pipe.predict_proba(X)[:, 1]
        assert score.shape == (len(df),)
        assert score[y == 1].mean() > score[y == 0].mean() + 0.2


def test_full_one_class_pipeline_uses_only_real_training_text():
    df, X = _xy()
    y = df["fraudulent"].to_numpy()
    model = models_mod.get_models()["ocsvm"].set_params(
        detector__pre__text__min_df=1,
        detector__pca__n_components=5,
    )
    model.fit(X, y)
    vocabulary = model.detector_.named_steps["pre"].named_transformers_["text"].vocabulary_
    assert "earn" not in vocabulary, "fraud rows must not fit even the TF-IDF stage"
    assert model.decision_function(X).shape == (len(df),)


def test_one_class_detector_scores_outliers_higher():
    from sklearn.svm import OneClassSVM

    rng = np.random.default_rng(0)
    normal = rng.normal(0, 1, size=(200, 2))
    outliers = rng.normal(6, 1, size=(20, 2))
    X = np.vstack([normal, outliers])
    y = np.r_[np.zeros(200, int), np.ones(20, int)]
    detector = models_mod.OneClassDetector(OneClassSVM(nu=0.05, gamma="scale")).fit(X, y)
    score = detector.decision_function(X)
    assert detector.detector_.fit_status_ == 0
    assert score[y == 1].min() > np.median(score[y == 0])
