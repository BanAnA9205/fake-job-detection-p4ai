"""Tests for EDA-motivated features and leakage-safe preprocessing."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.features import BINARY_COLS, NUMERIC_COLS, add_engineered_features
from modules.models import make_preprocessor


def _toy_df(n=12):
    rows = []
    for i in range(n):
        rows.append({
            "job_id": i,
            "title": "Data engineer",
            "location": "US, NY",
            "department": "Engineering",
            "salary_range": "50000-60000",
            "company_profile": "We build python data pipelines.",
            "description": "Build python data pipelines and sql reports.",
            "requirements": "Python and sql.",
            "benefits": "Health insurance.",
            "telecommuting": i % 2,
            "has_company_logo": 1,
            "has_questions": 0,
            "employment_type": "Full-time",
            "required_experience": "Mid-Senior level",
            "required_education": "Bachelor's Degree",
            "industry": "IT",
            "function": "Engineering",
            "fraudulent": int(i % 4 == 0),
        })
    return pd.DataFrame(rows)


def test_text_join_and_input_untouched():
    df = _toy_df(2)
    df.loc[0, "company_profile"] = np.nan
    out = add_engineered_features(df)
    assert list(df.columns) == list(_toy_df(2).columns)
    assert "nan" not in out.loc[0, "text"].lower()
    assert out.loc[0, "has_company_profile"] == 0
    assert out.loc[1, "has_company_profile"] == 1


def test_presence_and_length_features():
    df = _toy_df(3)
    df.loc[1, "salary_range"] = np.nan
    df.loc[2, ["requirements", "benefits"]] = np.nan
    out = add_engineered_features(df)
    assert out["has_salary"].tolist() == [1, 0, 1]
    assert out["has_requirements"].tolist() == [1, 1, 0]
    assert out["has_benefits"].tolist() == [1, 1, 0]
    assert out["requirements_len"].tolist() == [15, 15, 0]
    assert NUMERIC_COLS == ["desc_len", "requirements_len"]


def test_contact_flags():
    df = _toy_df(4)
    df.loc[0, "description"] = "Apply at http://spam.example now"
    df.loc[1, "description"] = "Send your cv to hr@example.com"
    df.loc[2, "description"] = "Call #PHONE_12ab34cd# now"
    out = add_engineered_features(df)
    assert out["has_url"].tolist() == [1, 0, 0, 0]
    assert out["has_email"].tolist() == [0, 1, 0, 0]
    assert out["has_phone"].tolist() == [0, 0, 1, 0]


def test_rare_and_constant_flags_are_not_pruned_here():
    """Single-value cleanup belongs to QC, not an additional variance filter."""
    df = add_engineered_features(_toy_df(40))
    names = list(make_preprocessor().fit(df).get_feature_names_out())
    assert "binary__has_company_logo" in names  # constant 1 in this direct feature-only fixture
    assert "binary__has_questions" in names     # constant 0 in this fixture
    assert "binary__has_salary" in names


def test_new_columns_have_no_nan():
    df = _toy_df(6)
    df.loc[1, ["description", "requirements"]] = np.nan
    out = add_engineered_features(df)
    new_cols = ["text", "desc_len", "requirements_len", "has_url", "has_email", "has_phone"]
    assert not out[new_cols].isna().any().any()


def test_preprocessor_shape_and_names():
    df = add_engineered_features(_toy_df(12))
    pre = make_preprocessor(max_features=500, ngram_range=(1, 1), min_df=2)
    X = pre.fit_transform(df)
    values = X.toarray() if sparse.issparse(X) else np.asarray(X)
    assert X.shape[0] == len(df)
    assert not np.isnan(values).any()
    names = list(pre.get_feature_names_out())
    assert "text__python" in names
    assert "numeric__requirements_len" in names
    assert "binary__telecommuting" in names
    assert any(name.startswith("nominal__employment_type") for name in names)
    assert any(name.startswith("nominal__required_experience") for name in names)


def test_text_only_preprocessor():
    df = add_engineered_features(_toy_df(12))
    pre = make_preprocessor(min_df=2, structured=False)
    pre.fit(df)
    names = list(pre.get_feature_names_out())
    assert "text__python" in names
    assert all(name.startswith("text__") for name in names)
    assert len(names) < len(make_preprocessor(min_df=2).fit(df).get_feature_names_out())


def test_unseen_nominal_category_transforms():
    df = add_engineered_features(_toy_df(12))
    pre = make_preprocessor().fit(df)
    other = df.iloc[:2].copy()
    other["employment_type"] = "Temporary"
    other["industry"] = "Space mining"
    X = pre.transform(other)
    values = X.toarray() if sparse.issparse(X) else np.asarray(X)
    assert X.shape[0] == 2
    assert not np.isnan(values).any()


def test_selected_binary_columns_have_candidate_coverage():
    assert set(BINARY_COLS) <= set(add_engineered_features(_toy_df(2)).columns)
