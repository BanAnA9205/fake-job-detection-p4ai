"""Shared EDA-motivated feature construction."""

import pandas as pd
from modules.data_helper import CAT_COLS, TEXT_COLS

NUMERIC_COLS = ["desc_len", "requirements_len"]
BINARY_COLS = [
    "telecommuting", "has_company_logo", "has_questions",
    "has_company_profile", "has_salary", "has_email", "has_phone",
]
# Candidate flags are available for EDA even when they are not selected for the model.
CANDIDATE_BINARY_COLS = BINARY_COLS + ["has_requirements", "has_benefits", "has_url"]
NOMINAL_COLS = CAT_COLS

EMAIL_PATTERN = r"(?:#EMAIL|[\w.+-]+@[\w-]+\.[A-Za-z]{2,})"
PHONE_PATTERN = r"#PHONE"
URL_PATTERN = r"(?:#URL|https?://|www\.)"


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Copy a cleaned frame and add row-local text, length, and presence features."""
    out = df.copy()
    text = out[TEXT_COLS[0]].fillna("").astype(str)
    for col in TEXT_COLS[1:]:
        text = text + " " + out[col].fillna("").astype(str)
    out["text"] = text

    for col in ["company_profile", "requirements", "benefits"]:
        out[f"has_{col}"] = out[col].fillna("").astype(str).str.strip().ne("").astype(int)
    out["has_salary"] = (
        out["salary_range"].notna()
        & out["salary_range"].astype(str).str.strip().ne("")
    ).astype(int)
    out["desc_len"] = out["description"].fillna("").astype(str).str.len().astype(int)
    out["requirements_len"] = out["requirements"].fillna("").astype(str).str.len().astype(int)
    out["has_url"] = text.str.contains(URL_PATTERN, regex=True).astype(int)
    out["has_email"] = text.str.contains(EMAIL_PATTERN, regex=True).astype(int)
    out["has_phone"] = text.str.contains(PHONE_PATTERN, regex=True).astype(int)
    return out
