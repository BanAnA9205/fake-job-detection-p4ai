"""Row-wise feature engineering and the column preprocessor for EMSCAD."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from modules.data_helper import TEXT_COLS

# Column groups the preprocessor works on.
NUMERIC_COLS = ["desc_len", "title_len", "n_exclaim", "n_upper_ratio"]
BINARY_COLS = [
    "telecommuting", "has_company_logo", "has_questions",
    "has_salary", "missing_company_profile", "missing_requirements",
    "missing_benefits", "missing_department",
    "has_url", "has_email", "has_phone", "has_money_words",
]
NOMINAL_COLS = ["employment_type", "industry", "function", "country"]
ORDINAL_COLS = ["required_experience", "required_education"]

# Explicit low-to-high order, matching the levels present in the CSV.
# "missing" is first, so the NaN fill level is code 0 and unknown levels are -1.
EXPERIENCE_LEVELS = [
    "missing", "Not Applicable", "Internship", "Entry level", "Associate",
    "Mid-Senior level", "Director", "Executive",
]
EDUCATION_LEVELS = [
    "missing", "Unspecified", "Some High School Coursework", "High School or equivalent",
    "Vocational - HS Diploma", "Vocational", "Vocational - Degree",
    "Some College Coursework Completed", "Associate Degree", "Certification",
    "Bachelor's Degree", "Master's Degree", "Professional", "Doctorate",
]

# Spam cues, kept in one regex so the flag stays easy to explain.
MONEY_PATTERN = (r"(?:\bearn\b|\bper week\b|\bno experience\b|\bwork from home\b"
                 r"|\bwire\b|\bwestern union\b|\$)")
# The dataset replaced real links, emails and phone numbers with placeholders such as
# "#URL_ab12cd#", so we look for the placeholder first and for a real address second.
URL_PATTERN = r"(?:#URL_|https?://|www\.)"
EMAIL_PATTERN = r"(?:#EMAIL_|[\w.+-]+@[\w-]+\.[A-Za-z]{2,})"
PHONE_PATTERN = r"#PHONE_"

# A 0/1 flag has variance p(1-p), roughly p, where p is the share of rows that fire it.
# 0.001 drops flags that fire on about 0.1% of rows or fewer (around 10 training rows).
MIN_FLAG_VARIANCE = 0.001


def _country(location):
    """Country part of a "City, State" location, or "missing" when there is none."""
    if pd.isna(location):
        return "missing"
    country = str(location).split(",")[0].strip()
    return country or "missing"


def _upper_ratio(text):
    """Share of letters that are upper case, 0.0 for text without letters."""
    letters = text.str.count(r"[A-Za-z]").clip(lower=1)
    return (text.str.count(r"[A-Z]") / letters).astype(float)


def add_engineered_features(df):
    """Copy df and add the text blob, country, missing flags and spam cues."""
    out = df.copy()
    text = out[TEXT_COLS[0]].fillna("").astype(str)
    for col in TEXT_COLS[1:]:
        text = text + " " + out[col].fillna("").astype(str)
    out["text"] = text
    out["country"] = out["location"].map(_country)
    out["has_salary"] = out["salary_range"].notna().astype(int)
    for col in ["company_profile", "requirements", "benefits", "department"]:
        out["missing_" + col] = out[col].isna().astype(int)
    out["desc_len"] = out["description"].fillna("").str.len().astype(int)
    out["title_len"] = out["title"].fillna("").str.len().astype(int)
    out["n_exclaim"] = text.str.count("!").astype(int)
    out["n_upper_ratio"] = _upper_ratio(text)
    out["has_url"] = text.str.contains(URL_PATTERN, regex=True).astype(int)
    out["has_email"] = text.str.contains(EMAIL_PATTERN, regex=True).astype(int)
    out["has_phone"] = text.str.contains(PHONE_PATTERN, regex=True).astype(int)
    out["has_money_words"] = text.str.contains(
        MONEY_PATTERN, case=False, regex=True).astype(int)
    assert len(out) == len(df), "feature engineering changed the row count"
    return out


def make_tfidf(max_features=20000, ngram_range=(1, 2), min_df=3):
    """The one TF-IDF definition shared by every text model."""
    return TfidfVectorizer(sublinear_tf=True, strip_accents="unicode",
                           stop_words="english", max_features=max_features,
                           ngram_range=ngram_range, min_df=min_df)


def _onehot(dense):
    """One-hot for nominal columns; rare values share one infrequent column."""
    return OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30,
                         sparse_output=not dense)


def _ordinal():
    """Fill NaN with the lowest level, then encode the explicit order as numbers."""
    return Pipeline([
        ("fill", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ordinal", OrdinalEncoder(categories=[EXPERIENCE_LEVELS, EDUCATION_LEVELS],
                                   handle_unknown="use_encoded_value",
                                   unknown_value=-1)),
    ])


def _meta_transformers(dense):
    """Non-text branch of the preprocessor; dense=True suits tree models."""
    return [
        ("nominal", _onehot(dense), NOMINAL_COLS),
        ("ordinal", _ordinal(), ORDINAL_COLS),
        ("numeric", StandardScaler(with_mean=False), NUMERIC_COLS),
        # Fitted on training rows only: drops flags that are (almost) always 0 or always 1.
        ("binary", VarianceThreshold(MIN_FLAG_VARIANCE), BINARY_COLS),
    ]


def make_preprocessor(kind="tfidf", max_features=20000, ngram_range=(1, 2), min_df=3):
    """ColumnTransformer that turns the engineered frame into a model matrix.

    kind="tfidf" keeps the text sparse for linear models; kind="meta" drops the
    text and returns one dense block, which tree models can consume.
    """
    if kind == "tfidf":
        text = ("text", make_tfidf(max_features, ngram_range, min_df), "text")
        return ColumnTransformer([text] + _meta_transformers(dense=False))
    if kind == "meta":
        return ColumnTransformer(_meta_transformers(dense=True), sparse_threshold=0)
    raise ValueError(f"unknown preprocessor kind: {kind!r}")
