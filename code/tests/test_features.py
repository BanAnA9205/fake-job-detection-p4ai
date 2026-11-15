"""Tests for modules.features on a handful of hand-written synthetic rows.

Run with pytest, or directly: .venv/bin/python tests/test_features.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.features import add_engineered_features, make_preprocessor


def _toy_df(n=12):
    """Tiny frame with the real column schema; every row starts identical."""
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
            "telecommuting": i % 2,  # varies, so the sparse-flag guard keeps at least one flag
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
    columns_before = list(df.columns)
    out = add_engineered_features(df)
    assert list(df.columns) == columns_before, "input frame was modified"
    assert "nan" not in out.loc[0, "text"].lower()
    assert out.loc[0, "text"] == (
        "Data engineer  Build python data pipelines and sql reports. "
        "Python and sql. Health insurance."
    )


def test_country_and_missing_flags():
    df = _toy_df(3)
    df.loc[0, "location"] = "GB, London"
    df.loc[1, "location"] = np.nan
    df.loc[2, "location"] = ""
    df.loc[1, "salary_range"] = np.nan
    for col in ["company_profile", "requirements", "benefits", "department",
                "description"]:
        df.loc[2, col] = np.nan
    out = add_engineered_features(df)
    assert out.loc[0, "country"] == "GB"
    assert out.loc[1, "country"] == "missing"
    assert out.loc[2, "country"] == "missing"
    assert out["has_salary"].tolist() == [1, 0, 1]
    flags = ["missing_company_profile", "missing_requirements",
             "missing_benefits", "missing_department"]
    assert out.loc[2, flags].tolist() == [1, 1, 1, 1]
    assert out.loc[0, flags].tolist() == [0, 0, 0, 0]
    assert out.loc[2, "desc_len"] == 0


def test_lengths_counts_and_upper_ratio():
    df = _toy_df(1)
    df.loc[0, "title"] = "Hi"
    df.loc[0, "description"] = "AB ab!!"
    for col in ["company_profile", "requirements", "benefits"]:
        df.loc[0, col] = ""
    out = add_engineered_features(df)
    assert out.loc[0, "title_len"] == 2
    assert out.loc[0, "desc_len"] == 7
    assert out.loc[0, "n_exclaim"] == 2
    text = out.loc[0, "text"]
    expected = sum(c.isupper() for c in text) / sum(c.isalpha() for c in text)
    assert abs(out.loc[0, "n_upper_ratio"] - expected) < 1e-12
    assert 0.0 < expected < 1.0


def test_regex_flags():
    df = _toy_df(4)
    df.loc[0, "description"] = "Apply at http://spam.example now"
    df.loc[1, "description"] = "Send your cv to hr@example.com"
    df.loc[2, "description"] = "Earn cash, work from home"
    df.loc[3, "description"] = "Normal office job"
    out = add_engineered_features(df)
    assert out["has_url"].tolist() == [1, 0, 0, 0]
    assert out["has_email"].tolist() == [0, 1, 0, 0]
    assert out["has_money_words"].tolist() == [0, 0, 1, 0]


def test_placeholder_flags():
    """The dataset hides links, emails and phones behind #URL_..#, #EMAIL_..#, #PHONE_..#."""
    df = _toy_df(4)
    df.loc[0, "description"] = "Apply here #URL_3f9a1c2b# today"
    df.loc[1, "description"] = "Mail #EMAIL_77be90aa# for details"
    df.loc[2, "description"] = "Call #PHONE_12ab34cd# now"
    out = add_engineered_features(df)
    assert out["has_url"].tolist() == [1, 0, 0, 0]
    assert out["has_email"].tolist() == [0, 1, 0, 0]
    assert out["has_phone"].tolist() == [0, 0, 1, 0]


def test_sparse_flag_guard_drops_constant_flags():
    """A flag that never fires must not reach the model; a flag that varies must."""
    df = add_engineered_features(_toy_df(40))
    df.loc[:19, "has_email"] = 0           # make sure it is constant
    df.loc[:, "has_url"] = [1, 0] * 20     # fires on half the rows
    pre = make_preprocessor("tfidf").fit(df)
    names = list(pre.get_feature_names_out())
    assert "binary__has_url" in names
    assert "binary__has_email" not in names


def test_new_columns_have_no_nan():
    df = _toy_df(6)
    df.loc[1, "description"] = np.nan
    out = add_engineered_features(df)
    new_cols = ["text", "country", "desc_len", "title_len", "n_exclaim",
                "n_upper_ratio", "has_url", "has_email", "has_money_words"]
    assert not out[new_cols].isna().any().any()
    assert out["desc_len"].dtype.kind in "iu"


def test_tfidf_preprocessor_shape_and_names():
    df = add_engineered_features(_toy_df(12))
    pre = make_preprocessor("tfidf", max_features=500, ngram_range=(1, 1), min_df=2)
    X = pre.fit_transform(df)
    assert X.shape[0] == len(df)
    assert X.shape[1] > 0
    values = X.toarray() if sparse.issparse(X) else np.asarray(X)
    assert not np.isnan(values).any()
    # The text branch itself stays sparse; on a tiny matrix the ColumnTransformer
    # may densify the union because its default density threshold is 0.3.
    text_branch = pre.named_transformers_["text"]
    assert sparse.issparse(text_branch.transform(df["text"]))
    names = list(pre.get_feature_names_out())
    assert "text__python" in names
    assert "ordinal__required_experience" in names
    assert "binary__telecommuting" in names  # varies in the toy data, so the guard keeps it
    assert "binary__has_salary" not in names  # constant in the toy data, so the guard drops it
    assert any(name.startswith("nominal__employment_type") for name in names)


def test_ordinal_codes_are_ordered():
    df = add_engineered_features(_toy_df(6))
    df.loc[0, "required_experience"] = "Internship"
    df.loc[1, "required_experience"] = "Director"
    df.loc[2, "required_experience"] = "Mid-Senior level"
    df.loc[3, "required_experience"] = np.nan
    df.loc[4, "required_experience"] = "Alien"
    df.loc[5, "required_education"] = "Doctorate"
    pre = make_preprocessor("tfidf")
    X = pre.fit_transform(df)
    X = X.toarray() if sparse.issparse(X) else np.asarray(X)
    assert not np.isnan(X).any()
    names = list(pre.get_feature_names_out())
    codes = X[:, names.index("ordinal__required_experience")]
    # The codes are rescaled by a positive factor, so signs and order are preserved.
    assert codes[3] == 0, "NaN must be filled with the first level"
    assert codes[4] < 0, "unknown level must be encoded below every real level"
    assert codes[0] < codes[2] < codes[1], "experience order not preserved"
    edu = X[:, names.index("ordinal__required_education")]
    assert edu[5] > edu[0], "doctorate must rank above bachelor"


def test_unseen_nominal_category_transforms():
    df = add_engineered_features(_toy_df(12))
    pre = make_preprocessor("tfidf")
    pre.fit(df)
    other = df.iloc[:2].copy()
    other["employment_type"] = "Temporary"
    other["industry"] = "Space mining"
    X = pre.transform(other)
    assert X.shape[0] == 2
    assert not np.isnan(X).any()


def test_bm25_known_answer():
    """Hand-computed BM25 on a 3-ad count matrix (k1=1.5, b=0.75)."""
    from modules.features import Bm25Transformer
    counts = np.array([[2, 0], [1, 1], [0, 3]])  # ad lengths 2, 2, 3; average 7/3
    bm = Bm25Transformer(k1=1.5, b=0.75).fit(sparse.csr_matrix(counts))
    idf = np.log(1 + (3 - 2 + 0.5) / (2 + 0.5))   # both words appear in 2 of 3 ads
    assert np.allclose(bm.idf_, [idf, idf])
    out = bm.transform(sparse.csr_matrix(counts)).toarray()
    avg = 7 / 3
    tf, dl = 2.0, 2.0                              # word 0 in ad 0
    expected = idf * tf * 2.5 / (tf + 1.5 * (1 - 0.75 + 0.75 * dl / avg))
    assert np.isclose(out[0, 0], expected)
    assert out[0, 1] == 0.0, "absent words stay zero"
    assert out[2, 1] > out[1, 1], "more repeats of a word weigh more (saturating)"


def test_bm25_preprocessor_matches_tfidf_shape():
    df = add_engineered_features(_toy_df(12))
    a = make_preprocessor("tfidf").fit_transform(df)
    b = make_preprocessor("bm25").fit_transform(df)
    b = b.toarray() if sparse.issparse(b) else np.asarray(b)
    assert a.shape == b.shape and not np.isnan(b).any()


def test_unknown_kind_raises():
    try:
        make_preprocessor("bogus")
    except ValueError:
        return
    raise AssertionError("expected ValueError for an unknown kind")


def main():
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print("ok", test.__name__)
    print(f"{len(tests)} features checks passed")


if __name__ == "__main__":
    main()
