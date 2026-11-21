"""Build the four project notebooks with nbformat. No data is read, nothing is fitted.

Run from the code/ folder (the script resolves its own folder, so cwd does not matter):
    python notebooks/build_notebooks.py

Writes 01_eda_preprocessing, 02_classical_ml, 03_deep_learning and 04_analysis (.ipynb), overwriting
existing copies, then re-reads each file and ast.parse-checks every code cell (magics and
! lines ignored). The notebooks are executed separately with jupyter nbconvert. All runs
happen inside the notebooks; the .py files in modules/ are importable helpers only.
"""

import ast
import os

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))


def build(cells, filename):
    """Turn a list of (kind, source) tuples into one .ipynb file; return its path."""
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python",
                                 "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    stem = os.path.splitext(filename)[0]
    for i, (kind, src) in enumerate(cells):
        if kind == "md":
            cell = nbf.v4.new_markdown_cell(src.strip("\n"))
        elif kind == "code":
            cell = nbf.v4.new_code_cell(src.strip("\n"))
        else:
            raise ValueError(f"unknown cell kind: {kind!r}")
        # Deterministic ids, so rebuilding the same sources gives the same file.
        cell["id"] = f"{stem}-{i:02d}"
        nb.cells.append(cell)
    nbf.validate(nb)
    path = os.path.join(HERE, filename)
    nbf.write(nb, path)
    return path


def check(path):
    """Re-read a notebook, ast.parse every code cell; return (total cells, code cells)."""
    nb = nbf.read(path, as_version=4)
    n_code = 0
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        n_code += 1
        # IPython magics (%) and shell (!) lines are not Python syntax; drop them.
        lines = [ln for ln in cell.source.splitlines()
                 if not ln.lstrip().startswith(("%", "!"))]
        ast.parse("\n".join(lines) or "pass", filename=path)
    return len(nb.cells), n_code


# ---------------------------------------------------------------------------
# Notebook 01: exploratory data analysis
# ---------------------------------------------------------------------------

NB1 = [
    ("md", """
# Fake job postings: EDA and preprocessing (notebook 01)

We get to know the EMSCAD dataset before touching any model: class balance, missing
data, duplicates, text lengths and simple spam-style flags, followed by data quality checks on
the text itself and on how related ads would be split. Every section feeds a decision: what to
fix, what to drop, how to split, and which metrics to trust. The second half builds the
preprocessing that notebooks 02 (classical ML) and 03 (deep learning) share: cleaning, engineered
features, the model matrix, and the five saved train/val/test splits.

Run time: about 2 minutes.
"""),
    ("code", """
import sys, os; os.chdir(os.path.join(os.getcwd(), "..")) if os.path.basename(os.getcwd()) == "notebooks" else None; sys.path.insert(0, "."); import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from modules import data_helper as dh
from modules import features as ft
from modules import qc
from modules.classical_runs import get_splits

DATA = "../data/fake_job_postings.csv"
"""),
    ("code", """
SEEDS = [0, 1, 2, 3, 4]  # one split per seed; every model is evaluated on all five
SEED = 0

dh.seed_everything(SEED)  # Session 5: fix every seed so runs are reproducible
sns.set_theme(style="whitegrid")
"""),
    ("md", """
## Load and audit the raw data

Load the CSV and run the audit from `modules/data_helper.py`: shape, class counts,
duplicate structure and label conflicts, before we change anything.
"""),
    ("code", """
df = dh.load_raw(DATA)
a = dh.audit(df)
pd.Series({k: v for k, v in a.items() if k != "missing_rate"}, name="audit").to_frame()
"""),
    ("md", """
## Class balance

The target is heavily imbalanced, which drives every later choice of metric and
threshold (Sessions 6/7).
"""),
    ("code", """
# Count ads per class.
counts = df["fraudulent"].value_counts().sort_index()
print(counts.to_dict())

# Bar chart of the two counts.
fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(["real (0)", "fraud (1)"], counts.values, color=["tab:blue", "tab:red"])
ax.set_title("Class balance of the target")
ax.set_ylabel("number of ads")
plt.show()
"""),
    ("md", """
Reading: bar heights are numbers of ads on one shared linear scale; the imbalance
seen here is why accuracy is useless and why PR-AUC plus a validation threshold are
our headline tools.
"""),
    ("md", """
## Missing values by class

Share of ads missing each column, real versus fraud. A gap between the two bars of
a pair means missingness itself is informative.
"""),
    ("code", """
# Share of missing values per column, one column per class, sorted by the real-ad share.
by_class = pd.DataFrame({
    "real": df.loc[df["fraudulent"] == 0].isna().mean(),
    "fraud": df.loc[df["fraudulent"] == 1].isna().mean(),
}).sort_values("real", ascending=False)

# Side-by-side bars: real on the left of each pair, fraud on the right.
fig, ax = plt.subplots(figsize=(10, 4))
x = np.arange(len(by_class))
ax.bar(x - 0.2, by_class["real"], 0.4, label="real")
ax.bar(x + 0.2, by_class["fraud"], 0.4, label="fraud")
ax.set_xticks(x, labels=by_class.index, rotation=45, ha="right")
ax.set_ylabel("share of ads missing the column")
ax.set_title("Missing share per column, real vs fraud")
ax.legend()
plt.show()
"""),
    ("md", """
Reading: in each pair the left bar is real ads and the right bar fraudulent ads;
both use the same 0 to 1 scale.
"""),
    ("md", """
## Duplicates and shared descriptions

The same posting can appear several times (exact duplicates), and different job_ids
can carry the same description text. If copies end up on both sides of a split,
evaluation leaks.
"""),
    ("code", """
# Duplicate statistics from the audit.
print("exact duplicate rows (ignoring job_id):", a["n_duplicate_rows"])
print("rows sharing their description with another row:", a["n_rows_shared_description"])
print("duplicate blocks with conflicting labels:", a["n_conflicting_duplicate_groups"])

# Group ads by identical (normalised) description: lowercase, collapse whitespace.
desc = (df["description"].fillna("").astype(str).str.lower()
        .str.replace(r"\\s+", " ", regex=True).str.strip())
sizes = desc.value_counts()
print("largest description group:", int(sizes.max()), "ads")

# Histogram of group sizes on a log y axis.
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(sizes.values, bins=np.arange(0.5, sizes.max() + 1.5, 1))
ax.set_yscale("log")
ax.set_xlabel("ads per identical description")
ax.set_ylabel("number of description groups")
ax.set_title("Description group sizes")
plt.show()
"""),
    ("md", """
Reading: the y axis is log-scaled, so each gridline up is a tenfold jump in the
number of groups; bars on the right are descriptions repeated many times.
"""),
    ("md", """
## Description length by class

Long versus short ads. The x axis is log-scaled because lengths span several orders
of magnitude.
"""),
    ("code", """
# Description lengths (non-empty only) and log-spaced bins to match the log x axis.
lens = df["description"].fillna("").str.len()
lens = lens[lens > 0]
bins = np.logspace(np.log10(lens.min()), np.log10(lens.max()), 40)

# Overlaid density histograms, one per class.
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(lens[df["fraudulent"] == 0], bins=bins, alpha=0.5, density=True, label="real")
ax.hist(lens[df["fraudulent"] == 1], bins=bins, alpha=0.5, density=True, label="fraud")
ax.set_xscale("log")
ax.set_xlabel("description length (characters, log scale)")
ax.set_ylabel("share of ads (density)")
ax.set_title("Description length by class")
ax.legend()
plt.show()
"""),
    ("md", """
Reading: both histograms are densities, so the two classes are directly comparable
despite their very different sizes.
"""),
    ("md", """
## Engineered flags by class

`modules/features.py` adds row-wise flags: missingness indicators, length and
capitalisation cues, and spam cues. Note that the dataset itself already replaced
real URLs, emails and phone numbers with placeholders such as `#URL_ab12cd#`, so the
flags look for those placeholders (plus a few plain-text patterns).
"""),
    ("code", """
# Engineer the features, then the share of ads firing each binary flag, per class.
feats = ft.add_engineered_features(df)
flags = pd.DataFrame({
    "real": feats.loc[feats["fraudulent"] == 0, ft.BINARY_COLS].mean(),
    "fraud": feats.loc[feats["fraudulent"] == 1, ft.BINARY_COLS].mean(),
}).sort_values("fraud", ascending=False)
flags.round(3)
"""),
    ("md", """
Reading: each number is the share of ads in that class where the flag fires; 1.0
means every ad of that class fires it.
"""),
    ("md", """
## Fraud rate by country

Only values seen on at least 100 ads, so the rates are not driven by a handful of
rows.
"""),
    ("code", """
def fraud_rate(col, y, min_n=100):
    \"\"\"Fraud rate per value of a column, keeping values with at least min_n ads.\"\"\"
    t = pd.DataFrame({"value": col, "fraud": y}).groupby("value")["fraud"]
    t = t.agg(n="count", fraud_rate="mean")
    return t[t["n"] >= min_n].sort_values("fraud_rate", ascending=False)


fraud_rate(feats["country"], feats["fraudulent"])
"""),
    ("md", """
## Fraud rate by employment type

Same table for employment type, with missing as its own value.
"""),
    ("code", """
fraud_rate(feats["employment_type"].fillna("missing"), feats["fraudulent"])
"""),
    ("md", """
## Fraud rate by the other categorical columns

Experience, education, function and industry, with missing as its own value and only values seen
on at least 100 ads.
"""),
    ("code", """
for col in ["required_experience", "required_education", "function", "industry"]:
    t = fraud_rate(feats[col].fillna("missing"), feats["fraudulent"])
    print(f"{col}: highest and lowest fraud rates (overall {feats['fraudulent'].mean():.3f})")
    print(pd.concat([t.head(4), t.tail(2)]).round(3).to_string(), "\\n")
"""),
    ("md", """
## Text length of every field

Median length in characters per class, and the share of ads where the field is empty.
"""),
    ("code", """
rows = {}
for col in dh.TEXT_COLS:
    length = df[col].fillna("").str.len()
    rows[col] = {"median real": length[df["fraudulent"] == 0].median(),
                 "median fraud": length[df["fraudulent"] == 1].median(),
                 "empty % real": round(100 * (length[df["fraudulent"] == 0] == 0).mean(), 1),
                 "empty % fraud": round(100 * (length[df["fraudulent"] == 1] == 0).mean(), 1)}
pd.DataFrame(rows).T
"""),
    ("md", """
## Titles and words

The most common titles among fraudulent ads, and the words whose presence differs most between
the classes (log-odds of an ad containing the word, with add-one smoothing, words in at least 50
ads). This is exploration on all ads; the models only ever learn from their training rows.
"""),
    ("code", """
titles = df["title"].str.strip().str.lower()
top = titles[df["fraudulent"] == 1].value_counts().head(8)
print(pd.DataFrame({"fraud ads": top, "fraud rate": [df["fraudulent"][titles == t].mean().round(2) for t in top.index]}))

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(binary=True, min_df=50, stop_words="english")
X = cv.fit_transform(feats["text"])
y_all = feats["fraudulent"].to_numpy()
in_fraud = (np.asarray(X[y_all == 1].sum(0)).ravel() + 1) / ((y_all == 1).sum() + 2)
in_real = (np.asarray(X[y_all == 0].sum(0)).ravel() + 1) / ((y_all == 0).sum() + 2)
log_odds = pd.Series(np.log(in_fraud / (1 - in_fraud)) - np.log(in_real / (1 - in_real)),
                     index=cv.get_feature_names_out()).sort_values()
print("\\nmost fraud-leaning words:", ", ".join(log_odds.tail(15).index[::-1]))
print("most real-leaning words: ", ", ".join(log_odds.head(15).index))
"""),
    ("md", """
Reading: on the raw text the strongest words are largely artefacts rather than scam language: a
single company's name (Aker, also glued into "overviewaker"), hashed URLs that identify individual
posters, and the "jobs in Asia, TEFL" template that alone accounts for hundreds of real ads. The
data quality checks below address exactly these: hashed contacts, glued words, and companies or
templates that would otherwise sit on both sides of a split.
"""),
    ("md", """
## Data quality checks

The sections above look at the columns. These checks look at the content and at how related
ads would be split, using `modules/qc.py`.

### Cells that look filled but are empty

Excel errors such as `#NAME?` and filler such as `.` count as text unless we treat them as
missing.
"""),
    ("code", """
qc.disguised_missing(df, dh.TEXT_COLS + ["department"])
"""),
    ("md", """
### Placeholders, glued text and very short ads

The dataset replaced every link, email and phone number with a placeholder carrying its own hash
(`#URL_3f9a...#`). A model can memorise one hash, i.e. one poster, instead of learning "has a
phone number". Removed HTML also glued sentences together ("place.We").
"""),
    ("code", """
raw_text = ft.add_engineered_features(df)["text"]
y_raw = df["fraudulent"]
print(qc.placeholder_counts(raw_text))
glue = qc.glued_boundaries(raw_text)
print(f"ads with glued boundaries: {(glue > 0).sum()} of {len(df)}")
for limit in (100, 300):
    short = raw_text.str.len() < limit
    print(f"joined text under {limit} characters: {short.sum()} ads, fraud rate {y_raw[short].mean():.3f}")
"""),
    ("md", """
Reading: the hashes repeat (more placeholders than distinct hashes), so the same poster's contact
details appear in several ads; cleaning replaces each hash by a generic `#URL`, `#EMAIL` or
`#PHONE`. Very short ads are mostly fraudulent, which the length features can pick up.

### Companies and near-copies: would related ads end up on both sides of a split?

Two ads can be related without sharing the exact description: the same company (identical
company profile), or the same template with small edits (near-copies, TF-IDF cosine >= 0.9).
"""),
    ("code", """
company = df["company_profile"].fillna("").str.lower().str.split().str.join(" ")
has_company = company != ""
print("companies with 2+ ads:", qc.group_purity(company[has_company], y_raw[has_company]))

desc = df["description"].fillna("").str.lower().str.split().str.join(" ")
near = qc.near_duplicate_clusters(desc)
other_group = pd.Series(near).groupby(near).transform(lambda s: desc.iloc[s.index].nunique() > 1)
print("ads in a near-copy cluster that spans several exact descriptions:", int(other_group.sum()))
"""),
    ("md", """
Reading: companies are label-pure (no company has both real and fraudulent ads), so a company seen
in training gives the label away in test. The check below splits the data the way this project
first did, by exact description only, and counts the test ads whose company or near-copy is in the
training rows.
"""),
    ("code", """
# The first version's grouping: exact (normalised) description only.
old = df.drop_duplicates(subset=[c for c in df.columns if c not in ("job_id", "fraudulent")]).reset_index(drop=True)
old["group"] = old["description"].fillna("").str.lower().str.split().str.join(" ")
old_splits = dh.make_splits(old, seed=SEED)
keys = {"same company": old["company_profile"].fillna("").str.lower().str.split().str.join(" "),
        "near-copy cluster": qc.near_duplicate_clusters(old["group"]).astype(str)}
print(f"exact-description split, seed {SEED}: {len(old_splits['test'])} test ads, "
      f"{int(old['fraudulent'].iloc[old_splits['test']].sum())} test frauds")
qc.split_overlap(old_splits, keys, old["fraudulent"].to_numpy())
"""),
    ("md", """
### Format, encoding and flag checks

Columns the models use only indirectly still need checking: the salary (only "given or not" is
used), location (the country is used), department (only "missing or not" is used), the text
encoding, and whether the engineered flags fire when they should.
"""),
    ("code", """
sal = df["salary_range"].dropna()
valid = sal.str.fullmatch(r"\\d+-\\d+")
print(f"salary: {len(sal)} given, {int(valid.sum())} as 'number-number', {int((~valid).sum())} other:",
      sal[~valid].value_counts().head(5).to_dict())
parts = df["location"].dropna().str.split(",").str.len()
print("location: values per number of comma parts", parts.value_counts().sort_index().to_dict())
dept = df["department"].dropna().str.strip()
print(f"department: {dept.nunique()} levels, {dept.str.lower().nunique()} after lower-casing")
text = feats["text"]
print(f"encoding damage (Ã, â€, Â): {int(text.str.contains('Ã|â€|Â').sum())} ads | more than 5% non-ASCII: "
      f"{int((text.str.count(r'[^\\x00-\\x7f]') / text.str.len().clip(lower=1) > 0.05).sum())} ads")
for flag, direct in [("has_url", r"#URL|https?://|www\\."), ("has_phone", "#PHONE")]:
    print(f"{flag}: flag and direct regex disagree on {int((feats[flag] != text.str.contains(direct).astype(int)).sum())} ads")
words = text[feats["has_money_words"] == 1].str.extract("(" + ft.MONEY_PATTERN + ")", flags=2)[0].str.lower()
print("money-word flag, what it matches:", words.value_counts().to_dict())
"""),
    ("md", """
Reading: the salary column holds dates where Excel converted ranges such as "10-10" (only "salary
given" is used, so this does not reach the models); department and location are messy but only
their missingness and the country are used; the URL and phone flags agree with a direct match; and
the money-word flag fires mostly on the dollar sign, so it is better read as "mentions an amount".
"""),
    ("md", """
Reading: under exact-description grouping most test ads, and a large share of the test frauds,
have their company or a near-copy in the training rows, so test scores would partly measure
recognising known companies. The cleaning below therefore groups related ads, not only identical
descriptions.

## Cleaning

Session 5: clean first, then split, then fit. `clean` fixes the text (junk cells become missing,
hashed placeholders become generic, glued sentences are separated), drops exact duplicates
(ignoring `job_id`) and gives each ad a `group`: ads sharing an exact description, a company
profile or a near-copy cluster belong to one group, and the relation chains. The splits keep each
group on one side.
"""),
    ("code", """
dfc = dh.clean(df)
print("rows before cleaning:", len(df), "| after:", len(dfc))
sizes = dfc["group"].value_counts()
print(f"groups: {len(sizes)} | largest: {sizes.max()} ads | groups with 2+ ads:",
      qc.group_purity(dfc["group"], dfc["fraudulent"]))
qc.assert_clean(dfc, dh.TEXT_COLS)
print("checked: no junk cells, hashed placeholders or duplicate ads left after cleaning")
"""),
    ("md", """
### Groups with both labels

A group mixes labels when related ads (same company or near-copies) are labelled differently,
which can be a labelling error. They are listed for reading; their labels are kept as given.
"""),
    ("code", """
g = dfc.groupby("group")["fraudulent"].agg(["size", "mean"])
mixed = g[(g["mean"] > 0) & (g["mean"] < 1)]
print(f"{len(mixed)} mixed groups, {int(mixed['size'].sum())} ads")
for gid, row in mixed.sort_values("size").iterrows():
    ads = dfc[dfc["group"] == gid]
    print(f"- {int(row['size'])} ads, {int(ads['fraudulent'].sum())} fraud | e.g.",
          "; ".join(ads.groupby("fraudulent")["title"].first().str.strip().str[:40]))
"""),
    ("md", """
Reading: almost every group with 2+ ads is label-pure. The mixed ones are few, but one is large: a
chain of related staffing-agency ads of which about half are labelled fraudulent. Keeping each
group on one side of the split is what matters for evaluation, so the labels are kept as given.
"""),
    ("md", """
### Save the processed data

Every later notebook (and the GPU server) reads this file instead of cleaning the raw CSV again,
so all of them use exactly the same rows, text and groups.
"""),
    ("code", """
dh.save_processed(dfc)
assert dh.load_processed().equals(dfc), "the saved file must read back identically"
print("saved", dh.PROCESSED, f"({os.path.getsize(dh.PROCESSED) / 1e6:.1f} MB)")
"""),
    ("md", """
## Feature engineering

`add_engineered_features` adds the joined `text` column (title, company profile,
description, requirements, benefits), missing-value flags, length and capitalisation
statistics, and spam cues. These are computed row by row from the ad itself, so they
cannot leak information between splits.
"""),
    ("code", """
dfe = ft.add_engineered_features(dfc)
new_cols = [c for c in dfe.columns if c not in df.columns]
print(len(new_cols), "new columns:", ", ".join(new_cols))
dfe[["title", "text"] + ft.NUMERIC_COLS[:3]].head(3)
"""),
    ("md", """
## Splits: by group of related ads, saved for all seeds

`make_splits` uses `StratifiedGroupKFold` on the group id (60/20/20, class ratio kept), so related
ads can never sit in two splits. Each seed's indices are saved to `splits/` with a fingerprint of
the cleaned data and its groups; every notebook and every machine reloads exactly these rows, and
splits made under another grouping are refused.
"""),
    ("code", """
y = dfe["fraudulent"].to_numpy()

# One row per (seed, split): size and fraud count.
rows = []
for seed in SEEDS:
    splits = get_splits(dfe, seed)  # loads splits/seed<N>.npz, creating it the first time
    for name, idx in splits.items():
        rows.append({"seed": seed, "split": name, "rows": len(idx),
                     "fraud": int(y[idx].sum()), "fraud rate": round(float(y[idx].mean()), 4)})

# Pivot to one line per seed.
pd.DataFrame(rows).pivot(index="seed", columns="split", values=["rows", "fraud"])
"""),
    ("code", """
for seed in SEEDS:
    splits = get_splits(dfe, seed)

    # Description groups present in each split.
    groups = {name: set(dfe["group"].iloc[idx]) for name, idx in splits.items()}

    # No group may appear in two splits (otherwise copies of an ad leak across the split).
    for a_name, b_name in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = groups[a_name] & groups[b_name]
        assert not shared, f"seed {seed}: {len(shared)} groups shared by {a_name} and {b_name}"
print("checked: no group appears in two splits, for every seed")

# The leak measured above under the old grouping, now on the saved splits: must be zero.
norm = lambda s: s.fillna("").str.lower().str.split().str.join(" ")
keys = {"same company": norm(dfe["company_profile"]),
        "near-copy cluster": qc.near_duplicate_clusters(norm(dfe["description"])).astype(str)}
for seed in SEEDS:
    ov = qc.split_overlap(get_splits(dfe, seed), keys, y)
    assert ov["test ads"].sum() == 0, f"seed {seed}: related ads cross from train to test: {ov.to_dict()}"
print("checked: no test ad shares its company or near-copy cluster with a training ad, for every seed")
"""),
    ("md", """
## The model matrix (fitted on the training rows only)

`make_preprocessor` is one `ColumnTransformer`: TF-IDF over the joined text, one-hot
metadata, ordinal levels, scaled numeric statistics and the binary flags. Session 5: it
is fitted on the training rows only and then applied to validation and test.
"""),
    ("code", """
split0 = get_splits(dfe, SEED)

# Fit on the training rows only, then transform validation with the fitted vocabulary and scales.
pre = ft.make_preprocessor()
X_train = pre.fit_transform(dfe.iloc[split0["train"]])
X_val = pre.transform(dfe.iloc[split0["val"]])
names = pre.get_feature_names_out()
print("train matrix:", X_train.shape, "| validation matrix:", X_val.shape)

# Columns contributed by each block of the ColumnTransformer.
pd.Series([n.split("__")[0] for n in names]).value_counts().rename("columns per block").to_frame()
"""),
    ("md", """
The binary-flag block passes through a `VarianceThreshold`: flags that are (almost)
always 0 or always 1 on the training rows carry no information and are dropped.
"""),
    ("code", """
kept = {n.split("__", 1)[1] for n in names if n.startswith("binary__")}
dropped = [c for c in ft.BINARY_COLS if c not in kept]
print("flags kept   :", sorted(kept))
print("flags dropped:", dropped or "none")
"""),
    ("md", """
## What the EDA tells us to do

- Fix the text first: junk cells become missing, hashed contacts become generic placeholders,
  glued sentences are separated.
- Drop exact duplicates before splitting: a copy of the same ad on both sides of a
  split would leak evaluation rows.
- Split by groups of related ads (same description, same company or near-copy), not by row:
  companies are label-pure, so a company seen in training would give the label away in test.
- The classes are heavily imbalanced: use PR-AUC as the headline metric, pick the
  decision threshold on validation, and never judge a model by accuracy alone.
- Missingness is informative: keep the missing-indicator flags instead of imputing
  them away.
- The text fields carry the strongest signal: keep the joined text central to the
  feature set and read model errors from it.
- Fit every transformer on the training rows only; the saved splits make that the same
  for every model, notebook and machine.
"""),
]

# ---------------------------------------------------------------------------
# Notebook 02: classical models walkthrough on the seed-0 split
# ---------------------------------------------------------------------------

NB2 = [
    ("md", """
# Classical ML: TF-IDF logistic regression and a one-class SVM (notebook 02)

Part A walks through one seed-0 split with the pipeline from the course slides: the dummy
baseline, TF-IDF logistic regression with a threshold picked on validation, and an unsupervised
one-class SVM, then explanations and error reading. Only training and validation rows are used
there. Part B runs the tuned experiment for all five split seeds (finished runs are
reused from `outputs/classical/`, so this is fast unless you delete them); the test split is
scored once per (model, seed) inside the run. Comparing and plotting the saved results is
notebook 04.

Run time: about 5 minutes for part A; part B is under a minute when the saved runs exist, or
about 40 minutes if it has to recompute all 15 runs (3 models x 5 seeds).
"""),
    ("code", """
import sys, os; os.chdir(os.path.join(os.getcwd(), "..")) if os.path.basename(os.getcwd()) == "notebooks" else None; sys.path.insert(0, "."); import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from modules import data_helper as dh
from modules import features as ft
from modules import models as md
from modules import eval_helper as ev
from modules import explain_helper as eh
from sklearn.inspection import permutation_importance
from modules import classical_runs as cr
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, precision_recall_curve

DATA = "../data/fake_job_postings.csv"
"""),
    ("code", """
SEED = 0
SEEDS = [0, 1, 2, 3, 4]
SPLIT_PATH = "splits/seed0.npz"
OUT_CLASSICAL = "outputs/classical"

dh.seed_everything(SEED)
sns.set_theme(style="whitegrid")
"""),
    ("md", """
# Part A: one split, validation only

## Setup: the seed-0 split

Load, clean, add features, and reuse the saved seed-0 split (Session 5: split
first, then fit on the training rows only).
"""),
    ("code", """
# Load, clean (drop duplicates), add features.
df = ft.add_engineered_features(dh.load_processed())

# Reuse the saved split, or create and save it the first time.
if os.path.exists(SPLIT_PATH):
    splits = dh.load_splits(SPLIT_PATH, df)
else:
    splits = dh.make_splits(df, seed=SEED)
    dh.save_splits(SPLIT_PATH, splits, df)
assert not set(splits["train"]) & set(splits["val"]), "train/val row overlap"

# Frames and labels for the two splits this notebook uses (test stays untouched).
train = df.iloc[splits["train"]]
val = df.iloc[splits["val"]]
y_train = train["fraudulent"].to_numpy()
y_val = val["fraudulent"].to_numpy()
"""),
    ("code", """
y_all = df["fraudulent"].to_numpy()
for name in ("train", "val", "test"):
    idx = splits[name]
    print(f"{name:5s}: {len(idx):6d} rows, {int(y_all[idx].sum()):4d} fraud ({y_all[idx].mean():.1%})")
print("test rows are listed for completeness; this notebook never scores them")
"""),
    ("md", """
## One helper for every model

The same three steps each time: fit on train, pick the decision threshold on
validation (never on test), then report train and validation side by side so the
generalisation gap is visible (Session 6). `ev.compare_table` names its second split
"test"; we pass validation metrics there and rename the columns below, so the test
rows are never touched.
"""),
    ("code", """
# Everything the later cells need, keyed by model name.
fitted, thresholds, scores, results = {}, {}, {}, {}


def report(name, model):
    \"\"\"Score a fitted model; the threshold is picked on validation only.\"\"\"
    # Validation scores, then the threshold chosen on them.
    s_val = ev.score_of(model, val)
    # A dummy scores every ad the same, so it keeps its own predictions (threshold 0.5).
    thr = 0.5 if name.startswith("dummy") else ev.pick_threshold(y_val, s_val)

    # Train and validation metrics at that threshold, stored for the comparison table.
    results[name] = {"train": ev.evaluate(y_train, ev.score_of(model, train), thr),
                     "test": ev.evaluate(y_val, s_val, thr)}
    fitted[name], thresholds[name], scores[name] = model, thr, s_val

    # One small table for display.
    out = pd.DataFrame(results[name]).T.rename(index={"test": "val"})  # "test" slot holds validation
    out = out[["pr_auc", "roc_auc", "f1", "precision", "recall"]]
    out.insert(0, "model", name)
    return out
"""),
    ("md", """
## The dummy baseline (Sessions 6/7)

Every model has to beat the dumb reference first: `dummy_majority` always predicts the
majority class (real). It reaches about 95% accuracy while catching no fraud at all, which is why
accuracy is not our metric; its PR-AUC equals the fraud rate (about 0.049), the floor for PR-AUC.
Its score is the same for every ad, so a tuned threshold would mean nothing; it keeps 0.5.
"""),
    ("code", """
frames = [report(name, est.fit(train, y_train))
          for name, est in md.get_baselines(SEED).items()]
pd.concat(frames)
"""),
    ("md", """
## Logistic regression on TF-IDF + engineered features

`modules/models.py` builds one Pipeline: TF-IDF over the joined text, one-hot and
ordinal encoded metadata, scaled numeric flags, then a `class_weight="balanced"`
logistic regression.
"""),
    ("code", """
report("logreg", md.get_models(SEED)["logreg"].fit(train, y_train))
"""),
    ("md", """
### Confusion matrix on validation
"""),
    ("code", """
# Hard predictions at the validation-chosen threshold, drawn with sklearn's display.
pred_val = (scores["logreg"] >= thresholds["logreg"]).astype(int)
ConfusionMatrixDisplay.from_predictions(y_val, pred_val, display_labels=["real", "fraud"],
                                        colorbar=False)
plt.title("Validation confusion matrix (logreg)")
plt.show()
"""),
    ("md", """
Reading: rows are the actual class, columns the predicted one; the diagonal holds
the correct calls.
"""),
    ("md", """
### The threshold trade-off

Session 6: with an imbalanced target there is no free lunch between precision and
recall; the threshold is a business decision, and we pick it on validation.
"""),
    ("code", """
# Precision and recall at every distinct score (the last point has no threshold).
prec, rec, thr_grid = precision_recall_curve(y_val, scores["logreg"])

# Two curves plus the threshold that was actually chosen.
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(thr_grid, prec[:-1], label="precision")
ax.plot(thr_grid, rec[:-1], label="recall")
ax.axvline(thresholds["logreg"], color="grey", linestyle="--", label="chosen threshold")
ax.set_xlabel("threshold on the fraud score")
ax.set_ylabel("metric value")
ax.set_title("Precision and recall vs threshold (logreg, validation)")
ax.legend()
plt.show()
"""),
    ("md", """
Reading: the dashed line is the threshold picked on validation; moving it right can
only shrink the set of flagged ads.
"""),
    ("md", """
## One-class SVM: can fraud be found without labels?

Anomaly detection view: learn what a *real* ad looks like from the real training ads only, then
flag ads that do not fit. The detector never sees a fraud example; the labels are used only to
leave the frauds out of its training data and, in part B, to choose its settings.

Two choices make this a fair test. First, the input is the **text only**: in the full feature
matrix four numeric columns (lengths, capitals) make up about half of each row's length, and an
RBF kernel works on distances, so it would mostly compare ad lengths. Second, the RBF kernel needs
short dense input, so the TF-IDF text is reduced to 100 dimensions with **PCA** (the `arpack`
solver works on the sparse matrix directly) and each row is rescaled to unit length. The score is
how far an ad lies outside the region of the real ads.
"""),
    ("code", """
report("ocsvm", md.get_models(SEED)["ocsvm"].fit(train, y_train))
"""),
    ("md", """
### gamma decides how "unusual" is measured

`gamma` sets how local the RBF kernel is. Grouped 3-fold CV on the training rows only, for a few
values (this takes a few minutes):
"""),
    ("code", """
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score

folds = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
for gamma in [0.3, 3, 30, 300]:
    pipe = md.get_models(SEED)["ocsvm"].set_params(clf__detector__svm__gamma=gamma)
    pr = cross_val_score(pipe, train, y_train, groups=train["group"], cv=folds,
                         scoring="average_precision")
    roc = cross_val_score(pipe, train, y_train, groups=train["group"], cv=folds, scoring="roc_auc")
    print(f"gamma {gamma:>5}: CV PR-AUC {pr.mean():.3f}  ROC-AUC {roc.mean():.3f}")
"""),
    ("md", """
Reading: with a broad kernel (small gamma) the ROC-AUC is far **below** 0.5, so frauds look *more*
typical than real ads: many are repeated templates that sit in dense regions. With a local kernel
(gamma around 30) the detector asks whether an ad has a close neighbour among the real ads, and
frauds often do not. The fraud base rate (PR-AUC floor) is about 0.049.
"""),
    ("md", """
Is the remaining gap to the supervised models a lack of information in these features, or a limit
of the anomaly view? Check: the same text, the same PCA, but a supervised logistic regression
that is allowed to see the labels.
"""),
    ("code", """
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pca_lr = Pipeline([("pre", ColumnTransformer([("text", ft.make_tfidf(), "text")])),
                   ("pca", PCA(n_components=100, svd_solver="arpack", random_state=SEED)),
                   ("clf", LogisticRegression(class_weight="balanced", max_iter=2000))])
report("pca_logreg (check)", pca_lr.fit(train, y_train))
"""),
    ("md", """
Reading: if the supervised model on the very same features scores far above the one-class SVM,
the information is there and the labels are what the anomaly view is missing.
"""),
    ("md", """
## All models side by side

Train versus validation PR-AUC per model; the gap column is the generalisation gap.
This is validation, not test: part B scores the test split once per (model, seed).
"""),
    ("code", """
table = ev.compare_table(results).rename(columns=lambda c: c.replace("test_", "val_"))
table.sort_values("val_pr_auc", ascending=False)
"""),
    ("md", """
## Hyper-parameter tuning (switched off)

Part B below runs the full random search over all models and five seeds:
grouped cross-validation on the training rows only, scoring PR-AUC (Session 7). The
flag below runs a tiny live version (`n_iter=3, cv=3`); keep it False during the
demo.
"""),
    ("code", """
RUN_TUNING = False  # flip to True to run a small random search live (slow)

if RUN_TUNING:
    # Grouped CV on the training rows so duplicate descriptions cannot leak across folds.
    search = md.tune(md.get_models(SEED)["logreg"], md.PARAM_GRIDS["logreg"],
                     train, y_train, train["group"].to_numpy(),
                     n_iter=3, cv=3, seed=SEED)
    print("best params:", search.best_params_)
    print("CV PR-AUC:", round(float(search.best_score_), 3))
"""),
    ("md", """
## Which words drive the logreg?

Session 6: read the model. `top_linear_terms` maps the coefficients back to feature
names recovered from the fitted ColumnTransformer.
"""),
    ("code", """
# The 15 strongest terms on each side, sorted so the bars read from blue to red.
terms = eh.top_linear_terms(fitted["logreg"], k=15).sort_values("coef")

# Horizontal bars: red pushes towards fraud, blue towards real.
fig, ax = plt.subplots(figsize=(8, 7))
ax.barh(terms["term"], terms["coef"],
        color=np.where(terms["coef"] > 0, "tab:red", "tab:blue"))
ax.set_xlabel("logistic regression coefficient")
ax.set_title("Strongest fraud (red) and real-job (blue) terms")
plt.show()
"""),
    ("md", """
Reading: red bars push an ad towards fraud and blue bars towards real; bar length is
the coefficient, not the importance of a whole input column.
"""),
    ("md", """
## Which columns matter? Permutation importance

Shuffle one raw input column at a time and see how much validation PR-AUC drops
(Session 6). We use a 1000-row validation sample and a single repeat to keep it
quick.
"""),
    ("code", """
# Small validation sample and one repeat keep this quick. sklearn shuffles each raw column
# of the frame in turn and measures the drop in PR-AUC of the whole fitted pipeline.
val_sample = val.sample(min(1000, len(val)), random_state=SEED)
perm = permutation_importance(fitted["logreg"], val_sample, val_sample["fraudulent"].to_numpy(),
                              scoring="average_precision", n_repeats=1, random_state=SEED)

# Top 10 columns as a bar chart.
top = pd.Series(perm.importances_mean, index=val_sample.columns).nlargest(10).iloc[::-1]
fig, ax = plt.subplots(figsize=(7, 5))
ax.barh(top.index, top.values)
ax.set_xlabel("drop in validation PR-AUC when the column is shuffled")
ax.set_title("Permutation importance (validation sample, 1 repeat)")
plt.show()
"""),
    ("md", """
Reading: the longer the bar, the more the model leaned on that single input column;
near-zero bars mean the column was barely used.
"""),
    ("md", """
## Read the errors (Session 6)

The five most confident false positives (flagged, but real) and false negatives
(not flagged, but fraud) of the logreg on validation.
"""),
    ("code", """
errors = eh.show_errors(val, y_val, scores["logreg"], thresholds["logreg"], k=5)
for _, e in errors.iterrows():
    print(f"[{e['error_type']}] score={e['score']:.3f}  title: {e['title']}")
    print(e["description"], "\\n")
"""),
    ("md", """
## Per-group check: employment type

Session 6: a model can look fine on average and still fail one group. Recall and
precision by employment type at the logreg's validation threshold.
"""),
    ("code", """
# One row per prediction, then counts per employment type (missing is its own group).
pred_val = (scores["logreg"] >= thresholds["logreg"]).astype(int)
g = pd.DataFrame({"group": val["employment_type"].fillna("missing").to_numpy(),
                  "y": y_val, "pred": pred_val})
g["tp"] = g["y"] & g["pred"]
tab = g.groupby("group").agg(count=("y", "size"), positives=("y", "sum"),
                             flagged=("pred", "sum"), tp=("tp", "sum"))

# Recall = caught frauds / all frauds; precision = caught frauds / flagged ads.
tab["recall"] = tab["tp"] / tab["positives"]
tab["precision"] = tab["tp"] / tab["flagged"]
tab[["count", "positives", "recall", "precision"]]
"""),
    ("md", """
Reading: rows with few ads have noisy recall and precision; the count column says
how much to trust each row.
"""),
    ("md", """
# Part B: tuned runs over five split seeds

For each seed and model `cr.run_classical` tunes on the training rows (random search, 4
candidates, 3-fold grouped CV, scoring PR-AUC), refits, picks the threshold on validation, and
scores validation and test once. One JSON + NPZ pair per (model, seed) goes to
`outputs/classical/`; pairs that already exist are skipped. Only validation numbers are printed.
"""),
    ("code", """
RUN_CLASSICAL = True  # False: only read the saved results below
N_JOBS = 1            # parallel CV fits per search; up to 4 on the server
if RUN_CLASSICAL:
    df_all = ft.add_engineered_features(dh.load_processed())
    cr.run_classical(df_all, SEEDS, out_dir=OUT_CLASSICAL, n_iter=4, cv=3, n_jobs=N_JOBS)
"""),
]

# ---------------------------------------------------------------------------
# Notebook 04: analysis of the saved results of notebooks 02 and 03
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Results cells of notebook 04: read saved files only, fit nothing
# ---------------------------------------------------------------------------

NB4_RESULTS = [
    ("code", """
DIRS = {"classical": OUT_CLASSICAL, "deep": OUT_DEEP, "stack": OUT_STACK}


class StopExecution(Exception):
    \"\"\"Stops the notebook quietly when there is nothing to report.\"\"\"

    def _render_traceback_(self):
        pass  # IPython hook: an empty traceback, so the notebook just stops
"""),
    ("md", """
## Load the saved results

`results_helper.load_results` collects one row per (model, seed) from the JSON
files of the groups listed in `GROUPS` (deep results only when `outputs/deep/` is non-empty). If
nothing is found we stop here with a message instead of crashing.
"""),
    ("code", """
# Result folders of the wanted groups that exist and are not empty.
dirs = [d for g, d in DIRS.items() if g in GROUPS and os.path.isdir(d) and os.listdir(d)]

# One row per (model, seed) from the saved json files.
res = rh.load_results(*dirs) if dirs else pd.DataFrame(columns=["group"])
res = res[res["group"].isin(GROUPS)]
if res.empty:
    print("No result files were found. Looked in:", dirs)
    print("Run the experiment cells of this notebook first, then re-run this section.")
    raise StopExecution
print(f"{len(res)} runs found: {res['model'].nunique()} models, "
      f"seeds {sorted(int(x) for x in res['seed'].unique())}")
"""),
    ("md", """
## Which runs were found

Each cell counts finished runs; a full grid has exactly one run per (model, seed).
"""),
    ("code", """
res.groupby(["model", "seed"]).size().unstack(fill_value=0)
"""),
    ("md", """
## Headline test metrics

Mean and standard deviation over seeds, sorted by mean test PR-AUC. The dummy rows
are the floor every real model has to beat. With a single seed there is no standard
deviation and only the mean is shown.
"""),
    ("code", """
def fmt(col):
    \"\"\"mean +- std over seeds for one metric; a lone seed has no std.\"\"\"
    g = res.groupby("model")[col]
    return [f"{m:.3f} +- {s:.3f}" if pd.notna(s) else f"{m:.3f}"
            for m, s in zip(g.mean(), g.std())]


# Models ordered by mean test PR-AUC, best first.
order = res.groupby("model")["test_pr_auc"].mean().sort_values(ascending=False).index

# The headline table: mean +- std over seeds.
pd.DataFrame({
    "seeds": res.groupby("model").size(),
    "test PR-AUC": fmt("test_pr_auc"),
    "test F1": fmt("test_f1"),
    "test precision": fmt("test_precision"),
    "test recall": fmt("test_recall"),
}).loc[order]
"""),
    ("md", """
The dummy predicts "real" for every ad: precision, recall and F1 are 0 and its PR-AUC equals the
share of fraud in the data, the floor every real model has to beat.
"""),
    ("md", """
## Validation vs test PR-AUC

Validation is what models, hyper-parameters and thresholds were chosen on; test is
the honest number. The difference is the price of that selection.
"""),
    ("code", """
# Mean validation and test PR-AUC per model, then grouped bars.
means = res.groupby("model")[["val_pr_auc", "test_pr_auc"]].mean().loc[order]
ax = means.plot(kind="bar", figsize=(9, 4))
ax.set_title("PR-AUC on validation vs test (mean over seeds)")
ax.set_xlabel("")
ax.set_ylabel("PR-AUC")
plt.show()
"""),
    ("md", """
Reading: each model has two bars, validation and test, on the same 0 to 1 scale;
the validation bar tends to sit at or above the test bar.
"""),
    ("md", """
## Generalisation gap

Train PR-AUC minus test PR-AUC per model (mean over seeds). A large gap means the
model memorised its training rows.
"""),
    ("code", """
g = res.groupby("model").agg(train_pr_auc=("train_pr_auc", "mean"),
                             test_pr_auc=("test_pr_auc", "mean"))
g["gap (train - test)"] = g["train_pr_auc"] - g["test_pr_auc"]
g.sort_values("gap (train - test)", ascending=False)
"""),
    ("md", """
## Per-seed spread

Session 7 rule of thumb: a model that wins by less than its own standard deviation
has not won. Each point is one split seed.
"""),
    ("code", """
# One point per (model, seed).
fig, ax = plt.subplots(figsize=(9, 4))
sns.stripplot(data=res, x="model", y="test_pr_auc", order=order, ax=ax, size=6)
ax.set_title("Test PR-AUC per split seed")
ax.set_xlabel("")
ax.set_ylabel("test PR-AUC")
ax.tick_params(axis="x", rotation=45)
plt.show()
"""),
    ("md", """
Reading: every point is one seed of one model; models whose point clouds overlap
are indistinguishable at this level of noise.
"""),
    ("md", """
## Seed 0 in detail

The NPZ files hold y and scores for validation and test in the same row order as the
split, so curves and matrices can be rebuilt without refitting anything. If no
seed-0 run exists we stop with a message.
"""),
    ("code", """
# Stop quietly when there is no seed-0 run.
seed0 = res[res["seed"] == 0]
if seed0.empty:
    print("No seed-0 runs found; the remaining sections need seed 0.")
    raise StopExecution

# Saved scores and threshold of every seed-0 model; y_test is the same for all of them.
seed_runs = {row["model"]: (rh.load_scores(DIRS[row["group"]], row["model"], 0), row["thr"])
             for _, row in seed0.iterrows()}
y_test = next(iter(seed_runs.values()))[0]["y_test"]
print("seed-0 models:", ", ".join(seed_runs))
"""),
    ("md", """
### Precision-recall curves on the test split (seed 0)

Built from the saved scores with sklearn's `precision_recall_curve`; the dashed line
is the fraud base rate, i.e. what random guessing scores.
"""),
    ("code", """
# One curve per model, rebuilt from the saved test scores, on shared axes.
fig, ax = plt.subplots(figsize=(9, 5))
for name, (s, thr) in seed_runs.items():
    PrecisionRecallDisplay.from_predictions(y_test, s["score_test"], name=name, ax=ax)

# Base rate: what random guessing scores.
ax.axhline(float(y_test.mean()), linestyle="--", color="grey", label="base rate")
ax.set_title("Test precision-recall curves (seed 0)")

# Legend outside the axes, to the right, so it does not cover the curves.
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize="small")
fig.tight_layout()
plt.show()
"""),
    ("md", """
Reading: the x axis is recall, the y axis precision; closer to the top-right corner
is better, and the dashed grey line is the base rate every curve should beat.
"""),
    ("md", """
### Confusion matrices at the validation-chosen threshold (seed 0)
"""),
    ("code", """
# Grid of panels, 3 per row, one per model.
n_cols = 3
n_rows = -(-len(seed_runs) // n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), squeeze=False)

# Hard predictions at each model's own validation-chosen threshold.
for ax, (name, (s, thr)) in zip(axes.ravel(), seed_runs.items()):
    pred = (s["score_test"] >= thr).astype(int)
    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["real", "fraud"],
                                            colorbar=False, ax=ax)
    ax.set_title(name)

# Hide unused panels.
for ax in axes.ravel()[len(seed_runs):]:
    ax.set_visible(False)
fig.suptitle("Test confusion matrices, validation-chosen threshold (seed 0)")
fig.tight_layout()
plt.show()
"""),
    ("md", """
Reading: rows are the actual class, columns the predicted class; the diagonal holds
the correct calls.
"""),
    ("md", """
## Fit time per model

Wall-clock seconds per run (mean over seeds and slowest seed), as recorded by the
runners.
"""),
    ("code", """
res.groupby("model")["seconds"].agg(["mean", "max"]).sort_values("mean").round(1)
"""),
    ("md", """
## Which frauds does each model catch? (seed 0)

At each model's own validation-chosen threshold: how many of the test frauds does
model A catch that model B misses. Plus the frauds caught by at least one model but
not by all.
"""),
    ("code", """
# Which test frauds each model catches at its own threshold.
SKIP_CATCH = ("dummy_majority",)
fraud_idx = np.where(y_test == 1)[0]
# Left out: the dummy flags nothing.
caught = {name: frozenset(fraud_idx[s["score_test"][fraud_idx] >= thr])
          for name, (s, thr) in seed_runs.items() if not name.startswith(SKIP_CATCH)}

# Pairwise matrix: frauds caught by the row model but missed by the column model.
names = list(caught)
overlap = pd.DataFrame(0, index=names, columns=names)
for a in names:
    for b in names:
        overlap.loc[a, b] = len(caught[a] - caught[b])

# Frauds caught by at least one model but not by all (union minus intersection).
union = set().union(*caught.values())
inter = set(fraud_idx)
for c in caught.values():
    inter &= c
print("test frauds caught by at least one model but not by all:", len(union - inter))
overlap
"""),
    ("md", """
Reading: cell (A, B) counts test frauds caught by A but missed by B, so the
diagonal is zero by construction; rows with large values catch much that the column
models let through.
"""),
    ("md", """
## The ads nobody catches (seed 0)

The last Session 6 step: read the test frauds that no model flagged at its own
threshold. These are the target for the next iteration. This cell needs the seed-0
split file to look the rows up.
"""),
    ("code", """
if not os.path.exists(SPLIT_PATH):
    print("splits/seed0.npz not found; cannot look up the missed ads.")
else:
    # The seed-0 test rows, in the same order as the saved scores.
    df = ft.add_engineered_features(dh.load_processed())
    test = df.iloc[dh.load_splits(SPLIT_PATH, df)["test"]]

    # Ads flagged by at least one real model at its own threshold.
    caught_any = np.zeros(len(test), dtype=bool)
    for name, (s, thr) in seed_runs.items():
        if not name.startswith(SKIP_CATCH):   # see the cell above for why these are left out
            caught_any |= (s["score_test"] >= thr)

    # The frauds nobody caught, printed for reading.
    missed = test[(y_test == 1) & ~caught_any]
    print(len(missed), "test frauds that no model caught:")
    for _, r in missed.iterrows():
        print("-", str(r["title"])[:80], "|", str(r["description"])[:150], "\\n")
"""),
]


NB4_HEAD = [
    ("md", """
# Analysis of the results (notebook 04)

Notebooks 02 and 03 run the models; this one only reads what they saved in `outputs/classical/`
and `outputs/deep/`. The only model fitted here is a two-input logistic regression that stacks two
saved runs (first section); it is CPU-only and takes seconds. Then it compares all models on the
same five splits:
headline metrics, validation versus test, generalisation gap, per-seed spread, curves and
confusion matrices of seed 0, the training curves of the fine-tuned model, and which frauds
each model catches or misses.

Run time: under a minute.
"""),
    ("code", """
import sys, os; os.chdir(os.path.join(os.getcwd(), "..")) if os.path.basename(os.getcwd()) == "notebooks" else None; sys.path.insert(0, "."); import warnings
import glob, json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay

from modules import data_helper as dh
from modules import features as ft
from modules import results_helper as rh

DATA = "../data/fake_job_postings.csv"
"""),
    ("code", """
SEED = 0
SPLIT_PATH = "splits/seed0.npz"
OUT_CLASSICAL = "outputs/classical"
OUT_DEEP = "outputs/deep"
OUT_STACK = "outputs/stack"
GROUPS = ["classical", "deep", "stack"]

dh.seed_everything(SEED)
sns.set_theme(style="whitegrid")
"""),
    ("md", """
## Stacking DistilBERT with TF-IDF logistic regression

The two models catch different frauds (see "Explaining DistilBERT, part 1" below), so we test
whether combining them helps. The standard way: a logistic regression that takes the two saved
scores (as logits) as its only two inputs. It is fitted on the validation rows, because both
models were trained on the training rows and their training scores are over-confident. The
threshold comes from 5-fold grouped out-of-fold scores inside validation, so it never sees a row
the stacker was fitted on; test is scored once. TF-IDF logistic regression is the partner because it
is the standard classical baseline and the model DistilBERT disagrees with in part 1 below.
"""),
    ("code", """
from modules import stacking

A = (OUT_DEEP, "bert_distilbert-base-uncased")
B = (OUT_CLASSICAL, "logreg")
if all(rh.result_exists(*m, s) for m in (A, B) for s in range(5)):
    df_stack = ft.add_engineered_features(dh.load_processed())
    stacking.run_stack(df_stack, range(5), A, B, out_dir=OUT_STACK)
else:
    print("needs all 5 seeds of both runs; skipping the stack")
"""),
]

NB4 = NB4_HEAD + NB4_RESULTS + [
    ("md", """
## Training curves of the fine-tuned model

Session 7: read the curve, not only the end number. Per epoch validation PR-AUC and training
loss, from the `history` stored with each saved run.
"""),
    ("code", """
import glob, json

# One row per (seed, epoch) from the history stored in each saved fine-tuning run.
rows = []
for path in sorted(glob.glob(os.path.join(OUT_DEEP, "bert_*_seed*.json"))):
    with open(path, encoding="utf-8") as f:
        rec = json.load(f)
    for h in rec.get("history", []):
        rows.append({"seed": rec["seed"], **h})
hist = pd.DataFrame(rows)

if hist.empty:
    print("no fine-tuning history found yet")
else:
    # Left: validation PR-AUC per epoch. Right: training loss per epoch. One line per seed.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for seed, g in hist.groupby("seed"):
        axes[0].plot(g["epoch"], g["val_pr_auc"], marker="o", label=f"seed {seed}")
        axes[1].plot(g["epoch"], g["train_loss"], marker="o", label=f"seed {seed}")
    axes[0].set_title("Validation PR-AUC per epoch"); axes[0].set_xlabel("epoch"); axes[0].set_ylabel("PR-AUC")
    axes[1].set_title("Training loss per epoch"); axes[1].set_xlabel("epoch"); axes[1].set_ylabel("weighted cross-entropy")
    axes[0].legend()
    plt.show()

    # Time per epoch decides whether to run all five seeds.
    print("seconds per epoch:", hist["seconds"].round(0).tolist())
"""),
    ("md", """
## Explaining DistilBERT, part 1: where it disagrees with TF-IDF logistic regression (seed 0)

Error analysis from the saved test scores only, nothing is refitted. At each model's own
validation-chosen threshold: the frauds one model catches and the other misses, and the real ads
DistilBERT is most sure are fraud (its most confident false alarms).
"""),
    ("code", """
bert_name = next((n for n in seed_runs if n.startswith("bert_")), None)
if bert_name is None or "logreg" not in seed_runs or not os.path.exists(SPLIT_PATH):
    print("needs seed-0 runs of DistilBERT and logreg, and splits/seed0.npz")
else:
    df = ft.add_engineered_features(dh.load_processed())
    test = df.iloc[dh.load_splits(SPLIT_PATH, df)["test"]].reset_index(drop=True)

    # Hard predictions of both models at their own thresholds.
    (sb, tb), (sl, tl) = seed_runs[bert_name], seed_runs["logreg"]
    flag_b, flag_l = sb["score_test"] >= tb, sl["score_test"] >= tl
    fraud = y_test == 1

    def titles(mask, k=8):
        for _, r in test[mask].head(k).iterrows():
            print("   -", str(r["title"])[:70], "|", str(r["description"])[:80])

    print(f"frauds caught by DistilBERT only: {int((fraud & flag_b & ~flag_l).sum())}")
    titles(fraud & flag_b & ~flag_l)
    print(f"frauds caught by logreg only: {int((fraud & flag_l & ~flag_b).sum())}")
    titles(fraud & flag_l & ~flag_b)

    # The real ads with the highest DistilBERT scores.
    worst = np.argsort(-np.where(fraud, -np.inf, sb["score_test"]))[:5]
    print("DistilBERT's most confident false alarms:")
    for i in worst:
        print(f"   - score {sb['score_test'][i]:.3f} |", str(test.loc[i, "title"])[:70])
"""),
    ("md", """
## Explaining DistilBERT, part 2: which words pushed it towards fraud (Integrated Gradients)

Computed on the server right after training seed 0 (notebook 03): for 10 test ads, Integrated
Gradients measures how much each token moved the fraud probability compared with an all-[PAD]
version of the same ad. Word pieces (`##...`) are merged back into words. Listed: the 8 words with
the largest total push towards fraud (a word that occurs several times counts once, with its
pushes added up), after dropping stop words and punctuation (they get many small pushes that say
nothing); `$` and numbers are kept because pay amounts are a known scam cue.
"""),
    ("code", """
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

paths = sorted(glob.glob(os.path.join(OUT_DEEP, "explain", "bert_*_seed0.json")))
if not paths:
    print("no explanation file yet; it is written by the DistilBERT run of notebook 03")
else:
    with open(paths[0], encoding="utf-8") as f:
        explained = json.load(f)
    for r in explained:
        # Merge word pieces into words and drop the special tokens.
        words, attrs = [], []
        for t, a in zip(r["tokens"], r["attributions"]):
            if t in ("[CLS]", "[SEP]", "[PAD]"):
                continue
            if t.startswith("##") and words:
                words[-1] += t[2:]
                attrs[-1] += a
            else:
                words.append(t)
                attrs.append(a)
        # Keep words with a letter or digit (and "$"), minus English stop words.
        # One entry per word: a word that occurs several times gets the sum of its attributions.
        total = {}
        for a, w in zip(attrs, words):
            # Keep words (and "$"); drop punctuation, stop words and one-letter fragments.
            if (w == "$" or (len(w) > 1 and any(c.isalnum() for c in w))) and w not in ENGLISH_STOP_WORDS:
                total[w] = total.get(w, 0.0) + a
        ranked = sorted(total.items(), key=lambda kv: kv[1], reverse=True)[:8]
        top = [w for w, a in ranked if a > 0]
        # Check: the attributions add up to (score of the ad) - (score of the all-[PAD] baseline).
        print(f"[{r['kind']}] score {r['score']:.3f}, attributions sum {sum(attrs):+.2f}: " + ", ".join(top))
"""),
    ("md", """
Reading: a word listed for a missed fraud still pushed towards fraud, just not enough to cross
the threshold; words for a false alarm show what misled the model.
"""),
("md", """
## Findings (classical models)

To be written from the results of the server run.
"""),
    ("md", """
## Findings (deep models)

To be written from the results of the server run.
"""),
]


# NB2 ends with the part-B run cell above.

# ---------------------------------------------------------------------------
# Notebook 03: fine-tuned DistilBERT
# ---------------------------------------------------------------------------

NB3 = [
    ("md", """
# Deep learning: fine-tuned DistilBERT (notebook 03)

The whole pretrained encoder is fine-tuned on the training rows and scored once on test per seed,
on the same five splits as notebook 02; comparing and plotting the saved results is notebook 04.
The run cell needs a GPU and the `torch`, `transformers` and `captum` packages, and is skipped
unless `ON_GPU_SERVER = True` (set it on the server only).

Run time: about 5 minutes per seed on one GPU at batch size 128 (5 epochs).
"""),
    ("code", """
import sys, os; os.chdir(os.path.join(os.getcwd(), "..")) if os.path.basename(os.getcwd()) == "notebooks" else None; sys.path.insert(0, "."); import warnings
import importlib.util

from modules import data_helper as dh
from modules import features as ft
from modules import deep_runs as dr
"""),
    ("code", """
SEED = 0
SEEDS = [0, 1, 2, 3, 4]  # seeds with existing result files are skipped
OUT_DEEP = "outputs/deep"

BERT_MODEL = "distilbert-base-uncased"
EPOCHS, BATCH_SIZE, MAX_LEN, LR = 5, 128, 256, 2e-5
LOG_EVERY = 10  # print the loss every this many steps
DEVICE = "cuda:1"  # which GPU to use; the server is shared, so pick an emptier one (nvidia-smi)

# SAFETY: fine-tuning is heavy GPU work. It runs only when this flag is True, and the flag must
# stay False on the laptop. Set it to True on the GPU server, nowhere else.
ON_GPU_SERVER = False

# Libraries present? (a missing one would make the run cell fail, not skip)
HAVE_TORCH = all(importlib.util.find_spec(m) for m in ("torch", "transformers", "captum"))
RUN_BERT = ON_GPU_SERVER and HAVE_TORCH
print({"ON_GPU_SERVER": ON_GPU_SERVER, "torch stack": HAVE_TORCH})

dh.seed_everything(SEED)
"""),
    ("md", """
## Data and splits

The same processed data and the same saved splits as the classical notebook, so every model is
scored on identical rows.
"""),
    ("code", """
df = ft.add_engineered_features(dh.load_processed())
print("rows:", len(df), "| fraud:", int(df["fraudulent"].sum()))
"""),
    ("md", """
## Fine-tune DistilBERT

The whole encoder is trained with a linear classification layer on top: AdamW, learning rate
2e-5 with warm-up and linear decay, square-root class weights in the loss, bf16 autocast on
CUDA, and early stopping on validation PR-AUC (the best epoch's weights are restored). Texts
are truncated to 256 tokens.

Model explanation: right after seed 0 is trained, Integrated Gradients scores how much each token
pushed the model towards fraud, for 10 seed-0 test ads (4 caught frauds, 3 missed frauds, 3 false
alarms). They are saved next to the results and shown in notebook 04.
"""),
    ("code", """
if RUN_BERT:
    dr.run_bert(df, SEEDS, OUT_DEEP, DEVICE, BERT_MODEL, epochs=EPOCHS,
                batch_size=BATCH_SIZE, max_len=MAX_LEN, lr=LR, log_every=LOG_EVERY)
else:
    print("skipped: set ON_GPU_SERVER = True (GPU server only) and install torch + transformers + captum")
"""),
]

def main():
    """Write the four notebooks, then re-read and parse-check them."""
    for cells, name in ((NB1, "01_eda_preprocessing.ipynb"),
                        (NB2, "02_classical_ml.ipynb"),
                        (NB3, "03_deep_learning.ipynb"),
                        (NB4, "04_analysis.ipynb")):
        path = build(cells, name)
        total, n_code = check(path)
        print(f"{path}: {total} cells ({n_code} code, {total - n_code} markdown)")


if __name__ == "__main__":
    main()
