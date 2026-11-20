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
data, duplicates, text lengths and simple spam-style flags. Every section feeds a
decision: what to drop, how to split, and which metrics to trust. The second half builds
the preprocessing that notebooks 02 (classical ML) and 03 (deep learning) share:
cleaning, engineered features, the model matrix, and the five saved train/val/test splits.

Run time: about 1 minute.
"""),
    ("code", """
import sys, os; os.chdir(os.path.join(os.getcwd(), "..")) if os.path.basename(os.getcwd()) == "notebooks" else None; sys.path.insert(0, "."); import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from modules import data_helper as dh
from modules import features as ft
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
## Cleaning

Session 5: clean first, then split, then fit. `clean` drops exact duplicates (ignoring
`job_id`) and hashes each normalised description into a `group` id, so near-copies can
be kept on one side of every split later.
"""),
    ("code", """
dfc = dh.clean(df)
print("rows before cleaning:", len(df), "| after:", len(dfc))
print("description groups:", dfc["group"].nunique())
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
## Splits: by description group, saved for all seeds

`make_splits` uses `StratifiedGroupKFold` on the group id (60/20/20, class ratio kept), so one
description can never sit in two splits. Each seed's indices are saved to `splits/` with a
fingerprint of the cleaned data; every notebook and every machine reloads exactly these rows.
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
print("checked: no description group appears in two splits, for every seed")
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
pre = ft.make_preprocessor("tfidf")
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

- Drop exact duplicates before splitting: a copy of the same ad on both sides of a
  split would leak evaluation rows.
- Split by description group, not by row: near-identical ads must stay on one side
  of the split.
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
# Classical ML: dummy, logistic regression, BM25, tree, MLP, Naive Bayes, one-class SVM (notebook 02)

Part A walks through one seed-0 split with the pipeline from the course slides: dummy
baseline, TF-IDF logistic regression with a threshold picked on validation, BM25, a decision
tree, an MLP, Complement Naive Bayes and an unsupervised one-class SVM, then explanations and
error reading. Only training and validation rows
are used there. Part B runs the tuned experiment for all five split seeds (finished runs are
reused from `outputs/classical/`, so this is fast unless you delete them); the test split is
scored once per (model, seed) inside the run. Comparing and plotting the saved results is
notebook 04.

Run time: about 5 minutes for part A; part B is under a minute when the saved runs exist, or
about an hour if it has to recompute all 35 runs.
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
df = ft.add_engineered_features(dh.clean(dh.load_raw(DATA)))

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
## BM25 instead of TF-IDF

BM25 is the ranking function behind classic search engines. It differs from TF-IDF
in two ways: repeating a word has diminishing returns (saturating term frequency),
and long ads are discounted relative to the average ad length (length normalisation).
The vocabulary and everything else in the pipeline are identical, so any difference
in the numbers comes from the weighting alone.
"""),
    ("code", """
report("logreg_bm25", md.get_models(SEED)["logreg_bm25"].fit(train, y_train))
"""),
    ("md", """
## Decision tree

Easy to read, prone to overfitting on sparse text. `tree_rules` prints the top
levels of the fitted tree; `top_tree_features` lists the features it leans on.
"""),
    ("code", """
report("tree", md.get_models(SEED)["tree"].fit(train, y_train))
"""),
    ("code", """
print(eh.tree_rules(fitted["tree"], max_depth=3))
"""),
    ("code", """
eh.top_tree_features(fitted["tree"], k=10)
"""),
    ("md", """
## MLP on the same features

A small sklearn neural network. It has no `class_weight`, so we rely on the ranking
of its scores plus the threshold picked on validation (Session 5).
"""),
    ("code", """
report("mlp", md.get_models(SEED)["mlp"].fit(train, y_train))
"""),
    ("md", """
## Complement Naive Bayes

The classic text-classification baseline: word counts per class, with no feature interactions.
Complement NB is the variant designed for imbalanced classes (it learns each class from the
documents of all *other* classes). It needs non-negative inputs, which TF-IDF, one-hot flags and
the scaled counts are.
"""),
    ("code", """
report("cnb", md.get_models(SEED)["cnb"].fit(train, y_train))
"""),
    ("md", """
## One-class SVM: can fraud be found without labels?

Anomaly detection view: learn what a *real* ad looks like from the real training ads only, then
flag ads that do not fit. The detector never sees a fraud example; the labels are used only to
leave the frauds out of its training data (and, in part B, to pick `nu`). An RBF kernel needs
dense, low-dimensional input, so the TF-IDF matrix is first reduced to 100 to 200 dimensions with
truncated SVD (as in LSA). The score is how far an ad lies outside the "normal" region.
"""),
    ("code", """
report("ocsvm", md.get_models(SEED)["ocsvm"].fit(train, y_train))
"""),
    ("md", """
Is the fraud signal missing from these features, or does the "unusual = fraud" assumption fail?
Check: the same preprocessing and the same 200-dimensional SVD, but a supervised logistic
regression that is allowed to see the labels.
"""),
    ("code", """
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

svd_lr = Pipeline([("pre", ft.make_preprocessor("tfidf")),
                   ("svd", TruncatedSVD(n_components=200, random_state=SEED)),
                   ("clf", LogisticRegression(class_weight="balanced", max_iter=2000))])
report("svd_logreg (check)", svd_lr.fit(train, y_train))
"""),
    ("md", """
Reading: if the supervised model on the very same features scores far above the one-class SVM,
the information is there and it is the anomaly assumption that fails: fraudulent ads are not
unusual ads; many are generic, template-like postings.
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
if RUN_CLASSICAL:
    df_all = ft.add_engineered_features(dh.clean(dh.load_raw(DATA)))
    cr.run_classical(df_all, SEEDS, out_dir=OUT_CLASSICAL, n_iter=4, cv=3)
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
DIRS = {"classical": OUT_CLASSICAL, "deep": OUT_DEEP}


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
SKIP_CATCH = ("dummy_majority", "ocsvm")
fraud_idx = np.where(y_test == 1)[0]
# Left out: the dummy flags nothing, and the one-class SVM flags about half of all ads,
# so it "catches" frauds by flagging everything near them.
caught = {name: frozenset(fraud_idx[s["score_test"][fraud_idx] >= thr])
          for name, (s, thr) in seed_runs.items() if name not in SKIP_CATCH}

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
    df = ft.add_engineered_features(dh.clean(dh.load_raw(DATA)))
    test = df.iloc[dh.load_splits(SPLIT_PATH, df)["test"]]

    # Ads flagged by at least one real model at its own threshold.
    caught_any = np.zeros(len(test), dtype=bool)
    for name, (s, thr) in seed_runs.items():
        if name not in SKIP_CATCH:            # see the cell above for why these two are left out
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

Notebooks 02 and 03 run the models; this one only reads what they saved in 
and . Nothing is fitted here. It compares all models on the same five splits:
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
GROUPS = ["classical", "deep"]

dh.seed_everything(SEED)
sns.set_theme(style="whitegrid")
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
    df = ft.add_engineered_features(dh.clean(dh.load_raw(DATA)))
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
the largest push towards fraud.
"""),
    ("code", """
paths = sorted(glob.glob(os.path.join(OUT_DEEP, "bert_*_explain_seed0.json")))
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
        top = [w for a, w in sorted(zip(attrs, words), reverse=True)[:8] if a > 0]
        print(f"[{r['kind']}] score {r['score']:.3f}: " + ", ".join(top))
"""),
    ("md", """
Reading: a word listed for a missed fraud still pushed towards fraud, just not enough to cross
the threshold; words for a false alarm show what misled the model.
"""),
("md", """
## Findings (classical models)

Numbers are test PR-AUC, means over the 5 split seeds in the first table.

- Every supervised text model beats the dummy floor (0.049) by a wide margin: 0.77 to 0.85, except
  the single decision tree (0.40).
- Logistic regression with TF-IDF (0.821), with BM25 (0.847) and the MLP (0.846) are within seed
  noise of each other: the spread across seeds (std about 0.06 to 0.08) is larger than the gaps
  between them. The MLP costs about 7 minutes per seed against 1 to 1.5 for the logistic regressions.
- Complement Naive Bayes reaches 0.773: below TF-IDF logistic regression on 4 of 5 seeds (mean
  difference -0.048), but with the smallest spread across seeds of any supervised model (std 0.029).
- The one-class SVM, trained on real ads only, scores 0.057 (ROC-AUC 0.47), no better than the
  dummy. A supervised logistic regression on the same SVD features does far better (part A of
  notebook 02), so the signal is in the features: fraudulent ads are simply not unusual ads. This is
  why the project treats the task as supervised classification, not anomaly detection.
- Train minus test PR-AUC is 0.15 to 0.17 for the logistic regressions, the MLP and Naive Bayes,
  and 0.24 for the tree: all of them fit their training rows much better than unseen ads.
- Caveat: the dataset marks real ads from some countries (for example Greece) as 0%
  fraud, so part of the signal may come from how the data was collected.
"""),
    ("md", """
## Findings (deep models)

Pending: DistilBERT is being rerun on the GPU server (its classification head is now seeded, so
the five seeds are reproducible) together with the new zero-shot language model. The findings are
written from those results once they are back.
"""),
]


# NB2 ends with the part-B run cell above.

# ---------------------------------------------------------------------------
# Notebook 03: frozen embeddings and fine-tuned BERT
# ---------------------------------------------------------------------------

NB3 = [
    ("md", """
# Deep learning: frozen embeddings, fine-tuned DistilBERT, zero-shot LLM (notebook 03)

Session 7 order: first freeze a pretrained encoder and train only a small head on its
embeddings (cheap, cached), then fine-tune the whole encoder (expensive). Last, a zero-shot
language model (Session 8) answers "is this ad fraudulent?" with no training at all. Each model is scored
once on test per (model, seed) on the same five splits as notebook 02; comparing and plotting
the saved results is notebook 04. Heavy cells need a GPU and the `torch`, `transformers` and
`sentence-transformers` packages (plus `captum` for the explanation); the run cells are skipped unless `ON_GPU_SERVER = True`
(set it on the server only).

Run time: one DistilBERT run took about 7 minutes on one GPU at batch size 128 (measured, seed 0,
5 epochs); the embedding heads take seconds once the embeddings are cached.
"""),
    ("code", """
import sys, os; os.chdir(os.path.join(os.getcwd(), "..")) if os.path.basename(os.getcwd()) == "notebooks" else None; sys.path.insert(0, "."); import warnings
import importlib.util

from modules import data_helper as dh
from modules import features as ft
from modules import deep_runs as dr

DATA = "../data/fake_job_postings.csv"
"""),
    ("code", """
SEED = 0
SEEDS = [0, 1, 2, 3, 4]
BERT_SEEDS = SEEDS  # seeds with existing result files are skipped
OUT_DEEP = "outputs/deep"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BERT_MODEL = "distilbert-base-uncased"
LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
EPOCHS, BATCH_SIZE, MAX_LEN, LR = 5, 128, 256, 2e-5
LOG_EVERY = 10  # print the loss every this many steps
DEVICE = "cuda:1"  # which GPU to use; the server is shared, so pick an emptier one (nvidia-smi), e.g. "cuda:1"

# SAFETY: the embedding and fine-tuning cells are heavy GPU work. They run only when this flag is
# True, and it must stay False on the laptop. Set it to True on the GPU server, nowhere else.
ON_GPU_SERVER = False

# Libraries present? (a missing one would make the heavy cells fail, not skip)
HAVE_TORCH = all(importlib.util.find_spec(m)
                 for m in ("torch", "transformers", "sentence_transformers", "captum"))
HAVE_EMBEDDINGS = os.path.exists("features/" + EMBED_MODEL.replace("/", "_") + "_emb.npy")

# Heavy cells need the explicit opt-in AND the libraries; the sklearn heads are light (CPU) and
# only need the cached embeddings.
RUN_EMBED = RUN_BERT = RUN_LLM = ON_GPU_SERVER and HAVE_TORCH
RUN_HEADS = RUN_EMBED or HAVE_EMBEDDINGS
print({"ON_GPU_SERVER": ON_GPU_SERVER, "torch stack": HAVE_TORCH, "cached embeddings": HAVE_EMBEDDINGS})

dh.seed_everything(SEED)
"""),
    ("md", """
## Data and splits

The same cleaned frame and the same saved splits as the classical notebook, so every model
is scored on identical rows.
"""),
    ("code", """
df = ft.add_engineered_features(dh.clean(dh.load_raw(DATA)))
print("rows:", len(df), "| fraud:", int(df["fraudulent"].sum()))
"""),
    ("md", """
## Step 1: frozen sentence embeddings

`all-MiniLM-L6-v2` turns each ad (the joined `text` column) into one 384-dimensional vector.
Nothing is trained here. The vectors are cached in `features/` keyed by `job_id`, so this
cell is a no-op the second time.
"""),
    ("code", """
if RUN_EMBED:
    # Encode every ad once; the result is cached in features/.
    dr.run_embed(df, DEVICE, EMBED_MODEL)
else:
    print("skipped: set ON_GPU_SERVER = True (GPU server only) and install torch + sentence-transformers")
"""),
    ("md", """
## Step 2: logistic regression on the frozen embeddings

A logistic regression (C picked on validation PR-AUC) trained on the cached vectors with CPU only.
It is the same classifier as the TF-IDF logistic regression in notebook 02, so the pair compares
the text representation alone. Run for all five seeds; the threshold is picked on validation and
the test rows are scored once.
"""),
    ("code", """
if RUN_HEADS:
    dr.run_heads(df, SEEDS, OUT_DEEP, EMBED_MODEL)
else:
    print("skipped: no cached embeddings here")
"""),
    ("md", """
## Step 3: fine-tune DistilBERT

The whole encoder is trained with a linear classification layer on top: AdamW, learning rate
2e-5 with warm-up and linear decay, square-root class weights in the loss, bf16 autocast on
CUDA, and early stopping on validation PR-AUC (the best epoch's weights are restored). Texts
are truncated to 256 tokens. `BERT_SEEDS` lists the seeds to fine-tune; finished ones are skipped.

Model explanation: right after seed 0 is trained, Integrated Gradients scores how much each token
pushed the model towards fraud, for 10 seed-0 test ads (4 caught frauds, 3 missed frauds, 3 false
alarms). They are saved next to the results and shown in notebook 04.
"""),
    ("code", """
if RUN_BERT:
    dr.run_bert(df, BERT_SEEDS, OUT_DEEP, DEVICE, BERT_MODEL, epochs=EPOCHS,
                batch_size=BATCH_SIZE, max_len=MAX_LEN, lr=LR, log_every=LOG_EVERY)
else:
    print("skipped: set ON_GPU_SERVER = True (GPU server only) and install torch + transformers")
"""),
    ("md", """
## Step 4: zero-shot language model

`Qwen2.5-1.5B-Instruct` reads each ad (cut to 512 tokens) in a short Yes/No prompt; the fraud
score is its probability of answering "Yes" (renormalised over Yes and No). Nothing is trained,
so every ad is scored once and the scores are cached in `features/`; each seed then only picks
its threshold on validation and scores its test rows, like every other model.
"""),
    ("code", """
if RUN_LLM:
    dr.run_zero_shot(df, SEEDS, OUT_DEEP, DEVICE, LLM_MODEL)
else:
    print("skipped: set ON_GPU_SERVER = True (GPU server only) and install torch + transformers")
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
