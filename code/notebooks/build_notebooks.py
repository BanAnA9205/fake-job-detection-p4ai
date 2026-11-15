"""Build the three demo notebooks with nbformat. No data is read, nothing is fitted.

Run from the code/ folder (the script resolves its own folder, so cwd does not matter):
    python notebooks/build_notebooks.py

Writes 01_eda.ipynb, 02_classical_demo.ipynb and 03_results.ipynb (overwriting any
existing copies), then re-reads each file and ast.parse-checks every code cell
(magics and ! lines ignored). The notebooks themselves are executed by the orchestrator.
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
# Fake job postings: exploratory data analysis (notebook 01)

We get to know the EMSCAD dataset before touching any model: class balance, missing
data, duplicates, text lengths and simple spam-style flags. Every section feeds a
decision in the modelling notebooks: what to drop, how to split, and which metrics
to trust.

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

DATA = "../data/fake_job_postings.csv"
"""),
    ("code", """
SEED = 0
SPLIT_PATH = "splits/seed0.npz"

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
counts = df["fraudulent"].value_counts().sort_index()
print(counts.to_dict())
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
by_class = pd.DataFrame({
    "real": df.loc[df["fraudulent"] == 0].isna().mean(),
    "fraud": df.loc[df["fraudulent"] == 1].isna().mean(),
}).sort_values("real", ascending=False)
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
print("exact duplicate rows (ignoring job_id):", a["n_duplicate_rows"])
print("rows sharing their description with another row:", a["n_rows_shared_description"])
print("duplicate blocks with conflicting labels:", a["n_conflicting_duplicate_groups"])
# Same normalisation as modules/data_helper: lowercase, collapse whitespace.
desc = (df["description"].fillna("").astype(str).str.lower()
        .str.replace(r"\\s+", " ", regex=True).str.strip())
sizes = desc.value_counts()
print("largest description group:", int(sizes.max()), "ads")
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
lens = df["description"].fillna("").str.len()
lens = lens[lens > 0]
bins = np.logspace(np.log10(lens.min()), np.log10(lens.max()), 40)
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
## Cleaning and the split

Session 5: clean first, then split, then fit. `clean` drops exact duplicates and
hashes each normalised description into a `group` id; `make_splits` uses
`StratifiedGroupKFold` on that group so one description can never sit in two splits,
while keeping the class ratio balanced. We reuse the saved seed-0 split when it
exists, so this notebook and the runners see the same rows.
"""),
    ("code", """
dfc = dh.clean(df)
print("rows before cleaning:", len(df), "| after:", len(dfc))
if os.path.exists(SPLIT_PATH):
    splits = dh.load_splits(SPLIT_PATH, dfc)
else:
    splits = dh.make_splits(dfc, seed=SEED)
y = dfc["fraudulent"].to_numpy()
for name, idx in splits.items():
    print(f"{name:5s}: {len(idx):6d} rows, {int(y[idx].sum()):4d} fraud ({y[idx].mean():.1%})")
"""),
    ("code", """
groups_by_split = {name: set(dfc["group"].iloc[idx]) for name, idx in splits.items()}
for a_name, b_name in (("train", "val"), ("train", "test"), ("val", "test")):
    shared = groups_by_split[a_name] & groups_by_split[b_name]
    assert not shared, f"{len(shared)} description groups shared between {a_name} and {b_name}"
print("checked: no description group appears in two splits")
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
"""),
]

# ---------------------------------------------------------------------------
# Notebook 02: classical models walkthrough on the seed-0 split
# ---------------------------------------------------------------------------

NB2 = [
    ("md", """
# Classical models: a live walkthrough (notebook 02)

One seed-0 split, one pipeline from the course slides end to end: dummy baselines,
TF-IDF logistic regression with a threshold picked on validation, BM25, a decision
tree and an MLP, then explanations, error reading and a per-group check. Only
training and validation rows are used here; the test split stays untouched and is
read from saved files in notebook 03.

Run time: about 5 minutes.
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

DATA = "../data/fake_job_postings.csv"
"""),
    ("code", """
SEED = 0
SPLIT_PATH = "splits/seed0.npz"

dh.seed_everything(SEED)
sns.set_theme(style="whitegrid")
"""),
    ("md", """
## Setup: the seed-0 split

Load, clean, add features, and reuse the saved seed-0 split (Session 5: split
first, then fit on the training rows only).
"""),
    ("code", """
df = ft.add_engineered_features(dh.clean(dh.load_raw(DATA)))
if os.path.exists(SPLIT_PATH):
    splits = dh.load_splits(SPLIT_PATH, df)
else:
    splits = dh.make_splits(df, seed=SEED)
    dh.save_splits(SPLIT_PATH, splits, df)
assert not set(splits["train"]) & set(splits["val"]), "train/val row overlap"
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
fitted, thresholds, scores, results = {}, {}, {}, {}

def report(name, model):
    \"\"\"Score a fitted model; the threshold is picked on validation only.\"\"\"
    s_val = ev.score_of(model, val)
    thr = ev.pick_threshold(y_val, s_val)
    results[name] = {"train": ev.evaluate(y_train, ev.score_of(model, train), thr),
                     "test": ev.evaluate(y_val, s_val, thr)}
    fitted[name], thresholds[name], scores[name] = model, thr, s_val
    out = pd.DataFrame(results[name]).T.rename(index={"test": "val"})  # "test" slot holds validation
    out = out[["pr_auc", "roc_auc", "f1", "precision", "recall"]]
    out.insert(0, "model", name)
    return out
"""),
    ("md", """
## The dummy baseline (Sessions 6/7)

Every model has to beat the dumb references first: `dummy_majority` always predicts
the majority class; `dummy_stratified` guesses the class at the training rates.
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
m = results["logreg"]["test"]  # validation metrics (see the helper above)
cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
ev.plot_confusions({"logreg": cm}, ["real", "fraud"],
                   title="Validation confusion matrix (logreg)")
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
sw = ev.sweep_thresholds(y_val, scores["logreg"])
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(sw["threshold"], sw["precision"], label="precision")
ax.plot(sw["threshold"], sw["recall"], label="recall")
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
## All models side by side

Train versus validation PR-AUC per model; the gap column is the generalisation gap.
This is validation, not test: `run_classical.py` scores the test split once and
notebook 03 reads it.
"""),
    ("code", """
table = ev.compare_table(results).rename(columns=lambda c: c.replace("test_", "val_"))
table.sort_values("val_pr_auc", ascending=False)
"""),
    ("md", """
## Hyper-parameter tuning (switched off)

The full random search over all models and five seeds lives in `run_classical.py`:
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
terms = eh.top_linear_terms(fitted["logreg"], k=15).sort_values("coef")
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
val_sample = val.sample(min(1000, len(val)), random_state=SEED)
perm = eh.permutation_table(fitted["logreg"], val_sample,
                            val_sample["fraudulent"].to_numpy(),
                            n_repeats=1, seed=SEED)
top = perm.head(10).iloc[::-1]  # reversed so the biggest bar lands on top
fig, ax = plt.subplots(figsize=(7, 5))
ax.barh(top["col"], top["importance_mean"])
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
pred_val = (scores["logreg"] >= thresholds["logreg"]).astype(int)
ev.group_breakdown(y_val, pred_val, val["employment_type"])
"""),
    ("md", """
Reading: rows with few ads have noisy recall and precision; the count column says
how much to trust each row.
"""),
    ("md", """
## Where to next

`run_classical.py` repeats this pipeline for all models with proper tuning over
five split seeds and saves one JSON + NPZ pair per (model, seed) into
`outputs/classical/`. Notebook 03 reads those files and reports the honest test
numbers.
"""),
]

# ---------------------------------------------------------------------------
# Notebook 03: results over seeds, read from saved files only
# ---------------------------------------------------------------------------

NB3 = [
    ("md", """
# Results: every model, every seed (notebook 03)

This notebook only reads the files saved by `run_classical.py` (and `run_deep.py` on
the GPU box): test metrics aggregated over seeds, validation versus test, per-seed
spread, curves and confusion matrices rebuilt from the saved scores, and the errors
every model misses. Nothing is fitted here.

Run time: under 1 minute.
"""),
    ("code", """
import sys, os; os.chdir(os.path.join(os.getcwd(), "..")) if os.path.basename(os.getcwd()) == "notebooks" else None; sys.path.insert(0, "."); import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve

from modules import data_helper as dh
from modules import features as ft
from modules import eval_helper as ev
from modules import results_helper as rh

DATA = "../data/fake_job_postings.csv"
"""),
    ("code", """
SEED = 0
SPLIT_PATH = "splits/seed0.npz"
OUT_CLASSICAL = "outputs/classical"
OUT_DEEP = "outputs/deep"
DIRS = {"classical": OUT_CLASSICAL, "deep": OUT_DEEP}


class StopExecution(Exception):
    \"\"\"Stops the notebook quietly when there is nothing to report.\"\"\"

    def _render_traceback_(self):
        pass  # IPython hook: an empty traceback, so the notebook just stops


dh.seed_everything(SEED)
sns.set_theme(style="whitegrid")
"""),
    ("md", """
## Load the saved results

`results_helper.load_results` collects one row per (model, seed) from the JSON
files. Deep results are included when `outputs/deep/` exists and is non-empty. If
nothing is found we stop here with a message instead of crashing.
"""),
    ("code", """
dirs = [OUT_CLASSICAL]
if os.path.isdir(OUT_DEEP) and os.listdir(OUT_DEEP):
    dirs.append(OUT_DEEP)
res = rh.load_results(*dirs)
res = res[res["group"].isin(DIRS)]
if res.empty:
    print("No result files were found. Looked in:", dirs)
    print("Run run_classical.py first (and run_deep.py for the deep models), then re-run.")
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

order = res.groupby("model")["test_pr_auc"].mean().sort_values(ascending=False).index
pd.DataFrame({
    "seeds": res.groupby("model").size(),
    "test PR-AUC": fmt("test_pr_auc"),
    "test F1": fmt("test_f1"),
    "test precision": fmt("test_precision"),
    "test recall": fmt("test_recall"),
}).loc[order]
"""),
    ("md", """
Why the two dummy rows show recall 1.0: a constant score has a single possible threshold,
and picking it by F1 flags every ad as fraud. Their precision therefore equals the
share of fraud in the data, which is the floor every real model has to beat.
"""),
    ("md", """
## Validation vs test PR-AUC

Validation is what models, hyper-parameters and thresholds were chosen on; test is
the honest number. The difference is the price of that selection.
"""),
    ("code", """
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
seed0 = res[res["seed"] == 0]
if seed0.empty:
    print("No seed-0 runs found; the remaining sections need seed 0.")
    raise StopExecution
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
curves = {name: precision_recall_curve(y_test, s["score_test"])[:2]
          for name, (s, thr) in seed_runs.items()}
pos_rate = float(y_test.mean())
ev.plot_pr_curves(curves, pos_rate=pos_rate,
                  title="Test precision-recall curves (seed 0)")
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
cms = {}
for name, (s, thr) in seed_runs.items():
    pred = (s["score_test"] >= thr).astype(int)
    cms[name] = confusion_matrix(y_test, pred)
ev.plot_confusions(cms, ["real", "fraud"],
                   title="Test confusion matrices, validation-chosen threshold (seed 0)")
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
fraud_idx = np.where(y_test == 1)[0]
# The dummies are left out: at their threshold they flag every ad, so they "catch" every fraud.
caught = {name: frozenset(fraud_idx[s["score_test"][fraud_idx] >= thr])
          for name, (s, thr) in seed_runs.items() if not name.startswith("dummy")}
names = list(caught)
overlap = pd.DataFrame(0, index=names, columns=names)
for a in names:
    for b in names:
        overlap.loc[a, b] = len(caught[a] - caught[b])
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
    df = ft.add_engineered_features(dh.clean(dh.load_raw(DATA)))
    test = df.iloc[dh.load_splits(SPLIT_PATH, df)["test"]]
    caught_any = np.zeros(len(test), dtype=bool)
    for name, (s, thr) in seed_runs.items():
        if not name.startswith("dummy"):      # dummies flag every ad, so they would hide all misses
            caught_any |= (s["score_test"] >= thr)
    missed = test[(y_test == 1) & ~caught_any]
    print(len(missed), "test frauds that no model caught:")
    for _, r in missed.iterrows():
        print("-", str(r["title"])[:80], "|", str(r["description"])[:150], "\\n")
"""),
    ("md", """
## Findings so far (classical models only; update after the deep results are in)

Numbers are the means over the 5 split seeds in the first table, test split.

- Every text-based model beats the dummy floor by a wide margin (test PR-AUC 0.82 to
  0.85 against 0.049).
- Logistic regression with TF-IDF, with BM25, and the MLP are within seed noise of each
  other: the spread across seeds (about 0.07) is larger than the gaps between them.
- A single decision tree on sparse TF-IDF is far weaker (test PR-AUC about 0.40).
- Every model fits the training rows almost perfectly, so there is a large
  generalisation gap (train minus test PR-AUC about 0.15 to 0.17 for the linear models
  and the MLP).
- Caveat: the dataset marks real ads from some countries (for example Greece) as 0%
  fraud, so part of the signal may come from how the data was collected.
"""),
]


def main():
    """Write the three notebooks, then re-read and parse-check them."""
    for cells, name in ((NB1, "01_eda.ipynb"),
                        (NB2, "02_classical_demo.ipynb"),
                        (NB3, "03_results.ipynb")):
        path = build(cells, name)
        total, n_code = check(path)
        print(f"{path}: {total} cells ({n_code} code, {total - n_code} markdown)")


if __name__ == "__main__":
    main()
