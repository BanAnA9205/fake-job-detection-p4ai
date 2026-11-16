# Spec for the demo notebooks. Read CODE_SPEC.md (Presentability + safety rules) and DEEP_SPEC.md (results contract) first.

You write THREE notebooks into `notebooks/` by running a builder script (`notebooks/build_notebooks.py`, using `nbformat`) that you
also write. You must NOT execute the notebooks and must NOT run any model fitting; the orchestrator executes them. You may
validate that every code cell parses (`ast.parse`, ignoring `%` magics and `!` lines) and that the notebooks load with
`nbformat.read`. Wrap any command as `timeout 120 taskset -c 0-1 nice -n 19 <cmd>`. Do not install anything.

## Audience and style
The owner will present these notebooks live to classmates and a lecturer (course: programming for AI and data science; it is a
CODING course, slides conventions in CODE_SPEC.md). So:
- Each notebook starts with a markdown cell: title, one-paragraph purpose, and "Run time: about N minutes".
- Many short sections: a markdown cell (1-3 sentences saying WHAT we do and WHY, naming the course session it comes from when
  relevant, e.g. "Session 5: split first, then fit") followed by a small code cell. No code cell longer than about 25 lines.
- Plain pandas / matplotlib / seaborn. Use the project modules (`modules.*`) instead of re-implementing anything.
- Every printed table/plot is followed by a one-line markdown "Reading:" cell ONLY IF it states something that will be true
  whatever the numbers are (for example how to read the axes). Do NOT write conclusions with specific numbers into markdown,
  the numbers will change; the orchestrator adds a findings cell after execution.
- First code cell of every notebook:
  `import sys, os; os.chdir(os.path.join(os.getcwd(), "..")) if os.path.basename(os.getcwd()) == "notebooks" else None;
  sys.path.insert(0, "."); import warnings` plus the imports; and constants `DATA = "../data/fake_job_postings.csv"`.
  (The notebooks live in `code/notebooks/`, modules are in `code/modules/`, so cwd must become `code/`.)
- Seeds and paths in one constants cell at the top. Matplotlib: `sns.set_theme(style="whitegrid")`, titles and axis labels on
  every plot, figures saved nowhere (inline only), `plt.show()` after each.
- Never touch the TEST split in notebooks 01 and 02 (validation only). Notebook 03 is the only one that reads test results, and
  it reads them from saved files, it never fits anything.

## Notebook 1: `01_eda.ipynb` (Run time about 1 minute)
Sections: load + `audit` printed as a small table; class balance (counts + bar); missing share per column for real vs fraud
(side-by-side bars, sorted); duplicates (exact duplicates ignoring job_id, rows sharing a description, size of the largest
description group, histogram of group sizes on a log y axis); text length of `description` by class (log-scaled x, overlaid
histograms); engineered flags: share of ads firing each flag by class (table from `add_engineered_features`) plus a note that the
dataset replaces URLs/emails/phones with `#URL_..#` placeholders; the most frequent `country` and `employment_type` values by
fraud rate (only values with at least 100 ads); cleaning + splits: run `clean`, `make_splits(seed=0)` (or load
`splits/seed0.npz` with `load_splits`), print sizes and class balance per split and assert no description group is shared.
Final markdown cell: "What the EDA tells us to do" as a bullet list of DECISIONS (drop duplicates before splitting, split by
description group, class imbalance so use PR-AUC and threshold on validation, missingness is informative so keep indicator flags),
without specific numbers.

## Notebook 2: `02_classical_demo.ipynb` (Run time about 5 minutes)
Uses the seed-0 saved split (`splits/seed0.npz`), VALIDATION only. Sections: setup and split sizes; the dummy baseline (why:
Session 6/7); logistic regression on TF-IDF + engineered features (fit, validation PR-AUC/ROC-AUC, `pick_threshold` on validation,
`evaluate`, confusion matrix with `plot_confusions`); the threshold trade-off (`sweep_thresholds` plotted as precision and recall
vs threshold); BM25 vs TF-IDF (same pipeline, `get_models()["logreg_bm25"]`; a short markdown explaining BM25 in 3 sentences:
saturating term frequency, length normalisation, same vocabulary); decision tree (fit, metrics, `tree_rules(max_depth=3)`,
`top_tree_features`); MLP (fit, metrics, `history`-free: just metrics); a comparison table of all models on validation using
`compare_table`-style rows (train vs validation PR-AUC to show the generalisation gap); hyper-parameter tuning demo: a flag
`RUN_TUNING = False` at the top of that section; when True it calls `tune()` on the logreg with `n_iter=3, cv=3` and prints the best
params and CV score (leave the flag False in the saved notebook, say in markdown that the full tuning is in `run_classical.py`);
explanation: `top_linear_terms` for the logreg as a horizontal bar chart (top 15 each side), `permutation_table` on a 1000-row
validation sample with `n_repeats=1` and the top 10 columns as a bar chart; error reading: `show_errors(k=5)` for the logreg
(Session 6: read the errors); per-group breakdown by `employment_type` with `group_breakdown`.

## Notebook 3: `03_results.ipynb` (Run time under 1 minute; reads saved files only)
Reads `outputs/classical/` and, if it exists and is non-empty, `outputs/deep/` via `modules.results_helper.load_results`.
If no results are found it prints a clear message and stops (do not crash). Sections: which (model, seed) runs were found
(pivot table of counts); the main table: TEST PR-AUC, F1, precision, recall per model as mean +- std over seeds (format as
"0.812 +- 0.015"), sorted by mean PR-AUC, with the dummy rows included; validation vs test PR-AUC per model (grouped bars, to
show the tuning/selection effect); generalisation gap = train PR-AUC minus test PR-AUC per model; per-seed strip/point plot of
test PR-AUC (does a model win by more than its seed spread? say that rule in markdown: Session 7 "a model that wins by less than
its own standard deviation has not won"); precision-recall curves on the test split for seed 0 from the saved scores
(`sklearn.metrics.precision_recall_curve`, `plot_pr_curves`, with the base-rate line); confusion matrices at the
validation-chosen threshold for seed 0 (`plot_confusions`, built from saved scores and `thr`); fit time per model (table from
`seconds`); error overlap on the seed-0 test split: for each pair of models, how many of the test frauds each catches at its own
threshold (a small matrix: rows = model A catches, columns = model B misses), plus the number caught by at least one model but
not by all; and a last cell that prints the few test ads (title + 150 chars) that NO model catches, to read them.
Everything model-specific must be discovered from the files (loop over the models found), so it works with or without the deep
results and with any subset of seeds.

## Builder script
`notebooks/build_notebooks.py` creates the three `.ipynb` files with `nbformat.v4` (cells defined as Python lists of
(type, source) tuples), writes them, and prints the list of files and cell counts. Re-running it must overwrite them cleanly.
Run only that script (it does no heavy work). Final report: files written, cell counts per notebook, and what you did NOT verify
(execution results).

> **Superseded.** The notebooks were restructured: `01_eda_preprocessing`, `02_classical_ml`, `03_deep_learning`.
> Runs now happen inside the notebooks via `modules/classical_runs.py` and `modules/deep_runs.py` (the `run_*.py`
> scripts are gone). The source of truth is `notebooks/build_notebooks.py`.
