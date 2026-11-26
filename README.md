# Fake job posting detection (EMSCAD), course project

Binary classification of job ads (fraudulent vs real) on the public EMSCAD dataset (Kaggle copy, CC0):
17,880 ads, of which 866 (4.8%) are fraudulent. The LaTeX report is in `materials/report/` (see
[Report](#report) to build the PDF).

The study compares nine model variants on five grouped train/validation/test partitions (45 runs):

- **Classical:** majority dummy; TF-IDF logistic regression with and without EDA-motivated structured
  features (the classical feature ablation); a decision tree and a one-class SVM on the same
  text-plus-structured features.
- **Deep learning:** DistilBERT fine-tuned in two budgets (top two encoder layers plus head, or the
  full model), each as a text-only and a structured late-fusion variant.

Every split keeps whole related-ad groups (shared descriptions, company profiles, emails or phone
numbers) on one side, so near-identical ads cannot leak between training and test. All preprocessing
is fitted on training rows, models and epochs are selected on validation AP, and thresholds on
validation F1; test labels are used only for the final report.

| Requirement | Where it is met |
| --- | --- |
| Problem definition | Report §1; dataset audit and cleaning in notebook 01 and report §3 |
| Literature review | Report §2 (EMSCAD, classical and transformer detectors, imbalance and evaluation) |
| Classical ML: preprocessing, feature engineering, tuning | Notebooks 01–02, `code/modules/`; report §3–4 |
| Deep learning: preprocessing, fine-tuning, explanation | Notebook 03; report §4 and §6 |
| Evaluation and comparison | Notebook 04; report §5 |

## Repository layout

- `data/fake_job_postings.csv`  raw data (18 columns, target `fraudulent`)
- `data/processed/emscad_clean.csv.gz`  cleaned data with related-ad groups: 17,597 rows (856 fraud)
- `code/notebooks/01_eda_preprocessing.ipynb`  quality control, cleaning, EDA and the five saved splits
- `code/notebooks/02_classical_ml.ipynb`  classical models on all five splits; seed-0 explanations
- `code/notebooks/03_deep_learning.ipynb`  DistilBERT fine-tuning, checkpoint checks and neural explanations
- `code/notebooks/04_analysis.ipynb`  verified 45-run comparison, paired ablations and diagnostics
- `code/modules/`  shared data/QC, features, models, evaluation, result I/O and report style
- `code/tests/`  unit tests for the shared modules
- `code/splits/`  the five saved partitions (row indices plus a data fingerprint)
- `code/outputs/`  one JSON + NPZ per model and seed, checkpoints, explanation files and the
  second-optimization-seed runs in `distilbert_top2_reseed/` and `finetuned_reseed/` (not tracked)
- `code/features/`  pinned Hugging Face base-model cache (not tracked)
- `materials/report/`  LaTeX report; the notebooks write its figures (`img/`) and generated tables

## Assignment specification

### Requirements

- **Problem definition:** Identify a specific problem and an appropriate dataset.
- **Literature review:** Review and summarize existing methods and approaches related to the selected problem.
- **Algorithm implementation:** Apply classical machine-learning algorithms, with a focus on preprocessing,
  feature engineering and hyperparameter tuning, and/or deep-learning methods, including preprocessing,
  training/fine-tuning techniques and model explanation, to build a complete pipeline for the selected problem.
- **Evaluation:** Compare and analyze the implemented approaches based on experimental results.

### Suggested topics

- Demand forecasting (energy demand, sales forecasting, etc.)
- Financial analysis
- Credit risk scoring
- Anomaly detection

## Setup and verification

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r code/requirements.txt pytest ruff
.venv/bin/python -m pytest code/tests
.venv/bin/ruff check code/modules code/tests
```

Run the notebooks from `code/notebooks/` in order; they change to `code/` internally. Each one saves
its report figures and tables into `materials/report/`. Notebooks 02 and 03 resume finished
model/seed pairs, and saved results carry the data fingerprint and exact split rows, so stale results
are rejected after any data or split change.

## DistilBERT fine-tuning (GPU)

Tested with `torch==2.8.0` and `transformers==4.57.6`:

```sh
uv pip install --python .venv/bin/python 'torch==2.8.0' 'transformers>=4.46,<5' safetensors
export HF_HOME="$PWD/code/features/hf_cache" HF_HUB_DISABLE_TELEMETRY=1
```

In notebook 03, set `ON_GPU_SERVER = True` and choose a free `DEVICE`. `RUN_TRAINING = True` trains the
20 runs (and runs the smoke tests); `VALIDATE_ON_GPU = True` reloads the seed-0 checkpoints and checks
that they reproduce the saved scores. With `ON_GPU_SERVER = False` the notebook only verifies saved
results. The committed notebook was last run with `ON_GPU_SERVER = True`, `RUN_TRAINING = False` and
`VALIDATE_ON_GPU = True`
(after training), so it shows the verification, the checkpoint reload check and the explanations;
the per-epoch training logs are kept in each run's JSON `history`. Run notebook 04 after notebooks 01–03.

## Report

```sh
cd materials/report
tectonic -X compile main.tex
```

The report build was tested with Tectonic 0.17.0.
