# Fake job posting detection (EMSCAD), course project

Binary classification of job ads (fraudulent vs real) on the public EMSCAD dataset (Kaggle copy, CC0).

The classical branch compares TF-IDF logistic regression with and without EDA-motivated structured
features, a decision tree, and a one-class SVM against a majority dummy. Notebook 03 compares two
DistilBERT fine-tuning budgets: top-two encoder layers plus a two-logit classification head, versus
full encoder/head fine-tuning, each with text-only and structured late-fusion arms. The final study
contains nine variants and 45 runs. PCA is used only to make classical TF-IDF one-class SVM tractable;
the DistilBERT pipelines omit it, and no precomputed embedding artifacts are generated or required.

- `data/fake_job_postings.csv`  raw data: 17,880 rows, 18 columns, target `fraudulent` (866 positives)
- `data/processed/emscad_clean.csv.gz`  cleaned data with related-ad groups: 17,597 rows (856 positives)
- `code/notebooks/01_eda_preprocessing.ipynb`  QC, cleaning, EDA, and five saved grouped splits
- `code/notebooks/02_classical_ml.ipynb`  classical models on all five splits; seed 0 explanations
- `code/notebooks/03_deep_learning.ipynb`  two DistilBERT fine-tuning budgets with text/fused heads, checkpoints, and adaptation ablation on five splits
- `code/notebooks/04_analysis.ipynb`  final comparison, ablation, diagnostics, and conclusions; run only after 01–03
- `code/modules/`  shared data/QC, feature construction, model definitions, evaluation, explanation, and result I/O
- `code/tests/`  unit tests for the shared modules, run with pytest
- `code/outputs/`  one JSON + NPZ per model/seed in `classical/`, `distilbert_top2/`, and `finetuned/`, for 45 runs total (ignored heavy artifacts; retain/restore them)
- `code/features/`  the pinned Hugging Face base-model cache (ignored heavy input; retain/restore it)
- `docs/literature_review.md`  literature-review draft
- `materials/report/`  current LaTeX report, generated figures, and generated result/explanation tables
- `code/make_report_figures.py`  report-asset generator for the complete 45-run result matrix

Split groups prevent exact descriptions, nonempty company profiles, and original email/phone identities
from crossing train/validation/test boundaries. Missing descriptions share an empty-description key,
whereas empty company profiles are not linked. URLs are not grouping keys. Saved indices include a
content-and-label fingerprint to reject stale splits. Result JSON/NPZ pairs also store that fingerprint and
exact validation/test positional indices, so lineage-aware resume and analysis reject old scores when the
cleaned data or split rows change.

## Setup and verification

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r code/requirements.txt pytest ruff
.venv/bin/python -m pytest code/tests
.venv/bin/ruff check code/modules code/tests code/make_report_figures.py
```

Run the notebooks from `code/notebooks/` in order. They change to `code/` internally and save outputs
there. Notebooks 02 and 03 resume completed model/seed pairs; use a new output directory when changing
a pipeline rather than mixing incompatible runs. Notebook 03 uses separate directories for selective
and full fine-tuning and verifies both sets of checkpoints.

## Optional DistilBERT fine-tuning

Run selective/full fine-tuning on the GPU server, tested here with `torch==2.8.0` and
`transformers==4.57.6`:

```sh
uv pip install --python .venv/bin/python 'torch==2.8.0' 'transformers>=4.46,<5' safetensors
export HF_HOME="$PWD/code/features/hf_cache" HF_HUB_DISABLE_TELEMETRY=1
```

In notebook 03, set both `ON_GPU_SERVER = True` and `RUN_TRAINING = True` for training and smoke tests,
and select an accessible CUDA device. Set `VALIDATE_ON_GPU = True` separately only for inference-time
checkpoint reload checks. These flags default to `False`, which also forces Hugging Face offline mode.
Metadata/score/lineage validation works without GPU training. Run notebook 04 only after notebooks
01–03 provide the complete 45-run result matrix. Current results are exploratory; use all five saved
splits and preserve the paired adaptation/fusion comparisons.

## Report assets

After notebooks 02–04 have produced the complete 45-run result matrix, regenerate the report's 11
vector figures and seven generated LaTeX tables from the repository root:

```sh
.venv/bin/python code/make_report_figures.py
```

The generator also refits only the selected seed-0 classical LR/tree models for explanation tables.
It does not rerun the five-split experiments.

Compile `materials/report/main.tex` with a LaTeX/BibTeX toolchain. For example, with Tectonic:

```sh
cd materials/report
tectonic -X compile main.tex
```

The report build was tested with Tectonic 0.17.0.

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

### Group strategy and course conventions

- Reuse the pipeline in older work at `/home/banana9205/Desktop/Main/Uni/P4DSAI/older_work/` to save effort,
  but ensure it still complies and is well suited to the requirements above.
- Since this course is more about coding than theory, see `syllabus.pdf` and the course slides for conventions.
