# Fake job posting detection (EMSCAD), course project

Binary classification of job ads (fraudulent vs real) on the public EMSCAD dataset (Kaggle copy, CC0).
Classical branch: TF-IDF or BM25 + engineered features with logistic regression, an MLP and Complement Naive
Bayes, plus an unsupervised one-class SVM on the text (anomaly detection view).
Deep branch: frozen sentence embeddings with a logistic-regression head, a fine-tuned DistilBERT (explained with
Integrated Gradients), and a zero-shot instruction-tuned LLM (Qwen2.5-1.5B-Instruct).
Combination: a stack of DistilBERT and TF-IDF logistic regression (logistic regression on their saved scores).

- `data/fake_job_postings.csv`  17,880 rows, 18 columns, target `fraudulent` (866 positives)
- `data/processed/emscad_clean.csv.gz`  the cleaned data with split groups (17,596 rows), written by notebook 01 and read by every later step
- `code/modules/`               importable helpers: data, quality checks (`qc.py`), features, models, evaluation, explanation, saved results, fine-tuning, stacking, and the run logic (`classical_runs.py`, `deep_runs.py`)
- `code/notebooks/`  01 EDA + preprocessing, 02 classical ML (runs), 03 deep learning (runs), 04 analysis (reads saved results: tables, plots, findings); everything is run from the notebooks, in that order
- `code/outputs/`               one JSON + NPZ per (model, seed): `classical/`, `deep/` and `stack/`, read by notebook 04
- `code/splits/`                saved train/val/test indices per seed; related ads (same description, company or near-copy) never cross splits
- `code/tests/`                 plain-assert tests on synthetic data (run as scripts; `test_finetune_helper.py` uses torch, so run it on the GPU server only)
- `docs/`                       assignment spec and the literature review draft

Setup: `python -m venv .venv && .venv/bin/pip install -r code/requirements.txt`, then run tests with
`cd code && for t in tests/test_*.py; do ../.venv/bin/python $t; done` (adjust the venv path).

Notebook 03 needs a GPU and `torch`, `transformers`, `sentence-transformers`, `captum`; its heavy cells run only when `ON_GPU_SERVER = True`.
`code/notebooks/build_notebooks.py` regenerates the four notebooks (it wipes their saved outputs).

Results: none saved yet. They are produced by running notebooks 02 and 03 (then 04) on the server after the data quality fixes.
