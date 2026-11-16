# Fake job posting detection (EMSCAD), course project

Binary classification of job ads (fraudulent vs real) on the public EMSCAD dataset (Kaggle copy, CC0).
Classical branch: TF-IDF or BM25 + engineered features with logistic regression, a decision tree and an MLP.
Deep branch: frozen sentence embeddings with LR/MLP heads, and a fine-tuned DistilBERT.

- `data/fake_job_postings.csv`  17,880 rows, 18 columns, target `fraudulent` (866 positives)
- `code/modules/`               data, features, models, evaluation, explanation, deep helpers
- `code/notebooks/`  01 EDA + preprocessing, 02 classical ML (runs), 03 deep learning (runs), 04 analysis (reads saved results: tables, plots, findings); everything is run from the notebooks
- `code/modules/classical_runs.py`, `deep_runs.py`  importable run logic called by notebooks 02 and 03
- `code/splits/`                saved train/val/test indices per seed (identical on every machine)
- `code/tests/`                 plain-assert tests on synthetic data (run as scripts)
- `docs/`                       assignment spec and the literature review draft

Setup: `python -m venv .venv && .venv/bin/pip install -r code/requirements.txt`, then run tests with
`cd code && for t in tests/test_*.py; do ../.venv/bin/python $t; done` (adjust the venv path).
