# Fake job posting detection (EMSCAD), course project

Binary classification of job ads (fraudulent vs real) on the public EMSCAD dataset (Kaggle copy, CC0).
Classical branch: TF-IDF or BM25 + engineered features with logistic regression, a decision tree and an MLP.
Deep branch: frozen sentence embeddings with LR/MLP heads, and a fine-tuned DistilBERT.

- `data/fake_job_postings.csv`  17,880 rows, 18 columns, target `fraudulent` (866 positives)
- `code/modules/`               importable helpers: data, features, models, evaluation, explanation, saved results, fine-tuning, and the run logic (`classical_runs.py`, `deep_runs.py`)
- `code/notebooks/`  01 EDA + preprocessing, 02 classical ML (runs), 03 deep learning (runs), 04 analysis (reads saved results: tables, plots, findings); everything is run from the notebooks, in that order
- `code/outputs/`               one JSON + NPZ per (model, seed): `classical/` and `deep/`, read by notebook 04
- `code/splits/`                saved train/val/test indices per seed (identical on every machine)
- `code/tests/`                 plain-assert tests on synthetic data (run as scripts)
- `docs/`                       assignment spec and the literature review draft

Setup: `python -m venv .venv && .venv/bin/pip install -r code/requirements.txt`, then run tests with
`cd code && for t in tests/test_*.py; do ../.venv/bin/python $t; done` (adjust the venv path).

Notebook 03 needs a GPU and `torch`, `transformers`, `sentence-transformers`; its heavy cells run only when `ON_GPU_SERVER = True`.
`code/notebooks/build_notebooks.py` regenerates the four notebooks (it wipes their saved outputs).

Headline result (test PR-AUC, mean of 5 split seeds): BM25 logistic regression 0.847, TF-IDF MLP 0.846, fine-tuned DistilBERT 0.831,
TF-IDF logistic regression 0.821, frozen-embedding MLP head 0.632, decision tree 0.402, dummy 0.049. See notebook 04.
