# Fake job posting detection (EMSCAD), course project

Binary classification of job ads (fraudulent vs real) on the public EMSCAD dataset (Kaggle copy, CC0).
Classical branch: TF-IDF or BM25 + engineered features with logistic regression, a decision tree and an MLP.
Deep branch: frozen sentence embeddings with LR/MLP heads, and a fine-tuned DistilBERT.

- `data/fake_job_postings.csv`  17,880 rows, 18 columns, target `fraudulent` (866 positives)
- `code/modules/`               data, features, models, evaluation, explanation, deep helpers
- `code/run_classical.py`       classical experiments, 5 split seeds (done; results in `code/outputs/classical/`)
- `code/run_deep.py`            embeddings + heads + BERT fine-tuning (for the GPU server; see `SERVER.md`)
- `code/splits/`                saved train/val/test indices per seed (identical on every machine)
- `code/notebooks/`             01 EDA, 02 classical demo, 03 results (executed, outputs saved)
- `code/tests/`                 plain-assert tests on synthetic data (run as scripts)
- `code/*_SPEC.md`              the contracts the modules were written against
- `SERVER.md`                   commands to run the deep part on the GPU server
- `docs/`                       assignment spec and the literature review draft
- `HANDOFF.md`                  full project state for a new session (read this first)

Setup: `python -m venv .venv && .venv/bin/pip install -r code/requirements.txt`, then run tests with
`cd code && for t in tests/test_*.py; do ../.venv/bin/python $t; done` (adjust the venv path).
