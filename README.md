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

## Reproducibility and historical artifacts

The committed `code/outputs/` JSON/NPZ files are historical results from the original
pipeline. They do **not** have run signatures or archive checksums and are available
for reading only; they are not accepted for resume. The legacy dummy baseline
thresholds are wrong (they flagged every row). Set `RUN_CLASSICAL=True` in notebook
02 to regenerate corrected, integrity-checked experiments on the verified splits.
This requires rerunning the classical models and may take significant CPU time.

Embedding caches now use a content-addressed filename based on the ordered texts,
job IDs, model name, revision (if provided) and truncation length. Older
`<model>_emb.npy` / `<model>_ids.npy` caches cannot be trusted and must be
recreated from notebook 03.

GPU execution is opt-in using `P4AI_ENABLE_GPU=1` and optionally
`P4AI_DEVICE=cuda:0` (or another valid device), not a hardcoded server GPU.
Rebuilding notebooks with `python code/notebooks/build_notebooks.py` overwrites
their saved outputs.

New split files hash every ordered field and target label. The original committed
`splits/seed*.npz` files use legacy ID-only fingerprints: loading them emits a
warning and re-checks group disjointness and class balance, but cannot establish
full original-data integrity. Strict callers can set `allow_legacy=False`.

For frozen MiniLM + logistic regression, C is now tuned by three-fold cross-validation
*inside training data* (group-aware for real EMSCAD runs); validation is reserved
for threshold choice. Classical and embedding MLPs no longer use the ungrouped
internal validation holdout in sklearn's `early_stopping=True`.
The committed deep-learning results were produced before this correction; rerun
them before making any updated performance claim.
