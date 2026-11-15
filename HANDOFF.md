# HANDOFF: state of the project for a new Claude session

Written at the end of a long session on the owner's laptop, so a session on the NRC GPU server can continue.
Read this, then `docs/spec.md`, `code/CODE_SPEC.md` and the code. Do not trust numbers below beyond what they say they are.

## Task and constraints (from the owner)
- University course P4AI-DS (programming for AI and data science). Deliverable: problem + dataset, literature review,
  implementation (classical ML and/or deep learning, with preprocessing, feature engineering, tuning, model explanation),
  and an experimental comparison. The requirements are in `docs/spec.md`.
- The course is about CODING, not theory. Follow the slides' conventions (Sessions 4-7): split first, everything fitted
  inside one sklearn `Pipeline`, drop duplicates before splitting, a `DummyClassifier` baseline first, report precision/
  recall/F1/PR-AUC/confusion matrix and the train-test gap, tune on validation only and touch the test set once,
  seed everything and report mean and std over seeds, cache expensive steps (embeddings), freeze the pretrained
  backbone before fine-tuning, read ten real errors.
- The owner must PRESENT this code. Keep it readable: short functions, plain names, WHY-comments, well-known tools.
  The owner is an experienced competitive programmer and AI researcher; do not explain basics.
- Owner style (from their global instructions): no flattery or filler, say what was run and measured, label reasoning as
  reasoning, never fabricate, verify by running, call things untested if they were not run, ask before consequential or
  hard-to-reverse choices, change only what the request needs.
- The laptop must not run heavy compute (a power surge once fried it). That rule was for the laptop. On NRC, GPU jobs
  are intended, but NRC is a SHARED server: check `nvidia-smi` free memory right before each launch, pick the emptiest
  GPU, set `CUDA_VISIBLE_DEVICES`, limit CPU threads (`OMP_NUM_THREADS`), use `nice`, run inside `tmux` with logs, never
  install system-wide (use a venv), and stay modest with disk (root disk was 97% full).

## Data
- `data/fake_job_postings.csv`: HuggingFace mirror of the Kaggle EMSCAD copy (cc0-1.0). 17,880 rows, 866 fraud (4.84%).
- Duplicates: 281 rows are duplicates ignoring `job_id` (no label conflicts); after `clean()` there are 17,599 rows.
  4,552 rows share a normalised description with another row; the largest shared description has 377 rows, all real.
- Missingness is high and informative: `company_profile` 18% missing (16% of real ads, 69% of fraud), `salary_range` 84%.
- The dataset replaced URLs, emails and phone numbers with placeholders (`#URL_xxxx#`, `#EMAIL_xxxx#`, `#PHONE_xxxx#`).

## Code (in `code/`)
- `modules/data_helper.py`: `load_raw`, `audit`, `clean` (drops duplicates, adds `group` = hash of normalised description),
  `make_splits` (StratifiedGroupKFold, test first then val, 60/20/20, no description group in two splits),
  `save_splits`/`load_splits` (with a data fingerprint, so every machine uses the identical rows), metadata, seeding.
- `modules/features.py`: `add_engineered_features`, `make_preprocessor("tfidf"|"bm25")`, `Bm25Transformer`.
  Binary flags go through `VarianceThreshold(0.001)`; regexes match the `#URL_`, `#EMAIL_`, `#PHONE_` placeholders.
- `modules/models.py`: baselines, `get_models` = logreg, logreg_bm25, tree, mlp (sklearn `MLPClassifier`), `PARAM_GRIDS`, `tune`.
  SVM and HistGradientBoosting were removed on the owner's request (lineup: simple method, classic ML, MLP, TF-IDF vs BM25, BERT).
- `modules/eval_helper.py`, `explain_helper.py`, `results_helper.py`: metrics, threshold on validation, tables, plots,
  linear coefficients, tree importances and rules, permutation importance, error reading, one shared results format.
- `modules/deep_helper.py` (`embed_texts`, cached), `modules/finetune_helper.py` (plain PyTorch fine-tuning loop).
- `run_classical.py` (done, results committed) and `run_deep.py` (embeddings, sklearn heads on embeddings, BERT; NOT yet run on
  real data). Both write `outputs/<group>/<model>_seed<N>.json/.npz`; the notebook `03_results` reads them.
- `notebooks/`: `01_eda`, `02_classical_demo` (validation only), `03_results` (reads saved results), built by `build_notebooks.py`
  and executed with saved outputs. Do not rebuild casually: rebuilding wipes the saved outputs and the findings cells.
- `tests/`: plain-assert scripts, all pass on the laptop (Python 3.14, scikit-learn 1.9.1, CPU torch for the tiny BERT smoke test).
- `SERVER.md`: step-by-step commands for the GPU server. `CODE_SPEC.md`, `DEEP_SPEC.md`, `NOTEBOOK_SPEC.md`: the contracts
  the modules were written against (the agents' safety rules in them were for the laptop).

## Measured (classical, 5 split seeds 0-4, tuned on train with 3-fold grouped CV, threshold chosen on validation, TEST set)
| model | test PR-AUC | test F1 | test recall | test precision |
|---|---|---|---|---|
| logreg + BM25 | 0.847 +- 0.073 | 0.804 +- 0.055 | 0.760 | 0.856 |
| MLP (TF-IDF) | 0.846 +- 0.077 | 0.787 +- 0.081 | 0.726 | 0.863 |
| logreg + TF-IDF | 0.821 +- 0.063 | 0.772 +- 0.042 | 0.726 | 0.827 |
| decision tree | 0.402 +- 0.066 | 0.501 +- 0.055 | 0.634 | 0.434 |
| dummy | 0.049 | 0.093 | | |
(mean +- std over seeds.) The spread between seeds (about 0.07) is larger than the gaps between logreg, BM25 and MLP, so those three are
statistically indistinguishable here. Train PR-AUC is about 1.0 for logreg and 0.99 for the MLP: large generalisation gap.
Honesty note: during the pilot of the runner the seed-0 test numbers for four models were printed once; no choice was made from them,
and the runner now prints validation only.
Dataset observation (unverified hypothesis): fraud rate differs sharply by country (Greece 940 ads and 0% fraud, Australia 19%,
US 6.9%); the country and "missing company profile" signals may reflect how the dataset was collected rather than fraud itself.

## Open issues / next steps
1. Run `SERVER.md` on NRC: embeddings, sklearn heads, then DistilBERT (one seed first to measure the real time). Push the small
   result files back; then re-run `03_results.ipynb` locally (it picks up `outputs/deep/`), and add a findings cell.
2. Not yet verified: downloading the real models, bf16 on CUDA, memory/time of the BERT run (all estimates).
3. Ordinal level orders in `features.py` are judgment calls ("Not Applicable" below "Internship", etc.).
4. Two false negatives looked like the same posting with different capitalisation (possible near-duplicates that exact dedup
   misses). Unconfirmed.
5. Writing: literature review (`docs/literature_review.md`, a draft, tags [V]/[M]), report, slides. The explanation story is:
   logreg coefficients, tree rules/importances, permutation importance, reading real errors; LIME on raw text for a few BERT examples
   is planned but not implemented.

## Other material not in this repo
- Slides (course PDFs) and syllabus, the previous course's project (`older_work`, an image-classification pipeline whose
  structure was reused), and the survey/paper PDFs under `surveys/` (gitignored: SBERT, SHAP, LIME, SMOTE, Fraud-BERT).
- `docs/literature_review.md` is a draft. Items tagged [V] were confirmed from abstracts/snippets; [M] are from memory.
  The Springer systematic review of recruitment-scam detection and the surveys the owner mentioned were not obtained.
