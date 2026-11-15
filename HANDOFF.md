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
  `make_splits` (StratifiedGroupKFold, test first then val, 60/20/20, no description group in two splits), metadata, seeding.
- `modules/features.py`: `add_engineered_features`, `make_preprocessor("tfidf"|"meta")`. Binary flags go through
  `VarianceThreshold(0.001)`; regexes match the placeholders.
- `modules/models.py`: baselines, `get_models` (logreg, linear_svm, hist_gb), `PARAM_GRIDS`, `tune`.
- `modules/eval_helper.py`, `modules/explain_helper.py`: metrics, threshold choice on validation, tables, plots,
  linear coefficients, permutation importance, error reading.
- `modules/deep_helper.py`: `embed_texts` (cached sentence-transformers embeddings), a small MLP head, `train_mlp`.
  NEVER RUN so far (torch was not installed on the laptop); only compiled and statically tested. Untested = assume bugs.
- `tests/`: 6 plain-assert scripts on synthetic data, all passing on the laptop (Python 3.14, scikit-learn 1.9.1).
  Run them again after installing dependencies on NRC.

## Measured so far (laptop, real data, split seed 0, VALIDATION set only, threshold picked on validation)
| model | val PR-AUC | val F1 | val precision | val recall |
|---|---|---|---|---|
| linear SVM | 0.893 | 0.855 | 0.846 | 0.865 |
| logistic regression | 0.825 | 0.773 | 0.852 | 0.708 |
| hist gradient boosting | 0.814 | 0.744 | 0.799 | 0.696 |
| dummy baselines | 0.049 | | | |
Train PR-AUC is 0.98-1.00 for all three, so there is a large generalisation gap. One seed, 171 validation positives:
differences of a few points are within noise. The TEST set has not been touched. A tiny tuning pilot (logreg, 2 candidates,
3 folds) picked C=10, ngram (1,2), 30000 features, CV PR-AUC 0.813.
For context only: the Fraud-BERT paper (Taneja et al. 2025, Discover Computing 28:9) reports fine-tuned BERT P 0.94 R 0.92
F1 0.93 vs SVM F1 0.89 on a single 80:20 split without dedup or seeds; its numbers are NOT comparable with ours.

## Open issues
1. `LinearSVC` prints "Liblinear failed to converge" at `max_iter=5000`. Tested on seed 0: `max_iter=50000` converges in
   14 s with the same scores (PR-AUC 0.893); scaling the ordinal columns did not help; `C=0.1` converges but PR-AUC drops
   to 0.853. Proposed fix (awaiting the owner's yes): set `max_iter=50000` in `models.py`.
2. Splits must be identical on every machine. `StratifiedGroupKFold` may split differently across scikit-learn versions,
   so save the train/val/test indices once (one file per seed) and load them everywhere. Not implemented yet.
3. Ordinal level orders in `features.py` are judgment calls ("Not Applicable" ranks below "Internship", etc.).
4. Two false negatives looked like the same posting with different capitalisation (possible near-duplicates that exact
   dedup misses). Unconfirmed.

## Agreed plan for the deep part (the owner chose option B)
Frozen `sentence-transformers/all-MiniLM-L6-v2` embeddings (Session 7 model), then compare on the same splits and seeds:
embeddings + LR, embeddings + SVM, MLP head, embeddings combined with TF-IDF and metadata; PLUS a fine-tuned DistilBERT
(plain PyTorch loop, no Trainer, so it is easy to explain; class weighting for the 4.8% positives; early stopping on
validation PR-AUC; test touched once). Explanation: compare which frauds each model catches, SVM top words, LIME on raw text
for about five examples. Run one seed first to measure real timing, then more seeds. Still to write: split save/load,
fine-tuning helper, hybrid-feature helper, the notebook(s), a seed-loop runner, final tables and plots.
Remaining classical work: full tuning (validation only), multi-seed comparison, then a single test evaluation.

## Other material not in this repo
- Slides (course PDFs) and syllabus, the previous course's project (`older_work`, an image-classification pipeline whose
  structure was reused), and the survey/paper PDFs under `surveys/` (gitignored: SBERT, SHAP, LIME, SMOTE, Fraud-BERT).
- `docs/literature_review.md` is a draft. Items tagged [V] were confirmed from abstracts/snippets; [M] are from memory.
  The Springer systematic review of recruitment-scam detection and the surveys the owner mentioned were not obtained.
