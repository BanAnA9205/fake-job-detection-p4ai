# Implementation contract: fake job posting detection (EMSCAD)

Working dir for everything: `assignment/code/`. Data: `../data/fake_job_postings.csv` (17,880 rows, 18 columns, target `fraudulent`, 866 positives = 4.84%).
Python: `assignment/code/.venv/bin/python` (pandas, scikit-learn, scipy, matplotlib, seaborn, joblib). torch / transformers / sentence-transformers are NOT installed and must NOT be installed.

## HARD SAFETY RULES (the owner's laptop is fragile; these override any other instruction)
1. Do NOT run any model training, hyper-parameter search, embedding extraction, fine-tuning, or anything that uses torch/transformers/GPU. Do not install packages.
2. Allowed to run: `python -m py_compile`, imports, and tiny smoke tests on SYNTHETIC data (<= 200 rows, finishes in < 5 s). Never fit anything on the full CSV. The orchestrator runs the full-data pilot.
3. Import torch / transformers / sentence_transformers ONLY inside functions (lazy), never at module top level.
4. Any command you run must be wrapped as `timeout 60 taskset -c 0-1 nice -n 19 <cmd>`.
5. Edit only the files assigned to you. Do not touch `../data`, `../surveys`, `../older_work`, or other agents' files.
6. No `n_jobs=-1`; default `n_jobs=1` (parameterised, max 4).

## Course conventions to follow (from the slides, Sessions 4-7)
- Split FIRST, then fit every preprocessing step inside one sklearn `Pipeline`/`ColumnTransformer` (fit on train only).
- Drop duplicates BEFORE splitting; assert no train/test overlap and print class balance of every split.
- Always compare against `DummyClassifier(strategy="most_frequent")` first.
- Imbalanced target: report precision, recall, F1, PR-AUC, ROC-AUC, confusion matrix, per-class report; never accuracy alone. Report train and test scores together (generalization gap).
- Tune on validation / CV on the training portion only; touch the test set once.
- Seed everything (`seed_everything`), report mean +- std over several split seeds.
- Use assertions on shape / alignment / NaNs. Cache expensive steps (embeddings) to disk with the row ids saved alongside.
- No em dashes in comments/docstrings. Keep code simple; no speculative abstractions. Docstrings short.

## Files and public API (exact names and signatures; other modules import these)

### `modules/data_helper.py`  (agent A)
- `TEXT_COLS = ["title","company_profile","description","requirements","benefits"]`
- `CAT_COLS = ["employment_type","required_experience","required_education","industry","function"]`
- `BIN_COLS = ["telecommuting","has_company_logo","has_questions"]`
- `load_raw(path) -> pd.DataFrame`  (asserts 18 columns, `job_id` unique, target in {0,1})
- `audit(df) -> dict`  (shape, class counts, positive rate, per-column missing rate, exact-duplicate count ignoring `job_id`, rows sharing a description, label conflicts among duplicates)
- `clean(df) -> pd.DataFrame`  (drop rows duplicated on all columns except `job_id` and keep first; assert no label conflicts remain; add column `group` = stable hash of whitespace/case-normalised `description` (use hashlib, not Python `hash`); reset_index(drop=True); do NOT impute or encode here)
- `make_splits(df, seed=0, val_size=0.2, test_size=0.2) -> dict[str, np.ndarray]`  keys `"train","val","test"` holding positional row indices. Use `StratifiedGroupKFold` on `group` so that no `group` appears in two splits. Assert: indices disjoint, union == all rows, no group overlap, positive rate of each split within 0.01 of the overall rate (if impossible for tiny data, raise a clear error).
- `save_metadata(path, df, splits, seed) -> None` / `load_metadata(path) -> dict`  JSON with seed, sizes, positives per split, column lists.
- `seed_everything(seed) -> None`  (random, numpy, os PYTHONHASHSEED; torch seeding only if torch is already importable, inside try/except ImportError).

### `modules/features.py`  (agent B)
- `add_engineered_features(df) -> pd.DataFrame`  returns a COPY with new columns: `text` (the TEXT_COLS joined by a space, NaN -> ""), `country` (first token of `location` before a comma, "missing" if NaN), `has_salary` (salary_range notna), `missing_company_profile`, `missing_requirements`, `missing_benefits`, `missing_department` (int 0/1), `desc_len`, `title_len` (chars), `n_exclaim`, `n_upper_ratio`, `has_url`, `has_email` (regex on text), `has_money_words` (regex for words like "earn", "$", "per week", "no experience", "work from home", "wire", "western union"). Pure row-wise, no fitting.
- `make_preprocessor(kind="tfidf", max_features=20000, ngram_range=(1,2), min_df=3) -> ColumnTransformer` for kind `"tfidf"`: `TfidfVectorizer(sublinear_tf=True, strip_accents="unicode", stop_words="english")` on `text`; `OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30)` on nominal cols (`employment_type`, `industry`, `function`, `country`); `OrdinalEncoder` with EXPLICIT category order and `handle_unknown="use_encoded_value", unknown_value=-1` on `required_experience` and `required_education` (document the order in a module constant; fill NaN with a "missing" level placed first); `StandardScaler(with_mean=False)` on numeric engineered cols; binary cols passthrough. For kind `"meta"`: everything except text (dense-friendly, for tree models).
- `NUMERIC_COLS`, `BINARY_COLS`, `NOMINAL_COLS`, `ORDINAL_COLS` module constants.

### `modules/models.py`  (agent B)
- `get_baselines(seed=0) -> dict[str, estimator]`  `{"dummy_majority", "dummy_stratified"}`
- `get_models(seed=0, kind_pre="tfidf") -> dict[str, Pipeline]`  `"logreg"` (LogisticRegression, class_weight="balanced", solver="liblinear" or saga), `"linear_svm"` (LinearSVC, class_weight="balanced", dual="auto"), `"hist_gb"` (a ColumnTransformer with the "meta" columns plus the TF-IDF text column reduced by `TruncatedSVD(100)`, then `HistGradientBoostingClassifier(class_weight="balanced")`; HistGradientBoosting needs dense input). Every entry is one `Pipeline([("pre", ...), ("clf", ...)])`.
- `PARAM_GRIDS: dict[str, dict]`  small grids keyed by model name using `pre__...` / `clf__...` names (e.g. `clf__C` in {0.1,1,10}, tfidf `max_features` in {10000, 30000}, `ngram_range` in {(1,1),(1,2)}; HGB `learning_rate`, `max_leaf_nodes`).
- `tune(pipe, grid, X, y, groups, n_iter=8, cv=5, scoring="average_precision", seed=0, n_jobs=1) -> RandomizedSearchCV`  using `StratifiedGroupKFold(n_splits=cv, shuffle=True, random_state=seed)`; returns the fitted search object. DO NOT RUN IT on real data.

### `modules/eval_helper.py`  (agent C)
- `score_of(model, X) -> np.ndarray`  uses `predict_proba[:,1]` if available else `decision_function`.
- `metrics_at_threshold(y, score, thr) -> dict`  precision, recall, f1, tp, fp, fn, tn.
- `evaluate(y, score, thr) -> dict`  PR-AUC (`average_precision_score`), ROC-AUC, plus `metrics_at_threshold`. `thr` is a required argument (no default) because probabilities and SVM decision scores need different thresholds.
- `pick_threshold(y_val, score_val, strategy="max_f1", min_recall=None) -> float`  chosen on VALIDATION only.
- `sweep_thresholds(y, score, n=50) -> pd.DataFrame`  precision/recall/flagged per threshold (Session 6).
- `compare_table(results: dict[str, dict]) -> pd.DataFrame`  one row per model, columns train/test PR-AUC, F1, recall, precision, gap.
- `aggregate_seeds(rows: list[dict]) -> pd.DataFrame`  mean and std per metric per model across seeds.
- `plot_confusions(cm_by_name, class_names, ...)`, `plot_pr_curves(curves_by_name, ...)`  matplotlib/seaborn, return the Figure, no plt.show().
- `group_breakdown(y, pred, groups: pd.Series) -> pd.DataFrame`  count, positives, recall, precision per group value (Session 6 per-group check).

### `modules/explain_helper.py`  (agent C)
- `top_linear_terms(pipe, k=20) -> pd.DataFrame`  top positive/negative coefficients of the final linear estimator, with feature names recovered from the `ColumnTransformer` (`get_feature_names_out`).
- `permutation_table(pipe, X, y, scoring="average_precision", n_repeats=5, seed=0, n_jobs=1, cols=None) -> pd.DataFrame`  permutation importance on RAW input columns (permute the dataframe columns, not transformed ones).
- `show_errors(df, y_true, score, thr, k=10) -> pd.DataFrame`  k most confident false positives and false negatives with title and first 200 chars of description (Session 6: read ten real errors).

### `modules/deep_helper.py`  (agent D; NEVER EXECUTE ANYTHING IN THIS FILE, only py_compile it)
- `embed_texts(texts, ids, model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=32, max_seq_length=256, cache_dir="features", device="cpu") -> np.ndarray`  lazy-import sentence_transformers; if `cache_dir/<model-slug>_emb.npy` and `_ids.npy` exist and ids match, load; else compute with `normalize_embeddings=True`, save both. Never shuffle.
- `MLPHead(nn.Module)` built lazily via `make_mlp(in_dim, hidden=128, dropout=0.3)`; `train_mlp(Xtr, ytr, Xva, yva, epochs=20, lr=1e-3, batch_size=64, pos_weight=None, seed=0, device="cpu") -> (model, history)` with: seed_everything, `DataLoader(..., shuffle=True, generator=g)`, `opt.zero_grad()` first in loop, `BCEWithLogitsLoss(pos_weight=...)` on raw logits, `model.eval()` + `torch.no_grad()` for validation, early stopping on validation PR-AUC. `predict_scores(model, X, device="cpu") -> np.ndarray` (probabilities).
- `fine_tune_note` docstring only: describe how a full BERT fine-tune would be done (Fraud-BERT setting) but do NOT implement a training loop for it.

## Tests
Each agent adds `tests/test_<module>.py` using pytest-style plain asserts on SYNTHETIC data (a handful of hand-written rows), and may run only: `timeout 60 taskset -c 0-1 nice -n 19 .venv/bin/python -m pytest -x -q tests/test_<module>.py` if pytest is present, else plain `python tests/test_<module>.py`. Agent D writes tests that only `py_compile`.

## Output of each agent
Final message: list of files written, anything in the contract you could not satisfy, and the exact commands you ran with their output. Do not claim a module is tested unless you ran its test.

## Presentability (the team must present and explain this code live)
- Standard, well-known scikit-learn / pandas / matplotlib tools only. Anything beyond the course slides (for example `StratifiedGroupKFold`, `TruncatedSVD`, permutation importance) must have a one-line comment saying what it does and why we use it.
- Short functions (about 30 lines max), plain names, one job each. No decorators, no metaclasses, no clever one-liners, no dynamic imports except the required lazy torch imports.
- Each function has a 1-2 line docstring in plain English. Comments explain WHY, not what.
- Prefer the simplest implementation that satisfies the contract. If two approaches work, pick the one that is easier to explain to classmates.
