"""Run the classical experiments over several split seeds (called from notebook 02).

Per seed and model: tune on the training rows with grouped cross-validation, refit,
pick the decision threshold on the validation rows, then score validation and test once.
A finished (model, seed) is skipped on the next call, so an interrupted run can resume.
"""

import os
import time
import warnings

from sklearn.model_selection import ParameterGrid

from modules import data_helper as dh
from modules import eval_helper as ev
from modules import models as md
from modules import results_helper as rh

ALL_MODELS = ["dummy_majority", "logreg", "ocsvm"]


def get_splits(df, seed, split_dir="splits"):
    """Load the saved split for this seed, or create and save it (same split on every machine)."""
    path = os.path.join(split_dir, f"seed{seed}.npz")
    if os.path.exists(path):
        return dh.load_splits(path, df)
    splits = dh.make_splits(df, seed=seed)
    dh.save_splits(path, splits, df)
    return splits


def fit_model(name, seed, train, y_train, n_iter, cv, n_jobs):
    """Return (fitted model, best params, cv score). Dummies are just fitted, the rest are tuned."""
    # Baselines have nothing to tune.
    if name.startswith("dummy"):
        return md.get_baselines(seed)[name].fit(train, y_train), {}, None

    # Real models: random search with grouped CV on the training rows only (the whole grid
    # for the models in FULL_GRID).
    pipe, grid = md.get_models(seed)[name], md.PARAM_GRIDS[name]
    if name in md.FULL_GRID:
        n_iter = len(ParameterGrid(grid))
    search = md.tune(pipe, grid, train, y_train, train["group"].to_numpy(),
                     n_iter=n_iter, cv=cv, seed=seed, n_jobs=n_jobs)
    return search.best_estimator_, search.best_params_, float(search.best_score_)


def run_one(name, seed, parts, out_dir, n_iter, cv, n_jobs):
    """Fit one model on one split and save its results."""
    train, val, test = parts
    y_train, y_val, y_test = (p["fraudulent"].to_numpy() for p in parts)

    # Fit (and tune) while recording warnings instead of hiding them.
    start = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model, best_params, cv_score = fit_model(name, seed, train, y_train, n_iter, cv, n_jobs)
    # Keep one line per distinct warning (for example "did not converge") in the notes.
    warned = sorted({str(w.message).splitlines()[0][:80] for w in caught})

    # Scores on every split, then the threshold from validation and the metrics at it.
    s_train, s_val, s_test = (ev.score_of(model, p) for p in parts)
    # Dummies give the same score to every row, so a max-F1 threshold would flag
    # everything as fraud; 0.5 keeps their own predictions (all-real for majority).
    if name.startswith("dummy"):
        thr = 0.5
    else:
        thr = ev.pick_threshold(y_val, s_val)      # chosen on validation only
    val_m = ev.evaluate(y_val, s_val, thr)
    test_m = ev.evaluate(y_test, s_test, thr)       # the test rows are scored once, here
    seconds = round(time.time() - start, 1)

    search_desc = (f"the full grid of {len(ParameterGrid(md.PARAM_GRIDS[name]))} candidates"
                   if name in md.FULL_GRID else f"{n_iter} random candidates")

    # Save the shared json + npz pair.
    rh.save_result(out_dir, name, "classical", seed, thr, val_m, test_m, y_val, s_val, y_test,
                   s_test, train_pr_auc=ev.evaluate(y_train, s_train, thr)["pr_auc"],
                   best_params=best_params, seconds=seconds,
                   notes=f"tuned with {search_desc}, {cv}-fold grouped CV; "
                         f"cv PR-AUC {cv_score}; warnings: {warned or 'none'}")

    # Only validation is printed on purpose: test numbers stay in the saved files until the final report.
    print(f"seed {seed} {name:16s} val PR-AUC {val_m['pr_auc']:.3f} F1 {val_m['f1']:.3f} | {seconds}s"
          f"{' | WARNINGS: ' + '; '.join(warned) if warned else ''}", flush=True)


def run_classical(df, seeds, out_dir="outputs/classical", models=ALL_MODELS,
                  split_dir="splits", n_iter=4, cv=3, n_jobs=1):
    """All models for all seeds on the cleaned, featurised frame df; finished runs are skipped."""
    assert 1 <= n_jobs <= 4, "n_jobs is capped at 4 to keep the machine cool"
    for seed in seeds:
        # The saved split for this seed, as three frames.
        dh.seed_everything(seed)
        splits = get_splits(df, seed, split_dir)
        parts = [df.iloc[splits[k]] for k in ("train", "val", "test")]
        print(f"seed {seed}: rows train/val/test = {[len(p) for p in parts]}, "
              f"positives = {[int(p['fraudulent'].sum()) for p in parts]}", flush=True)

        # Every model on this split; finished runs are skipped so a crash can resume.
        for name in models:
            if rh.result_exists(out_dir, name, seed):
                print(f"seed {seed} {name}: already done, skipping")
                continue
            run_one(name, seed, parts, out_dir, n_iter, cv, n_jobs)
