"""Run the classical experiments for several split seeds and save the results.

Run from the code/ folder, ideally with limited CPU:
    OMP_NUM_THREADS=2 nice -n 19 python run_classical.py --seeds 0 1 2 3 4

Per seed and model: tune on the training rows with grouped cross-validation, refit,
pick the decision threshold on the validation rows, then score validation and test once.
A finished (model, seed) is skipped on the next run, so an interrupted run can resume.
"""

import argparse
import os
import time
import warnings

import numpy as np

from modules import data_helper as dh
from modules import eval_helper as ev
from modules import features as ft
from modules import models as md
from modules import results_helper as rh

ALL_MODELS = ["dummy_majority", "dummy_stratified", "logreg", "logreg_bm25", "tree", "mlp"]


def get_splits(df, seed, split_dir):
    """Load the saved split for this seed, or create and save it (same split on every machine)."""
    path = os.path.join(split_dir, f"seed{seed}.npz")
    if os.path.exists(path):
        return dh.load_splits(path, df)
    splits = dh.make_splits(df, seed=seed)
    dh.save_splits(path, splits, df)
    return splits


def fit_model(name, seed, train, y_train, args):
    """Return (fitted model, best params, cv score). Dummies are just fitted, the rest are tuned."""
    if name.startswith("dummy"):
        return md.get_baselines(seed)[name].fit(train, y_train), {}, None
    pipe = md.get_models(seed)[name]
    search = md.tune(pipe, md.PARAM_GRIDS[name], train, y_train, train["group"].to_numpy(),
                     n_iter=args.n_iter, cv=args.cv, seed=seed, n_jobs=args.n_jobs)
    return search.best_estimator_, search.best_params_, float(search.best_score_)


def run_one(name, seed, parts, args):
    """Fit one model on one split and save its results."""
    train, val, test = parts
    y_train, y_val, y_test = (p["fraudulent"].to_numpy() for p in parts)
    start = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model, best_params, cv_score = fit_model(name, seed, train, y_train, args)
    # Keep one line per distinct warning (for example "did not converge") in the notes.
    warned = sorted({str(w.message).splitlines()[0][:80] for w in caught})
    s_train, s_val, s_test = (ev.score_of(model, p) for p in parts)
    thr = ev.pick_threshold(y_val, s_val)          # chosen on validation only
    val_m = ev.evaluate(y_val, s_val, thr)
    test_m = ev.evaluate(y_test, s_test, thr)       # the test rows are scored once, here
    seconds = round(time.time() - start, 1)
    rh.save_result(args.out, name, "classical", seed, thr, val_m, test_m, y_val, s_val, y_test,
                   s_test, train_pr_auc=ev.evaluate(y_train, s_train, thr)["pr_auc"],
                   best_params=best_params, seconds=seconds,
                   notes=f"tuned with {args.n_iter} random candidates, {args.cv}-fold grouped CV; "
                         f"cv PR-AUC {cv_score}; warnings: {warned or 'none'}")
    # Only validation is printed on purpose: test numbers stay in the saved files until the final report.
    print(f"seed {seed} {name:16s} val PR-AUC {val_m['pr_auc']:.3f} F1 {val_m['f1']:.3f} | {seconds}s"
          f"{' | WARNINGS: ' + '; '.join(warned) if warned else ''}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    parser.add_argument("--data", default="../data/fake_job_postings.csv")
    parser.add_argument("--out", default="outputs/classical")
    parser.add_argument("--splits", default="splits")
    parser.add_argument("--n-iter", type=int, default=4, help="random search candidates")
    parser.add_argument("--cv", type=int, default=3, help="CV folds on the training rows")
    parser.add_argument("--n-jobs", type=int, default=1, help="parallel CV fits (keep small)")
    args = parser.parse_args()
    assert 1 <= args.n_jobs <= 4, "n_jobs is capped at 4 to keep the machine cool"

    df = ft.add_engineered_features(dh.clean(dh.load_raw(args.data)))
    for seed in args.seeds:
        dh.seed_everything(seed)
        splits = get_splits(df, seed, args.splits)
        parts = [df.iloc[splits[k]] for k in ("train", "val", "test")]
        print(f"seed {seed}: rows train/val/test = {[len(p) for p in parts]}, "
              f"positives = {[int(p['fraudulent'].sum()) for p in parts]}", flush=True)
        for name in args.models:
            if rh.result_exists(args.out, name, seed):
                print(f"seed {seed} {name}: already done, skipping")
                continue
            run_one(name, seed, parts, args)


if __name__ == "__main__":
    main()
