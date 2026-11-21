"""Stack two finished models: a logistic regression on their saved scores (called from notebook 04).

Nothing is retrained except this two-input logistic regression. Per seed:
  1. Inputs are the logits of the two models' saved scores on the validation and test rows.
  2. Out-of-fold scores on validation (grouped CV inside the validation rows) pick the threshold,
     so the threshold never sees a row the stacker was fitted on.
  3. The stacker is refitted on all validation rows and scores the test rows once.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

from modules import data_helper as dh
from modules import eval_helper as ev
from modules import results_helper as rh


def _logit(p):
    """Scores near 0 or 1 (common for the fine-tuned model) become large finite numbers."""
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def stack_seed(score_a, score_b, y_val, groups_val, seed, cv=5):
    """Return (fitted stacker, out-of-fold validation scores, threshold)."""
    X_val = np.column_stack([_logit(score_a["score_val"]), _logit(score_b["score_val"])])
    stacker = LogisticRegression(class_weight="balanced")
    folds = StratifiedGroupKFold(n_splits=cv, shuffle=True, random_state=seed)
    oof = cross_val_predict(stacker, X_val, y_val, groups=groups_val, cv=folds,
                            method="predict_proba")[:, 1]
    thr = ev.pick_threshold(y_val, oof)
    return stacker.fit(X_val, y_val), oof, thr


def run_stack(df, seeds, a, b, out_dir="outputs/stack", split_dir="splits"):
    """Stack model a=(folder, name) with b=(folder, name) for every seed; finished seeds are skipped."""
    name = f"stack_{a[1].split('_')[0]}_{b[1]}"
    for seed in seeds:
        if rh.result_exists(out_dir, name, seed):
            print(f"seed {seed} {name}: already done, skipping")
            continue
        sa, sb = rh.load_scores(*a, seed), rh.load_scores(*b, seed)
        assert (sa["y_val"] == sb["y_val"]).all() and (sa["y_test"] == sb["y_test"]).all(), \
            "the two runs must use the same split rows"
        val_rows = dh.load_splits(f"{split_dir}/seed{seed}.npz", df)["val"]
        groups = df.iloc[val_rows]["group"].to_numpy()

        stacker, oof, thr = stack_seed(sa, sb, sa["y_val"], groups, seed)
        X_test = np.column_stack([_logit(sa["score_test"]), _logit(sb["score_test"])])
        s_test = stacker.predict_proba(X_test)[:, 1]          # the test rows are scored once, here

        val_m = ev.evaluate(sa["y_val"], oof, thr)
        test_m = ev.evaluate(sa["y_test"], s_test, thr)
        rh.save_result(out_dir, name, "stack", seed, thr, val_m, test_m, sa["y_val"], oof,
                       sa["y_test"], s_test,
                       best_params={"inputs": [a[1], b[1]], "weights": stacker.coef_[0].tolist(),
                                    "intercept": float(stacker.intercept_[0])},
                       notes="logistic regression on the logits of the two saved scores, fitted "
                             "on validation; threshold from 5-fold grouped out-of-fold scores")
        print(f"seed {seed} {name}: out-of-fold val PR-AUC {val_m['pr_auc']:.3f}")
