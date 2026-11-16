"""Save and load experiment results in one shared format (classical and deep runs alike).

Per (model, seed) two files are written to the output folder:
  <model>_seed<N>.json  metrics at the validation-chosen threshold, best params, timing
  <model>_seed<N>.npz   y_val, score_val, y_test, score_test (same row order as the saved splits)
"""

import glob
import json
import os

import numpy as np
import pandas as pd


def _plain(obj):
    """Make numpy numbers, tuples and arrays JSON-friendly."""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _paths(out_dir, model, seed):
    stem = os.path.join(out_dir, f"{model}_seed{seed}")
    return stem + ".json", stem + ".npz"


def result_exists(out_dir, model, seed):
    """True when both files of a (model, seed) run are already on disk (lets a run resume)."""
    return all(os.path.exists(p) for p in _paths(out_dir, model, seed))


def save_result(out_dir, model, group, seed, thr, val, test, y_val, score_val, y_test,
                score_test, train_pr_auc=None, best_params=None, seconds=None,
                notes="", history=None):
    """Write the json and npz files for one (model, seed) run."""
    os.makedirs(out_dir, exist_ok=True)
    json_path, npz_path = _paths(out_dir, model, seed)

    # Metrics and settings go to the json file...
    record = {"model": model, "group": group, "seed": int(seed), "thr": float(thr),
              "val": val, "test": test, "train_pr_auc": train_pr_auc,
              "best_params": best_params or {}, "seconds": seconds, "notes": notes}
    if history is not None:
        record["history"] = history
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_plain(record), f, indent=2)

    # ...and the raw scores to the npz, so curves and matrices can be rebuilt later.
    np.savez(npz_path, y_val=np.asarray(y_val), score_val=np.asarray(score_val),
             y_test=np.asarray(y_test), score_test=np.asarray(score_test))


def load_scores(out_dir, model, seed):
    """Load the saved scores of one run as a dict of arrays."""
    z = np.load(_paths(out_dir, model, seed)[1])
    return {k: z[k] for k in z.files}


def load_results(*out_dirs):
    """One row per (model, seed) with the headline metrics of every json found in the folders."""
    rows = []
    for out_dir in out_dirs:
        for path in sorted(glob.glob(os.path.join(out_dir, "*_seed*.json"))):
            with open(path, encoding="utf-8") as f:
                r = json.load(f)

            # Run-level fields first, then the headline metrics of both splits.
            row = {"model": r["model"], "group": r["group"], "seed": r["seed"], "thr": r["thr"],
                   "train_pr_auc": r.get("train_pr_auc"), "seconds": r.get("seconds")}
            for split in ("val", "test"):
                for k in ("pr_auc", "roc_auc", "f1", "precision", "recall"):
                    row[f"{split}_{k}"] = r[split][k]
            rows.append(row)
    return pd.DataFrame(rows)
