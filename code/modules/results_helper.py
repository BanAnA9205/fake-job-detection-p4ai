"""Save and load experiment results in one shared format (classical and deep runs alike).

Per (model, seed) two files are written to the output folder:
  <model>_seed<N>.json  metrics/settings/timing plus an optional cleaned-data fingerprint
  <model>_seed<N>.npz   y_val, score_val, y_test, score_test and optional split indices
"""

import glob
import json
import os

import numpy as np
import pandas as pd
from modules.eval_helper import evaluate, pick_threshold


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


def assert_result_lineage(record, scores, fingerprint, val_indices, test_indices):
    """Bind a record and its score arrays to one cleaned frame and exact saved split rows."""
    if record.get("data_fingerprint") != fingerprint:
        raise ValueError("result was fitted for different cleaned data; use a new output directory")
    if val_indices is None or test_indices is None:
        raise ValueError("exact validation and test split indices are required")
    for split, expected in (("val", val_indices), ("test", test_indices)):
        key = f"{split}_indices"
        if key not in scores:
            raise ValueError(f"result has no {key}; migrate it or use a new output directory")
        expected = np.asarray(expected, dtype=np.int64)
        if not np.array_equal(scores[key], expected):
            raise ValueError(f"result {key} does not match the current saved split")
        if len(scores[f"y_{split}"]) != len(expected):
            raise ValueError(f"result {split} labels and split indices have different lengths")


def result_exists(out_dir, model, seed, fingerprint=None, val_indices=None, test_indices=None):
    """Resume existing pairs, rejecting stale data/splits when lineage is supplied."""
    json_path, npz_path = _paths(out_dir, model, seed)
    if not (os.path.exists(json_path) and os.path.exists(npz_path)):
        return False
    if fingerprint is not None:
        with open(json_path, encoding="utf-8") as f:
            record = json.load(f)
        scores = load_scores(out_dir, model, seed)
        assert_result_lineage(record, scores, fingerprint, val_indices, test_indices)
    return True


def save_result(out_dir, model, group, seed, thr, val, test, y_val, score_val, y_test,
                score_test, train_pr_auc=None, best_params=None, seconds=None,
                notes="", history=None, data_fingerprint=None, val_indices=None, test_indices=None):
    """Write one run, optionally binding its scores to data and exact validation/test rows."""
    os.makedirs(out_dir, exist_ok=True)
    json_path, npz_path = _paths(out_dir, model, seed)

    # Metrics and settings go to the json file...
    record = {"model": model, "group": group, "seed": int(seed), "thr": float(thr),
              "val": val, "test": test, "train_pr_auc": train_pr_auc,
              "best_params": best_params or {}, "seconds": seconds, "notes": notes}
    if history is not None:
        record["history"] = history
    if data_fingerprint is not None:
        record["data_fingerprint"] = str(data_fingerprint)
        assert val_indices is not None and test_indices is not None
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_plain(record), f, indent=2)

    # ...and the raw scores, plus exact row positions whenever lineage is supplied.
    arrays = {"y_val": np.asarray(y_val), "score_val": np.asarray(score_val),
              "y_test": np.asarray(y_test), "score_test": np.asarray(score_test)}
    if data_fingerprint is not None:
        for split, indices in (("val", val_indices), ("test", test_indices)):
            indices = np.asarray(indices, dtype=np.int64)
            assert len(indices) == len(arrays[f"y_{split}"]) == len(arrays[f"score_{split}"])
            arrays[f"{split}_indices"] = indices
    np.savez(npz_path, **arrays)


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
                for k in ("pr_auc", "f1", "precision", "recall"):
                    row[f"{split}_{k}"] = r[split][k]
            rows.append(row)
    return pd.DataFrame(rows)


def verify_results(expected, y, split_cache, fingerprint):
    """Load an exact result matrix after recomputing every saved metric from its raw scores.

    `expected` maps each output folder to the model names it must hold, each for every seed in
    `split_cache`. Every run must match the cleaned-data fingerprint and the exact split rows and
    labels, reproduce its saved validation/test metrics, and use the validation-F1 threshold
    (the majority dummy keeps its fixed 0.5 threshold).
    """
    y = np.asarray(y)
    seeds = sorted(split_cache)
    for out_dir, models in expected.items():
        want = {(model, seed) for model in models for seed in seeds}
        for suffix in ("json", "npz"):
            have = {(os.path.basename(p).rsplit("_seed", 1)[0], int(p.rsplit("_seed", 1)[1].split(".")[0]))
                    for p in glob.glob(os.path.join(out_dir, f"*_seed*.{suffix}"))}
            if have != want:
                raise ValueError(f"{suffix} inventory mismatch in {out_dir}: "
                                 f"missing {sorted(want - have)}, extra {sorted(have - want)}")
        for model in models:
            for seed in seeds:
                with open(_paths(out_dir, model, seed)[0], encoding="utf-8") as f:
                    record = json.load(f)
                scores = load_scores(out_dir, model, seed)
                split = split_cache[seed]
                assert_result_lineage(record, scores, fingerprint, split["val"], split["test"])
                for name in ("val", "test"):
                    labels, values = scores[f"y_{name}"], scores[f"score_{name}"]
                    if not np.array_equal(labels, y[split[name]]):
                        raise ValueError(f"{model} seed {seed}: saved {name} labels do not match the split")
                    if not np.isfinite(values).all():
                        raise ValueError(f"{model} seed {seed}: non-finite {name} scores")
                    recomputed = evaluate(labels, values, record["thr"])
                    if any(not np.isclose(record[name][k], v) for k, v in recomputed.items()):
                        raise ValueError(f"{model} seed {seed}: saved {name} metrics differ from recomputed")
                fixed = 0.5 if model == "dummy_majority" else pick_threshold(scores["y_val"], scores["score_val"])
                if record["thr"] != fixed:
                    raise ValueError(f"{model} seed {seed}: threshold is not the validation-F1 threshold")
    return load_results(*expected)
