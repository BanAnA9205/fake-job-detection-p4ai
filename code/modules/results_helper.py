"""Versioned, integrity-checked experiment results and reproducible run signatures.

Each completed experiment has a JSON metadata file and an NPZ of scores. The
metadata is published LAST and includes a checksum of the NPZ, so interrupted
writes cannot accidentally be treated as completed runs.
"""

import glob
import hashlib
import json
import os
import platform
import tempfile

import numpy as np
import pandas as pd
import sklearn


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


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_fingerprint(df):
    """Hash *all* ordered rows and values, including labels, not only their IDs."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("dataset_fingerprint expects a DataFrame")
    h = hashlib.sha256()
    h.update(json.dumps([str(c) for c in df.columns]).encode("utf-8"))
    h.update(json.dumps([str(dt) for dt in df.dtypes]).encode("utf-8"))
    h.update(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes())
    return h.hexdigest()


def experiment_signature(data_hash, config, source_files=()):
    """Bind a run to its data, configuration, code and runtime versions."""
    meta = {
        "dataset_sha256": data_hash,
        "config": _plain(config),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "evaluation_protocol": 2,
    }
    h = hashlib.sha256(json.dumps(meta, sort_keys=True).encode("utf-8"))
    for path in sorted(source_files, key=os.path.basename):
        with open(path, "rb") as f:
            h.update(os.path.basename(path).encode("utf-8"))
            h.update(f.read())
    return h.hexdigest()


def _paths(out_dir, model, seed):
    stem = os.path.join(out_dir, f"{model}_seed{seed}")
    return stem + ".json", stem + ".npz"


def result_exists(out_dir, model, seed, expected_signature=None):
    """True only for a complete matching run with a valid scores checksum."""
    json_path, npz_path = _paths(out_dir, model, seed)
    if not (os.path.isfile(json_path) and os.path.isfile(npz_path)):
        return False
    try:
        with open(json_path, encoding="utf-8") as f:
            record = json.load(f)
        if (record.get("model") != model or record.get("seed") != int(seed)
                or not record.get("scores_sha256")
                or record["scores_sha256"] != _sha256_file(npz_path)):
            return False
        if expected_signature is not None and record.get("run_signature") != expected_signature:
            return False
        with np.load(npz_path, allow_pickle=False) as scores:
            needed = {"y_val", "score_val", "y_test", "score_test"}
            if not needed.issubset(scores.files):
                return False
            return (len(scores["y_val"]) == len(scores["score_val"])
                    and len(scores["y_test"]) == len(scores["score_test"]))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def save_result(out_dir, model, group, seed, thr, val, test, y_val, score_val, y_test,
                score_test, train_pr_auc=None, best_params=None, seconds=None,
                notes="", history=None, run_signature=None):
    """Atomically publish scores first and a checksum-bearing JSON record last."""
    os.makedirs(out_dir, exist_ok=True)
    json_path, npz_path = _paths(out_dir, model, seed)
    record = {
        "model": model, "group": group, "seed": int(seed), "thr": float(thr),
        "val": val, "test": test, "train_pr_auc": train_pr_auc,
        "best_params": best_params or {}, "seconds": seconds,
        "notes": notes, "run_signature": run_signature,
    }
    if history is not None:
        record["history"] = history

    temporary_npz = None
    temporary_json = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".npz",
                                         dir=out_dir, delete=False) as f:
            temporary_npz = f.name
            np.savez(f, y_val=np.asarray(y_val), score_val=np.asarray(score_val),
                     y_test=np.asarray(y_test), score_test=np.asarray(score_test))
        record["scores_sha256"] = _sha256_file(temporary_npz)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                         encoding="utf-8", dir=out_dir, delete=False) as f:
            temporary_json = f.name
            json.dump(_plain(record), f, indent=2)
        os.replace(temporary_npz, npz_path)
        os.replace(temporary_json, json_path)
    finally:
        for path in (temporary_npz, temporary_json):
            if path and os.path.exists(path):
                os.unlink(path)


def load_scores(out_dir, model, seed):
    """Load scores and verify checksums when the record has modern metadata."""
    json_path, npz_path = _paths(out_dir, model, seed)
    if os.path.isfile(json_path):
        with open(json_path, encoding="utf-8") as f:
            record = json.load(f)
        checksum = record.get("scores_sha256")
        if checksum is not None and checksum != _sha256_file(npz_path):
            raise ValueError(f"corrupted/mismatched score archive for {model} seed {seed}")
    with np.load(npz_path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def load_results(*out_dirs):
    """Load results, including legacy reports (which are NOT valid resume artifacts)."""
    rows = []
    for out_dir in out_dirs:
        for path in sorted(glob.glob(os.path.join(out_dir, "*_seed*.json"))):
            with open(path, encoding="utf-8") as f:
                r = json.load(f)
            row = {
                "model": r["model"], "group": r["group"], "seed": r["seed"],
                "thr": r["thr"], "train_pr_auc": r.get("train_pr_auc"),
                "seconds": r.get("seconds"), "verified_artifact": bool(r.get("scores_sha256")),
            }
            for split in ("val", "test"):
                for key in ("pr_auc", "roc_auc", "f1", "precision", "recall"):
                    row[f"{split}_{key}"] = r[split][key]
            rows.append(row)
    return pd.DataFrame(rows)
