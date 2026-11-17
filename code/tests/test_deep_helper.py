"""Static tests for modules/deep_helper.py, plus the cache-hit branch of embed_texts.

embed_texts is never run with a real model: the cache test pre-writes tiny
arrays, so sentence-transformers is not imported.
"""

import ast
import os
import py_compile
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "modules" / "deep_helper.py"
HEAVY = {"torch", "transformers", "sentence_transformers"}

sys.path.insert(0, str(ROOT))

from modules.deep_helper import cache_exists, cache_path, embed_texts, load_cached_embeddings


def test_module_compiles():
    """The module must compile without importing torch."""
    py_compile.compile(str(MODULE), doraise=True)


def test_heavy_imports_are_lazy():
    """Heavy machine-learning packages must not be imported at module level."""
    tree = ast.parse(MODULE.read_text())
    for node in tree.body:
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            assert name.split(".")[0] not in HEAVY, f"top-level import of {name}"


def test_public_api_is_defined():
    """embed_texts stays; the torch MLP head helpers and _PATIENCE must be gone."""
    tree = ast.parse(MODULE.read_text())
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assigned = {t.id for node in tree.body if isinstance(node, ast.Assign)
                for t in node.targets if isinstance(t, ast.Name)}
    assert "embed_texts" in functions, "missing embed_texts"
    removed = ["make_mlp", "_train_epoch", "_val_pr_auc", "train_mlp", "predict_scores"]
    for name in removed:
        assert name not in functions, f"{name} should have been deleted"
    for name in ["fine_tune_note", "_PATIENCE"]:
        assert name not in assigned, f"{name} should have been deleted"


def test_embed_texts_cache_hit():
    """A validated cached array is returned with no heavy library import."""
    import hashlib
    import json

    with tempfile.TemporaryDirectory() as tmp:
        texts = ["a", "b", "c"]
        ids = np.array([5, 6, 7])
        emb = np.arange(6, dtype=np.float32).reshape(3, 2)
        path = cache_path(texts, ids, model_name="tiny_model", cache_dir=tmp)
        np.save(path, emb)
        with open(path, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        with open(path + ".json", "w", encoding="utf-8") as f:
            json.dump({"schema": 2, "shape": [3, 2], "sha256": digest}, f)
        assert cache_exists(texts, ids, model_name="tiny_model", cache_dir=tmp)
        result = embed_texts(texts, ids, model_name="tiny_model", cache_dir=tmp)
        assert np.array_equal(result, emb)
        assert "sentence_transformers" not in sys.modules
        assert np.array_equal(load_cached_embeddings(
            texts, ids, model_name="tiny_model", cache_dir=tmp), emb)
        # Altering a single character, order, length or checkpoint changes cache identity.
        for changed_texts, changed_ids, length, revision in (
            (["x", "b", "c"], ids, 256, None),
            (texts, [6, 5, 7], 256, None),
            (texts, ids, 128, None),
            (texts, ids, 256, "new-revision"),
        ):
            assert not cache_exists(changed_texts, changed_ids, model_name="tiny_model",
                                    max_seq_length=length, cache_dir=tmp,
                                    model_revision=revision)
        # Even a matching path cannot be trusted after payload corruption.
        np.save(path, np.ones_like(emb))
        assert not cache_exists(texts, ids, model_name="tiny_model", cache_dir=tmp)


if __name__ == "__main__":
    test_module_compiles()
    test_heavy_imports_are_lazy()
    test_public_api_is_defined()
    test_embed_texts_cache_hit()
    print("all deep_helper static tests passed")
