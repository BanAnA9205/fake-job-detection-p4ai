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

from modules.deep_helper import embed_texts


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
    """The functions and the prompt that deep_runs relies on are defined at module level."""
    tree = ast.parse(MODULE.read_text())
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assigned = {t.id for node in tree.body if isinstance(node, ast.Assign)
                for t in node.targets if isinstance(t, ast.Name)}
    assert {"embed_texts", "zero_shot_scores"} <= functions
    assert "ZERO_SHOT_PROMPT" in assigned


def test_embed_texts_cache_hit():
    """Matching cached ids must be returned as is, without importing sentence-transformers."""
    with tempfile.TemporaryDirectory() as tmp:
        emb = np.arange(6, dtype=np.float32).reshape(3, 2)
        np.save(os.path.join(tmp, "tiny_model_emb.npy"), emb)
        np.save(os.path.join(tmp, "tiny_model_ids.npy"), np.array([5, 6, 7]))
        out = embed_texts(["a", "b", "c"], np.array([5, 6, 7]),
                          model_name="tiny_model", cache_dir=tmp)
        assert np.array_equal(out, emb), "cache hit must return the stored matrix"
        assert "sentence_transformers" not in sys.modules


if __name__ == "__main__":
    test_module_compiles()
    test_heavy_imports_are_lazy()
    test_public_api_is_defined()
    test_embed_texts_cache_hit()
    print("all deep_helper static tests passed")
