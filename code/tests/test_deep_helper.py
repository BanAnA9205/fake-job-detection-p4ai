"""Static tests for modules/deep_helper.py.

Torch and sentence-transformers are not installed, so this file only compiles
the module and checks the source; no function from the module is executed.
"""

import ast
import py_compile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "modules" / "deep_helper.py"
HEAVY = {"torch", "transformers", "sentence_transformers"}


def test_module_compiles():
    """The module must compile without importing torch."""
    py_compile.compile(str(MODULE), doraise=True)


def test_heavy_imports_are_lazy():
    """Torch, transformers and sentence-transformers must not be imported at module level."""
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
    """All contract names must exist as module-level functions or constants."""
    tree = ast.parse(MODULE.read_text())
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assigned = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assigned.add(target.id)
    for name in ["embed_texts", "make_mlp", "train_mlp", "predict_scores", "fine_tune_note"]:
        assert name in functions | assigned, f"missing {name}"


if __name__ == "__main__":
    test_module_compiles()
    test_heavy_imports_are_lazy()
    test_public_api_is_defined()
    print("all deep_helper static tests passed")
