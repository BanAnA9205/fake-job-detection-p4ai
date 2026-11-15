"""Checks for run_deep.py: the CLI help works without torch. (The results format is tested in
test_results_helper.py.) Nothing here loads data, embeddings or torch.

Run: .venv/bin/python tests/test_run_deep.py
"""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)



def test_help():
    """--help must work without torch installed, in a separate process."""
    out = subprocess.run([sys.executable, os.path.join(ROOT, "run_deep.py"), "--help"],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    for flag in ["--stage", "--seeds", "--out", "--embed-model", "--bert-model",
                 "--epochs", "--batch-size", "--max-len", "--lr", "--device", "--data"]:
        assert flag in out.stdout, f"missing {flag} in --help"


if __name__ == "__main__":
    test_help()
    print("all run_deep tests passed")
