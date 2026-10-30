"""Smoke tests that every notebook script runs to completion within its time budget."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

NOTEBOOK_DIR = Path(__file__).parent.parent / "notebooks"
NOTEBOOKS = sorted(NOTEBOOK_DIR.glob("[0-9][0-9]_*.py"))
TIME_LIMIT_SECONDS = 60


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=[path.stem for path in NOTEBOOKS])
def test_notebook_runs(notebook: Path) -> None:
    """Run the jupytext script of a notebook and check that it exits cleanly."""
    env = {**os.environ, "MPLBACKEND": "Agg"}
    result = subprocess.run(
        [sys.executable, notebook.name],
        cwd=NOTEBOOK_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=TIME_LIMIT_SECONDS,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
