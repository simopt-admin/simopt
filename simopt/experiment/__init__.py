"""Experiment classes."""

import time
from pathlib import Path

from .experiment import Experiment, check_compatibility, from_names

# The folder is not created here; each writer creates it when it writes.
EXPERIMENT_DIR = Path.cwd() / "experiments" / time.strftime("%Y-%m-%d_%H-%M-%S")

__all__ = [
    "EXPERIMENT_DIR",
    "Experiment",
    "check_compatibility",
    "from_names",
]
