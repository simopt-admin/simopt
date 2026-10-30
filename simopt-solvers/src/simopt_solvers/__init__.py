"""Simulation-optimization solvers for models built with simopt-dsl."""

from importlib import import_module
from typing import TYPE_CHECKING

from simopt_solvers.problem_types import ConstraintType, ObjectiveType, VariableType
from simopt_solvers.solution import Solution
from simopt_solvers.solver import (
    Budget,
    BudgetExhaustedError,
    Context,
    Solver,
    SolverConfig,
    SolveResult,
    solve,
)

if TYPE_CHECKING:
    from simopt_solvers.solvers.adam import ADAM
    from simopt_solvers.solvers.aloe import ALOE
    from simopt_solvers.solvers.astrodf import ASTRODF
    from simopt_solvers.solvers.dasso import DASSO
    from simopt_solvers.solvers.fcsa import FCSA
    from simopt_solvers.solvers.neldmd import NelderMead
    from simopt_solvers.solvers.randomsearch import RandomSearch
    from simopt_solvers.solvers.spsa import SPSA
    from simopt_solvers.solvers.strong import STRONG

# Solver classes are imported on first access so that optional dependencies
# (cvxpy for FCSA, pydoe for DASSO) are only required by the solvers using them.
_SOLVER_MODULES = {
    "ADAM": "adam",
    "ALOE": "aloe",
    "ASTRODF": "astrodf",
    "DASSO": "dasso",
    "FCSA": "fcsa",
    "NelderMead": "neldmd",
    "RandomSearch": "randomsearch",
    "SPSA": "spsa",
    "STRONG": "strong",
}


def __getattr__(name: str) -> type:
    if name in _SOLVER_MODULES:
        module = import_module(f"simopt_solvers.solvers.{_SOLVER_MODULES[name]}")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ADAM",
    "ALOE",
    "ASTRODF",
    "DASSO",
    "FCSA",
    "SPSA",
    "STRONG",
    "Budget",
    "BudgetExhaustedError",
    "ConstraintType",
    "Context",
    "NelderMead",
    "ObjectiveType",
    "RandomSearch",
    "Solution",
    "SolveResult",
    "Solver",
    "SolverConfig",
    "VariableType",
    "solve",
]
