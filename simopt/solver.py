"""Base classes for simulation optimization solvers.

Re-exported from :mod:`simopt_solvers.solver` for backward compatibility.
"""

from simopt_solvers.solver import (  # noqa: F401
    Budget,
    BudgetExhaustedError,
    Context,
    Solver,
    SolverConfig,
    SolveResult,
    solve,
)
