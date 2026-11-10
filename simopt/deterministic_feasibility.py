"""Deterministic feasibility and merit scoring utilities."""

import numpy as np
from numpy.linalg import norm as vec_norm

from simopt.problem import Problem


def deterministic_violation(problem: Problem, x: tuple) -> float:
    """Aggregate deterministic constraint violation magnitude for solution x."""
    c_eq = problem.get_deterministic_equality_constraints(tuple(x))
    c_ineq = problem.get_deterministic_inequality_constraints(tuple(x))

    parts = []
    if c_eq is not None:
        parts.append(np.atleast_1d(np.abs(c_eq)))
    if c_ineq is not None:
        parts.append(np.maximum(np.atleast_1d(c_ineq), 0))

    return float(vec_norm(np.concatenate(parts))) if parts else 0.0


def merit_from_objective_and_violation(
    objective: float,
    violation: float,
    obj_const: float,
    feas_tol_lower: float,
    feas_tol_upper: float,
) -> float:
    """Combine an objective value and feasibility violation into a merit score."""
    if violation <= feas_tol_lower:
        penalty = 0.0
    elif violation > feas_tol_upper:
        return np.inf
    else:
        penalty = violation
    return objective + obj_const * penalty
