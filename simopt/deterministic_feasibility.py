"""Deterministic feasibility and merit scoring utilities."""

import numpy as np
from numpy.linalg import norm as vec_norm


def deterministic_feas_violation(problem, x: tuple) -> float:
    """Aggregate deterministic constraint violation magnitude for solution x."""
    c_eq = problem.get_deterministic_equality_constraints(tuple(x))
    c_ineq = problem.get_deterministic_inequality_constraints(tuple(x))

    parts = []
    if c_eq is not None:
        parts.append(np.atleast_1d(np.abs(c_eq)))
    if c_ineq is not None:
        parts.append(np.maximum(np.atleast_1d(c_ineq), 0))

    return float(vec_norm(np.concatenate(parts))) if parts else 0.0


def merit_from_obj_and_feas(
    obj: float,
    feas: float,
    obj_const: float,
    feas_tol_lower: float,
    feas_tol_upper: float,
) -> float:
    """Combine an objective value and feasibility violation into a merit score."""
    if feas <= feas_tol_lower:
        penalty = 0.0
    elif feas > feas_tol_upper:
        return np.inf
    else:
        penalty = feas
    return obj + obj_const * penalty