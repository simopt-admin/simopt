"""Feasibility score functions."""

import numpy as np


def feasibility_score(
    lhs: np.ndarray, method: str, norm_degree: int, two_sided: bool
) -> float:
    """Compute feasibility score for a given set of constraints."""
    if lhs.size == 0:
        return 0.0

    is_feasible = not np.any(lhs > 0)
    if is_feasible:
        return -float(np.max(lhs)) if two_sided else 0.0

    if method == "inf_norm":
        return -float(np.max(lhs))

    if method == "norm":
        violations = np.where(lhs < 0, 0.0, lhs)
        return -float(np.linalg.norm(violations, ord=norm_degree))

    raise ValueError(f"invalid method: {method}")


def feasibility_score_history(
    history: list[np.ndarray],
    method: str,
    norm_degree: int,
    two_sided: bool,
) -> list[float]:
    """Compute feasibility score history."""
    scores = []
    for lhs in history:
        scores.append(feasibility_score(lhs, method, norm_degree, two_sided))
    return scores
