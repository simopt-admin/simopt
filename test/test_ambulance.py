"""Regression tests for the ambulance model."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from simopt.models.ambulance import Ambulance, AmbulanceMinAvgResponse


def test_deterministic_constraints_require_correct_dimension() -> None:
    """Ambulance base vectors must match the problem dimension."""
    problem = AmbulanceMinAvgResponse()

    assert not problem.check_deterministic_constraints(())
    assert not problem.check_deterministic_constraints((1.0,))
    assert not problem.check_deterministic_constraints((1.0, 2.0, 3.0, 4.0, 5.0))
    assert not problem.check_deterministic_constraints((-1.0, 2.0, 3.0, 4.0))
    assert not problem.check_deterministic_constraints((1.0, 2.0, 3.0, 21.0))
    assert problem.check_deterministic_constraints((1.0, 2.0, 3.0, 4.0))


@pytest.mark.parametrize(
    "fixed_factors",
    [
        {"call_loc_beta_x": (0.0, 1.0)},
        {"call_loc_beta_x": (-1.0, 1.0)},
        {"call_loc_beta_y": (1.0, 0.0)},
        {"call_loc_beta_y": (1.0, -1.0)},
    ],
)
def test_beta_shape_parameters_must_be_positive(
    fixed_factors: dict[str, tuple[float, float]],
) -> None:
    """Beta input-model shape parameters must be positive."""
    with pytest.raises(ValidationError, match="must be greater than 0"):
        Ambulance(fixed_factors)
