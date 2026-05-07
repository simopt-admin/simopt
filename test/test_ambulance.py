"""Regression tests for the ambulance model."""

from __future__ import annotations

from simopt.models.ambulance import AmbulanceMinAvgResponse


def test_deterministic_constraints_require_correct_dimension() -> None:
    """Ambulance base vectors must match the problem dimension."""
    problem = AmbulanceMinAvgResponse()

    assert not problem.check_deterministic_constraints(())
    assert not problem.check_deterministic_constraints((1.0,))
    assert not problem.check_deterministic_constraints((1.0, 2.0, 3.0, 4.0, 5.0))
    assert not problem.check_deterministic_constraints((-1.0, 2.0, 3.0, 4.0))
    assert not problem.check_deterministic_constraints((1.0, 2.0, 3.0, 21.0))
    assert problem.check_deterministic_constraints((1.0, 2.0, 3.0, 4.0))
