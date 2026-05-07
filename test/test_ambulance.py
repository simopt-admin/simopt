"""Regression tests for the ambulance model."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from simopt.models.ambulance import Ambulance, AmbulanceMinAvgResponse


class _SequenceInputModel:
    """Input model that returns a fixed sequence of values."""

    def __init__(self, values: list[float]) -> None:
        self._values = iter(values)

    def random(self, *_args: object, **_kwargs: object) -> float:
        """Return the next fixed value."""
        return next(self._values)


def test_queued_dispatch_keeps_ambulance_busy() -> None:
    """Queued service must prevent duplicate dispatch of one ambulance."""
    model = Ambulance(
        {
            "fixed_base_count": 0,
            "variable_base_count": 2,
            "fixed_locs": [],
            "variable_locs": [0, 0, 20, 20],
        }
    )
    model.arrival_time_model = _SequenceInputModel([0, 1, 1, 3, 2000])  # type: ignore
    model.scene_time_model = _SequenceInputModel([4, 2, 20, 1, 1])  # type: ignore
    model.beta_x_model = _SequenceInputModel([0, 1, 1, 1, 0])  # type: ignore
    model.beta_y_model = _SequenceInputModel([0, 1, 1, 1, 0])  # type: ignore

    responses, _gradients = model.replicate()

    assert responses["avg_response_time"] == pytest.approx(10.25)


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
