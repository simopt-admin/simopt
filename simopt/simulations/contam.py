"""Standalone Python simulation for contamination propagation."""

from __future__ import annotations

from typing import Annotated, Final, Self

import numpy as np
from pydantic import BaseModel, Field, model_validator

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt._markers import simulation
from simopt.input_models import InputModel

NUM_STAGES: Final[int] = 5


class ContaminationConfig(BaseModel):
    """Configuration model for Contamination simulation.

    A model that simulates a contamination problem with a beta distribution.
    Returns the probability of violating contamination upper limit in each level of
    supply chain.
    """

    contam_rate_alpha: Annotated[
        float,
        Field(
            default=1.0,
            description=(
                "alpha parameter of beta distribution for growth rate of "
                "contamination at each stage"
            ),
            gt=0,
        ),
    ]
    contam_rate_beta: Annotated[
        float,
        Field(
            default=round(17 / 3, 2),
            description=(
                "beta parameter of beta distribution for growth rate of contamination at each stage"
            ),
            gt=0,
        ),
    ]
    restore_rate_alpha: Annotated[
        float,
        Field(
            default=1.0,
            description=(
                "alpha parameter of beta distribution for rate that contamination "
                "decreases by after prevention effort"
            ),
            gt=0,
        ),
    ]
    restore_rate_beta: Annotated[
        float,
        Field(
            default=round(3 / 7, 3),
            description=(
                "beta parameter of beta distribution for rate that contamination "
                "decreases by after prevention effort"
            ),
            gt=0,
        ),
    ]
    initial_rate_alpha: Annotated[
        float,
        Field(
            default=1.0,
            description=("alpha parameter of beta distribution for initial contamination fraction"),
            gt=0,
        ),
    ]
    initial_rate_beta: Annotated[
        float,
        Field(
            default=30.0,
            description=("beta parameter of beta distribution for initial contamination fraction"),
            gt=0,
        ),
    ]
    stages: Annotated[
        int,
        Field(
            default=NUM_STAGES,
            description="stage of food supply chain",
            gt=0,
        ),
    ]
    prev_decision: Annotated[
        tuple[float, ...],
        Field(
            default=(0,) * NUM_STAGES,
            description="prevention decision",
        ),
    ]

    def _check_prev_decision(self) -> None:
        if not all(0 <= u <= 1 for u in self.prev_decision):
            raise ValueError("All elements in prev_decision must be in the range [0, 1].")

    @model_validator(mode="after")
    def _validate_model(self) -> Self:
        self._check_prev_decision()

        # Cross-validation: check for matching number of stages
        if len(self.prev_decision) != self.stages:
            raise ValueError(
                "The number of stages must be equal to the length of the previous decision tuple."
            )

        return self


@simulation
def replicate(
    factors: ContaminationConfig,
    rngs: list[MRG32k3a],
    contam_model: InputModel,
    restore_model: InputModel,
) -> np.ndarray:
    """Return contamination levels across all stages for one replication."""
    stages: int = factors.stages
    init_alpha: float = factors.initial_rate_alpha
    init_beta: float = factors.initial_rate_beta
    contam_alpha: float = factors.contam_rate_alpha
    contam_beta: float = factors.contam_rate_beta
    restore_alpha: float = factors.restore_rate_alpha
    restore_beta: float = factors.restore_rate_beta
    u: tuple = factors.prev_decision

    # Initialize levels with beta distribution.
    levels = np.zeros(stages)
    levels[0] = restore_model.random(rngs[1], init_alpha, init_beta)

    # Generate contamination and restoration values with beta distribution.
    rand_range = range(stages - 1)
    contamination_rates = [
        contam_model.random(rngs[0], contam_alpha, contam_beta) for _ in rand_range
    ]
    restoration_rates = [
        restore_model.random(rngs[1], restore_alpha, restore_beta) for _ in rand_range
    ]

    # Calculate contamination and restoration levels.
    # Start from stage 1; stage 0 was initialized separately.
    for i in range(1, stages):
        c = contamination_rates[i - 1]
        r = restoration_rates[i - 1]
        u_i = u[i]
        prev = levels[i - 1]

        contamination_change = c * (1 - u_i) * (1 - prev)
        restoration_change = (1 - r * u_i) * prev
        levels[i] = contamination_change + restoration_change
    return levels
