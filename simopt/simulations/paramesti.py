"""Standalone Python simulation for gamma parameter estimation."""

from __future__ import annotations

import math
from typing import Annotated, Self

import numpy as np
from pydantic import BaseModel, Field, model_validator

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt._markers import simulation
from simopt.input_models import InputModel


class ParameterEstimationConfig(BaseModel):
    """Configuration for the parameter estimation model."""

    xstar: Annotated[
        list[float],
        Field(
            default=[2, 5],
            description="x^*, the unknown parameter that maximizes g(x)",
        ),
    ]
    x: Annotated[
        list[float],
        Field(
            default=[1, 1],
            description="x, variable in pdf",
        ),
    ]

    def _check_xstar(self) -> None:
        if any(xstar_i <= 0 for xstar_i in self.xstar):
            raise ValueError("All elements in xstar must be greater than 0.")

    def _check_x(self) -> None:
        if any(x_i <= 0 for x_i in self.x):
            raise ValueError("All elements in x must be greater than 0.")

    @model_validator(mode="after")
    def _validate_model(self) -> Self:
        self._check_xstar()
        self._check_x()

        x_len = len(self.x)
        xstar_len = len(self.xstar)
        if x_len != 2:
            raise ValueError("The length of x must equal 2.")
        if xstar_len != 2:
            raise ValueError("The length of xstar must equal 2.")

        return self


@simulation
def replicate(
    factors: ParameterEstimationConfig,
    rngs: list[MRG32k3a],
    y1_model: InputModel,
    y2_model: InputModel,
) -> float:
    """Return the gamma-model log likelihood for one replication."""
    xstar = factors.xstar
    x = factors.x
    # Generate y1 and y2 from specified gamma distributions using input models.
    # Outputs will be coupled when generating Y_j's.
    y2 = y2_model.random(rngs[0], xstar[1], 1)
    y1 = y1_model.random(rngs[1], xstar[0] * y2, 1)
    # Compute Log Likelihood
    loglik = (
        -y1
        - y2
        + (x[0] * y2 - 1) * np.log(y1)
        + (x[1] - 1) * np.log(y2)
        - np.log(math.gamma(x[0] * y2))
        - np.log(math.gamma(x[1]))
    )
    return loglik  # noqa: RET504
