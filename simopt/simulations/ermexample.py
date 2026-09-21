"""Standalone Python simulation for the linear-regression ERM example."""

from __future__ import annotations

from typing import Annotated

import numpy as np
from pydantic import BaseModel, Field

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt._markers import input_model, simulation
from simopt.input_models import InputModel


class ERMExampleModelConfig(BaseModel):
    """Configuration model for ERMExample simulation.

    An empirical risk minimization model for linear regression.
    """

    beta: Annotated[
        tuple[float, ...],
        Field(
            default=(0.0, 0.0),
            description="(intercept, slope) coefficients",
        ),
    ]


@input_model
class FileInputModel(InputModel):
    """Input model that resamples observations from a NumPy data file."""

    def __init__(self, filename: str) -> None:
        """Load observations from the given file."""
        self.data = np.load(filename)

    def random(self, rng: MRG32k3a) -> tuple[float, float]:  # noqa: ARG002
        """Resample one observation from the loaded data."""
        n_rows = np.shape(self.data)[0]
        resample_idx = np.random.choice(n_rows, size=1, replace=True)
        resample_x = self.data[resample_idx, 0].item()
        resample_y = self.data[resample_idx, 1].item()
        return resample_x, resample_y


@simulation
def replicate(
    factors: ERMExampleModelConfig,
    rngs: list[MRG32k3a],
    resample_model: InputModel,
) -> tuple[float, tuple[float, float]]:
    """Return squared-error loss and its beta gradient for one observation."""
    beta0, beta1 = factors.beta
    x, y = resample_model.random(rngs[0])
    sq_error_loss = (y - beta0 - beta1 * x) ** 2
    error_loss = y - beta0 - beta1 * x
    # gradients wrt beta0 and beta1
    grad_sq_error_loss = (-2 * error_loss, -2 * x * error_loss)

    return sq_error_loss, grad_sq_error_loss
