"""Standalone Python simulation for facility sizing."""

from __future__ import annotations

from random import Random
from typing import Annotated, Final, Self

import numpy as np
from pydantic import BaseModel, Field, model_validator

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt._markers import input_model, simulation
from simopt.input_models import InputModel

NUM_FACILITIES: Final[int] = 3


class FacilitySizeConfig(BaseModel):
    """Configuration model for Facility Sizing simulation.

    A model that simulates a facility size problem with a multi-variate normal
    distribution. Returns the probability of violating demand in each scenario.
    """

    mean_vec: Annotated[
        list[float],
        Field(
            default_factory=lambda: [100] * NUM_FACILITIES,
            description=("location parameters of the multivariate normal distribution"),
        ),
    ]
    cov: Annotated[
        list[list[float]],
        Field(
            default_factory=lambda: [
                [2000, 1500, 500],
                [1500, 2000, 750],
                [500, 750, 2000],
            ],
            description="covariance of multivariate normal distribution",
        ),
    ]
    capacity: Annotated[
        list[float],
        Field(
            default=[150, 300, 400],
            description="capacity",
        ),
    ]
    n_fac: Annotated[
        int,
        Field(
            default=NUM_FACILITIES,
            description="number of facilities",
            gt=0,
            json_schema_extra={"isDatafarmable": False},
        ),
    ]

    def _check_mean_vec(self) -> None:
        if any(mean <= 0 for mean in self.mean_vec):
            raise ValueError("All elements in mean_vec must be greater than 0.")

    def _check_cov(self) -> None:
        try:
            np.linalg.cholesky(np.array(self.cov))
        except np.linalg.LinAlgError as err:
            if "Matrix is not positive definite" in str(err):
                raise ValueError("Covariance matrix is not positive definite.") from err

    def _check_capacity(self) -> None:
        if len(self.capacity) != self.n_fac:
            raise ValueError("The length of capacity must equal n_fac.")

    @model_validator(mode="after")
    def _validate_model(self) -> Self:
        self._check_mean_vec()
        self._check_cov()
        self._check_capacity()

        # Cross-validation: check dimensions match n_fac
        if len(self.capacity) != self.n_fac:
            raise ValueError("The length of capacity must be equal to n_fac.")
        if len(self.mean_vec) != self.n_fac:
            raise ValueError("The length of mean_vec must be equal to n_fac.")
        if len(self.cov) != self.n_fac:
            raise ValueError("The length of cov must be equal to n_fac.")
        if len(self.cov[0]) != self.n_fac:
            raise ValueError("The length of cov[0] must be equal to n_fac.")

        return self


@input_model
class DemandInputModel(InputModel):
    """Input model for multivariate normal demand at facilities."""

    def _mvnormalvariate(
        self,
        rng: Random,
        mean_vec: np.ndarray,
        cov: np.ndarray,
        factorized: bool = False,
    ) -> np.ndarray:
        chol = np.linalg.cholesky(cov) if not factorized else cov
        observations = [rng.normalvariate(0, 1) for _ in range(len(cov))]
        return np.dot(chol, observations).transpose() + mean_vec

    def random(self, rng: Random, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """Draw a nonnegative multivariate-normal demand vector."""
        while True:
            demand = np.array(self._mvnormalvariate(rng, mean, cov))
            if np.all(demand >= 0):
                return demand


@simulation
def replicate(
    factors: FacilitySizeConfig,
    rngs: list[MRG32k3a],
    demand_model: InputModel,
) -> tuple[int, int, int]:
    """Return stockout flag, number of stockouts, and unmet demand for one replication."""
    mean_vec = np.array(factors.mean_vec)
    cov = np.array(factors.cov)
    capacity = np.array(factors.capacity)
    demand = demand_model.random(rngs[0], mean_vec, cov)
    extra_demand = demand - capacity
    pos_excess_mask = extra_demand > 0
    n_fac_stockout = np.sum(pos_excess_mask).astype(int)
    n_cut = np.sum(extra_demand[pos_excess_mask]).astype(int)
    return int(n_fac_stockout > 0), n_fac_stockout, n_cut
