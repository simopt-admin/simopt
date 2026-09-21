"""Standalone Python simulation for revenue management with dependent demand."""

from __future__ import annotations

from typing import Annotated, Self

import numpy as np
from pydantic import BaseModel, Field, model_validator

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt._markers import input_model, simulation
from simopt.input_models import InputModel


class RMITDConfig(BaseModel):
    """Configuration for the RMITD model."""

    time_horizon: Annotated[
        int,
        Field(
            default=3,
            description="time horizon",
            gt=0,
        ),
    ]
    prices: Annotated[
        list[float],
        Field(
            default=[100, 300, 400],
            description="prices for each period",
        ),
    ]
    demand_means: Annotated[
        list[float],
        Field(
            default=[50, 20, 30],
            description="mean demand for each period",
        ),
    ]
    cost: Annotated[
        float,
        Field(
            default=80.0,
            description="cost per unit of capacity at t = 0",
            gt=0,
        ),
    ]
    gamma_shape: Annotated[
        float,
        Field(
            default=1.0,
            description="shape parameter of gamma distribution",
            gt=0,
        ),
    ]
    gamma_scale: Annotated[
        float,
        Field(
            default=1.0,
            description="scale parameter of gamma distribution",
            gt=0,
        ),
    ]
    initial_inventory: Annotated[
        int,
        Field(
            default=100,
            description="initial inventory",
            gt=0,
        ),
    ]
    reservation_qtys: Annotated[
        list[int],
        Field(
            default=[50, 30],
            description="inventory to reserve going into periods 2, 3, ..., T",
        ),
    ]

    def _check_prices(self) -> None:
        if any(price <= 0 for price in self.prices):
            raise ValueError("All elements in prices must be greater than 0.")

    def _check_demand_means(self) -> None:
        if any(demand_mean <= 0 for demand_mean in self.demand_means):
            raise ValueError("All elements in demand_means must be greater than 0.")

    def _check_reservation_qtys(self) -> None:
        if any(reservation_qty <= 0 for reservation_qty in self.reservation_qtys):
            raise ValueError("All elements in reservation_qtys must be greater than 0.")

    @model_validator(mode="after")
    def _validate_model(self) -> Self:
        self._check_prices()
        self._check_demand_means()
        self._check_reservation_qtys()

        if len(self.prices) != self.time_horizon:
            raise ValueError("The length of prices must be equal to time_horizon.")
        if len(self.demand_means) != self.time_horizon:
            raise ValueError("The length of demand_means must be equal to time_horizon.")
        if len(self.reservation_qtys) != self.time_horizon - 1:
            raise ValueError(
                "The length of reservation_qtys must be equal to the time_horizon minus 1."
            )

        if self.initial_inventory < self.reservation_qtys[0]:
            raise ValueError(
                "The initial_inventory must be greater than or equal to the first "
                "element in reservation_qtys."
            )

        if any(
            self.reservation_qtys[idx] < self.reservation_qtys[idx + 1]
            for idx in range(self.time_horizon - 2)
        ):
            raise ValueError(
                "Each value in reservation_qtys must be greater than the next value in the list."
            )

        if not np.isclose(self.gamma_shape * self.gamma_scale, 1):
            raise ValueError("gamma_shape times gamma_scale should be close to 1.")

        return self


@input_model
class DemandInputModel(InputModel):
    """Input model for temporally dependent demand components."""

    def random(
        self,
        rngs: list[MRG32k3a],
        demand_means: np.ndarray,
        gamma_shape: float,
        gamma_scale: float,
    ) -> np.ndarray:
        """Draw a temporally dependent demand vector."""
        x_demand = rngs[0].gammavariate(
            alpha=gamma_shape,
            beta=1.0 / gamma_scale,
        )
        y_demand = np.array([rngs[1].expovariate(1) for _ in range(len(demand_means))])
        return demand_means * x_demand * y_demand


@simulation
def replicate(
    factors: RMITDConfig,
    rngs: list[MRG32k3a],
    demand_model: DemandInputModel,
) -> float:
    """Return total revenue for one replication."""
    gamma_shape = factors.gamma_shape
    gamma_scale = factors.gamma_scale
    initial_inventory = factors.initial_inventory
    reservation_qtys: list = factors.reservation_qtys
    demand_means = np.array(factors.demand_means)
    prices = factors.prices
    cost = factors.cost
    # Generate X and Y (to use for computing demand).
    # random.gammavariate takes two inputs: alpha and beta.
    #     alpha = k = gamma_shape
    #     beta = 1/theta = 1/gamma_scale
    reservations = [*reservation_qtys, 0]
    demand_vec = demand_model.random(rngs, demand_means, gamma_shape, gamma_scale)

    # Set initial inventory and revenue
    remaining_inventory = initial_inventory
    revenue = 0.0

    # Compute revenue for each period.
    for reservation, demand, price in zip(reservations, list(demand_vec), prices, strict=False):
        available = max(remaining_inventory - reservation, 0)
        sell = min(available, demand)
        remaining_inventory -= sell
        revenue += sell * price

    revenue -= cost * initial_inventory

    return revenue
