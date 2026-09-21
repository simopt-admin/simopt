"""Standalone Python simulation for dual-sourcing inventory."""

from __future__ import annotations

from random import Random
from typing import Annotated, Self

import numpy as np
from pydantic import BaseModel, Field, model_validator

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt._markers import input_model, simulation
from simopt.input_models import InputModel


class DualSourcingConfig(BaseModel):
    """Configuration model for Dual Sourcing Inventory simulation.

    A model that simulates multiple periods of ordering and sales for a single-staged,
    dual sourcing inventory problem with stochastic demand. Returns average holding
    cost, average penalty cost, and average ordering cost per period.
    """

    n_days: Annotated[
        int,
        Field(
            default=1000,
            description="number of days to simulate",
            ge=1,
            json_schema_extra={"isDatafarmable": False},
        ),
    ]
    initial_inv: Annotated[
        int,
        Field(
            default=40,
            description="initial inventory",
            ge=0,
        ),
    ]
    cost_reg: Annotated[
        float,
        Field(
            default=100.00,
            description="regular ordering cost per unit",
            gt=0,
        ),
    ]
    cost_exp: Annotated[
        float,
        Field(
            default=110.00,
            description="expedited ordering cost per unit",
            gt=0,
        ),
    ]
    lead_reg: Annotated[
        int,
        Field(
            default=2,
            description="lead time for regular orders in days",
            ge=0,
        ),
    ]
    lead_exp: Annotated[
        int,
        Field(
            default=0,
            description="lead time for expedited orders in days",
            ge=0,
        ),
    ]
    holding_cost: Annotated[
        float,
        Field(
            default=5.00,
            description="holding cost per unit per period",
            gt=0,
        ),
    ]
    penalty_cost: Annotated[
        float,
        Field(
            default=495.00,
            description="penalty cost per unit per period for backlogging",
            gt=0,
        ),
    ]
    st_dev: Annotated[
        float,
        Field(
            default=10.0,
            description="standard deviation of demand distribution",
            gt=0,
        ),
    ]
    mu: Annotated[
        float,
        Field(
            default=30.0,
            description="mean of demand distribution",
            gt=0,
        ),
    ]
    order_level_reg: Annotated[
        int,
        Field(
            default=80,
            description="order-up-to level for regular orders",
            ge=0,
        ),
    ]
    order_level_exp: Annotated[
        int,
        Field(
            default=50,
            description="order-up-to level for expedited orders",
            ge=0,
        ),
    ]

    @model_validator(mode="after")
    def _validate_model(self) -> Self:
        # Cross-validation: check lead time and cost constraints
        if (self.lead_exp > self.lead_reg) or (self.cost_exp < self.cost_reg):
            raise ValueError(
                "lead_exp must be less than lead_reg and cost_exp must be greater than cost_reg"
            )

        return self


@input_model
class DemandInputModel(InputModel):
    """Input model for daily demand."""

    def random(self, rng: Random, mu: float, sigma: float) -> int:
        """Draw a rounded nonnegative daily demand value."""

        def round_and_clamp_non_neg(x: float | int) -> int:
            return round(max(0.0, float(x)))

        return round_and_clamp_non_neg(rng.normalvariate(mu, sigma))


@simulation
def replicate(
    factors: DualSourcingConfig,
    rngs: list[MRG32k3a],
    demand_model: InputModel,
) -> tuple[np.floating, np.floating, np.floating]:
    """Return average ordering, penalty, and holding costs for one replication."""
    n_days: int = factors.n_days
    n_days_range = range(n_days)
    lead_reg: int = factors.lead_reg
    lead_exp: int = factors.lead_exp
    order_level_reg: int = factors.order_level_reg
    order_level_exp: int = factors.order_level_exp
    mu: float = factors.mu
    st_dev: float = factors.st_dev
    initial_inv: int = factors.initial_inv
    cost_exp: float = factors.cost_exp
    cost_reg: float = factors.cost_reg
    penalty_cost: float = factors.penalty_cost
    holding_cost: float = factors.holding_cost

    def round_and_clamp_non_neg(x: float | int) -> int:
        return round(max(0.0, float(x)))

    # Vectors of regular orders to be received in periods n through n + lr - 1.
    orders_reg: list[int] = [0] * lead_reg
    # Vectors of expedited orders to be received in periods n through n + le - 1.
    orders_exp: list[int] = [0] * lead_exp

    # Generate demand.
    demand = [demand_model.random(rngs[0], mu, st_dev) for _ in n_days_range]

    # Track total expenses.
    total_holding_cost = np.zeros(n_days)
    total_penalty_cost = np.zeros(n_days)
    total_ordering_cost = np.zeros(n_days)
    inv: int = initial_inv

    # Run simulation over time horizon.
    for day in n_days_range:
        # Calculate inventory positions.
        inv_order_exp_sum = inv + sum(orders_exp)
        inv_position_exp = round(inv_order_exp_sum + sum(orders_reg[:lead_exp]))
        inv_position_reg = round(inv_order_exp_sum + sum(orders_reg))
        # Calculate how much to order.
        order_exp: int = round_and_clamp_non_neg(
            order_level_exp - inv_position_exp - orders_reg[lead_exp]
        )
        orders_exp.append(order_exp)
        order_reg: int = round_and_clamp_non_neg(
            order_level_reg - inv_position_reg - orders_exp[lead_exp]
        )
        orders_reg.append(order_reg)
        # Charge ordering cost.
        daily_cost_exp = cost_exp * order_exp
        daily_cost_reg = cost_reg * order_reg
        total_ordering_cost[day] = daily_cost_exp + daily_cost_reg
        # Orders arrive, update on-hand inventory.
        inv += orders_exp.pop(0) + orders_reg.pop(0)
        # Satisfy or backorder demand.
        # dn = max(0, demand[day]) THIS IS DONE TWICE
        # inv = inv - dn
        inv -= demand[day]
        # Calculate holding and penalty costs.
        total_penalty_cost[day] = -penalty_cost * min(0, int(inv))
        total_holding_cost[day] = holding_cost * max(0, int(inv))

    return (
        np.mean(total_ordering_cost),
        np.mean(total_penalty_cost),
        np.mean(total_holding_cost),
    )
