"""Standalone Python simulation for the (s, S) continuous inventory model."""

from __future__ import annotations

from typing import Annotated, Self

import numpy as np
from pydantic import BaseModel, Field, model_validator

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt._markers import simulation
from simopt.input_models import InputModel


class SSContConfig(BaseModel):
    """Configuration for the (s, S) continuous inventory model."""

    demand_mean: Annotated[
        float,
        Field(
            default=100.0,
            description="mean of exponentially distributed demand in each period",
            gt=0,
        ),
    ]
    lead_mean: Annotated[
        float,
        Field(
            default=6.0,
            description="mean of Poisson distributed order lead time",
            gt=0,
        ),
    ]
    backorder_cost: Annotated[
        float,
        Field(
            default=4.0,
            description="cost per unit of demand not met with in-stock inventory",
            gt=0,
        ),
    ]
    holding_cost: Annotated[
        float,
        Field(
            default=1.0,
            description="holding cost per unit per period",
            gt=0,
        ),
    ]
    fixed_cost: Annotated[
        float,
        Field(
            default=36.0,
            description="order fixed cost",
            gt=0,
        ),
    ]
    variable_cost: Annotated[
        float,
        Field(
            default=2.0,
            description="order variable cost per unit",
            gt=0,
        ),
    ]
    s: Annotated[
        float,
        Field(
            default=1000.0,
            description="inventory threshold for placing order",
            gt=0,
        ),
    ]
    S: Annotated[
        float,
        Field(
            default=2000.0,
            description="max inventory",
            gt=0,
        ),
    ]
    n_days: Annotated[
        int,
        Field(
            default=100,
            description="number of periods to simulate",
            ge=1,
            json_schema_extra={"isDatafarmable": False},
        ),
    ]
    warmup: Annotated[
        int,
        Field(
            default=20,
            description="number of periods as warmup before collecting statistics",
            ge=0,
            json_schema_extra={"isDatafarmable": False},
        ),
    ]

    @model_validator(mode="after")
    def _validate_model(self) -> Self:
        if self.s >= self.S:
            raise ValueError("s must be less than S.")
        return self


@simulation
def replicate(
    factors: SSContConfig,
    rngs: list[MRG32k3a],
    demand_model: InputModel,
    lead_model: InputModel,
) -> tuple[float, float, float, float, float, float, float, float]:
    """Simulate one replication using the supplied demand and lead-time samplers.

    Returns average backorder, order, and holding costs; on-time, order, and
    stockout rates; and average stockout and order amounts conditional on
    those events occurring.
    """
    demand_mean = factors.demand_mean
    n_days = factors.n_days
    warmup = factors.warmup
    fac_s = factors.s
    fac_S = factors.S  # noqa: N806
    lead_mean = factors.lead_mean
    fixed_cost = factors.fixed_cost
    variable_cost = factors.variable_cost
    holding_cost = factors.holding_cost
    backorder_cost = factors.backorder_cost

    periods = n_days + warmup
    # Generate exponential random demands.
    inv_demand_mean = 1 / demand_mean
    demands = np.array([demand_model.random(rngs[0], inv_demand_mean) for _ in range(periods)])
    # Initialize starting and ending inventories for each period.
    start_inv = np.zeros(periods)
    start_inv[0] = fac_s  # Start with s units at period 0.
    end_inv = np.zeros(periods)
    # Initialize other quantities to track:
    #   - Amount of product to be received in each period.
    #   - Inventory position each period.
    #   - Amount of product ordered in each period.
    #   - Amount of product outstanding in each period.
    orders_received = np.zeros(periods)
    inv_pos = np.zeros(periods)
    orders_placed = np.zeros(periods)
    orders_outstanding = np.zeros(periods)
    # Run simulation over time horizon.
    for day in range(periods):
        next_day = day + 1

        # Inventory position
        end_inv[day] = start_inv[day] - demands[day]
        inv_pos[day] = end_inv[day] + orders_outstanding[day]

        if inv_pos[day] < fac_s:
            order_qty = fac_S - inv_pos[day]
            orders_placed[day] = order_qty

            lead = lead_model.random(rngs[1], lead_mean)
            delivery_day = next_day + lead

            if delivery_day < periods:
                orders_received[delivery_day] += order_qty

            # Track future outstanding orders
            if next_day < periods:
                orders_outstanding[next_day : min(delivery_day, periods)] += order_qty

        if next_day < periods:
            start_inv[next_day] = end_inv[day] + orders_received[next_day]

    # Calculate responses from simulation data.
    orders_post_warmup = orders_placed[warmup:]
    pos_orders_post_warmup_mask = orders_post_warmup > 0
    inv_post_warmup = end_inv[warmup:]
    neg_inv_post_warmup_mask = inv_post_warmup < 0
    pos_inv_post_warmup_mask = inv_post_warmup > 0

    order_rate = np.mean(pos_orders_post_warmup_mask)
    stockout_rate = np.mean(neg_inv_post_warmup_mask)

    fixed_costs = fixed_cost * pos_orders_post_warmup_mask
    variable_costs = variable_cost * orders_post_warmup
    avg_order_costs = np.mean(fixed_costs + variable_costs)

    avg_holding_costs = np.mean(holding_cost * inv_post_warmup * pos_inv_post_warmup_mask)
    demands_post_warmup = demands[warmup:]
    demand_start_inv_diff = demands_post_warmup - start_inv[warmup:]

    shortage = np.minimum(demands_post_warmup, demand_start_inv_diff)
    shortage[demand_start_inv_diff <= 0] = 0
    on_time_rate = 1 - shortage.sum() / np.sum(demands_post_warmup)

    avg_backorder_costs = backorder_cost * (1 - on_time_rate) * np.sum(demands_post_warmup) / n_days
    # Calculate average stockout costs.
    neg_inv_post_warmup_mask = np.where(neg_inv_post_warmup_mask)
    if len(neg_inv_post_warmup_mask[0]) == 0:
        avg_stockout = 0
    else:
        avg_stockout = -np.mean(inv_post_warmup[neg_inv_post_warmup_mask])
    # Calculate average backorder costs.
    pos_orders_placed_post_warmup = np.where(pos_orders_post_warmup_mask)
    if len(pos_orders_placed_post_warmup[0]) == 0:
        avg_order = 0
    else:
        avg_order = np.mean(orders_post_warmup[pos_orders_placed_post_warmup])
    return (
        float(avg_backorder_costs),
        float(avg_order_costs),
        float(avg_holding_costs),
        float(on_time_rate),
        float(order_rate),
        float(stockout_rate),
        float(avg_stockout),
        float(avg_order),
    )
