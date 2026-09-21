"""Standalone Python simulation for iron-ore production and sales."""

from __future__ import annotations

from math import copysign, sqrt
from random import Random
from typing import Annotated, Self

import numpy as np
from pydantic import BaseModel, Field, model_validator

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt._markers import input_model, simulation
from simopt.input_models import InputModel


class IronOreConfig(BaseModel):
    """Configuration model for Iron Ore Inventory simulation.

    A model that simulates multiple periods of production and sales for an
    inventory problem with stochastic price determined by a mean-reverting
    random walk. Returns total profit, fraction of days producing iron, and
    mean stock.
    """

    mean_price: Annotated[
        float,
        Field(
            default=100.0,
            description="mean iron ore price per unit",
            gt=0,
        ),
    ]
    max_price: Annotated[
        float,
        Field(
            default=200.0,
            description="maximum iron ore price per unit",
            gt=0,
        ),
    ]
    min_price: Annotated[
        float,
        Field(
            default=0.0,
            description="minimum iron ore price per unit",
            ge=0,
        ),
    ]
    capacity: Annotated[
        int,
        Field(
            default=10000,
            description="maximum holding capacity",
            ge=0,
        ),
    ]
    st_dev: Annotated[
        float,
        Field(
            default=7.5,
            description="standard deviation of random walk steps for price",
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
    prod_cost: Annotated[
        float,
        Field(
            default=100.0,
            description="production cost per unit",
            gt=0,
        ),
    ]
    max_prod_perday: Annotated[
        int,
        Field(
            default=100,
            description="maximum units produced per day",
            gt=0,
        ),
    ]
    price_prod: Annotated[
        float,
        Field(
            default=80.0,
            description="price level to start production",
            gt=0,
        ),
    ]
    inven_stop: Annotated[
        int,
        Field(
            default=7000,
            description="inventory level to cease production",
            gt=0,
        ),
    ]
    price_stop: Annotated[
        float,
        Field(
            default=40.0,
            description="price level to stop production",
            gt=0,
        ),
    ]
    price_sell: Annotated[
        float,
        Field(
            default=100.0,
            description="price level to sell all stock",
            gt=0,
        ),
    ]
    n_days: Annotated[
        int,
        Field(
            default=365,
            description="number of days to simulate",
            ge=1,
            json_schema_extra={"isDatafarmable": False},
        ),
    ]

    @model_validator(mode="after")
    def _validate_model(self) -> Self:
        # Cross-validation: check price ordering constraint
        if (self.min_price > self.mean_price) or (self.mean_price > self.max_price):
            raise ValueError(
                "mean_price must be greater than or equal to min_price and less than "
                "or equal to max_price."
            )

        return self


@input_model
class MovementInputModel(InputModel):
    """Input model for mining movement and price shocks."""

    def random(self, rng: Random, mean: float, std: float) -> float:
        """Draw a normally distributed market-price movement."""
        return rng.normalvariate(mean, std)


@simulation
def replicate(
    factors: IronOreConfig,
    rngs: list[MRG32k3a],
    movement_model: InputModel,
) -> tuple[np.floating, np.floating, np.floating]:
    """Return profit, production rate, and mean stock for one replication."""
    n_days: int = factors.n_days
    min_price: float = factors.min_price
    mean_price: float = factors.mean_price
    max_price: float = factors.max_price
    st_dev: float = factors.st_dev
    price_stop: float = factors.price_stop
    inven_stop: int = factors.inven_stop
    max_prod_perday: int = factors.max_prod_perday
    capacity: int = factors.capacity
    prod_cost: float = factors.prod_cost
    price_prod: float = factors.price_prod
    price_sell: float = factors.price_sell
    holding_cost: float = factors.holding_cost
    # Initialize quantities to track:
    #   - Market price in each period (Pt).
    #   - Starting stock in each period.
    #   - Ending stock in each period.
    #   - Profit in each period.
    #   - Whether producing or not in each period.
    #   - Production in each period.
    mkt_price = np.zeros(n_days)
    mkt_price[0] = mean_price
    stock = np.zeros(n_days)
    prod_costs = np.zeros(n_days)
    hold_costs = np.zeros(n_days)
    sell_profit = np.zeros(n_days)

    # Run simulation over time horizon.
    for day in range(1, n_days):
        # === Initializatize values ===
        # Initialize today with values from yesterday
        prior_day = day - 1
        # Stock doesn't reset between days
        prev_stock = stock[prior_day]
        stock[day] = prev_stock
        # The market price is a random walk, but it's based off of the
        # previous day's price.
        prev_price = mkt_price[prior_day]
        mkt_price[day] = prev_price
        # We just need yesterday's producing status to help determine
        # if we should produce today.
        prev_producing = prod_costs[prior_day] != 0

        # === Price Update: mean-reverting random walk ===
        price_delta = mean_price - prev_price
        mean_move = copysign(sqrt(sqrt(abs(price_delta))), price_delta)
        move = movement_model.random(rngs[0], mean_move, st_dev)
        price_today = max(min(prev_price + move, max_price), min_price)
        mkt_price[day] = price_today

        # === Production Logic ===
        # If stock is below the inventory stop and either:
        # - if producing, price is above the price stop
        # - if not producing, price is above the price prod
        # then produce the maximum amount possible.
        if prev_stock < inven_stop and (
            (prev_producing and price_today >= price_stop)
            or (not prev_producing and price_today >= price_prod)
        ):
            missing_stock = capacity - prev_stock
            production_amount = min(max_prod_perday, missing_stock)
            stock[day] += production_amount
            prod_costs[day] = production_amount * prod_cost

        # === Selling Logic ===
        if price_today >= price_sell:
            sell_profit[day] = stock[day] * price_today
            stock[day] = 0

        # === Holding Cost ===
        hold_costs[day] = stock[day] * holding_cost

    # Calculate total profit
    profits = sell_profit - prod_costs - hold_costs
    net_profit = np.sum(profits)

    # Calculate fraction of days producing
    is_producing_mask = prod_costs != 0
    frac_producing = np.mean(is_producing_mask)

    return net_profit, frac_producing, np.mean(stock)
