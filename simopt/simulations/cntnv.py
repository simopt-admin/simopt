"""Standalone Python simulation for the continuous newsvendor model."""

from __future__ import annotations

from random import Random
from typing import Annotated, Self

import numpy as np
from pydantic import BaseModel, Field, model_validator

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt._markers import input_model, simulation
from simopt.input_models import InputModel


class CntNVConfig(BaseModel):
    """Configuration model for Continuous Newsvendor simulation.

    A model that simulates a day's worth of sales for a newsvendor with a Burr Type XII
    demand distribution. Returns the profit, after accounting for order costs and
    salvage.
    """

    purchase_price: Annotated[
        float,
        Field(
            default=5.0,
            description="purchasing cost per unit",
            gt=0,
        ),
    ]
    sales_price: Annotated[
        float,
        Field(
            default=9.0,
            description="sales price per unit",
            gt=0,
        ),
    ]
    salvage_price: Annotated[
        float,
        Field(
            default=1.0,
            description="salvage cost per unit",
            gt=0,
        ),
    ]
    order_quantity: Annotated[
        float,
        Field(
            default=0.5,
            description="order quantity",
            gt=0,
        ),
    ]
    burr_c: Annotated[
        float,
        Field(
            default=2.0,
            description="Burr Type XII cdf shape parameter",
            gt=0,
            alias="Burr_c",
        ),
    ]
    burr_k: Annotated[
        float,
        Field(
            default=20.0,
            description="Burr Type XII cdf shape parameter",
            gt=0,
            alias="Burr_k",
        ),
    ]

    @model_validator(mode="after")
    def _validate_model(self) -> Self:
        # Cross-validation: check price ordering constraint
        if self.salvage_price >= self.purchase_price:
            error_msg = (
                f"salvage_price ({self.salvage_price}) "
                "must be less than "
                f"purchase_price ({self.purchase_price})."
            )
            raise ValueError(error_msg)
        if self.purchase_price >= self.sales_price:
            error_msg = (
                f"purchase_price ({self.purchase_price}) "
                "must be less than "
                f"sales_price ({self.sales_price})."
            )
            raise ValueError(error_msg)

        return self


@input_model
class DemandInputModel(InputModel):
    """Input model for Burr Type XII demand."""

    def random(self, rng: Random, burr_c: float, burr_k: float) -> float:
        """Draw a Burr Type XII demand value."""

        # Generate random demand according to Burr Type XII distribution.
        # If U ~ Uniform(0,1) and the Burr Type XII has parameters c and k,
        #   X = ((1-U)**(-1/k) - 1)**(1/c) has the desired distribution.
        # https://en.wikipedia.org/wiki/Burr_distribution
        def nth_root(x: float, n: float) -> float:
            """Return the nth root of x."""
            return x ** (1 / n)

        u = rng.random()
        return nth_root(nth_root(1 - u, -burr_k) - 1, burr_c)


@simulation
def replicate(
    factors: CntNVConfig,
    rngs: list[MRG32k3a],
    demand_model: InputModel,
) -> tuple[float, float, int, float]:
    """Return profit, stockout measures, and the profit gradient for one replication."""
    ord_quant: float = factors.order_quantity
    purch_price: float = factors.purchase_price
    sales_price: float = factors.sales_price
    salvage_price: float = factors.salvage_price
    burr_k: float = factors.burr_k
    burr_c: float = factors.burr_c
    # Designate random number generator for demand variability.
    demand = demand_model.random(rngs[0], burr_c, burr_k)

    # Calculate units sold, as well as unsold/stockout
    units_sold = min(demand, ord_quant)
    order_diff = ord_quant - demand
    units_unsold = max(order_diff, 0)
    stockout_qty = max(-order_diff, 0)

    # Compute revenue and cost components
    order_cost = purch_price * ord_quant
    sales_revenue = units_sold * sales_price
    salvage_revenue = units_unsold * salvage_price

    # Build profit
    profit = sales_revenue + salvage_revenue - order_cost

    # Determine if there was a stockout.
    stockout = int(stockout_qty > 0)

    # Calculate gradient of profit w.r.t. order quantity.
    if order_diff < 0:
        grad_profit_order_quantity = sales_price - purch_price
    elif order_diff > 0:
        grad_profit_order_quantity = salvage_price - purch_price
    else:
        grad_profit_order_quantity = np.nan

    return profit, stockout_qty, stockout, grad_profit_order_quantity
