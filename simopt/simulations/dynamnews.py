"""Standalone Python simulation for the dynamic newsvendor model."""

from __future__ import annotations

import math
from random import Random
from typing import Annotated, Final, Self

import numpy as np
from pydantic import BaseModel, Field, model_validator

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt._markers import input_model, simulation
from simopt.input_models import InputModel

NUM_PRODUCTS: Final[int] = 10


class DynamNewsConfig(BaseModel):
    """Configuration model for Dynamic Newsvendor simulation.

    A model that simulates a day's worth of sales for a newsvendor
    with dynamic consumer substitution. Returns the profit and the
    number of products that stock out.
    """

    num_prod: Annotated[
        int,
        Field(
            default=NUM_PRODUCTS,
            description="number of products",
            gt=0,
        ),
    ]
    num_customer: Annotated[
        int,
        Field(
            default=30,
            description="number of customers",
            gt=0,
        ),
    ]
    c_utility: Annotated[
        list[float],
        Field(
            default_factory=lambda: [6 + j for j in range(NUM_PRODUCTS)],
            description="constant of each product's utility",
        ),
    ]
    mu: Annotated[
        float,
        Field(
            default=1.0,
            description="mu for calculating Gumbel random variable",
        ),
    ]
    init_level: Annotated[
        list[int],
        Field(
            default_factory=lambda: [3] * NUM_PRODUCTS,
            description="initial inventory level",
        ),
    ]
    price: Annotated[
        list[float],
        Field(
            default_factory=lambda: [9] * NUM_PRODUCTS,
            description="sell price of products",
        ),
    ]
    cost: Annotated[
        list[float],
        Field(
            default_factory=lambda: [5] * NUM_PRODUCTS,
            description="cost of products",
        ),
    ]

    def _check_c_utility(self) -> None:
        if len(self.c_utility) != self.num_prod:
            raise ValueError("The length of c_utility must be equal to num_prod.")

    def _check_init_level(self) -> None:
        if any(np.array(self.init_level) < 0) or (len(self.init_level) != self.num_prod):
            raise ValueError(
                "The length of init_level must be equal to num_prod and every element "
                "in init_level must be greater than or equal to zero."
            )

    def _check_price(self) -> None:
        if any(np.array(self.price) < 0) or (len(self.price) != self.num_prod):
            raise ValueError(
                "The length of price must be equal to num_prod and every element in "
                "price must be greater than or equal to zero."
            )

    def _check_cost(self) -> None:
        if any(np.array(self.cost) < 0) or (len(self.cost) != self.num_prod):
            raise ValueError(
                "The length of cost must be equal to num_prod and every element in "
                "cost must be greater than or equal to 0."
            )

    @model_validator(mode="after")
    def _validate_model(self) -> Self:
        self._check_c_utility()
        self._check_init_level()
        self._check_price()
        self._check_cost()

        # Cross-validation: check price > cost constraint
        if any(np.subtract(self.price, self.cost) < 0):
            raise ValueError(
                "Each element in price must be greater than its corresponding element in cost."
            )

        return self


@input_model
class Utility(InputModel):
    """Input model for customer utility sampling."""

    def _gumbelvariate(self, rng: Random, mu: float, beta: float) -> float:
        return mu - beta * math.log(-math.log(rng.random()))

    def random(
        self,
        rng: Random,
        mu: float,
        num_customer: int,
        num_prod: int,
        c_utility: list[float],
    ) -> np.ndarray:
        """Draw customer utilities for all products."""
        # Compute Gumbel rvs for the utility of the products.
        gumbel_mu = -mu * np.euler_gamma
        gumbel_beta = mu
        gumbel_flat = [
            self._gumbelvariate(rng, gumbel_mu, gumbel_beta) for _ in range(num_customer * num_prod)
        ]
        gumbel = np.reshape(gumbel_flat, (num_customer, num_prod))

        # Compute utility for each product and each customer.
        utility = np.zeros((num_customer, num_prod + 1))
        # Keep the first column of utility as 0, which indicates no purchase.
        utility[:, 1:] = np.array(c_utility) + gumbel
        return utility


@simulation
def replicate(
    factors: DynamNewsConfig,
    rngs: list[MRG32k3a],
    utility_model: InputModel,
) -> tuple[float, int, float, float]:
    """Return profit, stockout count, missed orders, and fill rate for one replication."""
    num_customer: int = factors.num_customer
    num_prod: int = factors.num_prod
    mu: float = factors.mu
    init_level: list = factors.init_level
    c_utility: list = factors.c_utility
    price: list = factors.price
    cost: list = factors.cost

    utility = utility_model.random(
        rngs[0],
        mu,
        num_customer,
        num_prod,
        c_utility,
    )

    # Initialize inventory.
    inventory = np.copy(init_level)
    itembought = np.zeros(num_customer)

    # Loop through customers
    for t in range(num_customer):
        # Figure out which producs are in stock
        instock = np.where(inventory > 0)[0]

        # If no products are in stock, no purchase is made.
        if len(instock) == 0:
            itembought[t] = 0
            continue

        # Shift indices to match utility (1-based product indices)
        utility_options = utility[t, instock + 1]

        # Pick index of max utility
        best_idx = np.argmax(utility_options)
        best_product = instock[best_idx] + 1

        # Record it and decrement inventory.
        itembought[t] = best_product
        inventory[best_product - 1] -= 1

    # Calculate profit.
    numsold = init_level - inventory
    total_sold = sum(numsold)
    revenue = numsold * np.array(price)
    costs = init_level * np.array(cost)
    profit = revenue - costs
    unmet_demand = num_customer - total_sold
    order_fill_rate = total_sold / num_customer

    return np.sum(profit), np.sum(inventory == 0), unmet_demand, order_fill_rate
