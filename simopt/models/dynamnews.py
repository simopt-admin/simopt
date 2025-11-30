"""Simulate sales for a newsvendor under dynamic consumer substitution."""

from __future__ import annotations

import math
from random import Random
from typing import Annotated, ClassVar, Final, Self

import numpy as np
from pydantic import BaseModel, Field, model_validator

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.base import (
    ConstraintType,
    Model,
    Objective,
    Problem,
    RepResult,
    VariableType,
)
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
        if any(np.array(self.init_level) < 0) or (
            len(self.init_level) != self.num_prod
        ):
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
                "Each element in price must be greater than its corresponding element "
                "in cost."
            )

        return self


class DynamNewsMaxProfitConfig(BaseModel):
    """Configuration model for Dynamic Newsvendor Max Profit Problem.

    A problem configuration that maximizes profit for a dynamic newsvendor
    with consumer substitution by optimizing initial inventory levels.
    """

    initial_solution: Annotated[
        tuple[int, ...],
        Field(
            default_factory=lambda: (3,) * NUM_PRODUCTS,
            description="initial solution",
        ),
    ]
    budget: Annotated[
        int,
        Field(
            default=1000,
            description="max # of replications for a solver to take",
            gt=0,
            json_schema_extra={"isDatafarmable": False},
        ),
    ]


class Utility(InputModel):
    """Input model for customer utility sampling."""

    rng: Random | None = None

    def _gumbelvariate(self, mu: float, beta: float) -> float:
        assert self.rng is not None
        return mu - beta * math.log(-math.log(self.rng.random()))

    def random(  # noqa: D102
        self,
        mu: float,
        num_customer: int,
        num_prod: int,
        c_utility: list[float],
    ) -> np.ndarray:
        # Compute Gumbel rvs for the utility of the products.
        gumbel_mu = -mu * np.euler_gamma
        gumbel_beta = mu
        gumbel_flat = [
            self._gumbelvariate(gumbel_mu, gumbel_beta)
            for _ in range(num_customer * num_prod)
        ]
        gumbel = np.reshape(gumbel_flat, (num_customer, num_prod))

        # Compute utility for each product and each customer.
        utility = np.zeros((num_customer, num_prod + 1))
        # Keep the first column of utility as 0, which indicates no purchase.
        utility[:, 1:] = np.array(c_utility) + gumbel
        return utility


class DynamNews(Model):
    """Dynamic Newsvendor Model.

    A model that simulates a day's worth of sales for a newsvendor
    with dynamic consumer substitution. Returns the profit and the
    number of products that stock out.
    """

    class_name_abbr: ClassVar[str] = "DYNAMNEWS"
    class_name: ClassVar[str] = "Dynamic Newsvendor"
    config_class: ClassVar[type[BaseModel]] = DynamNewsConfig
    n_rngs: ClassVar[int] = 1
    n_responses: ClassVar[int] = 4

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize the model.

        Args:
            fixed_factors (dict, optional): Fixed factors for the model.
                Defaults to None.
        """
        # Let the base class handle default arguments.
        super().__init__(fixed_factors)

        self.utility_model = Utility()

    def before_replicate(self, rng_list: list[MRG32k3a]) -> None:  # noqa: D102
        self.utility_model.set_rng(rng_list[0])

    def replicate(self) -> tuple[dict, dict]:
        """Simulate a single replication for the current model factors.

        Args:
            rng_list (list[MRG32k3a]): Random number generators used to simulate
                the replication.

        Returns:
            tuple[dict, dict]: A tuple containing:
                - responses (dict): Performance measures of interest, including:
                    - "profit": Profit in this scenario.
                    - "n_prod_stockout": Number of products that are out of stock.
                    - "n_missed_orders": Number of unmet customer orders.
                    - "fill_rate": Fraction of customer orders fulfilled.
                - gradients (dict): A dictionary of gradient estimates for
                    each response.
        """
        num_customer: int = self.factors["num_customer"]
        num_prod: int = self.factors["num_prod"]
        mu: float = self.factors["mu"]
        init_level: list = self.factors["init_level"]
        c_utility: list = self.factors["c_utility"]
        price: list = self.factors["price"]
        cost: list = self.factors["cost"]

        utility = self.utility_model.random(
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

        # Compose responses and gradients.
        responses = {
            "profit": np.sum(profit),
            "n_prod_stockout": np.sum(inventory == 0),
            "n_missed_orders": unmet_demand,
            "fill_rate": order_fill_rate,
        }
        return responses, {}


class DynamNewsMaxProfit(Problem):
    """Base class to implement simulation-optimization problems."""

    class_name_abbr: ClassVar[str] = "DYNAMNEWS-1"
    class_name: ClassVar[str] = "Max Profit for Dynamic Newsvendor"
    config_class: ClassVar[type[BaseModel]] = DynamNewsMaxProfitConfig
    model_class: ClassVar[type[Model]] = DynamNews
    n_objectives: ClassVar[int] = 1
    n_stochastic_constraints: ClassVar[int] = 0
    minmax: ClassVar[tuple[int, ...]] = (1,)
    constraint_type: ClassVar[ConstraintType] = ConstraintType.BOX
    variable_type: ClassVar[VariableType] = VariableType.CONTINUOUS
    gradient_available: ClassVar[bool] = False
    optimal_value: ClassVar[float | None] = None
    optimal_solution: tuple | None = None
    model_default_factors: ClassVar[dict] = {}
    model_decision_factors: ClassVar[set[str]] = {"init_level"}

    @property
    def dim(self) -> int:  # noqa: D102
        return self.model.factors["num_prod"]

    @property
    def lower_bounds(self) -> tuple:  # noqa: D102
        return (0,) * self.dim

    @property
    def upper_bounds(self) -> tuple:  # noqa: D102
        return (np.inf,) * self.dim

    def vector_to_factor_dict(self, vector: tuple) -> dict:  # noqa: D102
        return {"init_level": vector[:]}

    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:  # noqa: D102
        return tuple(factor_dict["init_level"])

    def replicate(self, _x: tuple) -> RepResult:  # noqa: D102
        responses, _ = self.model.replicate()
        objectives = [Objective(stochastic=responses["profit"])]
        return RepResult(objectives=objectives)

    def check_deterministic_constraints(self, x: tuple) -> bool:  # noqa: D102
        return all(x[j] > 0 for j in range(self.dim))

    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:  # noqa: D102
        return tuple([rand_sol_rng.uniform(0, 10) for _ in range(self.dim)])
