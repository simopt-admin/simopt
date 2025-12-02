"""Simulate production and sales over multiple periods for an iron ore inventory."""

# Changed get_random_solution quantiles
#     from 10 and 200 => mean=59.887, sd=53.338, p(X>100)=0.146
#     to 10 and 1000 => mean=199.384, sd=343.925, p(X>100)=0.5

from __future__ import annotations

from math import copysign, sqrt
from random import Random
from typing import Annotated, ClassVar, Self

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


class IronOreMaxRevCntConfig(BaseModel):
    """Configuration model for Iron Ore Max Revenue Continuous Problem.

    Max Revenue for Continuous Iron Ore simulation-optimization problem.
    """

    initial_solution: Annotated[
        tuple[float, ...],
        Field(
            default=(80, 40, 100),
            description="initial solution",
        ),
    ]
    budget: Annotated[
        int,
        Field(
            default=1000,
            description="max # of replications for a solver to take",
            gt=0,
        ),
    ]


class IronOreMaxRevConfig(BaseModel):
    """Configuration model for Iron Ore Max Revenue Problem.

    Max Revenue for Iron Ore simulation-optimization problem.
    """

    initial_solution: Annotated[
        tuple[float, ...],
        Field(
            default=(80, 7000, 40, 100),
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


class MovementInputModel(InputModel):
    """Input model for mining movement and price shocks."""

    rng: Random | None = None

    def random(self, mean: float, std: float) -> float:  # noqa: D102
        assert self.rng is not None
        return self.rng.normalvariate(mean, std)


class IronOre(Model):
    """Iron Ore Inventory Model.

    A model that simulates multiple periods of production and sales for an
    inventory problem with stochastic price determined by a mean-reverting
    random walk. Returns total profit, fraction of days producing iron, and
    mean stock.
    """

    class_name_abbr: ClassVar[str] = "IRONORE"
    class_name: ClassVar[str] = "Iron Ore"
    config_class: ClassVar[type[BaseModel]] = IronOreConfig
    n_rngs: ClassVar[int] = 1
    n_responses: ClassVar[int] = 3

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize the Iron Ore Inventory Model.

        Args:
            fixed_factors (dict, optional): Fixed factors for the model.
                Defaults to None.
        """
        # Let the base class handle default arguments.
        super().__init__(fixed_factors)

        self.movement_model = MovementInputModel()

    def before_replicate(self, rng_list: list[MRG32k3a]) -> None:  # noqa: D102
        self.movement_model.set_rng(rng_list[0])

    def replicate(self) -> tuple[dict, dict]:
        """Simulate a single replication for the current model factors.

        Args:
            rng_list (list[MRG32k3a]): Random number generators used to simulate
                the replication.

        Returns:
            tuple[dict, dict]: A tuple containing:
                - responses (dict): Performance measures of interest, including:
                    - "total_profit": The total profit over the time period.
                    - "frac_producing": The fraction of days spent producing iron ore.
                    - "mean_stock": The average stock over the time period.
                - gradients (dict): A dictionary of gradient estimates for each
                    response.
        """
        n_days: int = self.factors["n_days"]
        min_price: float = self.factors["min_price"]
        mean_price: float = self.factors["mean_price"]
        max_price: float = self.factors["max_price"]
        st_dev: float = self.factors["st_dev"]
        price_stop: float = self.factors["price_stop"]
        inven_stop: int = self.factors["inven_stop"]
        max_prod_perday: int = self.factors["max_prod_perday"]
        capacity: int = self.factors["capacity"]
        prod_cost: float = self.factors["prod_cost"]
        price_prod: float = self.factors["price_prod"]
        price_sell: float = self.factors["price_sell"]
        holding_cost: float = self.factors["holding_cost"]
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
            move = self.movement_model.random(mean_move, st_dev)
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

        # Calculate responses from simulation data.
        responses = {
            "total_profit": net_profit,
            "frac_producing": frac_producing,
            "mean_stock": np.mean(stock),
        }
        return responses, {}


class IronOreMaxRev(Problem):
    """Class to make iron ore inventory simulation-optimization problems."""

    class_name_abbr: ClassVar[str] = "IRONORE-1"
    class_name: ClassVar[str] = "Max Revenue for Iron Ore"
    config_class: ClassVar[type[BaseModel]] = IronOreMaxRevConfig
    model_class: ClassVar[type[Model]] = IronOre
    n_objectives: ClassVar[int] = 1
    n_stochastic_constraints: ClassVar[int] = 0
    minmax: ClassVar[tuple[int, ...]] = (1,)
    constraint_type: ClassVar[ConstraintType] = ConstraintType.BOX
    variable_type: ClassVar[VariableType] = VariableType.MIXED
    gradient_available: ClassVar[bool] = False
    optimal_value: ClassVar[float | None] = None
    optimal_solution: tuple | None = None
    model_default_factors: ClassVar[dict] = {}
    model_decision_factors: ClassVar[set[str]] = {
        "price_prod",
        "inven_stop",
        "price_stop",
        "price_sell",
    }

    @property
    def dim(self) -> int:  # noqa: D102
        return 4

    @property
    def lower_bounds(self) -> tuple:  # noqa: D102
        return (0,) * self.dim

    @property
    def upper_bounds(self) -> tuple:  # noqa: D102
        return (np.inf,) * self.dim

    def vector_to_factor_dict(self, vector: tuple) -> dict:  # noqa: D102
        return {
            "price_prod": vector[0],
            "inven_stop": vector[1],
            "price_stop": vector[2],
            "price_sell": vector[3],
        }

    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:  # noqa: D102
        return (
            factor_dict["price_prod"],
            factor_dict["inven_stop"],
            factor_dict["price_stop"],
            factor_dict["price_sell"],
        )

    def replicate(self, _x: tuple) -> RepResult:  # noqa: D102
        responses, _ = self.model.replicate()
        objectives = [Objective(stochastic=responses["total_profit"])]
        return RepResult(objectives=objectives)

    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:  # noqa: D102
        # return (
        #     rand_sol_rng.randint(70, 90),
        #     rand_sol_rng.randint(2000, 8000),
        #     rand_sol_rng.randint(30, 50),
        #     rand_sol_rng.randint(90, 110),
        # )
        return (
            rand_sol_rng.lognormalvariate(10, 200),
            rand_sol_rng.lognormalvariate(1000, 10000),
            rand_sol_rng.lognormalvariate(10, 200),
            rand_sol_rng.lognormalvariate(10, 200),
        )


class IronOreMaxRevCnt(Problem):
    """Class to make iron ore inventory simulation-optimization problems."""

    class_name_abbr: ClassVar[str] = "IRONORECONT-1"
    class_name: ClassVar[str] = "Max Revenue for Continuous Iron Ore"
    config_class: ClassVar[type[BaseModel]] = IronOreMaxRevCntConfig
    model_class: ClassVar[type[Model]] = IronOre
    n_objectives: ClassVar[int] = 1
    n_stochastic_constraints: ClassVar[int] = 0
    minmax: ClassVar[tuple[int, ...]] = (1,)
    constraint_type: ClassVar[ConstraintType] = ConstraintType.BOX
    variable_type: ClassVar[VariableType] = VariableType.CONTINUOUS
    gradient_available: ClassVar[bool] = False
    optimal_value: ClassVar[float | None] = None
    optimal_solution: tuple | None = None
    model_default_factors: ClassVar[dict] = {}
    model_decision_factors: ClassVar[set[str]] = {
        "price_prod",
        "price_stop",
        "price_sell",
    }

    @property
    def dim(self) -> int:  # noqa: D102
        return 3

    @property
    def lower_bounds(self) -> tuple:  # noqa: D102
        return (0.0,) * self.dim

    @property
    def upper_bounds(self) -> tuple:  # noqa: D102
        return (np.inf,) * self.dim

    def vector_to_factor_dict(self, vector: tuple) -> dict:  # noqa: D102
        return {
            "price_prod": vector[0],
            "price_stop": vector[1],
            "price_sell": vector[2],
        }

    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:  # noqa: D102
        return (
            factor_dict["price_prod"],
            factor_dict["price_stop"],
            factor_dict["price_sell"],
        )

    def replicate(self, _x: tuple) -> RepResult:  # noqa: D102
        responses, _ = self.model.replicate()
        objectives = [Objective(stochastic=responses["total_profit"])]
        return RepResult(objectives=objectives)

    def check_deterministic_constraints(self, x: tuple) -> bool:  # noqa: D102
        return x[0] >= 0 and x[1] >= 0 and x[2] >= 0

    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:  # noqa: D102
        # return (
        #     rand_sol_rng.randint(70, 90),
        #     rand_sol_rng.randint(30, 50),
        #     rand_sol_rng.randint(90, 110),
        # )
        return (
            rand_sol_rng.lognormalvariate(10, 1000),
            rand_sol_rng.lognormalvariate(10, 1000),
            rand_sol_rng.lognormalvariate(10, 1000),
        )
