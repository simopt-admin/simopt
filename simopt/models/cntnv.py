"""Simulate a day's worth of sales for a newsvendor."""

from __future__ import annotations

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


class CntNVMaxProfitConfig(BaseModel):
    """Configuration model for Continuous Newsvendor Max Profit Problem.

    A problem configuration that maximizes profit for a continuous newsvendor
    by optimizing the order quantity.
    """

    initial_solution: Annotated[
        tuple[float, ...],
        Field(
            default=(0,),
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


class DemandInputModel(InputModel):
    """Input model for Burr Type XII demand."""

    rng: Random | None = None

    def random(self, burr_c: float, burr_k: float) -> float:  # noqa: D102
        # Generate random demand according to Burr Type XII distribution.
        # If U ~ Uniform(0,1) and the Burr Type XII has parameters c and k,
        #   X = ((1-U)**(-1/k) - 1)**(1/c) has the desired distribution.
        # https://en.wikipedia.org/wiki/Burr_distribution
        def nth_root(x: float, n: float) -> float:
            """Return the nth root of x."""
            return x ** (1 / n)

        assert self.rng is not None
        u = self.rng.random()
        return nth_root(nth_root(1 - u, -burr_k) - 1, burr_c)


class CntNV(Model):
    """Continuous Newsvendor Model with a Burr Type XII demand distribution.

    A model that simulates a day's worth of sales for a newsvendor with a Burr Type XII
    demand distribution. Returns the profit, after accounting for order costs and
    salvage.
    """

    class_name_abbr: ClassVar[str] = "CNTNEWS"
    class_name: ClassVar[str] = "Continuous Newsvendor"
    config_class: ClassVar[type[BaseModel]] = CntNVConfig
    n_rngs: ClassVar[int] = 1
    n_responses: ClassVar[int] = 1

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize the Continuous Newsvendor model.

        Args:
            fixed_factors (dict, optional): Fixed factors for the model.
                Defaults to None.
        """
        # Let the base class handle default arguments.
        super().__init__(fixed_factors)

        self.demand_model = DemandInputModel()

    def before_replicate(self, rng_list: list[MRG32k3a]) -> None:  # noqa: D102
        self.demand_model.set_rng(rng_list[0])

    def replicate(self) -> tuple[dict, dict]:
        """Simulate a single replication for the current model factors.

        Args:
            rng_list (list[MRG32k3a]): Random number generators used to simulate the
                replication.

        Returns:
            tuple[dict, dict]: A tuple containing:
                - responses (dict): Performance measures of interest, including:
                    - "profit": Profit in this scenario.
                    - "stockout_qty": Amount by which demand exceeded supply.
                    - "stockout": Whether there was unmet demand ("Y" or "N").
                - gradients (dict): Gradient estimates for each response.
        """
        ord_quant: float = self.factors["order_quantity"]
        purch_price: float = self.factors["purchase_price"]
        sales_price: float = self.factors["sales_price"]
        salvage_price: float = self.factors["salvage_price"]
        burr_k: float = self.factors["Burr_k"]
        burr_c: float = self.factors["Burr_c"]
        # Designate random number generator for demand variability.
        demand = self.demand_model.random(burr_c, burr_k)

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

        # Compose responses and gradients.
        responses = {
            "profit": profit,
            "stockout_qty": stockout_qty,
            "stockout": stockout,
        }
        gradients = {
            response_key: dict.fromkeys(self.specifications, np.nan)
            for response_key in responses
        }
        gradients["profit"]["order_quantity"] = grad_profit_order_quantity
        return responses, gradients


class CntNVMaxProfit(Problem):
    """Base class to implement simulation-optimization problems."""

    class_name_abbr: ClassVar[str] = "CNTNEWS-1"
    class_name: ClassVar[str] = "Max Profit for Continuous Newsvendor"
    config_class: ClassVar[type[BaseModel]] = CntNVMaxProfitConfig
    model_class: ClassVar[type[Model]] = CntNV
    n_objectives: ClassVar[int] = 1
    n_stochastic_constraints: ClassVar[int] = 0
    minmax: ClassVar[tuple[int, ...]] = (1,)
    constraint_type: ClassVar[ConstraintType] = ConstraintType.BOX
    variable_type: ClassVar[VariableType] = VariableType.CONTINUOUS
    gradient_available: ClassVar[bool] = True
    optimal_value: ClassVar[float | None] = None
    optimal_solution: tuple | None = None
    model_default_factors: ClassVar[dict] = {
        "purchase_price": 5.0,
        "sales_price": 9.0,
        "salvage_price": 1.0,
        "Burr_c": 2.0,
        "Burr_k": 20.0,
    }
    model_decision_factors: ClassVar[set[str]] = {"order_quantity"}

    @property
    def dim(self) -> int:  # noqa: D102
        return 1

    @property
    def lower_bounds(self) -> tuple:  # noqa: D102
        return (0,)

    @property
    def upper_bounds(self) -> tuple:  # noqa: D102
        return (np.inf,)

    def vector_to_factor_dict(self, vector: tuple) -> dict:  # noqa: D102
        return {"order_quantity": vector[0]}

    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:  # noqa: D102
        return (factor_dict["order_quantity"],)

    def replicate(self, _x: tuple) -> RepResult:  # noqa: D102
        responses, gradients = self.model.replicate()
        return RepResult(
            objectives=[
                Objective(
                    stochastic=responses["profit"],
                    stochastic_gradients=gradients["profit"]["order_quantity"],
                )
            ],
        )

    def check_deterministic_constraints(self, x: tuple) -> bool:  # noqa: D102
        return x[0] > 0

    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:  # noqa: D102
        # Generate an Exponential(rate = 1) r.v.
        return (rand_sol_rng.expovariate(1),)
