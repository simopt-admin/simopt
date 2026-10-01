"""Simulate a day's worth of sales for a newsvendor."""

from __future__ import annotations

from typing import Annotated, ClassVar

import numpy as np
from pydantic import BaseModel, Field

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt import dsl
from simopt.base import (
    ConstraintType,
    Model,
    Problem,
    VariableType,
)
from simopt.simulations.cntnv import CntNVConfig, DemandInputModel, replicate
from simopt.utils import override


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


class CntNV(Model[CntNVConfig]):
    """Continuous Newsvendor Model with a Burr Type XII demand distribution.

    A model that simulates a day's worth of sales for a newsvendor with a Burr Type XII
    demand distribution. Returns the profit, after accounting for order costs and
    salvage.
    """

    class_name_abbr: ClassVar[str] = "CNTNEWS"
    class_name: ClassVar[str] = "Continuous Newsvendor"
    config_class: ClassVar[type[CntNVConfig]] = CntNVConfig
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

    def replicate(self, factors: CntNVConfig, rngs: list[MRG32k3a]) -> tuple[dict, dict]:
        """Simulate a single replication for the current model factors.

        Args:
            rngs (list[MRG32k3a]): Random number generators used to simulate the
                replication.

        Returns:
            tuple[dict, dict]: A tuple containing:
                - responses (dict): Performance measures of interest, including:
                    - "profit": Profit in this scenario.
                    - "stockout_qty": Amount by which demand exceeded supply.
                    - "stockout": Whether there was unmet demand ("Y" or "N").
                - gradients (dict): Gradient estimates for each response.
        """
        profit, stockout_qty, stockout, grad_profit_order_quantity = replicate(
            factors, rngs, self.demand_model
        )

        # Compose responses and gradients.
        responses = {
            "profit": profit,
            "stockout_qty": stockout_qty,
            "stockout": stockout,
        }
        gradients = {
            response_key: dict.fromkeys(self.specifications, np.nan) for response_key in responses
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

    @override
    def build(self) -> dsl.Model:
        problem = dsl.Model()
        order_quantity = problem.add_continuous_variable(
            lb=0.0, ub=np.inf, initial=self.factors["initial_solution"][0]
        )
        simulation = self.add_simulation(problem, {"order_quantity": order_quantity})
        problem.maximize(dsl.mean(simulation.metric("profit")))
        return problem

    def vector_to_factor_dict(self, vector: tuple) -> dict:
        return {"order_quantity": vector[0]}

    def check_deterministic_constraints(self, x: tuple) -> bool:
        return x[0] > 0

    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
        # Generate an Exponential(rate = 1) r.v.
        return (rand_sol_rng.expovariate(1),)
