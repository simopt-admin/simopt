"""Simulate sales for a (s,S) inventory problem with continuous inventory."""

from __future__ import annotations

from math import sqrt
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
from simopt.input_models import Exp, Poisson
from simopt.simulations.sscont import SSContConfig, replicate
from simopt.utils import override


class SSContMinCostConfig(BaseModel):
    """Configuration model for SSCont Min Cost Problem.

    Min Total Cost for (s, S) Inventory simulation-optimization problem.
    """

    initial_solution: Annotated[
        tuple[float, ...],
        Field(
            default=(600, 600),
            description="initial solution from which solvers start",
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


class SSCont(Model):
    """(s,S) Inventory Simulation Model.

    A model that simulates multiple periods' worth of sales for a (s,S)
    inventory problem with continuous inventory, exponentially distributed
    demand, and poisson distributed lead time. Returns the various types of
    average costs per period, order rate, stockout rate, fraction of demand
    met with inventory on hand, average amount backordered given a stockout
    occured, and average amount ordered given an order occured.
    """

    class_name_abbr: ClassVar[str] = "SSCONT"
    class_name: ClassVar[str] = "(s, S) Inventory"
    config_class: ClassVar[type[BaseModel]] = SSContConfig
    n_rngs: ClassVar[int] = 2
    n_responses: ClassVar[int] = 7

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize the (s,S) inventory simulation model.

        Args:
            fixed_factors (dict, optional): Fixed factors of the simulation model.
                Defaults to None.
        """
        # Let the base class handle default arguments.
        super().__init__(fixed_factors)

        self.demand_model = Exp()
        self.lead_model = Poisson()

    def replicate(self, factors: SSContConfig, rngs: list[MRG32k3a]) -> tuple[dict, dict]:
        """Simulate a single replication for the current model factors.

        Args:
            rngs (list[MRG32k3a]): Random number generators used to simulate
                the replication.

        Returns:
            tuple[dict, dict]: A tuple containing:
                - responses (dict): Performance measures of interest, including:
                    - "avg_backorder_costs": Average backorder costs per period.
                    - "avg_order_costs": Average order costs per period.
                    - "avg_holding_costs": Average holding costs per period.
                    - "on_time_rate": Fraction of demand met with stock on hand
                        in store.
                    - "order_rate": Fraction of periods in which an order was made.
                    - "stockout_rate": Fraction of periods with a stockout.
                    - "avg_stockout": Mean amount of product backordered given a
                        stockout occurred.
                    - "avg_order": Mean amount of product ordered given an
                        order occurred.
                - gradients (dict): A dictionary of gradient estimates for
                    each response.
        """
        (
            avg_backorder_costs,
            avg_order_costs,
            avg_holding_costs,
            on_time_rate,
            order_rate,
            stockout_rate,
            avg_stockout,
            avg_order,
        ) = replicate(factors, rngs, self.demand_model, self.lead_model)
        # Compose responses and gradients.
        responses = {
            "avg_backorder_costs": avg_backorder_costs,
            "avg_order_costs": avg_order_costs,
            "avg_holding_costs": avg_holding_costs,
            "on_time_rate": on_time_rate,
            "order_rate": order_rate,
            "stockout_rate": stockout_rate,
            "avg_stockout": avg_stockout,
            "avg_order": avg_order,
        }
        return responses, {}


class SSContMinCost(Problem):
    """Class to make (s,S) inventory simulation-optimization problems."""

    class_name_abbr: ClassVar[str] = "SSCONT-1"
    class_name: ClassVar[str] = "Min Total Cost for (s, S) Inventory"
    config_class: ClassVar[type[BaseModel]] = SSContMinCostConfig
    model_class: ClassVar[type[Model]] = SSCont
    n_objectives: ClassVar[int] = 1
    n_stochastic_constraints: ClassVar[int] = 0
    minmax: ClassVar[tuple[int, ...]] = (-1,)
    constraint_type: ClassVar[ConstraintType] = ConstraintType.BOX
    variable_type: ClassVar[VariableType] = VariableType.CONTINUOUS
    gradient_available: ClassVar[bool] = False
    optimal_value: ClassVar[float | None] = None
    optimal_solution: tuple | None = None
    model_default_factors: ClassVar[dict] = {"demand_mean": 100.0, "lead_mean": 6.0}
    model_decision_factors: ClassVar[set[str]] = {"s", "S"}

    @override
    def build(self) -> dsl.Model:
        problem = dsl.Model()
        reorder_point = problem.add_continuous_variable(
            lb=0.0, ub=np.inf, initial=self.factors["initial_solution"][0]
        )
        order_gap = problem.add_continuous_variable(
            lb=0.0, ub=np.inf, initial=self.factors["initial_solution"][1]
        )

        simulation = self.add_simulation(problem, {"s": reorder_point, "order_gap": order_gap})
        problem.minimize(
            dsl.mean(
                simulation.metric("avg_backorder_costs")
                + simulation.metric("avg_order_costs")
                + simulation.metric("avg_holding_costs")
            )
        )
        return problem

    def vector_to_factor_dict(self, vector: tuple) -> dict:
        return {"s": vector[0], "S": vector[0] + vector[1]}

    def check_deterministic_constraints(self, x: tuple) -> bool:
        return x[0] >= 0 and x[1] >= 0

    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
        # x = (rand_sol_rng.expovariate(1 / 300), rand_sol_rng.expovariate(1 / 300))
        # x = tuple(
        #     sorted(
        #         [
        #             rand_sol_rng.lognormalvariate(600, 1),
        #             rand_sol_rng.lognormalvariate(600, 1),
        #         ],
        #         key=float,
        #     )
        # )
        mu_d = self.model_default_factors["demand_mean"]
        mu_l = self.model_default_factors["lead_mean"]
        return (
            rand_sol_rng.lognormalvariate(
                mu_d * mu_l / 3, mu_d * mu_l + 2 * sqrt(2 * mu_d**2 * mu_l)
            ),
            rand_sol_rng.lognormalvariate(
                mu_d * mu_l / 3, mu_d * mu_l + 2 * sqrt(2 * mu_d**2 * mu_l)
            ),
        )


# If T is lead time and X is a single demand, then:
#   var(sum_{i=1}^T X_i) = E(T) var(X) + (E X))^2 var T
# var(S) = E var(S|T) + var E(S|T)
