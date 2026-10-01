"""Simulate periods of ordering and sales for a dual sourcing inventory problem."""

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
from simopt.simulations.dualsourcing import DemandInputModel, DualSourcingConfig, replicate
from simopt.utils import override


class DualSourcingMinCostConfig(BaseModel):
    """Configuration model for Dual Sourcing Min Cost Problem.

    A problem configuration that minimizes total cost for dual sourcing inventory
    by optimizing order levels for regular and expedited orders.
    """

    initial_solution: Annotated[
        tuple[int, int],
        Field(
            default=(50, 80),
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


class DualSourcing(Model[DualSourcingConfig]):
    """Dual Sourcing Inventory Model.

    A model that simulates multiple periods of ordering and sales for a single-staged,
    dual sourcing inventory problem with stochastic demand. Returns average holding
    cost, average penalty cost, and average ordering cost per period.
    """

    class_name_abbr: ClassVar[str] = "DUALSOURCING"
    class_name: ClassVar[str] = "Dual Sourcing"
    config_class: ClassVar[type[DualSourcingConfig]] = DualSourcingConfig
    n_rngs: ClassVar[int] = 1
    n_responses: ClassVar[int] = 3

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize the DualSourcing model.

        Args:
            fixed_factors (dict, optional): Fixed factors for the model.
                Defaults to None.
        """
        # Let the base class handle default arguments.
        super().__init__(fixed_factors)

        self.demand_model = DemandInputModel()

    def replicate(self, factors: DualSourcingConfig, rngs: list[MRG32k3a]) -> tuple[dict, dict]:
        """Simulate a single replication for the current model factors.

        Args:
            rngs (list[MRG32k3a]): Random number generators used to simulate
                the replication.

        Returns:
            tuple[dict, dict]: A tuple containing:
                - responses (dict): Performance measures of interest:
                    - "average_holding_cost": The average holding cost over the
                        time period.
                    - "average_penalty_cost": The average penalty cost over the
                        time period.
                    - "average_ordering_cost": The average ordering cost over the
                        time period.
                - gradients (dict): A dictionary of gradient estimates for
                    each response.
        """
        average_ordering_cost, average_penalty_cost, average_holding_cost = replicate(
            factors, rngs, self.demand_model
        )
        # Calculate responses from simulation data.
        responses = {
            "average_ordering_cost": average_ordering_cost,
            "average_penalty_cost": average_penalty_cost,
            "average_holding_cost": average_holding_cost,
        }
        return responses, {}


class DualSourcingMinCost(Problem):
    """Class to make dual-sourcing inventory simulation-optimization problems."""

    class_name_abbr: ClassVar[str] = "DUALSOURCING-1"
    class_name: ClassVar[str] = "Min Cost for Dual Sourcing"
    config_class: ClassVar[type[BaseModel]] = DualSourcingMinCostConfig
    model_class: ClassVar[type[Model]] = DualSourcing
    n_objectives: ClassVar[int] = 1
    n_stochastic_constraints: ClassVar[int] = 0
    minmax: ClassVar[tuple[int, ...]] = (-1,)
    constraint_type: ClassVar[ConstraintType] = ConstraintType.BOX
    variable_type: ClassVar[VariableType] = VariableType.DISCRETE
    gradient_available: ClassVar[bool] = False
    optimal_value: ClassVar[float | None] = None
    optimal_solution: tuple | None = None
    model_default_factors: ClassVar[dict] = {}
    model_decision_factors: ClassVar[set[str]] = {"order_level_exp", "order_level_reg"}

    @override
    def build(self) -> dsl.Model:
        problem = dsl.Model()
        initial_solution = self.factors["initial_solution"]
        order_level_exp = problem.add_integer_variable(lb=0, ub=np.inf, initial=initial_solution[0])
        order_level_reg = problem.add_integer_variable(lb=0, ub=np.inf, initial=initial_solution[1])

        simulation = self.add_simulation(
            problem, {"order_level_exp": order_level_exp, "order_level_reg": order_level_reg}
        )
        problem.minimize(
            dsl.mean(
                simulation.metric("average_ordering_cost")
                + simulation.metric("average_penalty_cost")
                + simulation.metric("average_holding_cost")
            )
        )
        return problem

    def vector_to_factor_dict(self, vector: tuple) -> dict:
        return {
            "order_level_exp": vector[0],
            "order_level_reg": vector[1],
        }

    def check_deterministic_constraints(self, x: tuple) -> bool:
        return x[0] >= 0 and x[1] >= 0

    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
        return (rand_sol_rng.randint(40, 60), rand_sol_rng.randint(70, 90))
