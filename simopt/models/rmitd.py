"""Simulate a multi-stage revenue management system with inter-temporal dependence."""

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
from simopt.simulations.rmitd import DemandInputModel, RMITDConfig, replicate
from simopt.utils import override


class RMITDMaxRevenueConfig(BaseModel):
    """Configuration model for RMITD Max Revenue Problem.

    Max Revenue for Revenue Management Temporal Demand simulation-optimization problem.
    """

    initial_solution: Annotated[
        tuple[int, ...],
        Field(
            default=(100, 50, 30),
            description="initial solution",
        ),
    ]
    budget: Annotated[
        int,
        Field(
            default=10000,
            description="max # of replications for a solver to take",
            gt=0,
            json_schema_extra={"isDatafarmable": False},
        ),
    ]


class RMITD(Model[RMITDConfig]):
    """Multi-stage Revenue Management with Inter-temporal Dependence (RMITD).

    A model that simulates a multi-stage revenue management system with
    inter-temporal dependence. Returns the total revenue.
    """

    class_name_abbr: ClassVar[str] = "RMITD"
    class_name: ClassVar[str] = "Revenue Management Temporal Demand"
    config_class: ClassVar[type[RMITDConfig]] = RMITDConfig
    n_rngs: ClassVar[int] = 2
    n_responses: ClassVar[int] = 1

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize the RMITD model.

        Args:
            fixed_factors (dict, optional): Dictionary of fixed factors for the model.
                Defaults to None.
        """
        # Let the base class handle default arguments.
        super().__init__(fixed_factors)

        self.demand_model = DemandInputModel()

    def replicate(self, factors: RMITDConfig, rngs: list[MRG32k3a]) -> tuple[dict, dict]:
        """Simulate a single replication for the current model factors.

        Args:
            rngs (list[MRG32k3a]): Random number generators used to simulate
                the replication.

        Returns:
            tuple[dict, dict]: A tuple containing:
                - responses (dict): Performance measures of interest, including:
                    - "revenue": Total revenue.
                - gradients (dict): A dictionary of gradient estimates for
                    each response.
        """
        revenue = replicate(factors, rngs, self.demand_model)
        # Compose responses and gradients.
        responses = {"revenue": revenue}
        return responses, {}


class RMITDMaxRevenue(Problem):
    """Base class to implement simulation-optimization problems."""

    class_name_abbr: ClassVar[str] = "RMITD-1"
    class_name: ClassVar[str] = "Max Revenue for Revenue Management Temporal Demand"
    config_class: ClassVar[type[BaseModel]] = RMITDMaxRevenueConfig
    model_class: ClassVar[type[Model]] = RMITD
    n_objectives: ClassVar[int] = 1
    n_stochastic_constraints: ClassVar[int] = 0
    minmax: ClassVar[tuple[int, ...]] = (1,)
    constraint_type: ClassVar[ConstraintType] = ConstraintType.DETERMINISTIC
    variable_type: ClassVar[VariableType] = VariableType.DISCRETE
    gradient_available: ClassVar[bool] = False
    optimal_value: ClassVar[float | None] = None
    optimal_solution: tuple | None = None
    model_default_factors: ClassVar[dict] = {}
    model_decision_factors: ClassVar[set[str]] = {
        "initial_inventory",
        "reservation_qtys",
    }

    @override
    def build(self) -> dsl.Model:
        problem = dsl.Model()
        initial_solution = tuple(self.factors["initial_solution"])
        initial_inventory = problem.add_integer_variable(
            lb=0, ub=np.inf, initial=initial_solution[0]
        )
        reservation_qtys = problem.add_integer_vector(
            lb=0,
            ub=np.inf,
            shape=(self.model.factors["time_horizon"] - 1,),
            initial=initial_solution[1:],
        )
        problem.add_linear_constraint(initial_inventory >= reservation_qtys[0])
        for idx in range(len(reservation_qtys) - 1):
            problem.add_linear_constraint(reservation_qtys[idx] >= reservation_qtys[idx + 1])

        simulation = self.add_simulation(
            problem,
            {
                "initial_inventory": initial_inventory,
                "reservation_qtys": reservation_qtys,
            },
        )
        problem.maximize(dsl.mean(simulation.metric("revenue")))
        return problem

    def vector_to_factor_dict(self, vector: tuple) -> dict:
        return {
            "initial_inventory": vector[0],
            "reservation_qtys": list(vector[1:]),
        }

    def check_deterministic_constraints(self, x: tuple) -> bool:
        return all(x[idx] >= x[idx + 1] for idx in range(self.dim - 1))

    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
        return tuple(
            sorted(
                (rand_sol_rng.randint(0, 199) for _ in range(self.dim)),
                reverse=True,
            )
        )
