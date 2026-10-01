"""Simulation of the average response time in a multi-base ambulance dispatch system."""

from __future__ import annotations

from typing import Annotated, ClassVar

from pydantic import BaseModel, Field

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt import dsl
from simopt.base import (
    ConstraintType,
    Model,
    Problem,
    VariableType,
)
from simopt.input_models import Beta, Exp
from simopt.simulations.ambulance import AmbulanceConfig, replicate
from simopt.utils import override


class AmbBaseAllocationConfig(BaseModel):
    """Configuration for the Ambulance optimization problem."""

    initial_solution: Annotated[
        tuple[float, ...],
        Field(
            default=(6, 6, 6, 6),
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


class Ambulance(Model[AmbulanceConfig]):
    """Simulate the average response time in a multi-base ambulance dispatch system.

    The system includes a set of fixed ambulance bases and a set of variable bases
    with decision-variable coordinates. The objective is to minimize the expected
    response time by optimizing the locations of the variable bases.
    """

    class_name_abbr: ClassVar[str] = "AMBULANCE"
    class_name: ClassVar[str] = "Ambulance Base Allocation"
    config_class: ClassVar[type[AmbulanceConfig]] = AmbulanceConfig
    n_rngs: ClassVar[int] = 4
    n_responses: ClassVar[int] = 1

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize the ambulance simulation model.

        Args:
            fixed_factors : dict
                fixed factors of the simulation model
        """
        super().__init__(fixed_factors)

        # Instantiate Input Models
        # 1. Exponential distributions for times
        self.arrival_time_model = Exp()
        self.scene_time_model = Exp()

        # 2. Beta distributions for locations
        self.beta_x_model = Beta()
        self.beta_y_model = Beta()

    def replicate(self, factors: AmbulanceConfig, rngs: list[MRG32k3a]) -> tuple[dict, dict]:
        """Run one replication of the ambulance dispatch simulation."""
        avg_time, grad_avg = replicate(
            factors,
            rngs,
            self.arrival_time_model,
            self.scene_time_model,
            self.beta_x_model,
            self.beta_y_model,
        )
        responses = {"avg_response_time": avg_time}
        gradients = {"avg_response_time": {"variable_locs": grad_avg.flatten().tolist()}}
        return responses, gradients


class AmbulanceMinAvgResponse(Problem):
    """Base class to implement simulation-optimization problems."""

    class_name_abbr: ClassVar[str] = "AMBULANCE-1"
    class_name: ClassVar[str] = "Minimum Average Waiting Time for Ambulance Dispatch"
    config_class: ClassVar[type[BaseModel]] = AmbBaseAllocationConfig
    model_class: ClassVar[type[Model]] = Ambulance
    n_objectives: ClassVar[int] = 1
    n_stochastic_constraints: ClassVar[int] = 0
    minmax: ClassVar[tuple[int, ...]] = (-1,)
    constraint_type: ClassVar[ConstraintType] = ConstraintType.BOX
    variable_type: ClassVar[VariableType] = VariableType.CONTINUOUS
    gradient_available: ClassVar[bool] = True
    optimal_value: ClassVar[float | None] = None
    optimal_solution: tuple | None = None

    # Define default factors used for problem instantiation
    model_default_factors: ClassVar[dict] = {
        "fixed_base_count": 3,
        "variable_base_count": 2,
        "fixed_locs": [15, 15, 5, 15, 5, 5],
        "variable_locs": [6, 6, 6, 6],
    }
    model_decision_factors: ClassVar[set[str]] = {"variable_locs"}

    @override
    def build(self) -> dsl.Model:
        problem = dsl.Model()
        initial = self.factors["initial_solution"]
        dim = len(initial)
        location = problem.add_continuous_vector(lb=0.0, ub=20.0, shape=(dim,), initial=initial)
        simulation = self.add_simulation(problem, {"variable_locs": location})
        problem.minimize(dsl.mean(simulation.metric("avg_response_time")))
        return problem

    @override
    def vector_to_factor_dict(self, vector: tuple) -> dict:
        return {"variable_locs": list(vector)}

    @override
    def check_deterministic_constraints(self, _x: tuple) -> bool:
        return len(_x) == self.dim and all(0 <= xi <= 20 for xi in _x)

    @override
    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
        return tuple(rand_sol_rng.uniform(0, 20) for _ in range(self.dim))
