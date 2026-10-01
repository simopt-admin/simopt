"""Simulate duration of a stochastic activity network (SAN)."""

from __future__ import annotations

from typing import Annotated, ClassVar, Final, Self

import numpy as np
from pydantic import BaseModel, Field, model_validator

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt import dsl
from simopt.base import (
    ConstraintType,
    Model,
    Problem,
    VariableType,
)
from simopt.input_models import Exp
from simopt.simulations.fixedsan import FixedSANConfig, replicate
from simopt.utils import override

# TODO: figure out if this should ever be anything other than 13
NUM_ARCS: Final[int] = 13


class FixedSANLongestPathConfig(BaseModel):
    """Configuration model for Fixed SAN Longest Path Problem.

    Min Mean Longest Path for Fixed Stochastic Activity Network problem.
    """

    initial_solution: Annotated[
        tuple[float, ...],
        Field(
            default=(10,) * 13,
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
    arc_costs: Annotated[
        tuple[float, ...],
        Field(
            default=(1,) * 13,
            description="cost associated to each arc",
        ),
    ]

    def _check_arc_costs(self) -> None:
        if len(self.arc_costs) != NUM_ARCS:
            raise ValueError(f"arc_costs must be of length {NUM_ARCS}.")

        if not all(x > 0 for x in list(self.arc_costs)):
            raise ValueError("All arc costs must be greater than 0.")

    @model_validator(mode="after")
    def _validate_model(self) -> Self:
        self._check_arc_costs()
        return self


class FixedSAN(Model[FixedSANConfig]):
    """Fixed Stochastic Activity Network (SAN) Model.

    A model that simulates a stochastic activity network problem with tasks
    that have exponentially distributed durations, and the selected means
    come with a cost.
    """

    class_name_abbr: ClassVar[str] = "FIXEDSAN"
    class_name: ClassVar[str] = "Fixed Stochastic Activity Network"
    config_class: ClassVar[type[FixedSANConfig]] = FixedSANConfig
    n_rngs: ClassVar[int] = 1
    n_responses: ClassVar[int] = 1

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize the Fixed Stochastic Activity Network model.

        Args:
            fixed_factors (dict, optional): Fixed factors for the model.
                Defaults to None.
        """
        # Let the base class handle default arguments.
        super().__init__(fixed_factors)

        self.time_model = Exp()

    def replicate(self, factors: FixedSANConfig, rngs: list[MRG32k3a]) -> tuple[dict, dict]:
        """Simulate a single replication for the current model factors.

        Args:
            rngs (list[MRG32k3a]): Random number generators used to simulate
                the replication.

        Returns:
            tuple[dict, dict]: A tuple containing:
                - responses (dict): Performance measures of interest, including:
                    - "longest_path_length": The length or duration of the longest path.
                - gradients (dict): A dictionary of gradient estimates for
                    each response.
        """
        longest_path, longest_path_gradient = replicate(factors, rngs, self.time_model)

        # Compose responses and gradients.
        responses = {"longest_path_length": longest_path}
        gradients = {
            response_key: {
                factor_key: np.zeros(len(self.specifications)) for factor_key in self.specifications
            }
            for response_key in responses
        }
        gradients["longest_path_length"]["arc_means"] = longest_path_gradient

        return responses, gradients


class FixedSANLongestPath(Problem):
    """Base class to implement simulation-optimization problems."""

    class_name_abbr: ClassVar[str] = "FIXEDSAN-1"
    class_name: ClassVar[str] = "Min Mean Longest Path for Fixed Stochastic Activity Network"
    config_class: ClassVar[type[BaseModel]] = FixedSANLongestPathConfig
    model_class: ClassVar[type[Model]] = FixedSAN
    n_objectives: ClassVar[int] = 1
    n_stochastic_constraints: ClassVar[int] = 0
    minmax: ClassVar[tuple[int, ...]] = (-1,)
    constraint_type: ClassVar[ConstraintType] = ConstraintType.BOX
    variable_type: ClassVar[VariableType] = VariableType.CONTINUOUS
    gradient_available: ClassVar[bool] = True
    optimal_value: ClassVar[float | None] = None
    optimal_solution: tuple | None = None
    model_default_factors: ClassVar[dict] = {}
    model_decision_factors: ClassVar[set[str]] = {"arc_means"}

    @override
    def build(self) -> dsl.Model:
        problem = dsl.Model()
        arc_means = problem.add_continuous_vector(
            lb=1e-2,
            ub=np.inf,
            shape=(self.model.factors["num_arcs"],),
            initial=tuple(self.factors["initial_solution"]),
        )
        simulation = self.add_simulation(problem, {"arc_means": arc_means})
        deterministic_cost = sum(
            cost / arc_mean
            for cost, arc_mean in zip(self.factors["arc_costs"], arc_means, strict=True)
        )
        problem.minimize(dsl.mean(simulation.metric("longest_path_length")) + deterministic_cost)
        return problem

    def check_arc_costs(self) -> bool:
        """Check if all arc costs are positive and match the number of arcs."""
        return len(self.factors["arc_costs"]) == self.model.factors["num_arcs"] and all(
            x > 0 for x in self.factors["arc_costs"]
        )

    def vector_to_factor_dict(self, vector: tuple) -> dict:
        return {"arc_means": vector[:]}

    def check_deterministic_constraints(self, x: tuple) -> bool:
        return all(x_i >= 1e-2 for x_i in x)

    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
        solution = []
        while len(solution) < self.dim:
            value = rand_sol_rng.lognormalvariate(lq=0.1, uq=10)
            if value >= 1e-2:
                solution.append(value)
        return tuple(solution)
