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
from simopt.simulations.san import SANConfig, replicate
from simopt.utils import override

NUM_ARCS: Final[int] = 13
CONST_NODES: Final[list[int]] = [6, 8]


class SANLongestPathConfig(BaseModel):
    """Configuration model for SAN Longest Path Problem.

    Min Mean Longest Path for Stochastic Activity Network
    simulation-optimization problem.
    """

    initial_solution: Annotated[
        tuple[float, ...],
        Field(
            default_factory=lambda: (8,) * NUM_ARCS,
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
            default_factory=lambda: (1,) * NUM_ARCS,
            description="Cost associated to each arc.",
        ),
    ]

    def _check_arc_costs(self) -> None:
        if len(self.arc_costs) != NUM_ARCS:
            raise ValueError(f"arc_costs must be of length {NUM_ARCS}.")

        positive = True
        for x in list(self.arc_costs):
            positive = positive and (x > 0)
        if not positive:
            raise ValueError("All elements in arc_costs must be greater than 0.")

    @model_validator(mode="after")
    def _validate_model(self) -> Self:
        self._check_arc_costs()
        return self


class SAN(Model[SANConfig]):
    """Stochastic Activity Network (SAN) Model.

    A model that simulates a stochastic activity network problem with
    tasks that have exponentially distributed durations, and the selected
    means come with a cost.
    """

    class_name_abbr: ClassVar[str] = "SAN"
    class_name: ClassVar[str] = "Stochastic Activity Network"
    config_class: ClassVar[type[SANConfig]] = SANConfig
    n_rngs: ClassVar[int] = 1
    n_responses: ClassVar[int] = 1

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize the SAN model.

        Args:
            fixed_factors : dict
                fixed factors of the simulation model
        """
        # Let the base class handle default arguments.
        super().__init__(fixed_factors)

        self.time_model = Exp()

    def __dfs(self, graph: dict[int, set], start: int, visited: set | None = None) -> set:
        if visited is None:
            visited = set()
        visited.add(start)

        for next_point in graph[start] - visited:
            self.__dfs(graph, next_point, visited)
        return visited

    def replicate(self, factors: SANConfig, rngs: list[MRG32k3a]) -> tuple[dict, dict]:
        """Simulate a single replication for the current model factors.

        Args:
            rngs (list[MRG32k3a]): Random number generators used to simulate
                the replication.

        Returns:
            tuple[dict, dict]: A tuple containing:
                - responses (dict): Performance measures of interest, including:
                    - "longest_path_length": Length or duration of the longest path.
                - gradients (dict): A dictionary of gradient estimates for
                    each response.
        """
        longest_path, path_length, topo_order, grads = replicate(factors, rngs, self.time_model)

        # Compose responses and gradients.
        responses = {
            "longest_path_length": longest_path,
            "longest_path_to_all_nodes": path_length,
            "topo_order": topo_order,
        }
        gradients = {
            "longest_path_length": {"arc_means": grads[-1]},
            "longest_path_to_all_nodes": {"arc_means": grads},
        }
        return responses, gradients


class SANLongestPath(Problem):
    """Base class to implement simulation-optimization problems."""

    class_name_abbr: ClassVar[str] = "SAN-1"
    class_name: ClassVar[str] = "Min Mean Longest Path for Stochastic Activity Network"
    config_class: ClassVar[type[BaseModel]] = SANLongestPathConfig
    model_class: ClassVar[type[Model]] = SAN
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
            shape=(len(self.model.factors["arcs"]),),
            initial=tuple(self.factors["initial_solution"]),
        )
        simulation = self.add_simulation(problem, {"arc_means": arc_means})
        deterministic_cost = sum(
            cost / arc_mean
            for cost, arc_mean in zip(self.factors["arc_costs"], arc_means, strict=True)
        )
        problem.minimize(dsl.mean(simulation.metric("longest_path_length")) + deterministic_cost)
        return problem

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


class SANLongestPathStochasticConfig(BaseModel):
    """Configuration model for SAN Longest Path Stochastic Problem."""

    initial_solution: Annotated[
        tuple[float, ...],
        Field(
            default_factory=lambda: (8.0,) * NUM_ARCS,
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
            default_factory=lambda: (1.0,) * NUM_ARCS,
            description="Cost associated to each arc.",
        ),
    ]
    constraint_nodes: Annotated[
        list[int],
        Field(
            default_factory=lambda: CONST_NODES.copy(),
            description="Nodes with corresponding stochastic constraints.",
            min_length=1,
        ),
    ]
    length_to_node_constraint: Annotated[
        list[float],
        Field(
            default_factory=lambda: [6.0] * len(CONST_NODES),
            description="Max allowable length to each constraint node.",
            min_length=1,
        ),
    ]

    def _check_arc_costs(self) -> None:
        if len(self.arc_costs) != NUM_ARCS:
            raise ValueError(f"arc_costs must be of length {NUM_ARCS}.")
        if any(cost <= 0 for cost in self.arc_costs):
            raise ValueError("All elements in arc_costs must be greater than 0.")

    @model_validator(mode="after")
    def _validate_model(self) -> Self:
        self._check_arc_costs()
        return self


class SANLongestPathStochastic(Problem):
    """Minimize total cost s.t. reaching certain nodes within an expected length."""

    class_name_abbr: ClassVar[str] = "SAN-2"
    class_name: ClassVar[str] = "Min Cost SAN with Stochastic Constraints"
    config_class: ClassVar[type[BaseModel]] = SANLongestPathStochasticConfig
    model_class: ClassVar[type[Model]] = SAN
    n_objectives: ClassVar[int] = 1
    n_stochastic_constraints: ClassVar[int] = len(CONST_NODES)
    minmax: ClassVar[tuple[int, ...]] = (-1,)
    constraint_type: ClassVar[ConstraintType] = ConstraintType.STOCHASTIC
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
            ub=100.0,
            shape=(len(self.model.factors["arcs"]),),
            initial=tuple(self.factors["initial_solution"]),
        )
        constraint_nodes = tuple(self.factors["constraint_nodes"])
        simulation = self.add_simulation(problem, {"arc_means": arc_means})
        deterministic_cost = sum(
            cost / arc_mean
            for cost, arc_mean in zip(self.factors["arc_costs"], arc_means, strict=True)
        )
        problem.minimize(dsl.mean(simulation.metric("longest_path_length")) + deterministic_cost)
        for node, limit in zip(
            constraint_nodes, self.factors["length_to_node_constraint"], strict=True
        ):
            problem.add_stochastic_constraint(
                simulation.metric("longest_path_to_all_nodes")[node - 1] <= limit
            )
        return problem

    def vector_to_factor_dict(self, vector: tuple) -> dict:
        return {"arc_means": vector[:]}

    def check_deterministic_constraints(self, x: tuple) -> bool:
        return all(1e-2 <= x_i <= 100.0 for x_i in x)

    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
        solution = []
        while len(solution) < self.dim:
            value = rand_sol_rng.lognormalvariate(lq=0.1, uq=10)
            if 1e-2 <= value <= 100.0:
                solution.append(value)
        return tuple(solution)
