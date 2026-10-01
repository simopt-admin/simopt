"""Simulate messages being processed in a queueing network."""

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
from simopt.input_models import Exp, Triangular
from simopt.simulations.network import NUM_NETWORKS, NetworkConfig, RouteInputModel, replicate
from simopt.utils import override


class NetworkMinTotalCostConfig(BaseModel):
    """Configuration model for Network Min Total Cost Problem.

    Min Total Cost for Communication Networks System simulation-optimization problem.
    """

    initial_solution: Annotated[
        tuple[float, ...],
        Field(
            default_factory=lambda: (0.1,) * NUM_NETWORKS,
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


class Network(Model[NetworkConfig]):
    """Simulate messages being processed in a queueing network."""

    class_name_abbr: ClassVar[str] = "NETWORK"
    class_name: ClassVar[str] = "Communication Networks System"
    config_class: ClassVar[type[NetworkConfig]] = NetworkConfig
    n_rngs: ClassVar[int] = 3
    n_responses: ClassVar[int] = 1

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize the Network model.

        Args:
            fixed_factors (dict): Fixed factors for the model.
        """
        # Let the base class handle default arguments.
        super().__init__(fixed_factors)

        self.arrival_model = Exp()
        self.route_model = RouteInputModel()
        self.service_model = Triangular()

    def replicate(self, factors: NetworkConfig, rngs: list[MRG32k3a]) -> tuple[dict, dict]:
        """Simulate a single replication for the current model factors.

        Args:
            rngs (list[MRG32k3a]): Random number generators used to simulate
                the replication.

        Returns:
            tuple[dict, dict]: A tuple containing:
                - responses (dict): Performance measure of interest, including:
                    - "total_cost": Total cost spent to route all messages.
                - gradients (dict): A dictionary of gradient estimates for
                    each response.
        """
        total_cost = replicate(
            factors, rngs, self.arrival_model, self.route_model, self.service_model
        )
        responses = {"total_cost": total_cost}
        return responses, {}


class NetworkMinTotalCost(Problem):
    """Base class to implement simulation-optimization problems."""

    class_name_abbr: ClassVar[str] = "NETWORK-1"
    class_name: ClassVar[str] = "Min Total Cost for Communication Networks System"
    config_class: ClassVar[type[BaseModel]] = NetworkMinTotalCostConfig
    model_class: ClassVar[type[Model]] = Network
    n_objectives: ClassVar[int] = 1
    n_stochastic_constraints: ClassVar[int] = 0
    minmax: ClassVar[tuple[int, ...]] = (-1,)
    constraint_type: ClassVar[ConstraintType] = ConstraintType.DETERMINISTIC
    variable_type: ClassVar[VariableType] = VariableType.CONTINUOUS
    gradient_available: ClassVar[bool] = False
    optimal_value: ClassVar[float | None] = None
    optimal_solution: tuple | None = None
    model_default_factors: ClassVar[dict] = {}
    model_decision_factors: ClassVar[set[str]] = {"process_prob"}

    @override
    def build(self) -> dsl.Model:
        problem = dsl.Model()
        n_networks = self.model.factors["n_networks"]
        process_prob = problem.add_continuous_vector(
            lb=0.0, ub=1.0, shape=(n_networks,), initial=self.factors["initial_solution"]
        )
        total_probability = dsl.sum(process_prob)
        problem.add_linear_constraint(total_probability <= 1.0 + 1e-10)
        problem.add_linear_constraint(total_probability >= 1.0 - 1e-10)
        simulation = self.add_simulation(problem, {"process_prob": process_prob})
        problem.minimize(dsl.mean(simulation.metric("total_cost")))
        return problem

    def vector_to_factor_dict(self, vector: tuple) -> dict:
        return {"process_prob": vector[:]}

    def check_deterministic_constraints(self, x: tuple) -> bool:
        # Check box constraints.
        box_feasible = super().check_deterministic_constraints(x)
        if not box_feasible:
            return False

        # Check constraint that probabilities sum to one.
        return round(sum(x), 10) == 1.0

    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
        # Generating a random pmf with length equal to number of networks.
        x = rand_sol_rng.continuous_random_vector_from_simplex(
            n_elements=self.model.factors["n_networks"],
            summation=1.0,
            exact_sum=True,
        )
        return tuple(x)
