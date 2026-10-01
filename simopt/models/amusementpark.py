"""Simulate a single day of operation for an amusement park queuing problem."""

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
from simopt.input_models import Exp, Gamma, WeightedChoice
from simopt.models._ext import patch_model
from simopt.simulations.amusementpark import (
    NUM_ATTRACTIONS,
    PARK_CAPACITY,
    AmusementParkConfig,
    replicate,
)
from simopt.utils import override


class AmusementParkMinDepartConfig(BaseModel):
    """Configuration model for Amusement Park Min Depart Problem.

    A problem configuration that minimizes the total number of departed
    visitors from an amusement park by optimizing queue capacities.
    """

    initial_solution: Annotated[
        tuple[int, ...],
        Field(
            default_factory=lambda: (
                (PARK_CAPACITY - NUM_ATTRACTIONS + 1,) + (1,) * (NUM_ATTRACTIONS - 1)
            ),
            description="Initial solution from which solvers start.",
        ),
    ]
    budget: Annotated[
        int,
        Field(
            default=100,
            description="Max # of replications for a solver to take.",
            gt=0,
            json_schema_extra={"isDatafarmable": False},
        ),
    ]


class AmusementPark(Model[AmusementParkConfig]):
    """Amusement Park Model.

    A model that simulates a single day of operation for an
    amusement park queuing problem based on a poisson distributed tourist
    arrival rate, a next attraction transition matrix, and attraction
    durations based on an Erlang distribution. Returns the total number
    and percent of tourists to leave the park due to full queues.
    """

    class_name_abbr: ClassVar[str] = "AMUSEMENTPARK"
    class_name: ClassVar[str] = "Amusement Park"
    config_class: ClassVar[type[AmusementParkConfig]] = AmusementParkConfig
    n_rngs: ClassVar[int] = 4
    n_responses: ClassVar[int] = 4

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize the Amusement Park Model."""
        # Let the base class handle default arguments.
        super().__init__(fixed_factors)

        self.arrival_model = Exp()
        self.attraction_model = WeightedChoice()
        self.destination_model = WeightedChoice()
        self.service_models = []
        for _ in range(self.factors["number_attractions"]):
            self.service_models.append(Gamma())

    def replicate(
        self, factors: AmusementParkConfig, rngs: list[MRG32k3a]
    ) -> tuple[dict[str, float | list[float]], dict]:
        """Simulate a single replication using current model factors.

        Args:
            rngs (list[MRG32k3a]): Random number generators used during the
                simulation.

        Returns:
            tuple: A tuple containing:
                - dict[str, float | list[float]]: Performance metrics from the simulation:
                    - "total_departed_tourists": Total number of tourists who left due to full queues.
                    - "percent_departed_tourists": Percentage of tourists who left due to full queues.
                    - "average_number_in_system": Average number of tourists in the park at a given time.
                    - "attraction_utilization_percentages": Utilization percentage of each attraction.
                - dict: Gradients of the performance measures with respect to model factors.
        """  # noqa: E501

        total_departed, percent_departed, average_number_in_system, cumulative_util = replicate(
            factors,
            rngs,
            self.arrival_model,
            self.attraction_model,
            self.destination_model,
            self.service_models,
        )
        responses = {
            "total_departed": total_departed,
            "percent_departed": percent_departed,
            "average_number_in_system": average_number_in_system,
            "attraction_utilization_percentages": cumulative_util,
        }
        return responses, {}


class AmusementParkMinDepart(Problem):
    """Class to make amusement park simulation-optimization problems."""

    class_name_abbr: ClassVar[str] = "AMUSEMENTPARK-1"
    class_name: ClassVar[str] = "Min Total Departed Visitors for Amusement Park"
    config_class: ClassVar[type[BaseModel]] = AmusementParkMinDepartConfig
    model_class: ClassVar[type[Model]] = AmusementPark
    n_objectives: ClassVar[int] = 1
    n_stochastic_constraints: ClassVar[int] = 0
    minmax: ClassVar[tuple[int, ...]] = (-1,)
    constraint_type: ClassVar[ConstraintType] = ConstraintType.DETERMINISTIC
    variable_type: ClassVar[VariableType] = VariableType.DISCRETE
    gradient_available: ClassVar[bool] = False
    optimal_value: ClassVar[float | None] = None
    optimal_solution: tuple | None = None
    model_default_factors: ClassVar[dict] = {}
    model_decision_factors: ClassVar[set[str]] = {"queue_capacities"}

    @override
    def build(self) -> dsl.Model:
        problem = dsl.Model()
        queue_capacities = problem.add_integer_vector(
            lb=0,
            ub=self.model.factors["park_capacity"],
            shape=(self.model.factors["number_attractions"],),
            initial=tuple(self.factors["initial_solution"]),
        )
        problem.add_linear_constraint(
            dsl.sum(queue_capacities) <= self.model.factors["park_capacity"]
        )
        simulation = self.add_simulation(problem, {"queue_capacities": queue_capacities})
        problem.minimize(dsl.mean(simulation.metric("total_departed")))
        return problem

    def vector_to_factor_dict(self, vector: tuple) -> dict[str, tuple]:
        return {
            "queue_capacities": vector[:],
        }

    def check_deterministic_constraints(self, x: tuple) -> bool:
        # Check box constraints.
        if not super().check_deterministic_constraints(x):
            return False
        # Check if sum of queue capacities is less than park capacity.
        return sum(x) <= self.model.factors["park_capacity"]

    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
        num_elements: int = self.model.factors["number_attractions"]
        summation: int = self.model.factors["park_capacity"]
        vector = rand_sol_rng.integer_random_vector_from_simplex(
            n_elements=num_elements, summation=summation, with_zero=False
        )
        return tuple(vector)


patch_model(AmusementPark)
