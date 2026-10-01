"""Simulate expected revenue for a hotel."""

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
from simopt.input_models import Exp
from simopt.simulations.hotel import HotelConfig, replicate
from simopt.utils import override


class HotelRevenueConfig(BaseModel):
    """Configuration model for Hotel Revenue Problem.

    Max Revenue for Hotel Booking simulation-optimization problem.
    """

    initial_solution: Annotated[
        tuple[int, ...],
        Field(
            default_factory=lambda: tuple([0 for _ in range(56)]),
            description="initial solution",
        ),
    ]
    budget: Annotated[
        int,
        Field(
            default=100,
            description="max # of replications for a solver to take",
            gt=0,
            json_schema_extra={"isDatafarmable": False},
        ),
    ]


class Hotel(Model[HotelConfig]):
    """A model that simulates business of a hotel with Poisson arrival rate."""

    class_name_abbr: ClassVar[str] = "HOTEL"
    class_name: ClassVar[str] = "Hotel Booking"
    config_class: ClassVar[type[HotelConfig]] = HotelConfig
    n_rngs: ClassVar[int] = 1
    n_responses: ClassVar[int] = 1

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize the Hotel model.

        Args:
            fixed_factors (dict, optional): Fixed factors for the model.
                Defaults to None.
        """
        # Let the base class handle default arguments.
        super().__init__(fixed_factors)

        self.arrival_model = Exp()

    def replicate(self, factors: HotelConfig, rngs: list[MRG32k3a]) -> tuple[dict, dict]:
        """Simulate a single replication for the current model factors.

        Args:
            rngs (list[MRG32k3a]): Random number generators used to simulate
                the replication.

        Returns:
            tuple[dict, dict]: A tuple containing:
                - responses (dict): Performance measures of interest, including:
                    - "revenue": Expected revenue.
                - gradients (dict): A dictionary of gradient estimates for each
                    response.
        """
        total_revenue = replicate(factors, rngs, self.arrival_model)
        # Compose responses and gradients.
        responses = {"revenue": total_revenue}
        return responses, {}


class HotelRevenue(Problem):
    """Base class to implement simulation-optimization problems."""

    class_name_abbr: ClassVar[str] = "HOTEL-1"
    class_name: ClassVar[str] = "Max Revenue for Hotel Booking"
    config_class: ClassVar[type[BaseModel]] = HotelRevenueConfig
    model_class: ClassVar[type[Model]] = Hotel
    n_objectives: ClassVar[int] = 1
    n_stochastic_constraints: ClassVar[int] = 0
    minmax: ClassVar[tuple[int, ...]] = (1,)
    constraint_type: ClassVar[ConstraintType] = ConstraintType.BOX
    variable_type: ClassVar[VariableType] = VariableType.DISCRETE
    gradient_available: ClassVar[bool] = False
    optimal_value: ClassVar[float | None] = None
    optimal_solution: tuple | None = None
    model_default_factors: ClassVar[dict] = {}
    model_decision_factors: ClassVar[set[str]] = {"booking_limits"}

    @override
    def build(self) -> dsl.Model:
        problem = dsl.Model()
        booking_limits = problem.add_integer_vector(
            lb=0,
            ub=self.model.factors["num_rooms"],
            shape=(self.model.factors["num_products"],),
            initial=self.factors["initial_solution"],
        )

        simulation = self.add_simulation(problem, {"booking_limits": booking_limits})
        problem.maximize(dsl.mean(simulation.metric("revenue")))
        return problem

    @override
    def vector_to_factor_dict(self, vector: tuple) -> dict:
        return {"booking_limits": vector[:]}

    @override
    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
        return tuple(
            [rand_sol_rng.randint(0, self.model.factors["num_rooms"]) for _ in range(self.dim)]
        )
