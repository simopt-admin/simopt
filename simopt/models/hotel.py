"""Simulate expected revenue for a hotel."""

from __future__ import annotations

import heapq
from typing import Annotated, ClassVar, Self

import numpy as np
from pydantic import BaseModel, Field, model_validator

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.base import ConstraintType, Model, Problem, VariableType
from simopt.input_models import Exp
from simopt.utils import override


def _double_up(values: list[float]) -> list[float]:
    """Duplicate each value in the list once."""
    return [x for x in values for _ in range(2)]


def _gen_binary_list(pattern: list[int]) -> list[int]:
    """Generate a binary list from alternating 0 and 1 runs.

    Args:
        pattern (list[int]): A list of run lengths. Even-indexed values
            correspond to 0s, odd-indexed to 1s. For example:
            bitstring([3, 2, 4]) → [0, 0, 0, 1, 1, 0, 0, 0, 0]

    Returns:
        list[int]: Expanded binary sequence.
    """
    result = []
    current_bit = 0
    for count in pattern:
        result.extend([current_bit] * count)
        current_bit = 1 - current_bit  # flip 0 to 1 or 1 to 0
    return result


class HotelConfig(BaseModel):
    """Configuration model for Hotel simulation.

    A model that simulates business of a hotel with Poisson arrival rate.
    """

    num_products: Annotated[
        int,
        Field(
            default=56,
            description="number of products: (rate, length of stay)",
            gt=0,
        ),
    ]
    lambda_: Annotated[
        list[float],
        Field(
            default_factory=lambda: [
                x / 168
                for x in _double_up(
                    [
                        1,
                        2,
                        3,
                        2,
                        1,
                        0.5,
                        0.25,
                        1,
                        2,
                        3,
                        2,
                        1,
                        0.5,
                        1,
                        2,
                        3,
                        2,
                        1,
                        1,
                        2,
                        3,
                        2,
                        1,
                        2,
                        3,
                        1,
                        2,
                        1,
                    ]
                )
            ],
            description="arrival rates for each product",
            alias="lambda",
        ),
    ]
    num_rooms: Annotated[
        int,
        Field(
            default=100,
            description="hotel capacity",
            gt=0,
        ),
    ]
    discount_rate: Annotated[
        int,
        Field(
            default=100,
            description="discount rate",
            gt=0,
        ),
    ]
    rack_rate: Annotated[
        int,
        Field(
            default=200,
            description="rack rate (full price)",
            gt=0,
        ),
    ]
    product_incidence: Annotated[
        list[list[int]],
        Field(
            default_factory=lambda: [
                _gen_binary_list([0, 14, 42]),
                _gen_binary_list([2, 24, 30]),
                _gen_binary_list([4, 10, 2, 20, 20]),
                _gen_binary_list([6, 8, 4, 8, 2, 16, 12]),
                _gen_binary_list([8, 6, 6, 6, 4, 6, 2, 12, 6]),
                _gen_binary_list([10, 4, 8, 4, 6, 4, 4, 4, 2, 8, 2]),
                _gen_binary_list([12, 2, 10, 2, 8, 2, 6, 2, 4, 2, 2, 4]),
            ],
            description="incidence matrix",
        ),
    ]
    time_limit: Annotated[
        list[int],
        Field(
            default_factory=lambda: (
                [27] * 14
                + [51] * 12
                + [75] * 10
                + [99] * 8
                + [123] * 6
                + [144] * 4
                + [168] * 2
            ),
            description=(
                "time after which orders of each product no longer arrive "
                "(e.g. Mon night stops at 3am Tues or t=27)"
            ),
        ),
    ]
    time_before: Annotated[
        int,
        Field(
            default=168,
            description=(
                "hours before t=0 to start running "
                "(e.g. 168 means start at time -168)"
            ),
            gt=0,
        ),
    ]
    runlength: Annotated[
        int,
        Field(
            default=168,
            description="runlength of simulation (in hours) after t=0",
            gt=0,
        ),
    ]
    booking_limits: Annotated[
        tuple[int, ...],
        Field(
            default_factory=lambda: tuple([100] * 56),
            description="booking limits",
        ),
    ]

    def _check_lambda(self) -> None:
        for i in self.lambda_:
            if i <= 0:
                raise ValueError("All elements in lambda must be greater than 0.")

    def _check_product_incidence(self) -> None:
        # TODO: fix check for product_incidence - keeping original implementation
        return

    def _check_time_limit(self) -> None:
        for i in self.time_limit:
            if i <= 0:
                raise ValueError("All elements in time_limit must be greater than 0.")

    def _check_booking_limits(self) -> None:
        for i in list(self.booking_limits):
            if i <= 0 or i > self.num_rooms:
                raise ValueError(
                    "All elements in booking_limits must be greater than 0 and less "
                    "than num_rooms."
                )

    @model_validator(mode="after")
    def _validate_model(self) -> Self:
        self._check_lambda()
        self._check_product_incidence()
        self._check_time_limit()
        self._check_booking_limits()

        # Cross-validation: check dimensions match num_products
        if len(self.lambda_) != self.num_products:
            raise ValueError("The length of lambda must equal num_products.")
        if len(self.time_limit) != self.num_products:
            raise ValueError("The length of time_limit must equal num_products.")
        if len(self.booking_limits) != self.num_products:
            raise ValueError("The length of booking_limits must equal num_products.")

        # Check product_incidence dimensions
        np_array = np.array(self.product_incidence)
        _, n = np_array.shape
        if n != self.num_products:
            raise ValueError(
                "The number of elements in product_incidence must equal num_products."
            )

        return self


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


class Hotel(Model):
    """A model that simulates business of a hotel with Poisson arrival rate."""

    config_class: ClassVar[type[BaseModel]] = HotelConfig
    class_name: str = "Hotel Booking"
    n_rngs: int = 1
    n_responses: int = 1

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize the Hotel model.

        Args:
            fixed_factors (dict, optional): Fixed factors for the model.
                Defaults to None.
        """
        # Let the base class handle default arguments.
        super().__init__(fixed_factors)

        self.arrival_model = Exp()

    @override
    def check_simulatable_factors(self) -> bool:
        if len(self.factors["lambda"]) != self.factors["num_products"]:
            raise ValueError("The length of lambda must equal num_products.")
        if len(self.factors["time_limit"]) != self.factors["num_products"]:
            raise ValueError("The length of time_limit must equal num_products.")
        if len(self.factors["booking_limits"]) != self.factors["num_products"]:
            raise ValueError("The length of booking_limits must equal num_products.")
        # TODO: get rid of this conversion to np.array
        np_array = np.array(self.factors["product_incidence"])
        # m, n = np_array.shape
        # if m * n != self.factors["num_products"]:
        _, n = np_array.shape
        if n != self.factors["num_products"]:
            raise ValueError(
                "The number of elements in product_incidence must equal num_products."
            )
        return True

    def before_replicate(self, rng_list: list[MRG32k3a]) -> None:  # noqa: D102
        self.arrival_model.set_rng(rng_list[0])

    def replicate(self) -> tuple[dict, dict]:
        """Simulate a single replication for the current model factors.

        Args:
            rng_list (list[MRG32k3a]): Random number generators used to simulate
                the replication.

        Returns:
            tuple[dict, dict]: A tuple containing:
                - responses (dict): Performance measures of interest, including:
                    - "revenue": Expected revenue.
                - gradients (dict): A dictionary of gradient estimates for each
                    response.
        """
        booking_limits = list(self.factors["booking_limits"])
        product_incidence = np.array(self.factors["product_incidence"])
        num_products: int = self.factors["num_products"]
        time_before: int = self.factors["time_before"]
        f_lambda = self.factors["lambda"]
        run_length: int = self.factors["runlength"]
        time_limit: list = self.factors["time_limit"]
        rack_rate: int = self.factors["rack_rate"]
        discount_rate: int = self.factors["discount_rate"]

        # Designate separate random number generators.
        total_revenue = 0

        # Generate interarrival times
        arr_bound = 10 * round(168 * np.sum(f_lambda))
        arr_time = np.array(
            [
                [self.arrival_model.random(f_lambda[i]) for _ in range(arr_bound)]
                for i in range(num_products)
            ]
        )

        # Initialize arrival times
        arrival = [-time_before + arr_time[i, 0] for i in range(num_products)]
        a_idx = np.ones(num_products, dtype=int)  # Next interarrival index

        # Precompute resource conflict matrix (bool)
        conflicts = (product_incidence.T @ product_incidence) >= 1

        # Min-heap for tracking next arrival events (arrival_time, product_idx)
        heap = [
            (arrival[i], i) for i in range(num_products) if arrival[i] <= time_limit[i]
        ]
        heapq.heapify(heap)

        while heap:
            current_time, product_idx = heapq.heappop(heap)
            if current_time > run_length:
                break
            if booking_limits[product_idx] > 0:
                rate = rack_rate if product_idx % 2 == 0 else discount_rate
                total_revenue += rate * np.sum(product_incidence[:, product_idx])
                for i in range(num_products):
                    if conflicts[product_idx, i] and booking_limits[i] > 0:
                        booking_limits[i] -= 1

            # Schedule next arrival for this product
            next_idx = a_idx[product_idx]
            if next_idx < arr_bound:
                next_time = current_time + arr_time[product_idx, next_idx]
                a_idx[product_idx] += 1
                if next_time <= time_limit[product_idx]:
                    heapq.heappush(heap, (next_time, product_idx))

        # Compose responses and gradients.
        responses = {"revenue": total_revenue}
        gradients = {
            response_key: dict.fromkeys(self.specifications, np.nan)
            for response_key in responses
        }
        return responses, gradients


class HotelRevenue(Problem):
    """Base class to implement simulation-optimization problems."""

    config_class: ClassVar[type[BaseModel]] = HotelRevenueConfig
    model_class: ClassVar[type[Model]] = Hotel
    class_name_abbr: str = "HOTEL-1"
    class_name: str = "Max Revenue for Hotel Booking"
    n_objectives: int = 1
    n_stochastic_constraints: int = 0
    minmax: tuple[int] = (1,)
    constraint_type: ConstraintType = ConstraintType.BOX
    variable_type: VariableType = VariableType.DISCRETE
    gradient_available: bool = False
    optimal_value: float | None = None
    optimal_solution: tuple | None = None
    model_default_factors: dict = {}
    model_decision_factors: set[str] = {"booking_limits"}

    @property
    @override
    def dim(self) -> int:
        return self.model.factors["num_products"]

    @property
    @override
    def lower_bounds(self) -> tuple:
        return (0,) * self.dim

    @property
    @override
    def upper_bounds(self) -> tuple:
        return (self.model.factors["num_rooms"],) * self.dim

    @override
    def check_initial_solution(self) -> bool:
        return len(self.factors["initial_solution"]) == self.dim

    # # TODO: figure out how Problem.check_simulatable_factors() works
    # def check_simulatable_factors(self) -> bool:
    #     return not (
    #         len(self.lower_bounds) != self.dim or len(self.upper_bounds) != self.dim
    #     )

    @override
    def vector_to_factor_dict(self, vector: tuple) -> dict:
        return {"booking_limits": vector[:]}

    @override
    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:
        return tuple(factor_dict["booking_limits"])

    @override
    def response_dict_to_objectives(self, response_dict: dict) -> tuple:
        return (response_dict["revenue"],)

    @override
    def deterministic_objectives_and_gradients(self, _x: tuple) -> tuple[tuple, tuple]:
        det_objectives = (0,)
        det_objectives_gradients = ((0,) * self.dim,)
        return det_objectives, det_objectives_gradients

    @override
    def check_deterministic_constraints(self, _x: tuple) -> bool:
        return True

    @override
    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
        return tuple(
            [
                rand_sol_rng.randint(0, self.model.factors["num_rooms"])
                for _ in range(self.dim)
            ]
        )
