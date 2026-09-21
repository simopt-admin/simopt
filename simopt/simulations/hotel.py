"""Standalone Python simulation for hotel booking revenue."""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
from typing import Annotated, Self

import numpy as np
import simpy
from pydantic import BaseModel, Field, model_validator

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt._markers import simulation
from simopt.input_models import InputModel


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
                [27] * 14 + [51] * 12 + [75] * 10 + [99] * 8 + [123] * 6 + [144] * 4 + [168] * 2
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
            description=("hours before t=0 to start running (e.g. 168 means start at time -168)"),
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
                    "All elements in booking_limits must be greater than 0 and less than num_rooms."
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
            raise ValueError("The number of elements in product_incidence must equal num_products.")

        return self


@dataclass
class State:
    """Revenue shared by booking processes within one replication."""

    total_revenue: int = 0


@simulation
def replicate(
    factors: HotelConfig,
    rngs: list[MRG32k3a],
    arrival_model: InputModel,
) -> int:
    """Simulate one replication and return revenue using the supplied arrival sampler."""
    booking_limits = list(factors.booking_limits)
    product_incidence = np.array(factors.product_incidence)
    num_products: int = factors.num_products
    time_before: int = factors.time_before
    f_lambda = factors.lambda_
    run_length: int = factors.runlength
    time_limit: list = factors.time_limit
    rack_rate: int = factors.rack_rate
    discount_rate: int = factors.discount_rate

    # Designate separate random number generators.
    state = State()

    # Generate interarrival times
    arr_bound = 10 * round(168 * sum(f_lambda))
    arr_time = np.array(
        [
            [arrival_model.random(rngs[0], f_lambda[i]) for _ in range(arr_bound)]
            for i in range(num_products)
        ]
    )

    # Precompute resource conflict matrix (bool)
    conflicts = (product_incidence.T @ product_incidence) >= 1

    env = simpy.Environment(initial_time=-time_before)
    booking_inventory = [
        simpy.Container(env, capacity=max(1, limit), init=limit) for limit in booking_limits
    ]

    def booking_stream(product_idx: int) -> Generator[simpy.Event, object, None]:
        for interarrival_time in arr_time[product_idx]:
            next_time = env.now + interarrival_time
            if next_time > time_limit[product_idx] or next_time > run_length:
                return
            yield env.timeout(interarrival_time)

            if booking_inventory[product_idx].level > 0:
                rate = rack_rate if product_idx % 2 == 0 else discount_rate
                state.total_revenue += rate * np.sum(product_incidence[:, product_idx])
                for i in range(num_products):
                    inventory = booking_inventory[i]
                    if conflicts[product_idx, i] and inventory.level > 0:
                        inventory.get(1)

    for product_idx in range(num_products):
        env.process(booking_stream(product_idx))
    env.run()

    return state.total_revenue
