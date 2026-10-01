"""Standalone Python simulation for an amusement park."""

from __future__ import annotations

from collections.abc import Generator, Sequence
from dataclasses import dataclass
from typing import Annotated, Final, Self

import simpy
from pydantic import BaseModel, Field, model_validator

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt._markers import simulation
from simopt.input_models import InputModel

# Default values for the model
PARK_CAPACITY: Final[int] = 350
NUM_ATTRACTIONS: Final[int] = 7


class AmusementParkConfig(BaseModel):
    """Configuration for the Amusement Park model."""

    park_capacity: Annotated[
        int,
        Field(
            default=PARK_CAPACITY,
            description=(
                "The total number of tourists waiting for attractions that can be "
                "maintained through park facilities, distributed across the "
                "attractions."
            ),
            ge=0,
        ),
    ]
    number_attractions: Annotated[
        int,
        Field(
            default=NUM_ATTRACTIONS,
            description="The number of attractions in the park.",
            # FIXME: strictly copying the original specification
            ge=0,
            json_schema_extra={"isDatafarmable": False},
        ),
    ]
    time_open: Annotated[
        float,
        Field(
            default=480.0,
            description="The number of minutes per day the park is open.",
            ge=0,
        ),
    ]
    erlang_shape: Annotated[
        list[int],
        Field(
            default_factory=lambda: [2] * NUM_ATTRACTIONS,
            description=(
                "The shape parameter of the Erlang distribution for each attraction duration."
            ),
        ),
    ]
    erlang_scale: Annotated[
        list[float],
        Field(
            default_factory=lambda: [1 / 9] * NUM_ATTRACTIONS,
            description=(
                "The rate parameter of the Erlang distribution for each attraction duration."
            ),
        ),
    ]
    queue_capacities: Annotated[
        list[int],
        Field(
            default_factory=lambda: [50] * NUM_ATTRACTIONS,
            description=(
                "The capacity of the queue for each attraction based on the "
                "portion of facilities allocated."
            ),
        ),
    ]
    depart_probabilities: Annotated[
        list[float],
        Field(
            default_factory=lambda: [0.2] * NUM_ATTRACTIONS,
            description=(
                "The probability that a tourist will depart the park after visiting an attraction."
            ),
        ),
    ]
    arrival_gammas: Annotated[
        list[int],
        Field(
            default_factory=lambda: [1] * NUM_ATTRACTIONS,
            description=(
                "The gamma values for the poisson distributions dictating the "
                "rates at which tourists entering the park arrive at each "
                "attraction"
            ),
        ),
    ]
    transition_probabilities: Annotated[
        list[list[float]],
        Field(
            default_factory=lambda: [
                [0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0],
                [0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0],
                [0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0],
                [0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0],
                [0.1, 0.1, 0.1, 0.1, 0, 0.1, 0.3],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0, 0.3],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2],
            ],
            description=(
                "The transition matrix that describes the probability of a tourist "
                "visiting each attraction after their current attraction."
            ),
        ),
    ]

    def _check_queue_capacities(self) -> None:
        if not all(cap >= 0 for cap in self.queue_capacities):
            raise ValueError("All queue capacities must be non-negative.")

    def _check_depart_probabilities(self) -> None:
        if len(self.depart_probabilities) != self.number_attractions:
            raise ValueError(
                "The number of departure probabilities must match the number of attractions."
            )
        if not all(0 <= prob <= 1 for prob in self.depart_probabilities):
            raise ValueError("All departure probabilities must be between 0 and 1.")

    def _check_arrival_gammas(self) -> None:
        if len(self.arrival_gammas) != self.number_attractions:
            raise ValueError("The number of arrivals must match the number of attractions.")
        if not all(gamma >= 0 for gamma in self.arrival_gammas):
            raise ValueError("All arrival gammas must be non-negative.")

    def _check_transition_probabilities(self) -> None:
        """Validate the structure and consistency of the transition matrix.

        Checks that the transition matrix is square (same number of rows and columns),
        and that the sum of each row and its corresponding departure probability equals
        1.

        Returns:
            bool: True if all checks pass.

        Raises:
            ValueError: If any row has the wrong shape or an invalid total probability.
        """
        transition_sums = [sum(row) for row in self.transition_probabilities]
        if not (
            all(
                len(row) == len(self.transition_probabilities)
                for row in self.transition_probabilities
            )
            and all(
                transition_sums[i] + self.depart_probabilities[i] == 1
                for i in range(self.number_attractions)
            )
        ):
            raise ValueError(
                "The values you entered are invalid. "
                "Check that each row and depart probability sums to 1."
            )

    def _check_erlang_shape(self) -> None:
        """Validate the Erlang shape parameters for each attraction.

        Checks that the number of shape parameters matches the number of attractions,
        and that all shape values are non-negative.

        Returns:
            bool: True if all shape parameters are valid.

        Raises:
            ValueError: If the number of shape parameters is incorrect.
        """
        if len(self.erlang_shape) != self.number_attractions:
            raise ValueError(
                "The number of attractions must equal the number of Erlang shape parameters."
            )
        if not all(gamma >= 0 for gamma in self.erlang_shape):
            raise ValueError("All Erlang shape parameters must be non-negative.")

    def _check_erlang_scale(self) -> None:
        """Validate the Erlang scale parameters for each attraction.

        Checks that the number of scale parameters matches the number of attractions,
        and that all scale values are non-negative.

        Returns:
            bool: True if all scale parameters are valid.

        Raises:
            ValueError: If the number of scale parameters is incorrect.
        """
        if len(self.erlang_scale) != self.number_attractions:
            raise ValueError("The number of attractions must equal the number of Erlang scales.")
        if not all(gamma >= 0 for gamma in self.erlang_scale):
            raise ValueError("All Erlang scale parameters must be non-negative.")

    @model_validator(mode="after")
    def _validate_model(self) -> Self:
        self._check_queue_capacities()
        self._check_depart_probabilities()
        self._check_arrival_gammas()
        self._check_transition_probabilities()
        self._check_erlang_shape()
        self._check_erlang_scale()

        if sum(self.queue_capacities) > self.park_capacity:
            raise ValueError(
                "The sum of the queue capacities must be less than or equal to the park capacity"
            )
        return self


@dataclass
class State:
    """Visitor counts and time statistics shared within one replication."""

    total_visitors: int = 0
    total_departed: int = 0
    time_average: float = 0.0
    previous_clock: float = 0.0


@simulation
def replicate(
    factors: AmusementParkConfig,
    rngs: list[MRG32k3a],
    arrival_model: InputModel,
    attraction_model: InputModel,
    destination_model: InputModel,
    service_models: Sequence[InputModel],
) -> tuple[int, float, float, list[float]]:
    """Return departures, departure fraction, mean occupancy, and attraction utilization."""
    # Keep local copies of factors to prevent excessive lookups
    num_attractions: int = factors.number_attractions
    arrival_gammas: list[int] = factors.arrival_gammas
    time_open: float = factors.time_open
    erlang_shape: list[int] = factors.erlang_shape
    erlang_scale: list[float] = factors.erlang_scale
    queue_capacities: list[int] = factors.queue_capacities
    transition_probabilities: list[list[float]] = factors.transition_probabilities
    depart_probabilities: list[float] = factors.depart_probabilities

    # initialize list of attractions to be selected upon arrival.
    attraction_range = range(num_attractions)
    destination_range = range(num_attractions + 1)
    depart_idx = destination_range[-1]

    # create external arrival probabilities for each attraction.
    arrival_prob_sum: float = float(sum(arrival_gammas))
    arrival_probabilities: list[float] = [
        arrival_gammas[i] / arrival_prob_sum for i in attraction_range
    ]

    # Initialize quantities to track:
    state = State()
    # initialize time average and utilization quantities.
    cumulative_util: list[float] = [0.0] * num_attractions

    env = simpy.Environment()
    queues = [simpy.Store(env, capacity=max(1, queue_capacities[i])) for i in attraction_range]
    busy = [False] * num_attractions

    def update_statistics() -> None:
        delta_time = env.now - state.previous_clock
        for i in attraction_range:
            if busy[i]:
                cumulative_util[i] += delta_time
        in_system = sum(len(queue.items) for queue in queues) + sum(busy)
        state.time_average += in_system * delta_time
        state.previous_clock = env.now

    def admit(attraction: int) -> None:
        if not busy[attraction]:
            busy[attraction] = True
            queues[attraction].put(None)
        elif len(queues[attraction].items) < queue_capacities[attraction]:
            queues[attraction].put(None)
        else:
            state.total_departed += 1

    def attraction_server(
        finished_attraction: int,
    ) -> Generator[simpy.Event, object, None]:
        while True:
            yield queues[finished_attraction].get()
            service_time = service_models[finished_attraction].random(
                rngs[3],
                alpha=erlang_shape[finished_attraction],
                beta=erlang_scale[finished_attraction],
            )
            while True:
                yield env.timeout(service_time)
                update_statistics()

                has_queued_visitor = bool(queues[finished_attraction].items)
                if has_queued_visitor:
                    yield queues[finished_attraction].get()
                    service_time = service_models[finished_attraction].random(
                        rngs[3],
                        alpha=erlang_shape[finished_attraction],
                        beta=erlang_scale[finished_attraction],
                    )
                else:
                    busy[finished_attraction] = False

                next_destination = destination_model.random(
                    rngs[2],
                    destination_range,
                    transition_probabilities[finished_attraction]
                    + [depart_probabilities[finished_attraction]],
                )
                if next_destination != depart_idx:
                    admit(next_destination)

                if not has_queued_visitor:
                    break

    def external_arrivals() -> Generator[simpy.Event, object, None]:
        while True:
            interarrival_time = arrival_model.random(rngs[0], arrival_prob_sum)
            yield env.timeout(interarrival_time)
            update_statistics()
            state.total_visitors += 1

            attraction_selection = attraction_model.random(
                rngs[1], attraction_range, arrival_probabilities
            )
            admit(attraction_selection)

    for i in attraction_range:
        env.process(attraction_server(i))
    env.process(external_arrivals())
    env.run(until=time_open)
    update_statistics()

    # Calculate overall percent utilization calculation for each attraction.
    cumulative_util = [cumulative_util[i] / time_open for i in attraction_range]

    # Calculate responses from simulation data.
    percent_departed = state.total_departed / state.total_visitors if state.total_visitors else 0
    return (
        state.total_departed,
        percent_departed,
        state.time_average / time_open,
        cumulative_util,
    )
