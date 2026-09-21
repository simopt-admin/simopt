"""Standalone Python simulation for the communication network."""

from __future__ import annotations

from collections.abc import Generator
from enum import IntEnum
from random import Random
from typing import Annotated, Final, Self

import numpy as np
import simpy
from pydantic import BaseModel, Field, model_validator

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt._markers import input_model, simulation
from simopt.input_models import InputModel

NUM_NETWORKS: Final = 10


class NetworkConfig(BaseModel):
    """Configuration for the queueing network model."""

    process_prob: Annotated[
        list[float],
        Field(
            default_factory=lambda: [0.1] * NUM_NETWORKS,
            description=("probability that a message will go through a particular network i"),
        ),
    ]
    cost_process: Annotated[
        list[float],
        Field(
            default_factory=lambda: [0.1 / (x + 1) for x in range(NUM_NETWORKS)],
            description="message processing cost of network i",
        ),
    ]
    cost_time: Annotated[
        list[float],
        Field(
            default_factory=lambda: [0.005] * NUM_NETWORKS,
            description=(
                "cost for the length of time a message spends in a network i per each unit of time"
            ),
        ),
    ]
    mode_transit_time: Annotated[
        list[float],
        Field(
            default_factory=lambda: [x + 1 for x in range(NUM_NETWORKS)],
            description=("mode time of transit for network i following a triangular distribution"),
        ),
    ]
    lower_limits_transit_time: Annotated[
        list[float],
        Field(
            default_factory=lambda: [0.5 + x for x in range(NUM_NETWORKS)],
            description=("lower limits for the triangular distribution for the transit time"),
        ),
    ]
    upper_limits_transit_time: Annotated[
        list[float],
        Field(
            default_factory=lambda: [1.5 + x for x in range(NUM_NETWORKS)],
            description=("upper limits for the triangular distribution for the transit time"),
        ),
    ]
    arrival_rate: Annotated[
        float,
        Field(
            default=1.0,
            description="arrival rate of messages following a Poisson process",
            gt=0,
        ),
    ]
    n_messages: Annotated[
        int,
        Field(
            default=1000,
            description="number of messages that arrives and needs to be routed",
            gt=0,
        ),
    ]
    n_networks: Annotated[
        int,
        Field(
            default=NUM_NETWORKS,
            description="number of networks",
            gt=0,
        ),
    ]

    def _check_process_prob(self) -> None:
        # Make sure probabilities are between 0 and 1.
        # Make sure probabilities sum up to 1.
        if (
            any(prob_i > 1.0 or prob_i < 0 for prob_i in self.process_prob)
            or abs(sum(self.process_prob) - 1.0) > 1e-10
        ):
            raise ValueError(
                "All elements in process_prob must be between 0 and 1 and the sum of "
                "all of the elements in process_prob must equal 1."
            )

    def _check_cost_process(self) -> None:
        if any(cost_i <= 0 for cost_i in self.cost_process):
            raise ValueError("All elements in cost_process must be greater than 0.")

    def _check_cost_time(self) -> None:
        if any(cost_time_i <= 0 for cost_time_i in self.cost_time):
            raise ValueError("All elements in cost_time must be greater than 0.")

    def _check_mode_transit_time(self) -> None:
        if any(transit_time_i <= 0 for transit_time_i in self.mode_transit_time):
            raise ValueError("All elements in mode_transit_time must be greater than 0.")

    def _check_lower_limits_transit_time(self) -> None:
        if any(lower_i <= 0 for lower_i in self.lower_limits_transit_time):
            raise ValueError("All elements in lower_limits_transit_time must be greater than 0.")

    def _check_upper_limits_transit_time(self) -> None:
        if any(upper_i <= 0 for upper_i in self.upper_limits_transit_time):
            raise ValueError("All elements in upper_limits_transit_time must be greater than 0.")

    @model_validator(mode="after")
    def _validate_model(self) -> Self:
        self._check_process_prob()
        self._check_cost_process()
        self._check_cost_time()
        self._check_mode_transit_time()
        self._check_lower_limits_transit_time()
        self._check_upper_limits_transit_time()

        if len(self.process_prob) != self.n_networks:
            raise ValueError("The length of process_prob must equal n_networks.")
        if len(self.cost_process) != self.n_networks:
            raise ValueError("The length of cost_process must equal n_networks.")
        if len(self.cost_time) != self.n_networks:
            raise ValueError("The length of cost_time must equal n_networks.")
        if len(self.mode_transit_time) != self.n_networks:
            raise ValueError("The length of mode_transit_time must equal n_networks.")
        if len(self.lower_limits_transit_time) != self.n_networks:
            raise ValueError("The length of lower_limits_transit_time must equal n_networks.")
        if len(self.upper_limits_transit_time) != self.n_networks:
            raise ValueError("The length of upper_limits_transit_time must equal n_networks.")

        if any(
            self.mode_transit_time[i] < self.lower_limits_transit_time[i]
            for i in range(self.n_networks)
        ):
            raise ValueError(
                "The mode_transit time must be greater than or equal to the "
                "corresponding lower_limits_transit_time for each network."
            )
        if any(
            self.upper_limits_transit_time[i] < self.mode_transit_time[i]
            for i in range(self.n_networks)
        ):
            raise ValueError(
                "The mode_transit time must be less than or equal to the corresponding "
                "upper_limits_transit_time for each network."
            )

        return self


@input_model
class RouteInputModel(InputModel):
    """Input model for routing choices in the network."""

    def random(self, rng: Random, choices: list[int], weights: list[float], k: int) -> list[int]:
        """Sample network routes using the supplied weights."""
        return rng.choices(choices, weights, k=k)


@simulation
def replicate(
    factors: NetworkConfig,
    rngs: list[MRG32k3a],
    arrival_model: InputModel,
    route_model: InputModel,
    service_model: InputModel,
) -> float:
    """Simulate one replication and return total cost using the supplied samplers."""
    # Determine total number of arrivals to simulate.
    total_arrivals = factors.n_messages
    arrival_rate = factors.arrival_rate
    n_networks = factors.n_networks
    process_prob = factors.process_prob
    lower_limits_transit_time = factors.lower_limits_transit_time
    upper_limits_transit_time = factors.upper_limits_transit_time
    mode_transit_time = factors.mode_transit_time
    cost_process = factors.cost_process
    cost_time = factors.cost_time

    # Generate all interarrival, network routes, and service times before the
    # simulation run.
    arrival_times = [arrival_model.random(rngs[0], arrival_rate) for _ in range(total_arrivals)]
    network_routes = route_model.random(
        rngs[1],
        list(range(n_networks)),
        weights=process_prob,
        k=total_arrivals,
    )
    service_times = [
        service_model.random(
            rngs[2],
            low=lower_limits_transit_time[route],
            high=upper_limits_transit_time[route],
            mode=mode_transit_time[route],
        )
        for route in network_routes
    ]

    # Alias columns by index
    class Col(IntEnum):
        ARR = 0  # arrival time to queue
        ROUTE = 1  # network route
        SVC = 2  # service time
        DONE = 3  # service completion time
        SOJ = 4  # sojourn time
        WAIT = 5  # waiting time
        PROC_COST = 6  # processing cost
        TIME_COST = 7  # time cost
        TOTAL_COST = 8  # total cost

    message_mat = np.zeros((total_arrivals, 9))
    message_mat[:, Col.ARR] = np.cumsum(arrival_times)
    message_mat[:, Col.ROUTE] = network_routes
    message_mat[:, Col.SVC] = service_times
    # Fill in entries for messages' metrics.
    routes = message_mat[:, Col.ROUTE].astype(int)
    arrival = message_mat[:, Col.ARR]
    service = message_mat[:, Col.SVC]

    env = simpy.Environment()
    networks = [simpy.Resource(env, capacity=1) for _ in range(n_networks)]

    def message(i: int) -> Generator[simpy.Event, object, None]:
        net = routes[i]
        arr_i = arrival[i]
        svc_i = service[i]
        curr_message = message_mat[i]

        yield env.timeout(arr_i)
        with networks[net].request() as request:
            yield request
            curr_message[Col.WAIT] = env.now - arr_i
            yield env.timeout(svc_i)

        curr_message[Col.DONE] = env.now
        curr_message[Col.SOJ] = curr_message[Col.DONE] - arr_i

    for i in range(total_arrivals):
        env.process(message(i))
    env.run()

    # Vectorized cost computations after SOJ is known
    message_mat[:, Col.PROC_COST] = np.array(cost_process)[routes]
    message_mat[:, Col.TIME_COST] = np.array(cost_time)[routes] * message_mat[:, Col.SOJ]
    message_mat[:, Col.TOTAL_COST] = message_mat[:, Col.PROC_COST] + message_mat[:, Col.TIME_COST]

    # Compute total costs for the simulation run.
    total_cost = np.sum(message_mat[:, Col.TOTAL_COST])
    return total_cost  # noqa: RET504 - Preserve the extracted calculation and variable name.
