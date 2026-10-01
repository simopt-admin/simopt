"""Standalone Python simulation for the M/M/1 queue."""

from __future__ import annotations

from collections.abc import Generator
from enum import IntEnum
from typing import Annotated

import numpy as np
import simpy
from pydantic import BaseModel, Field

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt._markers import simulation
from simopt.input_models import InputModel


class MM1QueueConfig(BaseModel):
    """Configuration model for MM1 Queue simulation.

    A model that simulates an M/M/1 queue with an Exponential(lambda)
    interarrival time distribution and an Exponential(x) service time
    distribution. Returns:
    - the average sojourn time
    - the average waiting time
    - the fraction of customers who wait
    for customers after a warmup period.
    """

    lambda_: Annotated[
        float,
        Field(
            default=1.5,
            description="rate parameter of interarrival time distribution",
            gt=0,
            alias="lambda",
        ),
    ]
    mu: Annotated[
        float,
        Field(
            default=3.0,
            description="rate parameter of service time distribution",
            gt=0,
        ),
    ]
    epsilon: Annotated[
        float,
        Field(
            default=0.001,
            description="the minimum value of mu",
            gt=0,
        ),
    ]
    warmup: Annotated[
        int,
        Field(
            default=20,
            description="number of people as warmup before collecting statistics",
            ge=0,
        ),
    ]
    people: Annotated[
        int,
        Field(
            default=50,
            description=("number of people from which to calculate the average sojourn time"),
            ge=1,
        ),
    ]


@simulation
def replicate(
    factors: MM1QueueConfig,
    rngs: list[MRG32k3a],
    arrival_model: InputModel,
    service_model: InputModel,
) -> tuple[float, float, float, float, float, float, float]:
    """Simulate one replication using the supplied arrival and service samplers.

    Returns the mean sojourn time and its mu/lambda gradients, mean waiting
    time and its mu/lambda gradients, and fraction of customers who wait.
    The retained IPA service gradients assume exponential service times;
    arbitrary compatible replacement samplers are supported in Python but
    do not necessarily satisfy that gradient assumption.
    """
    mu: float = factors.mu
    epsilon: float = factors.epsilon
    warmup: int = factors.warmup
    people: int = factors.people
    f_lambda: float = factors.lambda_
    # Designate separate RNGs for interarrival and serivce times.
    # Set mu to be at least epsilon.
    mu_floor = max(mu, epsilon)
    # Calculate total number of arrivals to simulate.
    total = warmup + people
    # Generate all interarrival and service times up front.
    arrival_times = [arrival_model.random(rngs[0], f_lambda) for _ in range(total)]
    service_times = [service_model.random(rngs[1], mu_floor) for _ in range(total)]

    # Create matrix storing times and metrics for each customer:
    #     column 0 : arrival time to queue;
    #     column 1 : service time;
    #     column 2 : service completion time;
    #     column 3 : sojourn time;
    #     column 4 : waiting time;
    #     column 5 : number of customers in system at arrival;
    #     column 6 : IPA gradient of sojourn time w.r.t. mu;
    #     column 7 : IPA gradient of waiting time w.r.t. mu;
    #     column 8 : IPA gradient of sojourn time w.r.t. lambda;
    #     column 9 : IPA gradient of waiting time w.r.t. lambda.
    # Alias columns by index
    class Col(IntEnum):
        ARR = 0
        SVC = 1
        DONE = 2
        SOJ = 3
        WAIT = 4
        IN_SYS = 5
        G_SOJ_MU = 6
        G_WAIT_MU = 7
        G_SOJ_LAM = 8
        G_WAIT_LAM = 9

    cust_mat = np.zeros((total, 10))
    cust_mat[:, Col.ARR] = np.cumsum(arrival_times)
    cust_mat[:, Col.SVC] = service_times

    env = simpy.Environment()
    server = simpy.Resource(env, capacity=1)

    def customer(i: int) -> Generator[simpy.Event, object, None]:
        curr_cust = cust_mat[i]
        arrival = curr_cust[Col.ARR]
        yield env.timeout(arrival)

        # Number in system at arrival
        curr_cust[Col.IN_SYS] = server.count + len(server.queue)

        with server.request() as request:
            yield request
            yield env.timeout(curr_cust[Col.SVC])

        curr_cust[Col.DONE] = env.now
        curr_cust[Col.SOJ] = curr_cust[Col.DONE] - arrival
        curr_cust[Col.WAIT] = curr_cust[Col.SOJ] - curr_cust[Col.SVC]

        # Gradients w.r.t lambda
        # cust_mat[i, 8] = 0.0
        # cust_mat[i, 9] = 0.0

    for i in range(total):
        env.process(customer(i))
    env.run()

    # Calculate IPA gradients with respect to mu. A customer's completion-time
    # gradient carries through the entire busy period, including customers who
    # have already departed. Below the service-rate floor, the simulated service
    # times do not depend on mu; at the floor, use the zero-valued left derivative.
    if mu > epsilon:
        prev_done_grad_mu = 0.0
        for i in range(total):
            arrival = cust_mat[i, Col.ARR]
            service = cust_mat[i, Col.SVC]
            busy = i > 0 and cust_mat[i - 1, Col.DONE] > arrival

            grad_wait_mu = prev_done_grad_mu if busy else 0.0
            grad_service_mu = -service / mu
            grad_sojourn_mu = grad_wait_mu + grad_service_mu

            cust_mat[i, Col.G_WAIT_MU] = grad_wait_mu
            cust_mat[i, Col.G_SOJ_MU] = grad_sojourn_mu
            prev_done_grad_mu = grad_sojourn_mu

    cust_mat_warmup = cust_mat[warmup:]
    # Compute average sojourn time and its gradient.
    mean_sojourn_time = np.mean(cust_mat_warmup[:, Col.SOJ])
    grad_mean_sojourn_time_mu = np.mean(cust_mat_warmup[:, Col.G_SOJ_MU])
    grad_mean_sojourn_time_lambda = np.mean(cust_mat_warmup[:, Col.G_SOJ_LAM])
    # Compute average waiting time and its gradient.
    mean_waiting_time = np.mean(cust_mat_warmup[:, Col.WAIT])
    grad_mean_waiting_time_mu = np.mean(cust_mat_warmup[:, Col.G_WAIT_MU])
    grad_mean_waiting_time_lambda = np.mean(cust_mat_warmup[:, Col.G_WAIT_LAM])
    # Compute fraction of customers who wait.
    fraction_wait = np.mean(cust_mat_warmup[:, Col.IN_SYS] > 0)
    return (
        float(mean_sojourn_time),
        float(grad_mean_sojourn_time_mu),
        float(grad_mean_sojourn_time_lambda),
        float(mean_waiting_time),
        float(grad_mean_waiting_time_mu),
        float(grad_mean_waiting_time_lambda),
        float(fraction_wait),
    )
