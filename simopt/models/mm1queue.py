"""Simulate an M/M/1 queue."""

from __future__ import annotations

from collections.abc import Generator
from enum import IntEnum
from typing import Annotated, ClassVar

import numpy as np
import simpy
from pydantic import BaseModel, Field

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.base import (
    ConstraintType,
    Model,
    Objective,
    Problem,
    RepResult,
    VariableType,
)
from simopt.input_models import Exp


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


class MM1MinMeanSojournTimeConfig(BaseModel):
    """Configuration model for MM1 Min Mean Sojourn Time Problem.

    Min Mean Sojourn Time for MM1 Queue simulation-optimization problem.
    """

    initial_solution: Annotated[
        tuple[float, ...],
        Field(
            default=(5,),
            description="initial solution from which solvers start",
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
    cost: Annotated[
        float,
        Field(
            default=0.1,
            description="cost for increasing service rate",
            gt=0,
        ),
    ]


class MM1Queue(Model):
    """MM1 Queue Simulation Model.

    A model that simulates an M/M/1 queue with an Exponential(lambda)
    interarrival time distribution and an Exponential(x) service time
    distribution. Returns:
    - the average sojourn time
    - the average waiting time
    - the fraction of customers who wait
    for customers after a warmup period.
    """

    class_name_abbr: ClassVar[str] = "MM1"
    class_name: ClassVar[str] = "MM1 Queue"
    config_class: ClassVar[type[BaseModel]] = MM1QueueConfig
    n_rngs: ClassVar[int] = 2
    n_responses: ClassVar[int] = 3

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize the MM1Queue model.

        Args:
            fixed_factors (dict, optional): fixed factors of the simulation model.
                Defaults to None.
        """
        # Let the base class handle default arguments.
        super().__init__(fixed_factors)
        self.arrival_model = Exp()
        self.service_model = Exp()

    def before_replicate(self, rng_list: list[MRG32k3a]) -> None:
        self.arrival_model.set_rng(rng_list[0])
        self.service_model.set_rng(rng_list[1])

    def replicate(self) -> tuple[dict, dict]:
        """Simulate a single replication for the current model factors.

        Args:
            rng_list (list[MRG32k3a]): Random number generators used to simulate
                the replication.

        Returns:
            tuple[dict, dict]: A tuple containing:
                - responses (dict): Performance measures of interest, including:
                    - "avg_sojourn_time": Average sojourn time.
                    - "avg_waiting_time": Average waiting time.
                    - "frac_cust_wait": Fraction of customers who wait.
                - gradients (dict): A dictionary of gradient estimates for
                    each response.
        """
        mu: float = self.factors["mu"]
        epsilon: float = self.factors["epsilon"]
        warmup: int = self.factors["warmup"]
        people: int = self.factors["people"]
        f_lambda: float = self.factors["lambda"]
        # Designate separate RNGs for interarrival and serivce times.
        # Set mu to be at least epsilon.
        mu_floor = max(mu, epsilon)
        # Calculate total number of arrivals to simulate.
        total = warmup + people
        # Generate all interarrival and service times up front.
        arrival_times = [self.arrival_model.random(f_lambda) for _ in range(total)]
        service_times = [self.service_model.random(mu_floor) for _ in range(total)]

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
        # Compose responses and gradients.
        responses = {
            "avg_sojourn_time": mean_sojourn_time,
            "avg_waiting_time": mean_waiting_time,
            "frac_cust_wait": fraction_wait,
        }
        gradients = {
            response_key: dict.fromkeys(self.specifications, np.nan) for response_key in responses
        }
        gradients["avg_sojourn_time"]["mu"] = float(grad_mean_sojourn_time_mu)
        gradients["avg_sojourn_time"]["lambda"] = float(grad_mean_sojourn_time_lambda)
        gradients["avg_waiting_time"]["mu"] = float(grad_mean_waiting_time_mu)
        gradients["avg_waiting_time"]["lambda"] = float(grad_mean_waiting_time_lambda)
        return responses, gradients


class MM1MinMeanSojournTime(Problem):
    """Base class to implement simulation-optimization problems."""

    class_name_abbr: ClassVar[str] = "MM1-1"
    class_name: ClassVar[str] = "Min Mean Sojourn Time for MM1 Queue"
    config_class: ClassVar[type[BaseModel]] = MM1MinMeanSojournTimeConfig
    model_class: ClassVar[type[Model]] = MM1Queue
    n_objectives: ClassVar[int] = 1
    n_stochastic_constraints: ClassVar[int] = 0
    minmax: ClassVar[tuple[int, ...]] = (-1,)
    constraint_type: ClassVar[ConstraintType] = ConstraintType.BOX
    variable_type: ClassVar[VariableType] = VariableType.CONTINUOUS
    gradient_available: ClassVar[bool] = True
    optimal_value: ClassVar[float | None] = None
    optimal_solution: tuple | None = None
    model_default_factors: ClassVar[dict] = {"warmup": 50, "people": 200}
    model_decision_factors: ClassVar[set[str]] = {"mu"}

    @property
    def dim(self) -> int:
        return 1

    @property
    def lower_bounds(self) -> tuple:
        return (0,) * self.dim

    @property
    def upper_bounds(self) -> tuple:
        return (np.inf,) * self.dim

    def vector_to_factor_dict(self, vector: tuple) -> dict:
        return {"mu": vector[0]}

    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:
        return (factor_dict["mu"],)

    def replicate(self, x: tuple) -> RepResult:
        responses, gradients = self.model.replicate()
        objectives = [
            Objective(
                stochastic=responses["avg_sojourn_time"],
                stochastic_gradients=gradients["avg_sojourn_time"]["mu"],
                deterministic=self.factors["cost"] * (x[0] ** 2),
                deterministic_gradients=2 * self.factors["cost"] * x[0],
            )
        ]
        return RepResult(objectives=objectives)

    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
        # Generate an Exponential(rate = 1/3) r.v.
        return (rand_sol_rng.expovariate(1 / 3),)
