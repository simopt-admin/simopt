"""Simulate an M/M/1 queue."""

from __future__ import annotations

from typing import Annotated, ClassVar

import numpy as np
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
from simopt.simulations.mm1queue import MM1QueueConfig, replicate
from simopt.utils import override


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

    def replicate(self, factors: MM1QueueConfig, rngs: list[MRG32k3a]) -> tuple[dict, dict]:
        """Simulate a single replication for the current model factors.

        Args:
            rngs (list[MRG32k3a]): Random number generators used to simulate
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
        (
            mean_sojourn_time,
            grad_mean_sojourn_time_mu,
            grad_mean_sojourn_time_lambda,
            mean_waiting_time,
            grad_mean_waiting_time_mu,
            grad_mean_waiting_time_lambda,
            fraction_wait,
        ) = replicate(factors, rngs, self.arrival_model, self.service_model)
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

    @override
    def build(self) -> dsl.Model:
        problem = dsl.Model()
        mu = problem.add_continuous_variable(
            lb=0.0, ub=np.inf, initial=self.factors["initial_solution"][0]
        )
        simulation = self.add_simulation(problem, {"mu": mu})
        problem.minimize(
            dsl.mean(simulation.metric("avg_sojourn_time")) + self.factors["cost"] * mu * mu
        )
        return problem

    def vector_to_factor_dict(self, vector: tuple) -> dict:
        return {"mu": vector[0]}

    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
        # Generate an Exponential(rate = 1/3) r.v.
        return (rand_sol_rng.expovariate(1 / 3),)
