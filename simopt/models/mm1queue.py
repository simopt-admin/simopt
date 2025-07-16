"""Simulate an M/M/1 queue."""

from __future__ import annotations

from enum import IntEnum
from typing import Callable

import numpy as np

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.base import ConstraintType, Model, Problem, VariableType
from simopt.input_models import Exp
from simopt.utils import classproperty, override


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

    @classproperty
    @override
    def class_name_abbr(cls) -> str:
        return "MM1"

    @classproperty
    @override
    def class_name(cls) -> str:
        return "MM1 Queue"

    @classproperty
    @override
    def n_rngs(cls) -> int:
        return 2

    @classproperty
    @override
    def n_responses(cls) -> int:
        return 3

    @classproperty
    @override
    def specifications(cls) -> dict[str, dict]:
        return {
            "lambda": {
                "description": "rate parameter of interarrival time distribution",
                "datatype": float,
                "default": 1.5,
            },
            "mu": {
                "description": "rate parameter of service time distribution",
                "datatype": float,
                "default": 3.0,
            },
            "epsilon": {
                "description": "the minimum value of mu",
                "datatype": float,
                "default": 0.001,
            },
            "warmup": {
                "description": (
                    "number of people as warmup before collecting statistics"
                ),
                "datatype": int,
                "default": 20,
            },
            "people": {
                "description": (
                    "number of people from which to calculate the average sojourn time"
                ),
                "datatype": int,
                "default": 50,
            },
        }

    @property
    @override
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "lambda": self._check_lambda,
            "mu": self._check_mu,
            "epsilon": self._check_epsilon,
            "warmup": self._check_warmup,
            "people": self._check_people,
        }

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

    def _check_lambda(self) -> None:
        if self.factors["lambda"] <= 0:
            raise ValueError("lambda must be greater than 0.")

    def _check_mu(self) -> None:
        if self.factors["mu"] <= 0:
            raise ValueError("mu must be greater than 0.")

    def _check_warmup(self) -> None:
        if self.factors["warmup"] < 0:
            raise ValueError("warmup must be greater than or equal to 0.")

    def _check_people(self) -> None:
        if self.factors["people"] < 1:
            raise ValueError("people must be greater than or equal to 1.")

    def _check_epsilon(self) -> bool:
        return self.factors["epsilon"] > 0

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
        # Input entries for first customer's queueing experience.
        first_cust = cust_mat[0]
        first_cust[Col.DONE] = first_cust[Col.ARR] + first_cust[Col.SVC]
        first_cust[Col.SOJ] = first_cust[Col.SVC]
        # first_cust[Col.WAIT] = 0
        # cfirst_cust[Col.IN_SYS] = 0
        first_cust[Col.G_SOJ_MU] = -first_cust[Col.SVC] / mu_floor
        # first_cust[Col.G_WAIT_MU] = 0
        # first_cust[Col.G_SOJ_LAM] = 0
        # first_cust[Col.G_WAIT_LAM] = 0
        # Fill in entries for remaining customers' experiences.
        for i in range(1, total):
            # Views into the customer matrix.
            # NOT copies, so be careful!
            curr_cust = cust_mat[i]
            prev_cust = cust_mat[i - 1]

            arrival = curr_cust[Col.ARR]
            prev_departure = prev_cust[Col.DONE]

            # Completion time
            curr_cust[Col.DONE] = max(arrival, prev_departure) + curr_cust[Col.SVC]
            # Sojourn and waiting times
            curr_cust[Col.SOJ] = curr_cust[Col.DONE] - arrival
            curr_cust[Col.WAIT] = curr_cust[Col.SOJ] - curr_cust[Col.SVC]

            # Number in system at arrival
            lookback = int(prev_cust[Col.IN_SYS]) + 1
            arrivals_in_window = cust_mat[i - lookback : i, Col.DONE]
            curr_cust[Col.IN_SYS] = np.count_nonzero(arrivals_in_window > arrival)

            # Gradients w.r.t mu
            n_in_sys = int(curr_cust[Col.IN_SYS])
            grad_range = cust_mat[i - n_in_sys : i + 1, Col.SVC]
            curr_cust[Col.G_SOJ_MU] = -np.sum(grad_range) / mu_floor
            curr_cust[Col.G_WAIT_MU] = -np.sum(grad_range[:-1]) / mu_floor

            # Gradients w.r.t lambda
            # cust_mat[i, 8] = 0.0
            # cust_mat[i, 9] = 0.0
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
            response_key: dict.fromkeys(self.specifications, np.nan)
            for response_key in responses
        }
        gradients["avg_sojourn_time"]["mu"] = float(grad_mean_sojourn_time_mu)
        gradients["avg_sojourn_time"]["lambda"] = float(grad_mean_sojourn_time_lambda)
        gradients["avg_waiting_time"]["mu"] = float(grad_mean_waiting_time_mu)
        gradients["avg_waiting_time"]["lambda"] = float(grad_mean_waiting_time_lambda)
        return responses, gradients


class MM1MinMeanSojournTime(Problem):
    """Base class to implement simulation-optimization problems."""

    @classproperty
    @override
    def class_name_abbr(cls) -> str:
        return "MM1-1"

    @classproperty
    @override
    def class_name(cls) -> str:
        return "Min Mean Sojourn Time for MM1 Queue"

    @classproperty
    @override
    def n_objectives(cls) -> int:
        return 1

    @classproperty
    @override
    def n_stochastic_constraints(cls) -> int:
        return 0

    @classproperty
    @override
    def minmax(cls) -> tuple[int]:
        return (-1,)

    @classproperty
    @override
    def constraint_type(cls) -> ConstraintType:
        return ConstraintType.BOX

    @classproperty
    @override
    def variable_type(cls) -> VariableType:
        return VariableType.CONTINUOUS

    @classproperty
    @override
    def gradient_available(cls) -> bool:
        return True

    @classproperty
    @override
    def optimal_value(cls) -> float | None:
        return None

    @classproperty
    @override
    def optimal_solution(cls) -> tuple | None:
        return None

    @classproperty
    @override
    def model_default_factors(cls) -> dict:
        return {"warmup": 50, "people": 200}

    @classproperty
    @override
    def model_decision_factors(cls) -> set[str]:
        return {"mu"}

    @classproperty
    @override
    def specifications(cls) -> dict[str, dict]:
        return {
            "initial_solution": {
                "description": "initial solution from which solvers start",
                "datatype": tuple,
                "default": (5,),
            },
            "budget": {
                "description": "max # of replications for a solver to take",
                "datatype": int,
                "default": 1000,
                "isDatafarmable": False,
            },
            "cost": {
                "description": "cost for increasing service rate",
                "datatype": float,
                "default": 0.1,
            },
        }

    @property
    @override
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "cost": self._check_cost,
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget,
        }

    @classproperty
    @override
    def dim(cls) -> int:
        return 1

    @classproperty
    @override
    def lower_bounds(cls) -> tuple:
        return (0,) * cls.dim

    @classproperty
    @override
    def upper_bounds(cls) -> tuple:
        return (np.inf,) * cls.dim

    def __init__(
        self,
        name: str = "MM1-1",
        fixed_factors: dict | None = None,
        model_fixed_factors: dict | None = None,
    ) -> None:
        """Initialize the MM1MinMeanSojournTime problem.

        Args:
            name (str, optional): user-specified name for problem. Defaults to "MM1-1".
            fixed_factors (dict, optional): fixed factors of the simulation model.
                Defaults to None.
            model_fixed_factors (dict, optional): subset of user-specified
                non-decision factors to pass through to the model. Defaults to None.
        """
        # Let the base class handle default arguments.
        super().__init__(
            name=name,
            fixed_factors=fixed_factors,
            model_fixed_factors=model_fixed_factors,
            model=MM1Queue,
        )

    def _check_cost(self) -> None:
        if self.factors["cost"] <= 0:
            raise ValueError("cost must be greater than 0.")

    @override
    def vector_to_factor_dict(self, vector: tuple) -> dict:
        return {"mu": vector[0]}

    @override
    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:
        return (factor_dict["mu"],)

    @override
    def response_dict_to_objectives(self, response_dict: dict) -> tuple:
        return (response_dict["avg_sojourn_time"],)

    @override
    def deterministic_objectives_and_gradients(self, x: tuple) -> tuple[tuple, tuple]:
        det_objectives = (self.factors["cost"] * (x[0] ** 2),)
        det_objectives_gradients = ((2 * self.factors["cost"] * x[0],),)
        return det_objectives, det_objectives_gradients

    @override
    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
        # Generate an Exponential(rate = 1/3) r.v.
        return (rand_sol_rng.expovariate(1 / 3),)
