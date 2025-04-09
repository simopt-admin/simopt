"""Simulate an M/M/1 queue.

A detailed description of the model/problem can be found
`here <https://simopt.readthedocs.io/en/latest/mm1queue.html>`__.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Callable

import numpy as np

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.base import ConstraintType, Model, Problem, VariableType
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

    Attributes:
    ----------
    name : string
        name of model
    n_rngs : int
        number of random-number generators used to run a simulation replication
    n_responses : int
        number of responses (performance measures)
    factors : dict
        changeable factors of the simulation model
    specifications : dict
        details of each factor (for GUI, data validation, and defaults)
    check_factor_list : dict
        switch case for checking factor simulatability

    Arguments:
    ---------
    fixed_factors : nested dict
        fixed factors of the simulation model

    See Also:
    --------
    base.Model
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

    def replicate(self, rng_list: list[MRG32k3a]) -> tuple[dict, dict]:
        """Simulate a single replication for the current model factors.

        Arguments:
        ---------
        rng_list : list of mrg32k3a.mrg32k3a.MRG32k3a objects
            rngs for model to use when simulating a replication

        Returns:
        -------
        responses : dict
            performance measures of interest
            "avg_sojourn_time" = average sojourn time
            "avg_waiting_time" = average waiting time
            "frac_cust_wait" = fraction of customers who wait
        gradients : dict of dicts
            gradient estimates for each response
        """
        mu: float = self.factors["mu"]
        epsilon: float = self.factors["epsilon"]
        warmup: int = self.factors["warmup"]
        people: int = self.factors["people"]
        f_lambda: float = self.factors["lambda"]
        # Designate separate RNGs for interarrival and serivce times.
        arrival_rng = rng_list[0]
        service_rng = rng_list[1]
        # Set mu to be at least epsilon.
        mu_floor = max(mu, epsilon)
        # Calculate total number of arrivals to simulate.
        total = warmup + people
        # Generate all interarrival and service times up front.
        arrival_times = [arrival_rng.expovariate(f_lambda) for _ in range(total)]
        service_times = [service_rng.expovariate(mu_floor) for _ in range(total)]

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


"""
Summary
-------
Minimize the mean sojourn time of an M/M/1 queue plus a cost term.
"""


class MM1MinMeanSojournTime(Problem):
    """Base class to implement simulation-optimization problems.

    Attributes:
    ----------
    name : string
        name of problem
    dim : int
        number of decision variables
    n_objectives : int
        number of objectives
    n_stochastic_constraints : int
        number of stochastic constraints
    minmax : tuple of int (+/- 1)
        indicator of maximization (+1) or minimization (-1) for each objective
    constraint_type : string
        description of constraints types:
            "unconstrained", "box", "deterministic", "stochastic"
    variable_type : string
        description of variable types:
            "discrete", "continuous", "mixed"
    lower_bounds : tuple
        lower bound for each decision variable
    upper_bounds : tuple
        upper bound for each decision variable
    gradient_available : bool
        indicates if gradient of objective function is available
    optimal_value : tuple
        optimal objective function value
    optimal_solution : tuple
        optimal solution
    model : Model object
        associated simulation model that generates replications
    model_default_factors : dict
        default values for overriding model-level default factors
    model_fixed_factors : dict
        combination of overriden model-level factors and defaults
    model_decision_factors : set of str
        set of keys for factors that are decision variables
    rng_list : list of mrg32k3a.mrg32k3a.MRG32k3a objects
        list of RNGs used to generate a random initial solution
        or a random problem instance
    factors : dict
        changeable factors of the problem
    specifications : dict
        details of each factor (for GUI, data validation, and defaults)

    Arguments:
    ---------
    name : str
        user-specified name for problem
    fixed_factors : dict
        dictionary of user-specified problem factors
    model_fixed_factors : dict
        subset of user-specified non-decision factors to pass through to the model

    See Also:
    --------
    base.Problem
    """

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

    def vector_to_factor_dict(self, vector: tuple) -> dict:
        """Convert a vector of variables to a dictionary with factor keys.

        Arguments:
        ---------
        vector : tuple
            vector of values associated with decision variables

        Returns:
        -------
        factor_dict : dictionary
            dictionary with factor keys and associated values
        """
        return {"mu": vector[0]}

    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:
        """Convert a dictionary with factor keys to a vector of variables.

        Arguments:
        ---------
        factor_dict : dictionary
            dictionary with factor keys and associated values

        Returns:
        -------
        vector : tuple
            vector of values associated with decision variables
        """
        return (factor_dict["mu"],)

    def response_dict_to_objectives(self, response_dict: dict) -> tuple:
        """Convert a dictionary with response keys to a vector of objectives.

        Arguments:
        ---------
        response_dict : dictionary
            dictionary with response keys and associated values

        Returns:
        -------
        objectives : tuple
            vector of objectives
        """
        return (response_dict["avg_sojourn_time"],)

    def deterministic_objectives_and_gradients(self, x: tuple) -> tuple[tuple, tuple]:
        """Compute deterministic components of objectives for a solution `x`.

        Arguments:
        ---------
        x : tuple
            vector of decision variables

        Returns:
        -------
        det_objectives : tuple
            vector of deterministic components of objectives
        det_objectives_gradients : tuple
            vector of gradients of deterministic components of objectives
        """
        det_objectives = (self.factors["cost"] * (x[0] ** 2),)
        det_objectives_gradients = ((2 * self.factors["cost"] * x[0],),)
        return det_objectives, det_objectives_gradients

    def check_deterministic_constraints(self, x: tuple) -> bool:
        """Check if a solution `x` satisfies the problem's deterministic constraints.

        Arguments:
        ---------
        x : tuple
            vector of decision variables

        Returns:
        -------
        satisfies : bool
            indicates if solution `x` satisfies the deterministic constraints.
        """
        # Check box constraints.
        return super().check_deterministic_constraints(x)

    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
        """Generate a random solution for starting or restarting solvers.

        Arguments:
        ---------
        rand_sol_rng : mrg32k3a.mrg32k3a.MRG32k3a object
            random-number generator used to sample a new random solution

        Returns:
        -------
        x : tuple
            vector of decision variables
        """
        # Generate an Exponential(rate = 1/3) r.v.
        return (rand_sol_rng.expovariate(1 / 3),)
