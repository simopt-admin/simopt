"""
Summary
-------
Simulate a M/M/1 queue.
A detailed description of the model/problem can be found
`here <https://simopt.readthedocs.io/en/latest/mm1queue.html>`__.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from mrg32k3a.mrg32k3a import MRG32k3a

from simopt.base import ConstraintType, Model, Problem, VariableType


class MM1Queue(Model):
    """
    A model that simulates an M/M/1 queue with an Exponential(lambda)
    interarrival time distribution and an Exponential(x) service time
    distribution. Returns:
    - the average sojourn time
    - the average waiting time
    - the fraction of customers who wait
    for customers after a warmup period.

    Attributes
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

    Arguments
    ---------
    fixed_factors : nested dict
        fixed factors of the simulation model

    See also
    --------
    base.Model
    """

    @property
    def name(self) -> str:
        return "MM1"

    @property
    def n_rngs(self) -> int:
        return 2

    @property
    def n_responses(self) -> int:
        return 3

    @property
    def specifications(self) -> dict[str, dict]:
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
                "description": "number of people as warmup before collecting statistics",
                "datatype": int,
                "default": 20,
            },
            "people": {
                "description": "number of people from which to calculate the average sojourn time",
                "datatype": int,
                "default": 50,
            },
        }

    @property
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "lambda": self.check_lambda,
            "mu": self.check_mu,
            "epsilon": self.check_epsilon,
            "warmup": self.check_warmup,
            "people": self.check_people,
        }

    def __init__(self, fixed_factors: dict | None = None) -> None:
        # Let the base class handle default arguments.
        super().__init__(fixed_factors)

    def check_lambda(self) -> None:
        if self.factors["lambda"] <= 0:
            raise ValueError("lambda must be greater than 0.")

    def check_mu(self) -> None:
        if self.factors["mu"] <= 0:
            raise ValueError("mu must be greater than 0.")

    def check_warmup(self) -> None:
        if self.factors["warmup"] < 0:
            raise ValueError("warmup must be greater than or equal to 0.")

    def check_people(self) -> None:
        if self.factors["people"] < 1:
            raise ValueError("people must be greater than or equal to 1.")

    def check_epsilon(self) -> bool:
        return self.factors["epsilon"] > 0

    def replicate(self, rng_list: list[MRG32k3a]) -> tuple[dict, dict]:
        """
        Simulate a single replication for the current model factors.

        Arguments
        ---------
        rng_list : list of mrg32k3a.mrg32k3a.MRG32k3a objects
            rngs for model to use when simulating a replication

        Returns
        -------
        responses : dict
            performance measures of interest
            "avg_sojourn_time" = average sojourn time
            "avg_waiting_time" = average waiting time
            "frac_cust_wait" = fraction of customers who wait
        gradients : dict of dicts
            gradient estimates for each response
        """
        # Set mu to be at least epsilon.
        self.factors["mu"] = max(self.factors["mu"], self.factors["epsilon"])
        # Calculate total number of arrivals to simulate.
        total = self.factors["warmup"] + self.factors["people"]
        # Designate separate RNGs for interarrival and serivce times.
        arrival_rng = rng_list[0]
        service_rng = rng_list[1]
        # Generate all interarrival and service times up front.
        arrival_times = [
            arrival_rng.expovariate(self.factors["lambda"])
            for _ in range(total)
        ]
        service_times = [
            service_rng.expovariate(self.factors["mu"]) for _ in range(total)
        ]
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
        cust_mat = np.zeros((total, 10))
        cust_mat[:, 0] = np.cumsum(arrival_times)
        cust_mat[:, 1] = service_times
        # Input entries for first customer's queueing experience.
        cust_mat[0, 2] = cust_mat[0, 0] + cust_mat[0, 1]
        cust_mat[0, 3] = cust_mat[0, 1]
        cust_mat[0, 4] = 0
        cust_mat[0, 5] = 0
        cust_mat[0, 6] = -cust_mat[0, 1] / self.factors["mu"]
        cust_mat[0, 7] = 0
        cust_mat[0, 8] = 0
        cust_mat[0, 9] = 0
        # Fill in entries for remaining customers' experiences.
        for i in range(1, total):
            cust_mat[i, 2] = (
                max(cust_mat[i, 0], cust_mat[i - 1, 2]) + cust_mat[i, 1]
            )
            cust_mat[i, 3] = cust_mat[i, 2] - cust_mat[i, 0]
            cust_mat[i, 4] = cust_mat[i, 3] - cust_mat[i, 1]
            cust_mat[i, 5] = sum(
                cust_mat[i - int(cust_mat[i - 1, 5]) - 1 : i, 2]
                > cust_mat[i, 0]
            )
            cust_mat[i, 6] = (
                -sum(cust_mat[i - int(cust_mat[i, 5]) : i + 1, 1])
                / self.factors["mu"]
            )
            cust_mat[i, 7] = (
                -sum(cust_mat[i - int(cust_mat[i, 5]) : i, 1])
                / self.factors["mu"]
            )
            cust_mat[i, 8] = np.nan  # ... to be derived
            cust_mat[i, 9] = np.nan  # ... to be derived
        # Compute average sojourn time and its gradient.
        mean_sojourn_time = np.mean(cust_mat[self.factors["warmup"] :, 3])
        grad_mean_sojourn_time_mu = np.mean(
            cust_mat[self.factors["warmup"] :, 6]
        )
        grad_mean_sojourn_time_lambda = np.mean(
            cust_mat[self.factors["warmup"] :, 8]
        )
        # Compute average waiting time and its gradient.
        mean_waiting_time = np.mean(cust_mat[self.factors["warmup"] :, 4])
        grad_mean_waiting_time_mu = np.mean(
            cust_mat[self.factors["warmup"] :, 7]
        )
        grad_mean_waiting_time_lambda = np.mean(
            cust_mat[self.factors["warmup"] :, 9]
        )
        # Compute fraction of customers who wait.
        fraction_wait = np.mean(cust_mat[self.factors["warmup"] :, 5] > 0)
        # Compose responses and gradients.
        responses = {
            "avg_sojourn_time": mean_sojourn_time,
            "avg_waiting_time": mean_waiting_time,
            "frac_cust_wait": fraction_wait,
        }
        gradients = {
            response_key: {
                factor_key: np.nan for factor_key in self.specifications
            }
            for response_key in responses
        }
        gradients["avg_sojourn_time"]["mu"] = float(grad_mean_sojourn_time_mu)
        gradients["avg_sojourn_time"]["lambda"] = float(
            grad_mean_sojourn_time_lambda
        )
        gradients["avg_waiting_time"]["mu"] = float(grad_mean_waiting_time_mu)
        gradients["avg_waiting_time"]["lambda"] = float(
            grad_mean_waiting_time_lambda
        )
        return responses, gradients


"""
Summary
-------
Minimize the mean sojourn time of an M/M/1 queue plus a cost term.
"""


class MM1MinMeanSojournTime(Problem):
    """
    Base class to implement simulation-optimization problems.

    Attributes
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

    Arguments
    ---------
    name : str
        user-specified name for problem
    fixed_factors : dict
        dictionary of user-specified problem factors
    model_fixed_factors : dict
        subset of user-specified non-decision factors to pass through to the model

    See also
    --------
    base.Problem
    """

    @property
    def n_objectives(self) -> int:
        return 1

    @property
    def n_stochastic_constraints(self) -> int:
        return 0

    @property
    def minmax(self) -> tuple[int]:
        return (-1,)

    @property
    def constraint_type(self) -> ConstraintType:
        return ConstraintType.BOX

    @property
    def variable_type(self) -> VariableType:
        return VariableType.CONTINUOUS

    @property
    def gradient_available(self) -> bool:
        return True

    @property
    def optimal_value(self) -> float | None:
        return None

    @property
    def optimal_solution(self) -> tuple | None:
        return None

    @property
    def model_default_factors(self) -> dict:
        return {"warmup": 50, "people": 200}

    @property
    def model_decision_factors(self) -> set[str]:
        return {"mu"}

    @property
    def specifications(self) -> dict[str, dict]:
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
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "cost": self.check_cost,
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget,
        }

    @property
    def dim(self) -> int:
        return 1

    @property
    def lower_bounds(self) -> tuple:
        return (0,) * self.dim

    @property
    def upper_bounds(self) -> tuple:
        return (np.inf,) * self.dim

    def __init__(
        self,
        name: str = "MM1-1",
        fixed_factors: dict | None = None,
        model_fixed_factors: dict | None = None,
    ) -> None:
        # Let the base class handle default arguments.
        super().__init__(
            name=name,
            fixed_factors=fixed_factors,
            model_fixed_factors=model_fixed_factors,
            model=MM1Queue,
        )

    def check_cost(self) -> None:
        if self.factors["cost"] <= 0:
            raise ValueError("cost must be greater than 0.")

    def vector_to_factor_dict(self, vector: tuple) -> dict:
        """
        Convert a vector of variables to a dictionary with factor keys

        Arguments
        ---------
        vector : tuple
            vector of values associated with decision variables

        Returns
        -------
        factor_dict : dictionary
            dictionary with factor keys and associated values
        """
        factor_dict = {"mu": vector[0]}
        return factor_dict

    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:
        """
        Convert a dictionary with factor keys to a vector
        of variables.

        Arguments
        ---------
        factor_dict : dictionary
            dictionary with factor keys and associated values

        Returns
        -------
        vector : tuple
            vector of values associated with decision variables
        """
        vector = (factor_dict["mu"],)
        return vector

    def response_dict_to_objectives(self, response_dict: dict) -> tuple:
        """
        Convert a dictionary with response keys to a vector
        of objectives.

        Arguments
        ---------
        response_dict : dictionary
            dictionary with response keys and associated values

        Returns
        -------
        objectives : tuple
            vector of objectives
        """
        objectives = (response_dict["avg_sojourn_time"],)
        return objectives

    def response_dict_to_stoch_constraints(self, response_dict: dict) -> tuple:
        """
        Convert a dictionary with response keys to a vector
        of left-hand sides of stochastic constraints: E[Y] <= 0

        Arguments
        ---------
        response_dict : dictionary
            dictionary with response keys and associated values

        Returns
        -------
        stoch_constraints : tuple
            vector of LHSs of stochastic constraint
        """
        stoch_constraints = ()
        return stoch_constraints

    def deterministic_objectives_and_gradients(
        self, x: tuple
    ) -> tuple[tuple, tuple]:
        """
        Compute deterministic components of objectives for a solution `x`.

        Arguments
        ---------
        x : tuple
            vector of decision variables

        Returns
        -------
        det_objectives : tuple
            vector of deterministic components of objectives
        det_objectives_gradients : tuple
            vector of gradients of deterministic components of objectives
        """
        det_objectives = (self.factors["cost"] * (x[0] ** 2),)
        det_objectives_gradients = ((2 * self.factors["cost"] * x[0],),)
        return det_objectives, det_objectives_gradients

    def deterministic_stochastic_constraints_and_gradients(
        self, x: tuple
    ) -> tuple[tuple, tuple]:
        """
        Compute deterministic components of stochastic constraints
        for a solution `x`.

        Arguments
        ---------
        x : tuple
            vector of decision variables

        Returns
        -------
        det_stoch_constraints : tuple
            vector of deterministic components of stochastic
            constraints
        det_stoch_constraints_gradients : tuple
            vector of gradients of deterministic components of
            stochastic constraints
        """
        det_stoch_constraints = ()
        det_stoch_constraints_gradients = ()
        return det_stoch_constraints, det_stoch_constraints_gradients

    def check_deterministic_constraints(self, x: tuple) -> bool:
        """
        Check if a solution `x` satisfies the problem's deterministic
        constraints.

        Arguments
        ---------
        x : tuple
            vector of decision variables

        Returns
        -------
        satisfies : bool
            indicates if solution `x` satisfies the deterministic constraints.
        """
        # Check box constraints.
        box_feasible = super().check_deterministic_constraints(x)
        return box_feasible

    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
        """
        Generate a random solution for starting or restarting solvers.

        Arguments
        ---------
        rand_sol_rng : mrg32k3a.mrg32k3a.MRG32k3a object
            random-number generator used to sample a new random solution

        Returns
        -------
        x : tuple
            vector of decision variables
        """
        # Generate an Exponential(rate = 1/3) r.v.
        x = (rand_sol_rng.expovariate(1 / 3),)
        return x
