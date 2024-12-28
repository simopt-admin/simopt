"""
Summary
-------
Simulate expected revenue for a hotel.
A detailed description of the model/problem can be found
`here <https://simopt.readthedocs.io/en/latest/hotel.html>`__.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from mrg32k3a.mrg32k3a import MRG32k3a

from simopt.base import ConstraintType, Model, Problem, VariableType


class Hotel(Model):
    """
    A model that simulates business of a hotel with Poisson arrival rate.

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
        details of each factor (for GUI and data validation)
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
        return "HOTEL"

    @property
    def n_rngs(self) -> int:
        return 1

    @property
    def n_responses(self) -> int:
        return 1

    @property
    def specifications(self) -> dict[str, dict]:
        return {
            "num_products": {
                "description": "number of products: (rate, length of stay)",
                "datatype": int,
                "default": 56,
            },
            "lambda": {
                "description": "arrival rates for each product",
                "datatype": list,
                "default": (
                    (1 / 168)
                    * np.array(
                        [
                            1,
                            1,
                            2,
                            2,
                            3,
                            3,
                            2,
                            2,
                            1,
                            1,
                            0.5,
                            0.5,
                            0.25,
                            0.25,
                            1,
                            1,
                            2,
                            2,
                            3,
                            3,
                            2,
                            2,
                            1,
                            1,
                            0.5,
                            0.5,
                            1,
                            1,
                            2,
                            2,
                            3,
                            3,
                            2,
                            2,
                            1,
                            1,
                            1,
                            1,
                            2,
                            2,
                            3,
                            3,
                            2,
                            2,
                            1,
                            1,
                            2,
                            2,
                            3,
                            3,
                            1,
                            1,
                            2,
                            2,
                            1,
                            1,
                        ]
                    )
                ).tolist(),
            },
            "num_rooms": {
                "description": "hotel capacity",
                "datatype": int,
                "default": 100,
            },
            "discount_rate": {
                "description": "discount rate",
                "datatype": int,
                "default": 100,
            },
            "rack_rate": {
                "description": "rack rate (full price)",
                "datatype": int,
                "default": 200,
            },
            "product_incidence": {
                "description": "incidence matrix",
                "datatype": list,
                "default": [
                    [
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ],
                    [
                        0,
                        0,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ],
                    [
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ],
                    [
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ],
                    [
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ],
                    [
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        0,
                        0,
                    ],
                    [
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        0,
                        0,
                        1,
                        1,
                        1,
                        1,
                    ],
                ],
            },
            "time_limit": {
                "description": "time after which orders of each product no longer arrive (e.g. Mon night stops at 3am Tues or t=27)",
                "datatype": list,
                "default": np.concatenate(
                    (
                        27 * np.ones(14),
                        51 * np.ones(12),
                        75 * np.ones(10),
                        99 * np.ones(8),
                        123 * np.ones(6),
                        144 * np.ones(4),
                        168 * np.ones(2),
                    ),
                    axis=None,
                ).tolist(),
            },
            "time_before": {
                "description": "hours before t=0 to start running (e.g. 168 means start at time -168)",
                "datatype": int,
                "default": 168,
            },
            "runlength": {
                "description": "runlength of simulation (in hours) after t=0",
                "datatype": int,
                "default": 168,
            },
            "booking_limits": {
                "description": "booking limits",
                "datatype": tuple,
                "default": tuple([100 for _ in range(56)]),
            },
        }

    @property
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "num_products": self.check_num_products,
            "lambda": self.check_lambda,
            "num_rooms": self.check_num_rooms,
            "discount_rate": self.check_discount_rate,
            "rack_rate": self.check_rack_rate,
            "product_incidence": self.check_product_incidence,
            "time_limit": self.check_time_limit,
            "time_before": self.check_time_before,
            "runlength": self.check_runlength,
            "booking_limits": self.check_booking_limits,
        }

    def __init__(self, fixed_factors: dict | None = None) -> None:
        # Let the base class handle default arguments.
        super().__init__(fixed_factors)

    def check_num_products(self) -> None:
        if self.factors["num_products"] <= 0:
            raise ValueError("num_products must be greater than 0.")

    def check_lambda(self) -> None:
        for i in self.factors["lambda"]:
            if i <= 0:
                raise ValueError(
                    "All elements in lambda must be greater than 0."
                )

    def check_num_rooms(self) -> None:
        if self.factors["num_rooms"] <= 0:
            raise ValueError("num_rooms must be greater than 0.")

    def check_discount_rate(self) -> None:
        if self.factors["discount_rate"] <= 0:
            raise ValueError("discount_rate must be greater than 0.")

    def check_rack_rate(self) -> None:
        if self.factors["rack_rate"] <= 0:
            raise ValueError("rack_rate must be greater than 0.")

    def check_product_incidence(self) -> None:
        # TODO: fix check for product_incidence
        return
        # is_positive = [[i > 0 for i in j] for j in self.factors["product_incidence"]]
        # if not all(all(i) for i in is_positive):
        #     raise ValueError(
        #         "All elements in product_incidence must be greater than 0."
        #     )

    def check_time_limit(self) -> None:
        for i in self.factors["time_limit"]:
            if i <= 0:
                raise ValueError(
                    "All elements in time_limit must be greater than 0."
                )

    def check_time_before(self) -> None:
        if self.factors["time_before"] <= 0:
            raise ValueError("time_before must be greater than 0.")

    def check_runlength(self) -> None:
        if self.factors["runlength"] <= 0:
            raise ValueError("runlength must be greater than 0.")

    def check_booking_limits(self) -> None:
        for i in list(self.factors["booking_limits"]):
            if i <= 0 or i > self.factors["num_rooms"]:
                raise ValueError(
                    "All elements in booking_limits must be greater than 0 and less than num_rooms."
                )

    def check_simulatable_factors(self) -> bool:
        if len(self.factors["lambda"]) != self.factors["num_products"]:
            raise ValueError("The length of lambda must equal num_products.")
        if len(self.factors["time_limit"]) != self.factors["num_products"]:
            raise ValueError(
                "The length of time_limit must equal num_products."
            )
        if len(self.factors["booking_limits"]) != self.factors["num_products"]:
            raise ValueError(
                "The length of booking_limits must equal num_products."
            )
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
            "revenue" = expected revenue
        gradients : dict of dicts
            gradient estimates for each response
        """
        # Designate separate random number generators.
        arr_rng = rng_list[0]

        total_revenue = 0
        b = list(self.factors["booking_limits"])
        a_array = np.array(self.factors["product_incidence"])
        # Vector of next arrival time per product.
        # (Starts at time = -1*time_before, e.g., t = -168.)
        arrival = (
            np.zeros(self.factors["num_products"]) - self.factors["time_before"]
        )
        # Upper bound on number of arrivals over the time period.
        arr_bound = 10 * round(168 * np.sum(self.factors["lambda"]))
        arr_time = np.zeros((self.factors["num_products"], arr_bound))
        # Index of which arrival time to use next for each product.
        a = np.zeros(self.factors["num_products"], dtype=int)
        # Generate all interarrival times in advance.
        for i in range(self.factors["num_products"]):
            arr_time[i] = np.array(
                [
                    arr_rng.expovariate(self.factors["lambda"][i])
                    for _ in range(arr_bound)
                ]
            )
        # Extract first arrivals.
        for i in range(self.factors["num_products"]):
            arrival[i] = arrival[i] + arr_time[i, a[i]]
            a[i] = 1
        min_time = (
            0  # Keeps track of minimum time of the orders not yet received.
        )
        min_idx = 0
        while min_time <= self.factors["runlength"]:
            min_time = self.factors["runlength"] + 1
            for i in range(self.factors["num_products"]):
                if (arrival[i] < min_time) and (
                    arrival[i] <= self.factors["time_limit"][i]
                ):
                    min_time = arrival[i]
                    min_idx = i
            if min_time > self.factors["runlength"]:
                break
            if b[min_idx] > 0:
                if min_idx % 2 == 0:  # Rack_rate.
                    total_revenue += sum(
                        self.factors["rack_rate"] * a_array[:, min_idx]
                    )
                else:  # Discount_rate.
                    total_revenue += sum(
                        self.factors["discount_rate"] * a_array[:, min_idx]
                    )
                # Reduce the inventory of products sharing the same resource.
                for i in range(self.factors["num_products"]):
                    if np.dot(a_array[:, i].T, a_array[:, min_idx]) >= 1:
                        if b[i] != 0:
                            b[i] -= 1
            arrival[min_idx] += arr_time[min_idx, a[min_idx]]
            a[min_idx] = a[min_idx] + 1
        # Compose responses and gradients.
        responses = {"revenue": total_revenue}
        gradients = {
            response_key: {
                factor_key: np.nan for factor_key in self.specifications
            }
            for response_key in responses
        }
        return responses, gradients


"""
Summary
-------
Maximize the expected revenue.
"""


class HotelRevenue(Problem):
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
            initial_solution : list
                default initial solution from which solvers start
            budget : int > 0
                max number of replications (fn evals) for a solver to take
    specifications : dict
        details of each factor (for GUI, data validation, and defaults)

    Arguments
    ---------
    name : str
        user-specified name for problem
    fixed_factors : dict
        dictionary of user-specified problem factors
    model_fixed factors : dict
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
        return (1,)

    @property
    def constraint_type(self) -> ConstraintType:
        return ConstraintType.BOX

    @property
    def variable_type(self) -> VariableType:
        return VariableType.DISCRETE

    @property
    def gradient_available(self) -> bool:
        return False

    @property
    def optimal_value(self) -> float | None:
        return None

    @property
    def optimal_solution(self) -> tuple | None:
        return None

    @property
    def model_default_factors(self) -> dict:
        return {}

    @property
    def model_decision_factors(self) -> set[str]:
        return {"booking_limits"}

    @property
    def specifications(self) -> dict[str, dict]:
        return {
            "initial_solution": {
                "description": "initial solution",
                "datatype": tuple,
                "default": tuple([0 for _ in range(56)]),
            },
            "budget": {
                "description": "max # of replications for a solver to take",
                "datatype": int,
                "default": 100,
                "isDatafarmable": False,
            },
        }

    @property
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget,
        }

    @property
    def dim(self) -> int:
        return self.model.factors["num_products"]

    @property
    def lower_bounds(self) -> tuple:
        return (0,) * self.dim

    @property
    def upper_bounds(self) -> tuple:
        return (self.model.factors["num_rooms"],) * self.dim

    def __init__(
        self,
        name: str = "HOTEL-1",
        fixed_factors: dict | None = None,
        model_fixed_factors: dict | None = None,
    ) -> None:
        # Let the base class handle default arguments.
        super().__init__(
            name=name,
            fixed_factors=fixed_factors,
            model_fixed_factors=model_fixed_factors,
            model=Hotel,
        )

    def check_initial_solution(self) -> bool:
        return len(self.factors["initial_solution"]) == self.dim

    def check_budget(self) -> bool:
        if self.factors["budget"] <= 0:
            raise ValueError("budget must be greater than 0.")
        return True

    def check_simulatable_factors(self) -> bool:
        if len(self.lower_bounds) != self.dim:
            return False
        elif len(self.upper_bounds) != self.dim:
            return False
        else:
            return True

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
        factor_dict = {"booking_limits": vector[:]}
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
        vector = tuple(factor_dict["booking_limits"])
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
        objectives = (response_dict["revenue"],)
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

    def deterministic_stochastic_constraints_and_gradients(
        self, x: tuple
    ) -> tuple[tuple, tuple]:
        """
        Compute deterministic components of stochastic constraints for a solution `x`.

        Arguments
        ---------
        x : tuple
            vector of decision variables

        Returns
        -------
        det_stoch_constraints : tuple
            vector of deterministic components of stochastic constraints
        det_stoch_constraints_gradients : tuple
            vector of gradients of deterministic components of stochastic constraints
        """
        det_stoch_constraints = ()
        det_stoch_constraints_gradients = ()
        return det_stoch_constraints, det_stoch_constraints_gradients

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
        det_objectives = (0,)
        det_objectives_gradients = ((0,) * self.dim,)
        return det_objectives, det_objectives_gradients

    def check_deterministic_constraints(self, x: tuple) -> bool:
        """
        Check if a solution `x` satisfies the problem's deterministic constraints.

        Arguments
        ---------
        x : tuple
            vector of decision variables

        Returns
        -------
        satisfies : bool
            indicates if solution `x` satisfies the deterministic constraints.
        """
        return True

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
        x = tuple(
            [
                rand_sol_rng.randint(0, self.model.factors["num_rooms"])
                for _ in range(self.dim)
            ]
        )
        return x
