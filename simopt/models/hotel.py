"""Hotel Expected Revenue Model.

Simulate expected revenue for a hotel.
A detailed description of the model/problem can be found
`here <https://simopt.readthedocs.io/en/latest/hotel.html>`__.
"""

from __future__ import annotations

import heapq
from typing import Callable

import numpy as np

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.base import ConstraintType, Model, Problem, VariableType
from simopt.utils import classproperty, override


class Hotel(Model):
    """A model that simulates business of a hotel with Poisson arrival rate.

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
        details of each factor (for GUI and data validation)
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
    def class_name(cls) -> str:
        return "Hotel Booking"

    @classproperty
    @override
    def n_rngs(cls) -> int:
        return 1

    @classproperty
    @override
    def n_responses(cls) -> int:
        return 1

    @classproperty
    @override
    def specifications(cls) -> dict[str, dict]:
        def double_up(values: list[float]) -> list[float]:
            """Duplicate each value in the list once."""
            return [x for x in values for _ in range(2)]

        def gen_binary_list(pattern: list[int]) -> list[int]:
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

        return {
            "num_products": {
                "description": "number of products: (rate, length of stay)",
                "datatype": int,
                "default": 56,
            },
            "lambda": {
                "description": "arrival rates for each product",
                "datatype": list,
                "default": [
                    x / 168
                    for x in double_up(
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
                    gen_binary_list([0, 14, 42]),
                    gen_binary_list([2, 24, 30]),
                    gen_binary_list([4, 10, 2, 20, 20]),
                    gen_binary_list([6, 8, 4, 8, 2, 16, 12]),
                    gen_binary_list([8, 6, 6, 6, 4, 6, 2, 12, 6]),
                    gen_binary_list([10, 4, 8, 4, 6, 4, 4, 4, 2, 8, 2]),
                    gen_binary_list([12, 2, 10, 2, 8, 2, 6, 2, 4, 2, 2, 4]),
                ],
            },
            "time_limit": {
                "description": (
                    "time after which orders of each product no longer arrive "
                    "(e.g. Mon night stops at 3am Tues or t=27)"
                ),
                "datatype": list,
                "default": (
                    [27] * 14
                    + [51] * 12
                    + [75] * 10
                    + [99] * 8
                    + [123] * 6
                    + [144] * 4
                    + [168] * 2
                ),
            },
            "time_before": {
                "description": (
                    "hours before t=0 to start running "
                    "(e.g. 168 means start at time -168)"
                ),
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
                "default": tuple([100] * 56),
            },
        }

    @property
    @override
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "num_products": self._check_num_products,
            "lambda": self._check_lambda,
            "num_rooms": self._check_num_rooms,
            "discount_rate": self._check_discount_rate,
            "rack_rate": self._check_rack_rate,
            "product_incidence": self._check_product_incidence,
            "time_limit": self._check_time_limit,
            "time_before": self._check_time_before,
            "runlength": self._check_runlength,
            "booking_limits": self._check_booking_limits,
        }

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize the Hotel model.

        Args:
            fixed_factors (dict, optional): Fixed factors for the model.
                Defaults to None.
        """
        # Let the base class handle default arguments.
        super().__init__(fixed_factors)

    def _check_num_products(self) -> None:
        if self.factors["num_products"] <= 0:
            raise ValueError("num_products must be greater than 0.")

    def _check_lambda(self) -> None:
        for i in self.factors["lambda"]:
            if i <= 0:
                raise ValueError("All elements in lambda must be greater than 0.")

    def _check_num_rooms(self) -> None:
        if self.factors["num_rooms"] <= 0:
            raise ValueError("num_rooms must be greater than 0.")

    def _check_discount_rate(self) -> None:
        if self.factors["discount_rate"] <= 0:
            raise ValueError("discount_rate must be greater than 0.")

    def _check_rack_rate(self) -> None:
        if self.factors["rack_rate"] <= 0:
            raise ValueError("rack_rate must be greater than 0.")

    def _check_product_incidence(self) -> None:
        # TODO: fix check for product_incidence
        return
        # is_positive = [[i > 0 for i in j] for j in self.factors["product_incidence"]]
        # if not all(all(i) for i in is_positive):
        #     raise ValueError(
        #         "All elements in product_incidence must be greater than 0."
        #     )

    def _check_time_limit(self) -> None:
        for i in self.factors["time_limit"]:
            if i <= 0:
                raise ValueError("All elements in time_limit must be greater than 0.")

    def _check_time_before(self) -> None:
        if self.factors["time_before"] <= 0:
            raise ValueError("time_before must be greater than 0.")

    def _check_runlength(self) -> None:
        if self.factors["runlength"] <= 0:
            raise ValueError("runlength must be greater than 0.")

    def _check_booking_limits(self) -> None:
        for i in list(self.factors["booking_limits"]):
            if i <= 0 or i > self.factors["num_rooms"]:
                raise ValueError(
                    "All elements in booking_limits must be greater than 0 and less "
                    "than num_rooms."
                )

    @override
    def check_simulatable_factors(self) -> bool:
        if len(self.factors["lambda"]) != self.factors["num_products"]:
            raise ValueError("The length of lambda must equal num_products.")
        if len(self.factors["time_limit"]) != self.factors["num_products"]:
            raise ValueError("The length of time_limit must equal num_products.")
        if len(self.factors["booking_limits"]) != self.factors["num_products"]:
            raise ValueError("The length of booking_limits must equal num_products.")
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
        """Simulate a single replication for the current model factors.

        Arguments:
        ---------
        rng_list : list of mrg32k3a.mrg32k3a.MRG32k3a objects
            rngs for model to use when simulating a replication

        Returns:
        -------
        responses : dict
            performance measures of interest
            "revenue" = expected revenue
        gradients : dict of dicts
            gradient estimates for each response
        """
        booking_limits = list(self.factors["booking_limits"])
        product_incidence = np.array(self.factors["product_incidence"])
        num_products: int = self.factors["num_products"]
        time_before: int = self.factors["time_before"]
        f_lambda = self.factors["lambda"]
        run_length: int = self.factors["runlength"]
        time_limit: list = self.factors["time_limit"]
        rack_rate: int = self.factors["rack_rate"]
        discount_rate: int = self.factors["discount_rate"]

        # Designate separate random number generators.
        arr_rng = rng_list[0]
        total_revenue = 0

        # Generate interarrival times
        arr_bound = 10 * round(168 * np.sum(f_lambda))
        arr_time = np.array(
            [
                [arr_rng.expovariate(f_lambda[i]) for _ in range(arr_bound)]
                for i in range(num_products)
            ]
        )

        # Initialize arrival times
        arrival = [-time_before + arr_time[i, 0] for i in range(num_products)]
        a_idx = np.ones(num_products, dtype=int)  # Next interarrival index

        # Precompute resource conflict matrix (bool)
        conflicts = (product_incidence.T @ product_incidence) >= 1

        # Min-heap for tracking next arrival events (arrival_time, product_idx)
        heap = [
            (arrival[i], i) for i in range(num_products) if arrival[i] <= time_limit[i]
        ]
        heapq.heapify(heap)

        while heap:
            current_time, product_idx = heapq.heappop(heap)
            if current_time > run_length:
                break
            if booking_limits[product_idx] > 0:
                rate = rack_rate if product_idx % 2 == 0 else discount_rate
                total_revenue += rate * np.sum(product_incidence[:, product_idx])
                for i in range(num_products):
                    if conflicts[product_idx, i] and booking_limits[i] > 0:
                        booking_limits[i] -= 1

            # Schedule next arrival for this product
            next_idx = a_idx[product_idx]
            if next_idx < arr_bound:
                next_time = current_time + arr_time[product_idx, next_idx]
                a_idx[product_idx] += 1
                if next_time <= time_limit[product_idx]:
                    heapq.heappush(heap, (next_time, product_idx))

        # Compose responses and gradients.
        responses = {"revenue": total_revenue}
        gradients = {
            response_key: dict.fromkeys(self.specifications, np.nan)
            for response_key in responses
        }
        return responses, gradients


"""
Summary
-------
Maximize the expected revenue.
"""


class HotelRevenue(Problem):
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
            initial_solution : list
                default initial solution from which solvers start
            budget : int > 0
                max number of replications (fn evals) for a solver to take
    specifications : dict
        details of each factor (for GUI, data validation, and defaults)

    Arguments:
    ---------
    name : str
        user-specified name for problem
    fixed_factors : dict
        dictionary of user-specified problem factors
    model_fixed factors : dict
        subset of user-specified non-decision factors to pass through to the model

    See Also:
    --------
    base.Problem
    """

    @classproperty
    @override
    def class_name_abbr(cls) -> str:
        return "HOTEL-1"

    @classproperty
    @override
    def class_name(cls) -> str:
        return "Max Revenue for Hotel Booking"

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
        return (1,)

    @classproperty
    @override
    def constraint_type(cls) -> ConstraintType:
        return ConstraintType.BOX

    @classproperty
    @override
    def variable_type(cls) -> VariableType:
        return VariableType.DISCRETE

    @classproperty
    @override
    def gradient_available(cls) -> bool:
        return False

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
        return {}

    @classproperty
    @override
    def model_decision_factors(cls) -> set[str]:
        return {"booking_limits"}

    @classproperty
    @override
    def specifications(cls) -> dict[str, dict]:
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
    @override
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget,
        }

    @property
    @override
    def dim(self) -> int:
        return self.model.factors["num_products"]

    @property
    @override
    def lower_bounds(self) -> tuple:
        return (0,) * self.dim

    @property
    @override
    def upper_bounds(self) -> tuple:
        return (self.model.factors["num_rooms"],) * self.dim

    def __init__(
        self,
        name: str = "HOTEL-1",
        fixed_factors: dict | None = None,
        model_fixed_factors: dict | None = None,
    ) -> None:
        """Initialize the HotelRevenue problem.

        Args:
            name (str, optional): Name of the problem. Defaults to "HOTEL-1".
            fixed_factors (dict, optional): Fixed factors for the problem.
                Defaults to None.
            model_fixed_factors (dict, optional): Fixed factors for the model.
                Defaults to None.
        """
        # Let the base class handle default arguments.
        super().__init__(
            name=name,
            fixed_factors=fixed_factors,
            model_fixed_factors=model_fixed_factors,
            model=Hotel,
        )

    @override
    def check_initial_solution(self) -> bool:
        return len(self.factors["initial_solution"]) == self.dim

    # # TODO: figure out how Problem.check_simulatable_factors() works
    # def check_simulatable_factors(self) -> bool:
    #     return not (
    #         len(self.lower_bounds) != self.dim or len(self.upper_bounds) != self.dim
    #     )

    @override
    def vector_to_factor_dict(self, vector: tuple) -> dict:
        return {"booking_limits": vector[:]}

    @override
    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:
        return tuple(factor_dict["booking_limits"])

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
        return (response_dict["revenue"],)

    def deterministic_objectives_and_gradients(self, _x: tuple) -> tuple[tuple, tuple]:
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
        det_objectives = (0,)
        det_objectives_gradients = ((0,) * self.dim,)
        return det_objectives, det_objectives_gradients

    def check_deterministic_constraints(self, _x: tuple) -> bool:
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
        return True

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
        return tuple(
            [
                rand_sol_rng.randint(0, self.model.factors["num_rooms"])
                for _ in range(self.dim)
            ]
        )
