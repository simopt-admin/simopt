"""
Summary
-------
Simulate a multi-stage revenue management system with inter-temporal dependence.
A detailed description of the model/problem can be found
`here <https://simopt.readthedocs.io/en/latest/rmitd.html>`__.

"""

from __future__ import annotations

from typing import Callable

import numpy as np
from mrg32k3a.mrg32k3a import MRG32k3a

from simopt.base import ConstraintType, Model, Problem, VariableType


class RMITD(Model):
    """
    A model that simulates a multi-stage revenue management system with
    inter-temporal dependence.
    Returns the total revenue.

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
        return "RMITD"

    @property
    def n_rngs(self) -> int:
        return 2

    @property
    def n_responses(self) -> int:
        return 1

    @property
    def specifications(self) -> dict[str, dict]:
        return {
            "time_horizon": {
                "description": "time horizon",
                "datatype": int,
                "default": 3,
            },
            "prices": {
                "description": "prices for each period",
                "datatype": list,
                "default": [100, 300, 400],
            },
            "demand_means": {
                "description": "mean demand for each period",
                "datatype": list,
                "default": [50, 20, 30],
            },
            "cost": {
                "description": "cost per unit of capacity at t = 0",
                "datatype": float,
                "default": 80.0,
            },
            "gamma_shape": {
                "description": "shape parameter of gamma distribution",
                "datatype": float,
                "default": 1.0,
            },
            "gamma_scale": {
                "description": "scale parameter of gamma distribution",
                "datatype": float,
                "default": 1.0,
            },
            "initial_inventory": {
                "description": "initial inventory",
                "datatype": int,
                "default": 100,
            },
            "reservation_qtys": {
                "description": "inventory to reserve going into periods 2, 3, ..., T",
                "datatype": list,
                "default": [50, 30],
            },
        }

    @property
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "time_horizon": self.check_time_horizon,
            "prices": self.check_prices,
            "demand_means": self.check_demand_means,
            "cost": self.check_cost,
            "gamma_shape": self.check_gamma_shape,
            "gamma_scale": self.check_gamma_scale,
            "initial_inventory": self.check_initial_inventory,
            "reservation_qtys": self.check_reservation_qtys,
        }

    def __init__(self, fixed_factors: dict | None = None) -> None:
        # Let the base class handle default arguments.
        super().__init__(fixed_factors)

    def check_time_horizon(self) -> None:
        if self.factors["time_horizon"] <= 0:
            raise ValueError("time_horizon must be greater than 0.")

    def check_prices(self) -> None:
        if any(price <= 0 for price in self.factors["prices"]):
            raise ValueError("All elements in prices must be greater than 0.")

    def check_demand_means(self) -> None:
        if any(
            demand_mean <= 0 for demand_mean in self.factors["demand_means"]
        ):
            raise ValueError(
                "All elements in demand_means must be greater than 0."
            )

    def check_cost(self) -> None:
        if self.factors["cost"] <= 0:
            raise ValueError("cost must be greater than 0.")

    def check_gamma_shape(self) -> None:
        if self.factors["gamma_shape"] <= 0:
            raise ValueError("gamma_shape must be greater than 0.")

    def check_gamma_scale(self) -> None:
        if self.factors["gamma_scale"] <= 0:
            raise ValueError("gamma_scale must be greater than 0.")

    def check_initial_inventory(self) -> None:
        if self.factors["initial_inventory"] <= 0:
            raise ValueError("initial_inventory must be greater than 0.")

    def check_reservation_qtys(self) -> None:
        if any(
            reservation_qty <= 0
            for reservation_qty in self.factors["reservation_qtys"]
        ):
            raise ValueError(
                "All elements in reservation_qtys must be greater than 0."
            )

    def check_simulatable_factors(self) -> bool:
        # Check for matching number of periods.
        if len(self.factors["prices"]) != self.factors["time_horizon"]:
            raise ValueError(
                "The length of prices must be equal to time_horizon."
            )
        elif len(self.factors["demand_means"]) != self.factors["time_horizon"]:
            raise ValueError(
                "The length of demand_means must be equal to time_horizon."
            )
        elif (
            len(self.factors["reservation_qtys"])
            != self.factors["time_horizon"] - 1
        ):
            raise ValueError(
                "The length of reservation_qtys must be equal to the time_horizon minus 1."
            )
        # Check that first reservation level is less than initial inventory.
        elif (
            self.factors["initial_inventory"]
            < self.factors["reservation_qtys"][0]
        ):
            raise ValueError(
                "The initial_inventory must be greater than or equal to the first element in reservation_qtys."
            )
        # Check for non-increasing reservation levels.
        elif any(
            self.factors["reservation_qtys"][idx]
            < self.factors["reservation_qtys"][idx + 1]
            for idx in range(self.factors["time_horizon"] - 2)
        ):
            raise ValueError(
                "Each value in reservation_qtys must be greater than the next value in the list."
            )
        # Check that gamma_shape*gamma_scale = 1.
        elif (
            np.isclose(
                self.factors["gamma_shape"] * self.factors["gamma_scale"], 1
            )
            is False
        ):
            raise ValueError(
                "gamma_shape times gamma_scale should be close to 1."
            )
        else:
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
            "revenue" = total revenue
        gradients : dict of dicts
            gradient estimates for each response
        """
        # Designate separate random number generators.
        # Outputs will be coupled when generating demand.
        x_rng = rng_list[0]
        y_rng = rng_list[1]
        # Generate X and Y (to use for computing demand).
        # random.gammavariate takes two inputs: alpha and beta.
        #     alpha = k = gamma_shape
        #     beta = 1/theta = 1/gamma_scale
        x_demand = x_rng.gammavariate(
            alpha=self.factors["gamma_shape"],
            beta=1.0 / self.factors["gamma_scale"],
        )
        y_demand = [
            y_rng.expovariate(1) for _ in range(self.factors["time_horizon"])
        ]
        # Track inventory over time horizon.
        remaining_inventory = self.factors["initial_inventory"]
        # Append "no reservations" for decision-making in final period.
        reservations = self.factors["reservation_qtys"]
        reservations.append(0)
        # Simulate over the time horizon and calculate the realized revenue.
        revenue = 0
        for period in range(self.factors["time_horizon"]):
            demand = (
                self.factors["demand_means"][period]
                * x_demand
                * y_demand[period]
            )
            sell = min(
                max(remaining_inventory - reservations[period], 0), demand
            )
            remaining_inventory = remaining_inventory - sell
            revenue += sell * self.factors["prices"][period]
        revenue -= self.factors["cost"] * self.factors["initial_inventory"]
        # Compose responses and gradients.
        responses = {"revenue": revenue}
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
Maximize the total revenue of a multi-stage revenue management
with inter-temporal dependence problem.
"""


class RMITDMaxRevenue(Problem):
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
            initial_solution : tuple
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
        return (1,)

    @property
    def constraint_type(self) -> ConstraintType:
        return ConstraintType.DETERMINISTIC

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
        # return (90, 50, 0)
        return None

    @property
    def model_default_factors(self) -> dict:
        return {}

    @property
    def model_decision_factors(self) -> set[str]:
        return {"initial_inventory", "reservation_qtys"}

    @property
    def specifications(self) -> dict[str, dict]:
        return {
            "initial_solution": {
                "description": "initial solution",
                "datatype": tuple,
                "default": (100, 50, 30),
            },
            "budget": {
                "description": "max # of replications for a solver to take",
                "datatype": int,
                "default": 10000,
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
        return 3

    @property
    def lower_bounds(self) -> tuple:
        return (0,) * self.dim

    @property
    def upper_bounds(self) -> tuple:
        return (np.inf,) * self.dim

    def __init__(
        self,
        name: str = "RMITD-1",
        fixed_factors: dict | None = None,
        model_fixed_factors: dict | None = None,
    ) -> None:
        # Let the base class handle default arguments.
        super().__init__(
            name=name,
            fixed_factors=fixed_factors,
            model_fixed_factors=model_fixed_factors,
            model=RMITD,
        )

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
        factor_dict = {
            "initial_inventory": vector[0],
            "reservation_qtys": list(vector[0:]),
        }
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
        vector = (
            factor_dict["initial_inventory"],
            *tuple(factor_dict["reservation_qtys"]),
        )
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
        return all(x[idx] >= x[idx + 1] for idx in range(self.dim - 1))

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
        # Generate random solution using acceptable/rejection.
        while True:
            x = tuple([200 * rand_sol_rng.random() for _ in range(self.dim)])
            if self.check_deterministic_constraints(x):
                break
        return x

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
        tuple
            vector of LHSs of stochastic constraint
        """
        raise NotImplementedError
