"""
Summary
-------
Simulate MLE estimation for the parameters of a two-dimensional gamma distribution.
A detailed description of the model/problem can be found
`here <https://simopt.readthedocs.io/en/latest/paramesti.html>`__.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
from mrg32k3a.mrg32k3a import MRG32k3a

from simopt.base import ConstraintType, Model, Problem, VariableType


class ParameterEstimation(Model):
    """
    A model that simulates MLE estimation for the parameters of
    a two-dimensional gamma distribution.

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
    base.model
    """

    @property
    def name(self) -> str:
        return "PARAMESTI"

    @property
    def n_rngs(self) -> int:
        return 2

    @property
    def n_responses(self) -> int:
        return 1

    @property
    def specifications(self) -> dict[str, dict]:
        return {
            "xstar": {
                "description": "x^*, the unknown parameter that maximizes g(x)",
                "datatype": list,
                "default": [2, 5],
            },
            "x": {
                "description": "x, variable in pdf",
                "datatype": list,
                "default": [1, 1],
            },
        }

    @property
    def check_factor_list(self) -> dict[str, Callable]:
        return {"xstar": self.check_xstar, "x": self.check_x}

    def __init__(self, fixed_factors: dict | None = None) -> None:
        # Let the base class handle default arguments.
        super().__init__(fixed_factors)

    def check_xstar(self) -> None:
        if any(xstar_i <= 0 for xstar_i in self.factors["xstar"]):
            raise ValueError("All elements in xstar must be greater than 0.")

    def check_x(self) -> None:
        if any(x_i <= 0 for x_i in self.factors["x"]):
            raise ValueError("All elements in x must be greater than 0.")

    def check_simulatable_factors(self) -> bool:
        # Check for dimension of x and xstar.
        x_len = len(self.factors["x"])
        xstar_len = len(self.factors["xstar"])
        if x_len != 2:
            raise ValueError("The length of x must equal 2.")
        elif xstar_len != 2:
            raise ValueError("The length of xstar must equal 2.")
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
            "loglik" = the corresponding loglikelihood
        gradients : dict of dicts
            gradient estimates for each response
        """
        # Designate separate random number generators.
        # Outputs will be coupled when generating Y_j's.
        y2_rng = rng_list[0]
        y1_rng = rng_list[1]
        # Generate y1 and y2 from specified gamma distributions.
        y2 = y2_rng.gammavariate(self.factors["xstar"][1], 1)
        y1 = y1_rng.gammavariate(self.factors["xstar"][0] * y2, 1)
        # Compute Log Likelihood
        loglik = (
            -y1
            - y2
            + (self.factors["x"][0] * y2 - 1) * np.log(y1)
            + (self.factors["x"][1] - 1) * np.log(y2)
            - np.log(math.gamma(self.factors["x"][0] * y2))
            - np.log(math.gamma(self.factors["x"][1]))
        )
        # Compose responses and gradients.
        responses = {"loglik": loglik}
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
Minimize the log likelihood of 2-D gamma random variable.
"""


class ParamEstiMaxLogLik(Problem):
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
    model : model object
        associated simulation model that generates replications
    model_default_factors : dict
        default values for overriding model-level default factors
    model_fixed_factors : dict
        combination of overriden model-level factors and defaults
    rng_list : list of mrg32k3a.mrg32k3a.MRG32k3a objects
        list of RNGs used to generate a random initial solution
        or a random problem instance
    factors : dict
        changeable factors of the problem
            initial_solution : list
                default initial solution from which solvers start
            budget : int > 0
                max number of replications (fn evals) for a solver to take
            prev_cost : list
                cost of prevention
            upper_thres : float > 0
                upper limit of amount of contamination
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
    def minmax(self) -> tuple:
        return (1,)

    @property
    def constraint_type(self) -> ConstraintType:
        return ConstraintType.BOX

    @property
    def variable_type(self) -> VariableType:
        return VariableType.CONTINUOUS

    @property
    def gradient_available(self) -> bool:
        return False

    @property
    def optimal_value(self) -> float | None:
        return None

    @property
    def optimal_solution(self) -> tuple | None:
        solution = self.model.factors["xstar"]
        if isinstance(solution, list):
            return tuple(solution)
        return solution

    @property
    def model_default_factors(self) -> dict:
        return {}

    @property
    def model_decision_factors(self) -> set[str]:
        return {"x"}

    @property
    def specifications(self) -> dict[str, dict]:
        return {
            "initial_solution": {
                "description": "initial solution",
                "datatype": tuple,
                "default": (1, 1),
            },
            "budget": {
                "description": "max # of replications for a solver to take",
                "datatype": int,
                "default": 1000,
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
        return 2

    @property
    def lower_bounds(self) -> tuple:
        return (0.1,) * self.dim

    @property
    def upper_bounds(self) -> tuple:
        return (10,) * self.dim

    def __init__(
        self,
        name: str = "PARAMESTI-1",
        fixed_factors: dict | None = None,
        model_fixed_factors: dict | None = None,
    ) -> None:
        # Let the base class handle default arguments.
        super().__init__(
            name=name,
            fixed_factors=fixed_factors,
            model_fixed_factors=model_fixed_factors,
            model=ParameterEstimation,
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
        factor_dict = {"x": vector[:]}
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
        vector = tuple(factor_dict["x"])
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
        objectives = (response_dict["loglik"],)
        return objectives

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
        det_objectives_gradients = ((0, 0),)
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
                rand_sol_rng.uniform(
                    self.lower_bounds[idx], self.upper_bounds[idx]
                )
                for idx in range(self.dim)
            ]
        )
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
