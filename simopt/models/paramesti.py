"""Simulate MLE estimation for the parameters of a 2D gamma distribution."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.base import ConstraintType, Model, Problem, VariableType
from simopt.utils import classproperty, override


class ParameterEstimation(Model):
    """MLE estimation model for the parameters of a 2D gamma distribution."""

    @classproperty
    @override
    def class_name_abbr(cls) -> str:
        return "PARAMESTI"

    @classproperty
    @override
    def class_name(cls) -> str:
        return "Gamma Parameter Estimation"

    @classproperty
    @override
    def n_rngs(cls) -> int:
        return 2

    @classproperty
    @override
    def n_responses(cls) -> int:
        return 1

    @classproperty
    @override
    def specifications(cls) -> dict[str, dict]:
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
    @override
    def check_factor_list(self) -> dict[str, Callable]:
        return {"xstar": self._check_xstar, "x": self._check_x}

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize the model.

        Args:
            fixed_factors (dict, optional): Fixed factors of the simulation model.
                Defaults to None.
        """
        # Let the base class handle default arguments.
        super().__init__(fixed_factors)

    def _check_xstar(self) -> None:
        if any(xstar_i <= 0 for xstar_i in self.factors["xstar"]):
            raise ValueError("All elements in xstar must be greater than 0.")

    def _check_x(self) -> None:
        if any(x_i <= 0 for x_i in self.factors["x"]):
            raise ValueError("All elements in x must be greater than 0.")

    @override
    def check_simulatable_factors(self) -> bool:
        # Check for dimension of x and xstar.
        x_len = len(self.factors["x"])
        xstar_len = len(self.factors["xstar"])
        if x_len != 2:
            raise ValueError("The length of x must equal 2.")
        if xstar_len != 2:
            raise ValueError("The length of xstar must equal 2.")
        return True

    def replicate(self, rng_list: list[MRG32k3a]) -> tuple[dict, dict]:
        """Simulate a single replication for the current model factors.

        Args:
            rng_list (list[MRG32k3a]): Random number generators used to simulate
                the replication.

        Returns:
            tuple[dict, dict]: A tuple containing:
                - responses (dict): Performance measures of interest, including:
                    - "loglik": The corresponding log-likelihood.
                - gradients (dict): A dictionary of gradient estimates for
                    each response.
        """
        xstar = self.factors["xstar"]
        x = self.factors["x"]
        # Designate separate random number generators.
        # Outputs will be coupled when generating Y_j's.
        y2_rng = rng_list[0]
        y1_rng = rng_list[1]
        # Generate y1 and y2 from specified gamma distributions.
        y2 = y2_rng.gammavariate(xstar[1], 1)
        y1 = y1_rng.gammavariate(xstar[0] * y2, 1)
        # Compute Log Likelihood
        loglik = (
            -y1
            - y2
            + (x[0] * y2 - 1) * np.log(y1)
            + (x[1] - 1) * np.log(y2)
            - np.log(math.gamma(x[0] * y2))
            - np.log(math.gamma(x[1]))
        )
        # Compose responses and gradients.
        responses = {"loglik": loglik}
        gradients = {
            response_key: dict.fromkeys(self.specifications, np.nan)
            for response_key in responses
        }
        return responses, gradients


class ParamEstiMaxLogLik(Problem):
    """Base class to implement simulation-optimization problems."""

    @classproperty
    @override
    def class_name_abbr(cls) -> str:
        return "PARAMESTI-1"

    @classproperty
    @override
    def class_name(cls) -> str:
        return "Max Log Likelihood for Gamma Parameter Estimation"

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
    def minmax(cls) -> tuple:
        return (1,)

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
        return False

    @classproperty
    @override
    def optimal_value(cls) -> float | None:
        return None

    @property
    @override
    def optimal_solution(self) -> tuple | None:
        solution = self.model.factors["xstar"]
        if isinstance(solution, list):
            return tuple(solution)
        return solution

    @classproperty
    @override
    def model_default_factors(cls) -> dict:
        return {}

    @classproperty
    @override
    def model_decision_factors(cls) -> set[str]:
        return {"x"}

    @classproperty
    @override
    def specifications(cls) -> dict[str, dict]:
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
    @override
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget,
        }

    @classproperty
    @override
    def dim(cls) -> int:
        return 2

    @classproperty
    @override
    def lower_bounds(cls) -> tuple:
        return (0.1,) * cls.dim

    @classproperty
    @override
    def upper_bounds(cls) -> tuple:
        return (10,) * cls.dim

    def __init__(
        self,
        name: str = "PARAMESTI-1",
        fixed_factors: dict | None = None,
        model_fixed_factors: dict | None = None,
    ) -> None:
        """Initialize the problem.

        Args:
            name (str, optional): User-specified name for problem.
                Defaults to "PARAMESTI-1".
            fixed_factors (dict, optional): Fixed factors of the simulation model.
                Defaults to None.
            model_fixed_factors (dict, optional): Fixed factors of the simulation
                model. Defaults to None.
        """
        # Let the base class handle default arguments.
        super().__init__(
            name=name,
            fixed_factors=fixed_factors,
            model_fixed_factors=model_fixed_factors,
            model=ParameterEstimation,
        )

    @override
    def vector_to_factor_dict(self, vector: tuple) -> dict:
        return {"x": vector[:]}

    @override
    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:
        return tuple(factor_dict["x"])

    @override
    def response_dict_to_objectives(self, response_dict: dict) -> tuple:
        return (response_dict["loglik"],)

    @override
    def deterministic_objectives_and_gradients(self, _x: tuple) -> tuple[tuple, tuple]:
        det_objectives = (0,)
        det_objectives_gradients = ((0, 0),)
        return det_objectives, det_objectives_gradients

    @override
    def check_deterministic_constraints(self, _x: tuple) -> bool:
        return True

    @override
    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
        return tuple(
            [
                rand_sol_rng.uniform(self.lower_bounds[idx], self.upper_bounds[idx])
                for idx in range(self.dim)
            ]
        )
