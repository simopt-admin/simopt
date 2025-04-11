"""Example problem of deterministic function with noise.

Simulate a synthetic problem with a deterministic objective function
evaluated with noise.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.base import ConstraintType, Model, Problem, VariableType
from simopt.utils import classproperty, override


class ExampleModel(Model):
    """A model that is a deterministic function evaluated with noise."""

    @classproperty
    @override
    def class_name_abbr(cls) -> str:
        return "EXAMPLE"

    @classproperty
    @override
    def class_name(cls) -> str:
        return "Deterministic Function + Noise"

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
        return {
            "x": {
                "description": "point to evaluate",
                "datatype": tuple,
                "default": (2.0, 2.0),
            }
        }

    @property
    @override
    def check_factor_list(self) -> dict[str, Callable]:
        return {"x": lambda: True}

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize the model.

        Args:
            fixed_factors (dict | None): fixed factors of the model.
                If None, use default values.
        """
        # Let the base class handle default arguments.
        super().__init__(fixed_factors)

    def replicate(self, rng_list: list[MRG32k3a]) -> tuple[dict, dict]:
        """Evaluate a deterministic function f(x) with stochastic noise.

        Args:
            rng_list (list[MRG32k3a]): Random number generators used to simulate
                the replication.

        Returns:
            tuple[dict, dict]: A tuple containing:
                - responses (dict): Performance measures of interest, including:
                    - "est_f(x)": Estimate of f(x) with added stochastic noise.
                - gradients (dict): A dictionary of gradient estimates for
                    each response.
        """
        # Designate random number generator for stochastic noise.
        noise_rng = rng_list[0]
        x = np.array(self.factors["x"])
        fn_eval_at_x = np.linalg.norm(x) ** 2 + noise_rng.normalvariate()

        # Compose responses and gradients.
        responses = {"est_f(x)": fn_eval_at_x}
        gradients = {"est_f(x)": {"x": tuple(2 * x)}}
        return responses, gradients


class ExampleProblem(Problem):
    """Base class to implement simulation-optimization problems."""

    @classproperty
    @override
    def class_name_abbr(cls) -> str:
        return "EXAMPLE-1"

    @classproperty
    @override
    def class_name(cls) -> str:
        return "Min Deterministic Function + Noise"

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
        return ConstraintType.UNCONSTRAINED

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
        # Change if f is changed
        # TODO: figure out what f is
        return 0.0

    @property
    @override
    def optimal_solution(self) -> tuple:
        # Change if f is changed
        # TODO: figure out what f is
        return (0,) * self.dim

    @classproperty
    @override
    def model_default_factors(cls) -> dict:
        return {}

    @property
    @override
    def model_fixed_factors(self) -> dict:
        return {}

    @model_fixed_factors.setter
    def model_fixed_factors(self, value: dict | None) -> None:
        # TODO: figure out if fixed factors should change
        pass

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
                "default": (2.0, 2.0),
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

    @property
    @override
    def dim(self) -> int:
        return len(self.factors["initial_solution"])

    @property
    @override
    def lower_bounds(self) -> tuple:
        return (-np.inf,) * self.dim

    @property
    @override
    def upper_bounds(self) -> tuple:
        return (np.inf,) * self.dim

    def __init__(
        self,
        name: str = "EXAMPLE-1",
        fixed_factors: dict | None = None,
        model_fixed_factors: dict | None = None,  # noqa: ARG002
    ) -> None:
        """Initialize the problem.

        Args:
            name (str): user-specified name for problem
            fixed_factors (dict | None): fixed factors of the problem.
                If None, use default values.
            model_fixed_factors (dict | None): fixed factors of the model.
                If None, use default values.
        """
        # Let the base class handle default arguments.
        # TODO: check if model_fixed_factors should be passed to the model
        super().__init__(
            name=name,
            fixed_factors=fixed_factors,
            model_fixed_factors=None,
            model=ExampleModel,
        )

    @override
    def vector_to_factor_dict(self, vector: tuple) -> dict:
        return {"x": vector[:]}

    @override
    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:
        return tuple(factor_dict["x"])

    @override
    def response_dict_to_objectives(self, response_dict: dict) -> tuple:
        return (response_dict["est_f(x)"],)

    @override
    def deterministic_objectives_and_gradients(self, _x: tuple) -> tuple:
        det_objectives = (0,)
        det_objectives_gradients = ((0,) * self.dim,)
        return det_objectives, det_objectives_gradients

    @override
    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
        # x = tuple([rand_sol_rng.uniform(-2, 2) for _ in range(self.dim)])
        return tuple(
            rand_sol_rng.mvnormalvariate(
                mean_vec=[0] * self.dim,
                cov=np.eye(self.dim),
                factorized=False,
            )
        )
