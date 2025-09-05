"""Simulate MLE estimation for the parameters of a 2D gamma distribution."""

from __future__ import annotations

import math
from typing import Annotated, ClassVar, Self

import numpy as np
from pydantic import BaseModel, Field, model_validator

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.base import (
    ConstraintType,
    Model,
    Objective,
    Problem,
    RepResult,
    VariableType,
)
from simopt.input_models import Gamma
from simopt.utils import override


class ParameterEstimationConfig(BaseModel):
    xstar: Annotated[
        list[float],
        Field(
            default=[2, 5],
            description="x^*, the unknown parameter that maximizes g(x)",
        ),
    ]
    x: Annotated[
        list[float],
        Field(
            default=[1, 1],
            description="x, variable in pdf",
        ),
    ]

    def _check_xstar(self) -> None:
        if any(xstar_i <= 0 for xstar_i in self.xstar):
            raise ValueError("All elements in xstar must be greater than 0.")

    def _check_x(self) -> None:
        if any(x_i <= 0 for x_i in self.x):
            raise ValueError("All elements in x must be greater than 0.")

    @model_validator(mode="after")
    def _validate_model(self) -> Self:
        self._check_xstar()
        self._check_x()

        x_len = len(self.x)
        xstar_len = len(self.xstar)
        if x_len != 2:
            raise ValueError("The length of x must equal 2.")
        if xstar_len != 2:
            raise ValueError("The length of xstar must equal 2.")

        return self


class ParamEstiMaxLogLikConfig(BaseModel):
    """Configuration model for Parameter Estimation Max Log Likelihood Problem.

    Max Log Likelihood for Gamma Parameter Estimation simulation-optimization problem.
    """

    initial_solution: Annotated[
        tuple[float, ...],
        Field(
            default=(1, 1),
            description="initial solution",
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


class ParameterEstimation(Model):
    """MLE estimation model for the parameters of a 2D gamma distribution."""

    config_class: ClassVar[type[BaseModel]] = ParameterEstimationConfig
    class_name_abbr: str = "PARAMESTI"
    class_name: str = "Gamma Parameter Estimation"
    n_rngs: int = 2
    n_responses: int = 1

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize the model.

        Args:
            fixed_factors (dict, optional): Fixed factors of the simulation model.
                Defaults to None.
        """
        # Let the base class handle default arguments.
        super().__init__(fixed_factors)
        self.y1_model = Gamma()
        self.y2_model = Gamma()

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

    def before_replicate(self, rng_list: list[MRG32k3a]) -> None:  # noqa: D102
        self.y2_model.set_rng(rng_list[0])
        self.y1_model.set_rng(rng_list[1])

    def replicate(self) -> tuple[dict, dict]:
        """Simulate a single replication for the current model factors.

        Returns:
            tuple[dict, dict]: A tuple containing:
                - responses (dict): Performance measures of interest, including:
                    - "loglik": The corresponding log-likelihood.
                - gradients (dict): A dictionary of gradient estimates for
                    each response.
        """
        xstar = self.factors["xstar"]
        x = self.factors["x"]
        # Generate y1 and y2 from specified gamma distributions using input models.
        # Outputs will be coupled when generating Y_j's.
        y2 = self.y2_model.random(xstar[1], 1)
        y1 = self.y1_model.random(xstar[0] * y2, 1)
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
        return responses, None


class ParamEstiMaxLogLik(Problem):
    """Base class to implement simulation-optimization problems."""

    config_class: ClassVar[type[BaseModel]] = ParamEstiMaxLogLikConfig
    model_class: ClassVar[type[Model]] = ParameterEstimation
    class_name_abbr: str = "PARAMESTI-1"
    class_name: str = "Max Log Likelihood for Gamma Parameter Estimation"
    n_objectives: int = 1
    n_stochastic_constraints: int = 0
    minmax: tuple[int] = (1,)
    constraint_type: ConstraintType = ConstraintType.BOX
    variable_type: VariableType = VariableType.CONTINUOUS
    gradient_available: bool = False
    optimal_value: float | None = None

    @property
    @override
    def optimal_solution(self) -> tuple | None:
        solution = self.model.factors["xstar"]
        if isinstance(solution, list):
            return tuple(solution)
        return solution

    model_default_factors: dict = {}
    model_decision_factors: set[str] = {"x"}
    dim: int = 2
    lower_bounds: tuple = (0.1,) * dim
    upper_bounds: tuple = (10,) * dim

    @override
    def vector_to_factor_dict(self, vector: tuple) -> dict:
        return {"x": vector[:]}

    @override
    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:
        return tuple(factor_dict["x"])

    def replicate(self, x: tuple) -> RepResult:
        responses, _ = self.model.replicate()
        objectives = [Objective(stochastic=responses["loglik"])]
        return RepResult(objectives=objectives)

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
