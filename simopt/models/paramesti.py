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


class ParameterEstimationConfig(BaseModel):
    """Configuration for the parameter estimation model."""

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

    class_name_abbr: ClassVar[str] = "PARAMESTI"
    class_name: ClassVar[str] = "Gamma Parameter Estimation"
    config_class: ClassVar[type[BaseModel]] = ParameterEstimationConfig
    n_rngs: ClassVar[int] = 2
    n_responses: ClassVar[int] = 1

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
        return responses, {}


class ParamEstiMaxLogLik(Problem):
    """Base class to implement simulation-optimization problems."""

    class_name_abbr: ClassVar[str] = "PARAMESTI-1"
    class_name: ClassVar[str] = "Max Log Likelihood for Gamma Parameter Estimation"
    config_class: ClassVar[type[BaseModel]] = ParamEstiMaxLogLikConfig
    model_class: ClassVar[type[Model]] = ParameterEstimation
    n_objectives: ClassVar[int] = 1
    n_stochastic_constraints: ClassVar[int] = 0
    minmax: ClassVar[tuple[int, ...]] = (1,)
    constraint_type: ClassVar[ConstraintType] = ConstraintType.BOX
    variable_type: ClassVar[VariableType] = VariableType.CONTINUOUS
    gradient_available: ClassVar[bool] = False
    model_default_factors: ClassVar[dict] = {}
    model_decision_factors: ClassVar[set[str]] = {"x"}

    @property
    def optimal_value(self) -> float | None:  # noqa: D102
        return None

    @property
    def optimal_solution(self) -> tuple | None:  # noqa: D102
        solution = self.model.factors["xstar"]
        if isinstance(solution, list):
            return tuple(solution)
        return solution

    @property
    def dim(self) -> int:  # noqa: D102
        return 2

    @property
    def lower_bounds(self) -> tuple:  # noqa: D102
        return (0.1,) * self.dim

    @property
    def upper_bounds(self) -> tuple:  # noqa: D102
        return (10,) * self.dim

    def vector_to_factor_dict(self, vector: tuple) -> dict:  # noqa: D102
        return {"x": vector[:]}

    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:  # noqa: D102
        return tuple(factor_dict["x"])

    def replicate(self, _x: tuple) -> RepResult:  # noqa: D102
        responses, _ = self.model.replicate()
        objectives = [Objective(stochastic=responses["loglik"])]
        return RepResult(objectives=objectives)

    def check_deterministic_constraints(self, _x: tuple) -> bool:  # noqa: D102
        return True

    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:  # noqa: D102
        return tuple(
            [
                rand_sol_rng.uniform(self.lower_bounds[idx], self.upper_bounds[idx])
                for idx in range(self.dim)
            ]
        )
