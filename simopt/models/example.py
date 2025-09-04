"""Example problem of deterministic function with noise.

Simulate a synthetic problem with a deterministic objective function
evaluated with noise.
"""

from __future__ import annotations

from typing import Annotated, ClassVar

import numpy as np
from pydantic import BaseModel, Field

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.base import (
    ConstraintType,
    Model,
    Objective,
    Problem,
    RepResult,
    VariableType,
)
from simopt.input_models import Normal


class ExampleModelConfig(BaseModel):
    """Configuration model for Example simulation.

    A model that is a deterministic function evaluated with noise.
    """

    x: Annotated[
        tuple[float, ...],
        Field(
            default=(2.0, 2.0),
            description="point to evaluate",
        ),
    ]


class ExampleProblemConfig(BaseModel):
    """Configuration model for Example Problem.

    Base class to implement simulation-optimization problems.
    """

    initial_solution: Annotated[
        tuple[float, ...],
        Field(
            default=(2.0, 2.0),
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


class ExampleModel(Model):
    """A model that is a deterministic function evaluated with noise."""

    class_name_abbr: ClassVar[str] = "EXAMPLE"
    class_name: ClassVar[str] = "Deterministic Function + Noise"
    config_class: ClassVar[type[BaseModel]] = ExampleModelConfig
    n_rngs: ClassVar[int] = 1
    n_responses: ClassVar[int] = 1

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize the model.

        Args:
            fixed_factors (dict | None): fixed factors of the model.
                If None, use default values.
        """
        # Let the base class handle default arguments.
        super().__init__(fixed_factors)
        self.noise_model = Normal()

    def before_replicate(self, rng_list: list[MRG32k3a]) -> None:  # noqa: D102
        self.noise_model.set_rng(rng_list[0])

    def replicate(self) -> tuple[dict, dict]:
        """Evaluate a deterministic function f(x) with stochastic noise.

        Returns:
            tuple[dict, dict]: A tuple containing:
                - responses (dict): Performance measures of interest, including:
                    - "est_f(x)": Estimate of f(x) with added stochastic noise.
                - gradients (dict): A dictionary of gradient estimates for
                    each response.
        """
        x = np.array(self.factors["x"])
        fn_eval_at_x = np.linalg.norm(x) ** 2 + self.noise_model.random()

        # Compose responses and gradients.
        responses = {"est_f(x)": fn_eval_at_x}
        gradients = {"est_f(x)": {"x": tuple(2 * x)}}
        return responses, gradients


class ExampleProblem(Problem):
    """Base class to implement simulation-optimization problems."""

    class_name_abbr: ClassVar[str] = "EXAMPLE-1"
    class_name: ClassVar[str] = "Min Deterministic Function + Noise"
    config_class: ClassVar[type[BaseModel]] = ExampleProblemConfig
    model_class: ClassVar[type[Model]] = ExampleModel
    n_objectives: ClassVar[int] = 1
    n_stochastic_constraints: ClassVar[int] = 0
    minmax: ClassVar[tuple[int, ...]] = (-1,)
    constraint_type: ClassVar[ConstraintType] = ConstraintType.UNCONSTRAINED
    variable_type: ClassVar[VariableType] = VariableType.CONTINUOUS
    gradient_available: ClassVar[bool] = True
    model_default_factors: ClassVar[dict] = {}
    model_decision_factors: ClassVar[set[str]] = {"x"}

    @property
    def optimal_value(self) -> float | None:  # noqa: D102
        return 0.0

    @property
    def optimal_solution(self) -> tuple | None:  # noqa: D102
        # Change if f is changed
        # TODO: figure out what f is
        return (0,) * self.dim

    @property
    def dim(self) -> int:  # noqa: D102
        return len(self.factors["initial_solution"])

    @property
    def lower_bounds(self) -> tuple:  # noqa: D102
        return (-np.inf,) * self.dim

    @property
    def upper_bounds(self) -> tuple:  # noqa: D102
        return (np.inf,) * self.dim

    def vector_to_factor_dict(self, vector: tuple) -> dict:  # noqa: D102
        return {"x": vector[:]}

    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:  # noqa: D102
        return tuple(factor_dict["x"])

    def replicate(self, _x: tuple) -> RepResult:  # noqa: D102
        responses, gradients = self.model.replicate()
        objectives = [
            Objective(
                stochastic=responses["est_f(x)"],
                stochastic_gradients=gradients["est_f(x)"]["x"],
            )
        ]
        return RepResult(objectives=objectives)

    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:  # noqa: D102
        # x = tuple([rand_sol_rng.uniform(-2, 2) for _ in range(self.dim)])
        return tuple(
            rand_sol_rng.mvnormalvariate(
                mean_vec=[0.0] * self.dim,
                cov=np.eye(self.dim).tolist(),
                factorized=False,
            )
        )
