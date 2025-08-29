"""Example problem of deterministic function with noise.

Simulate a synthetic problem with a deterministic objective function
evaluated with noise.
"""

from __future__ import annotations

from typing import Annotated, ClassVar

import numpy as np
from pydantic import BaseModel, Field

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.base import ConstraintType, Model, Problem, VariableType
from simopt.input_models import Normal
from simopt.utils import override


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

    config_class: ClassVar[type[BaseModel]] = ExampleModelConfig
    class_name_abbr: str = "EXAMPLE"
    class_name: str = "Deterministic Function + Noise"
    n_rngs: int = 1
    n_responses: int = 1

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

    config_class: ClassVar[type[BaseModel]] = ExampleProblemConfig
    model_class: ClassVar[type[Model]] = ExampleModel
    class_name_abbr: str = "EXAMPLE-1"
    class_name: str = "Min Deterministic Function + Noise"
    n_objectives: int = 1
    n_stochastic_constraints: int = 0
    minmax: tuple[int] = (-1,)
    constraint_type: ConstraintType = ConstraintType.UNCONSTRAINED
    variable_type: VariableType = VariableType.CONTINUOUS
    gradient_available: bool = True
    optimal_value: float | None = 0.0

    @property
    @override
    def optimal_solution(self) -> tuple:
        # Change if f is changed
        # TODO: figure out what f is
        return (0,) * self.dim

    model_default_factors: dict = {}

    @property
    @override
    def model_fixed_factors(self) -> dict:
        return {}

    @model_fixed_factors.setter
    def model_fixed_factors(self, value: dict | None) -> None:
        # TODO: figure out if fixed factors should change
        pass

    model_decision_factors: set[str] = {"x"}

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
