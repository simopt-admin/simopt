"""Example problem of deterministic function with noise.

Simulate a synthetic problem with a deterministic objective function
evaluated with noise.
"""

from __future__ import annotations

from typing import Annotated, ClassVar

import numpy as np
from pydantic import BaseModel, Field

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt import dsl
from simopt.base import (
    ConstraintType,
    Model,
    Problem,
    VariableType,
)
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

    def replicate(self, factors: dict, rngs: list[MRG32k3a]) -> tuple[dict, dict]:
        """Evaluate a deterministic function f(x) with stochastic noise.

        Returns:
            tuple[dict, dict]: A tuple containing:
                - responses (dict): Performance measures of interest, including:
                    - "est_f(x)": Estimate of f(x) with added stochastic noise.
                - gradients (dict): A dictionary of gradient estimates for
                    each response.
        """
        x = np.array(factors["x"])
        fn_eval_at_x = np.linalg.norm(x) ** 2 + self.noise_model.random(rngs[0])

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
    def optimal_value(self) -> float | None:
        return 0.0

    @property
    def optimal_solution(self) -> tuple | None:
        # Change if f is changed
        # TODO: figure out what f is
        return (0,) * self.dim

    @override
    def build(self) -> dsl.Model:
        problem = dsl.Model()
        initial_solution = self.factors["initial_solution"]
        x = problem.add_continuous_vector(
            lb=-np.inf, ub=np.inf, shape=(len(initial_solution),), initial=initial_solution
        )
        simulation = self.add_simulation(problem, {"x": x})
        problem.minimize(dsl.mean(simulation.metric("est_f(x)")))
        return problem

    def vector_to_factor_dict(self, vector: tuple) -> dict:
        return {"x": vector[:]}

    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
        # x = tuple([rand_sol_rng.uniform(-2, 2) for _ in range(self.dim)])
        return tuple(
            rand_sol_rng.mvnormalvariate(
                mean_vec=[0.0] * self.dim,
                cov=np.eye(self.dim).tolist(),
                factorized=False,
            )
        )


class Example2ModelConfig(BaseModel):
    """Configuration model for Example-2 simulation.

    A model that is a deterministic quadratic function evaluated with noise.
    """

    x: Annotated[
        tuple[int, ...],
        Field(
            default=(0, 0, 0, 0),
            description="point to evaluate",
        ),
    ]


class Example2ProblemConfig(BaseModel):
    """Configuration model for Example-2 Problem.

    Base class to implement simulation-optimization problems.
    """

    initial_solution: Annotated[
        tuple[int, ...],
        Field(
            default=(0, 0, 0, 0),
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


class Example2Model(Model):
    """A model that is a deterministic quadratic function evaluated with noise."""

    class_name_abbr: ClassVar[str] = "EXAMPLE-2-MODEL"
    class_name: ClassVar[str] = "Quadratic Function + Noise (Discrete)"
    config_class: ClassVar[type[BaseModel]] = Example2ModelConfig
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

    def replicate(self, factors: dict, rngs: list[MRG32k3a]) -> tuple[dict, dict]:
        """Evaluate a quadratic function f(x) with stochastic noise."""
        x = np.array(factors["x"])
        target = np.array([1, 2, 3, 4])
        fn_eval_at_x = np.sum((x - target) ** 2) + self.noise_model.random(rngs[0])

        responses = {"est_f(x)": fn_eval_at_x}
        return responses, {}


class Example2Problem(Problem):
    """Discrete quadratic minimization example with noise."""

    class_name_abbr: ClassVar[str] = "EXAMPLE-2"
    class_name: ClassVar[str] = "Min Quadratic Function + Noise (Discrete)"
    config_class: ClassVar[type[BaseModel]] = Example2ProblemConfig
    model_class: ClassVar[type[Model]] = Example2Model
    n_objectives: ClassVar[int] = 1
    n_stochastic_constraints: ClassVar[int] = 0
    minmax: ClassVar[tuple[int, ...]] = (-1,)
    constraint_type: ClassVar[ConstraintType] = ConstraintType.UNCONSTRAINED
    variable_type: ClassVar[VariableType] = VariableType.DISCRETE
    gradient_available: ClassVar[bool] = False
    model_default_factors: ClassVar[dict] = {}
    model_decision_factors: ClassVar[set[str]] = {"x"}

    @property
    def optimal_value(self) -> float | None:
        return 0.0

    @property
    def optimal_solution(self) -> tuple | None:
        return (1, 2, 3, 4)

    @override
    def build(self) -> dsl.Model:
        problem = dsl.Model()
        initial_solution = self.factors["initial_solution"]
        x = problem.add_integer_vector(
            lb=-4, ub=4, shape=(len(initial_solution),), initial=initial_solution
        )
        simulation = self.add_simulation(problem, {"x": x})
        problem.minimize(dsl.mean(simulation.metric("est_f(x)")))
        return problem

    def vector_to_factor_dict(self, vector: tuple) -> dict:
        return {"x": vector[:]}

    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
        return tuple(rand_sol_rng.randint(-4, 4) for _ in range(self.dim))
