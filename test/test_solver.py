"""Tests for base solver behavior."""

from typing import ClassVar

import numpy as np
from pydantic import BaseModel

from simopt.model import Model
from simopt.problem import Objective, Problem, RepResult
from simopt.problem_types import ConstraintType, ObjectiveType, VariableType
from simopt.solver import Budget, Solver, SolverConfig


class StaticModelConfig(BaseModel):
    pass


class StaticProblemConfig(BaseModel):
    pass


class StaticModel(Model):
    class_name_abbr: ClassVar[str] = "STATIC"
    class_name: ClassVar[str] = "Static"
    config_class: ClassVar[type[BaseModel]] = StaticModelConfig
    n_rngs: ClassVar[int] = 0
    n_responses: ClassVar[int] = 1

    def before_replicate(self, rng_list) -> None:
        pass

    def replicate(self) -> tuple[dict, dict]:
        return {}, {}


class StaticProblem(Problem):
    class_name_abbr: ClassVar[str] = "STATIC"
    class_name: ClassVar[str] = "Static"
    config_class: ClassVar[type[BaseModel]] = StaticProblemConfig
    model_class: ClassVar[type[Model]] = StaticModel
    constraint_type: ClassVar[ConstraintType] = ConstraintType.BOX
    variable_type: ClassVar[VariableType] = VariableType.CONTINUOUS
    gradient_available: ClassVar[bool] = True
    n_objectives: ClassVar[int] = 1
    n_stochastic_constraints: ClassVar[int] = 0
    model_default_factors: ClassVar[dict] = {}
    model_decision_factors: ClassVar[set[str]] = {"x"}

    @property
    def dim(self) -> int:
        return 1

    @property
    def lower_bounds(self) -> tuple[float, ...]:
        return (-10.0,)

    @property
    def upper_bounds(self) -> tuple[float, ...]:
        return (10.0,)

    def vector_to_factor_dict(self, vector: tuple) -> dict:
        return {"x": vector[0]}

    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:
        return (factor_dict["x"],)

    def get_random_solution(self, rand_sol_rng) -> tuple:
        return (1.0,)

    def replicate(self, x: tuple, /) -> RepResult:
        return RepResult([Objective(stochastic=x[0], stochastic_gradients=[2.0])])


class StaticMinProblem(StaticProblem):
    minmax: ClassVar[tuple[int, ...]] = (-1,)


class StaticMaxProblem(StaticProblem):
    minmax: ClassVar[tuple[int, ...]] = (1,)


class StaticSolver(Solver):
    name: str = "STATIC"
    class_name_abbr: ClassVar[str] = "STATIC"
    class_name: ClassVar[str] = "Static"
    config_class: ClassVar[type[SolverConfig]] = SolverConfig
    objective_type: ClassVar[ObjectiveType] = ObjectiveType.SINGLE
    constraint_type: ClassVar[ConstraintType] = ConstraintType.BOX
    variable_type: ClassVar[VariableType] = VariableType.CONTINUOUS
    gradient_needed: ClassVar[bool] = False

    def solve(self, problem: Problem) -> None:
        pass


def test_evaluate_standardizes_objectives_to_minimization() -> None:
    solver = StaticSolver()

    solver.budget = Budget(2)
    min_solution = solver.evaluate((3.0,), StaticMinProblem(), 2)

    solver.budget = Budget(2)
    max_solution = solver.evaluate((3.0,), StaticMaxProblem(), 2)

    assert min_solution.objectives_mean[0] == 3.0
    assert max_solution.objectives_mean[0] == -3.0
    np.testing.assert_array_equal(
        min_solution.objectives_gradients_mean[0], np.array([2.0])
    )
    np.testing.assert_array_equal(
        max_solution.objectives_gradients_mean[0], np.array([-2.0])
    )
