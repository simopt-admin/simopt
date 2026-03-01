"""First-order gradient-based optimization of stochastic objective functions.

An algorithm for first-order gradient-based optimization of
stochastic objective functions, based on adaptive estimates of lower-order moments.
"""

from __future__ import annotations

from typing import Annotated, ClassVar

import numpy as np
from pydantic import Field

from simopt.base import (
    ConstraintType,
    ObjectiveType,
    Problem,
    Solver,
    SolverConfig,
    VariableType,
)
from simopt.solvers.utils import fd


class ADAMConfig(SolverConfig):
    """Configuration for ADAM solver."""

    r: Annotated[
        int,
        Field(
            default=30,
            gt=0,
            description="number of replications taken at each solution",
        ),
    ]
    beta_1: Annotated[
        float,
        Field(
            default=0.9,
            gt=0,
            lt=1,
            description="exponential decay of the rate for the first moment estimates",
        ),
    ]
    beta_2: Annotated[
        float,
        Field(
            default=0.999,
            lt=1,
            description="exponential decay rate for the second-moment estimates",
        ),
    ]
    alpha: Annotated[float, Field(default=0.5, gt=0, description="step size")]
    epsilon: Annotated[
        float,
        Field(default=1e-8, gt=0, description="a small value to prevent zero-division"),
    ]
    sensitivity: Annotated[
        float,
        Field(default=1e-7, gt=0, description="shrinking scale for variable bounds"),
    ]


class ADAM(Solver):
    """First-order gradient-based optimization of stochastic objective functions.

    An algorithm for first-order gradient-based optimization of
    stochastic objective functions, based on adaptive estimates of lower-order moments.
    """

    name: str = "ADAM"
    config_class: ClassVar[type[SolverConfig]] = ADAMConfig
    class_name_abbr: ClassVar[str] = "ADAM"
    class_name: ClassVar[str] = "ADAM"
    objective_type: ClassVar[ObjectiveType] = ObjectiveType.SINGLE
    constraint_type: ClassVar[ConstraintType] = ConstraintType.BOX
    variable_type: ClassVar[VariableType] = VariableType.CONTINUOUS
    gradient_needed: ClassVar[bool] = False

    def solve(self, problem: Problem) -> None:
        # Default values.
        r: int = self.factors["r"]
        beta_1: float = self.factors["beta_1"]
        beta_2: float = self.factors["beta_2"]
        alpha: float = self.factors["alpha"]
        epsilon: float = self.factors["epsilon"]

        # Upper bound and lower bound.
        lower_bound = np.array(problem.lower_bounds)
        upper_bound = np.array(problem.upper_bounds)

        # Start with the initial solution.
        new_solution = self.create_new_solution(problem.factors["initial_solution"], problem)
        self.recommended_solns.append(new_solution)
        self.intermediate_budgets.append(self.budget.used)

        self.budget.request(r)
        problem.simulate(new_solution, r)

        best_solution = new_solution

        # Initialize the first moment vector, the second moment vector,
        # and the timestep.
        m = np.zeros(problem.dim)
        v = np.zeros(problem.dim)
        t = 0

        while True:
            # Update timestep.
            t += 1
            # Check variable bounds.
            x = np.array(new_solution.x, dtype=float)
            forward = np.isclose(x, lower_bound, atol=self.factors["sensitivity"]).astype(int)
            backward = np.isclose(x, upper_bound, atol=self.factors["sensitivity"]).astype(int)
            # 1 stands for forward, -1 stands for backward, 0 means central diff.
            bounds_check = np.subtract(forward, backward)
            if problem.gradient_available:
                # Use IPA gradient if available.
                grad = -problem.minmax[0] * new_solution.objectives_gradients_mean[0]
            else:
                # Use finite difference to estimate gradient if IPA gradient is
                # not available.
                def fn(x: np.ndarray) -> float:
                    candidate_solution = self.create_new_solution(tuple(x), problem)
                    self.budget.request(r)
                    problem.simulate(candidate_solution, r)
                    value = -problem.minmax[0] * candidate_solution.objectives_mean
                    return float(value[0])

                fn_value = float((-problem.minmax[0] * new_solution.objectives_mean)[0])

                grad = fd(
                    fn,
                    x,
                    self.factors["alpha"],
                    fn_value,
                    bounds_check,
                    lower_bound,
                    upper_bound,
                )

            # Update biased first moment estimate.
            m = beta_1 * m + (1 - beta_1) * grad
            # Update biased second raw moment estimate.
            v = beta_2 * v + (1 - beta_2) * grad**2
            # Compute bias-corrected first moment estimate.
            mhat = m / (1 - beta_1**t)
            # Compute bias-corrected second raw moment estimate.
            vhat = v / (1 - beta_2**t)
            # Update new_x (vectorized) and apply box constraints
            new_x = np.clip(x - alpha * mhat / (np.sqrt(vhat) + epsilon), lower_bound, upper_bound)

            # Create new solution based on new x
            new_solution = self.create_new_solution(tuple(new_x), problem)
            # Use r simulated observations to estimate the objective value.
            self.budget.request(r)
            problem.simulate(new_solution, r)

            if (new_solution.objectives_mean > best_solution.objectives_mean) ^ (
                problem.minmax[0] < 0
            ):
                best_solution = new_solution
                self.recommended_solns.append(new_solution)
                self.intermediate_budgets.append(self.budget.used)
