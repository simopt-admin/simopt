"""Stochastic line search algorithm with gradient estimation.

The solver is a stochastic line search algorithm  with the gradient estimate recomputed
in each iteration, whether or not a step is accepted. The algorithm includes the
relaxation of the Armijo condition by an additive constant. A detailed description of
the solver can be found `here <https://simopt.readthedocs.io/en/latest/aloe.html>`__.
"""

from __future__ import annotations

from typing import Annotated, ClassVar

import numpy as np
from pydantic import Field

from simopt.base import (
    ConstraintType,
    Context,
    ObjectiveType,
    Problem,
    Solver,
    SolverConfig,
    VariableType,
)
from simopt.solvers.utils import fd


class ALOEConfig(SolverConfig):
    """Configuration for ALOE solver."""

    r: Annotated[
        int,
        Field(
            default=30,
            gt=0,
            description="number of replications taken at each solution",
        ),
    ]
    theta: Annotated[
        float,
        Field(
            default=0.2,
            gt=0,
            lt=1,
            description="constant in the Armijo condition",
        ),
    ]
    gamma: Annotated[
        float,
        Field(
            default=0.8,
            gt=0,
            lt=1,
            description="constant for shrinking the step size",
        ),
    ]
    alpha_max: Annotated[int, Field(default=10, gt=0, description="maximum step size")]
    alpha_0: Annotated[int, Field(default=1, gt=0, description="initial step size")]
    epsilon_f: Annotated[
        int,
        Field(default=1, gt=0, description="additive constant in the Armijo condition"),
    ]
    sensitivity: Annotated[
        float,
        Field(default=1e-7, gt=0, description="shrinking scale for variable bounds"),
    ]
    lambda_: Annotated[
        int,
        Field(
            default=2,
            gt=0,
            description="magnifying factor for n_r in the finite difference function",
            alias="lambda",
        ),
    ]


class ALOE(Solver):
    """Adaptive Line-search with Oracle Estimations."""

    name: str = "ALOE"
    config_class: ClassVar[type[SolverConfig]] = ALOEConfig
    class_name_abbr: ClassVar[str] = "ALOE"
    class_name: ClassVar[str] = "ALOE"
    objective_type: ClassVar[ObjectiveType] = ObjectiveType.SINGLE
    constraint_type: ClassVar[ConstraintType] = ConstraintType.BOX
    variable_type: ClassVar[VariableType] = VariableType.CONTINUOUS
    gradient_needed: ClassVar[bool] = False

    def solve(self, problem: Problem, ctx: Context) -> None:
        # Default values.
        r = self.factors["r"]
        theta = self.factors["theta"]
        gamma = self.factors["gamma"]
        alpha_max = self.factors["alpha_max"]
        alpha = self.factors["alpha_0"]
        epsilon_f = self.factors["epsilon_f"]

        # Upper and lower bounds.
        lower_bound = np.array(problem.lower_bounds)
        upper_bound = np.array(problem.upper_bounds)

        # Start with the initial solution.
        new_solution = ctx.evaluate(problem.factors["initial_solution"], 0)
        ctx.log(new_solution)
        new_solution = ctx.evaluate(new_solution, r)

        best_solution = new_solution

        while True:
            new_x = np.array(new_solution.x, dtype=float)

            # Check variable bounds
            forward = np.isclose(new_x, lower_bound, atol=self.factors["sensitivity"]).astype(
                np.int64
            )
            backward = np.isclose(new_x, upper_bound, atol=self.factors["sensitivity"]).astype(
                np.int64
            )
            bounds_check = forward - backward

            if problem.gradient_available:
                grad = new_solution.objectives_gradients_mean[0]
            else:

                def fn(x: np.ndarray, reps: int) -> float:
                    candidate_solution = ctx.evaluate(tuple(x), reps)
                    value = candidate_solution.objectives_mean
                    return float(value[0])

                fn_value = float(new_solution.objectives_mean[0])

                grad = fd(
                    lambda x, reps=r: fn(x, reps),
                    new_x,
                    alpha,
                    fn_value,
                    bounds_check,
                    lower_bound,
                    upper_bound,
                )

                while np.all(grad == 0):
                    grad = fd(
                        lambda x, reps=r: fn(x, reps),
                        new_x,
                        alpha,
                        fn_value,
                        bounds_check,
                        lower_bound,
                        upper_bound,
                    )
                    r = int(self.factors["lambda"] * r)  # Update sample size

            # Compute candidate solution and apply box constraints (vectorized).
            candidate_x = np.clip(new_x - alpha * grad, lower_bound, upper_bound)
            candidate_solution = ctx.evaluate(tuple(candidate_x), r)

            # Check modified Armijo condition
            if candidate_solution.objectives_mean <= (
                new_solution.objectives_mean
                - alpha * theta * np.linalg.norm(grad) ** 2
                + 2 * epsilon_f
            ):
                new_solution = candidate_solution
                alpha = min(alpha_max, alpha / gamma)
            else:
                alpha = gamma * alpha

            if new_solution.objectives_mean < best_solution.objectives_mean:
                best_solution = new_solution
                ctx.log(new_solution)
