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
    ObjectiveType,
    Problem,
    Solution,
    Solver,
    SolverConfig,
    VariableType,
)
from simopt.solvers.utils import finite_diff


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

    def solve(self, problem: Problem) -> None:  # noqa: D102
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
        new_solution = self.create_new_solution(
            problem.factors["initial_solution"], problem
        )
        self.recommended_solns.append(new_solution)
        self.intermediate_budgets.append(self.budget.used)

        self.budget.request(r)
        problem.simulate(new_solution, r)

        best_solution = new_solution

        while True:
            new_x = np.array(new_solution.x, dtype=float)

            # Check variable bounds
            forward = np.isclose(
                new_x, lower_bound, atol=self.factors["sensitivity"]
            ).astype(np.int64)
            backward = np.isclose(
                new_x, upper_bound, atol=self.factors["sensitivity"]
            ).astype(np.int64)
            bounds_check = forward - backward

            if problem.gradient_available:
                grad = -problem.minmax[0] * new_solution.objectives_gradients_mean[0]
            else:
                finite_diff_budget = (
                    2 * problem.dim - np.count_nonzero(bounds_check)
                ) * r
                self.budget.request(int(finite_diff_budget))
                grad = finite_diff(self, new_solution, bounds_check, problem, alpha, r)

                while np.all(grad == 0):
                    finite_diff_budget = (
                        2 * problem.dim - np.count_nonzero(bounds_check)
                    ) * r
                    self.budget.request(int(finite_diff_budget))
                    grad = finite_diff(
                        self, new_solution, bounds_check, problem, alpha, r
                    )
                    r = int(self.factors["lambda"] * r)  # Update sample size

            # Compute candidate solution and apply box constraints (vectorized).
            candidate_x = np.clip(new_x - alpha * grad, lower_bound, upper_bound)
            candidate_solution = self.create_new_solution(tuple(candidate_x), problem)

            self.budget.request(r)
            problem.simulate(candidate_solution, r)

            # Check modified Armijo condition
            if (-problem.minmax[0] * candidate_solution.objectives_mean) <= (
                -problem.minmax[0] * new_solution.objectives_mean
                - alpha * theta * np.linalg.norm(grad) ** 2
                + 2 * epsilon_f
            ):
                new_solution = candidate_solution
                alpha = min(alpha_max, alpha / gamma)
            else:
                alpha = gamma * alpha

            if (
                problem.minmax[0] * new_solution.objectives_mean
                > problem.minmax[0] * best_solution.objectives_mean
            ):
                best_solution = new_solution
                self.recommended_solns.append(new_solution)
                self.intermediate_budgets.append(self.budget.used)

    def _finite_diff(
        self,
        new_solution: Solution,
        bounds_check: np.ndarray,
        problem: Problem,
        stepsize: float,
        r: int,
    ) -> np.ndarray:
        """Compute the finite difference approximation of the gradient for a solution.

        Args:
            new_solution (Solution): The current solution to perturb.
            bounds_check (np.ndarray): Array indicating which perturbation method to
                use per dimension.
            problem (Problem): The problem instance providing bounds and function
                evaluations.
            stepsize (float): The step size used for finite difference calculations.
            r (int): The number of replications used for each function evaluation.

        Returns:
            np.ndarray: The approximated gradient of the function at the given solution.
        """
        lower_bound = problem.lower_bounds
        upper_bound = problem.upper_bounds
        fn = -1 * problem.minmax[0] * new_solution.objectives_mean
        new_x = np.array(new_solution.x, dtype=np.float64)
        # Store values for each dimension.
        function_diff = np.zeros((problem.dim, 3))

        # Compute step sizes
        step_forward = np.minimum(stepsize, upper_bound - new_x)
        step_backward = np.minimum(stepsize, new_x - lower_bound)

        # Create perturbed variables
        x1 = np.tile(new_x, (problem.dim, 1))
        x2 = np.tile(new_x, (problem.dim, 1))

        central_mask = bounds_check == 0
        forward_mask = bounds_check == 1
        backward_mask = bounds_check == -1

        # Assign step sizes
        function_diff[:, 2] = np.where(
            central_mask,
            np.minimum(step_forward, step_backward),
            np.where(forward_mask, step_forward, step_backward),
        )

        # Apply step updates
        np.fill_diagonal(x1, new_x + function_diff[:, 2])
        np.fill_diagonal(x2, new_x - function_diff[:, 2])

        # Identify indices where x1 and x2 solutions are needed
        x1_indices = np.where(bounds_check != -1)[0]
        x2_indices = np.where(bounds_check != 1)[0]

        # Simulate only required solutions
        for i in x1_indices:
            x1_solution = self.create_new_solution(tuple(x1[i]), problem)
            problem.simulate_up_to([x1_solution], r)
            fn1 = -problem.minmax[0] * x1_solution.objectives_mean
            function_diff[i, 0] = fn1[0] if isinstance(fn1, np.ndarray) else fn1

        for i in x2_indices:
            x2_solution = self.create_new_solution(tuple(x2[i]), problem)
            problem.simulate_up_to([x2_solution], r)
            fn2 = -problem.minmax[0] * x2_solution.objectives_mean
            function_diff[i, 1] = fn2[0] if isinstance(fn2, np.ndarray) else fn2

        # Compute gradient
        fn_divisor = function_diff[:, 2].copy()
        fn_divisor[central_mask] *= 2

        fn_diff = np.zeros(problem.dim)
        if np.any(central_mask):
            fn_diff[central_mask] = function_diff[:, 0] - function_diff[:, 1]
        if np.any(forward_mask):
            fn_diff[forward_mask] = function_diff[forward_mask, 0] - fn
        if np.any(backward_mask):
            fn_diff[backward_mask] = fn - function_diff[backward_mask, 1]

        return fn_diff / fn_divisor
