"""STRONG Solver.

STRONG: A trust-region-based algorithm that fits first- or second-order models through
function evaluations taken within a neighborhood of the incumbent solution.
A detailed description of the solver can be found
`here <https://simopt.readthedocs.io/en/latest/strong.html>`__.
"""

from __future__ import annotations

import math
from typing import Annotated, ClassVar, Self

import numpy as np
from numpy.linalg import norm
from pydantic import Field, model_validator

from simopt.base import (
    ConstraintType,
    ObjectiveType,
    Problem,
    Solver,
    SolverConfig,
    VariableType,
)
from simopt.solvers.utils import bfgs_hessian_approx, finite_diff
from simopt.utils import make_nonzero


class STRONGConfig(SolverConfig):
    """Configuration for STRONG solver."""

    n0: Annotated[int, Field(default=10, gt=0, description="initial sample size")]
    n_r: Annotated[
        int,
        Field(
            default=10,
            gt=0,
            description="number of replications taken at each solution",
        ),
    ]
    sensitivity: Annotated[
        float, Field(default=1e-7, gt=0, description="shrinking scale for VarBds")
    ]
    delta_threshold: Annotated[
        float,
        Field(default=1.2, gt=0, description="maximum value of the radius"),
    ]
    delta_t: Annotated[
        float,
        Field(default=2.0, description="initial size of trust region", alias="delta_T"),
    ]
    eta_0: Annotated[
        float,
        Field(default=0.01, gt=0, lt=1, description="constant for accepting"),
    ]
    eta_1: Annotated[
        float,
        Field(
            default=0.3,
            lt=1,
            description="constant for more confident accepting",
        ),
    ]
    gamma_1: Annotated[
        float,
        Field(
            default=0.9,
            gt=0,
            lt=1,
            description="constant for shrinking the trust region",
        ),
    ]
    gamma_2: Annotated[
        float,
        Field(
            default=1.11,
            gt=1,
            description="constant for expanding the trust region",
        ),
    ]
    lambda_: Annotated[
        int,
        Field(
            default=2,
            gt=1,
            alias="lambda",
            description="magnifying factor for n_r in finite difference function",
        ),
    ]
    lambda_2: Annotated[
        float,
        Field(
            default=1.01,
            gt=1,
            description="magnifying factor for n_r in stage I and stage II (>1)",
        ),
    ]

    @model_validator(mode="after")
    def _validate_cross_field_constraints(self) -> Self:
        if self.delta_t <= self.delta_threshold:
            raise ValueError("delta_t must be greater than delta_threshold")
        if self.eta_1 <= self.eta_0:
            raise ValueError("eta_1 must be greater than eta_0")
        return self


class STRONG(Solver):
    """STRONG Solver.

    A trust-region-based algorithm that fits first- or second-order models through
    function evaluations taken within a neighborhood of the incumbent solution.
    """

    name: str = "STRONG"
    config_class: ClassVar[type[SolverConfig]] = STRONGConfig
    class_name_abbr: ClassVar[str] = "STRONG"
    class_name: ClassVar[str] = "STRONG"
    objective_type: ClassVar[ObjectiveType] = ObjectiveType.SINGLE
    constraint_type: ClassVar[ConstraintType] = ConstraintType.BOX
    variable_type: ClassVar[VariableType] = VariableType.CONTINUOUS
    gradient_needed: ClassVar[bool] = False

    def solve(self, problem: Problem) -> None:  # noqa: D102
        # Default values.
        n0: int = self.factors["n0"]
        n_r: int = self.factors["n_r"]
        delta_threshold: float = self.factors["delta_threshold"]
        delta_t: float = self.factors["delta_T"]
        eta_0: float = self.factors["eta_0"]
        eta_1: float = self.factors["eta_1"]
        gamma_1: float = self.factors["gamma_1"]
        gamma_2: float = self.factors["gamma_2"]
        lam: int = self.factors["lambda"]

        # Upper bound and lower bound.
        lower_bound = np.array(problem.lower_bounds)
        upper_bound = np.array(problem.upper_bounds)

        # Start with the initial solution.
        new_solution = self.create_new_solution(
            problem.factors["initial_solution"], problem
        )

        self.budget.request(n0)
        problem.simulate(new_solution, n0)

        best_solution = new_solution
        self.recommended_solns.append(new_solution)
        self.intermediate_budgets.append(self.budget.used)

        # Precompute factorials
        factorials = np.array([math.factorial(i) for i in range(1, problem.dim + 1)])
        # Precompute other variables
        neg_minmax = -problem.minmax[0]
        dim_sq = problem.dim**2

        while True:
            new_x = np.array(new_solution.x)
            # Check variable bounds.
            forward = np.isclose(
                new_x, lower_bound, atol=self.factors["sensitivity"]
            ).astype(np.int32)
            backward = np.isclose(
                new_x, upper_bound, atol=self.factors["sensitivity"]
            ).astype(np.int32)
            # bounds_check:
            #   1 stands for forward, -1 stands for backward, 0 means central diff.
            bounds_check = forward - backward

            # Stage I.
            if delta_t > delta_threshold:
                # Step 1: Build the linear model.
                num_evals = 2 * problem.dim - np.sum(bounds_check != 0)
                # Generate a new gradient and Hessian matrix.
                num_generated_grads = 0
                while True:
                    grad = finite_diff(
                        self,
                        new_solution,
                        bounds_check,
                        problem,
                        self.factors["delta_T"],
                        n_r,
                    )
                    self.budget.request(num_evals * n_r)
                    num_generated_grads += 1
                    if num_generated_grads > 2:
                        # Update n_r and counter after each loop.
                        n_r *= lam
                    # Accept any non-zero gradient, or exit if the budget is exceeded.
                    if norm(grad) != 0:
                        break

                # Step 2: Solve the subproblem.
                # Cauchy reduction.
                hessian = np.zeros((problem.dim, problem.dim))
                candidate_x = self.cauchy_point(grad, hessian, new_x, problem)
                candidate_solution = self.create_new_solution(
                    tuple(candidate_x), problem
                )

                # Step 3: Compute the ratio.
                # Use n_r simulated observations to estimate g_new.
                self.budget.request(n_r)
                problem.simulate(candidate_solution, n_r)
                # Find the old objective value and the new objective value.
                g_old = neg_minmax * new_solution.objectives_mean
                g_new = neg_minmax * candidate_solution.objectives_mean
                g_diff = g_old - g_new
                # Construct the polynomial.
                x_diff = candidate_x - new_x
                r_old = g_old
                r_new = g_old + (x_diff @ grad) + 0.5 * ((x_diff @ hessian) @ x_diff)

                r_diff = (r_old - r_new)[0]
                r_diff = make_nonzero(r_diff, "r_diff (stage I)")
                rho = g_diff / r_diff

                # Step 4: Update the trust region size and determine to accept or
                # reject the solution.
                if (rho < eta_0) or (g_diff <= 0) or (r_diff <= 0):
                    # The solution fails either the RC or SR test, the center point
                    # remains and the trust region shrinks.
                    delta_t = gamma_1 * delta_t
                elif (eta_0 <= rho) and (rho < eta_1):
                    # The center point moves to the new solution and the trust
                    # region remains.
                    new_solution = candidate_solution
                    # Update incumbent best solution.
                    if (
                        problem.minmax * new_solution.objectives_mean
                        > problem.minmax * best_solution.objectives_mean
                    ):
                        best_solution = new_solution
                        self.recommended_solns.append(new_solution)
                        self.intermediate_budgets.append(self.budget.used)
                else:
                    # The center point moves to the new solution and the trust
                    # region enlarges.
                    delta_t = gamma_2 * delta_t
                    new_solution = candidate_solution
                    # Update incumbent best solution.
                    if (
                        problem.minmax * new_solution.objectives_mean
                        > problem.minmax * best_solution.objectives_mean
                    ):
                        best_solution = new_solution
                        self.recommended_solns.append(new_solution)
                        self.intermediate_budgets.append(self.budget.used)
                n_r = int(np.ceil(self.factors["lambda_2"] * n_r))

            # Stage II.
            # When trust region size is very small, use the quadratic design.
            else:
                n_onbound = np.sum(bounds_check != 0)
                if n_onbound <= 1:
                    num_evals = dim_sq
                else:
                    # TODO: Check the formula, it seems to be dividing an
                    # integer by a tuple.
                    num_evals = (
                        dim_sq
                        + problem.dim
                        - factorials[n_onbound] / (2, factorials[n_onbound - 2])
                    )
                # Step 1: Build the quadratic model.
                num_generated_grads = 0
                while True:
                    grad = finite_diff(
                        self,
                        new_solution,
                        bounds_check,
                        problem,
                        self.factors["delta_T"],
                        n_r,
                    )
                    hessian = bfgs_hessian_approx(
                        self, new_solution, bounds_check, problem, n_r
                    )
                    self.budget.request(num_evals * n_r)
                    num_generated_grads += 1
                    if num_generated_grads > 2:
                        # Update n_r and counter after each loop.
                        n_r *= lam
                    # Accept any non-zero gradient, or exit if the budget is exceeded.
                    if norm(grad) != 0 or self.budget.remaining <= 0:
                        break

                # Step 2: Solve the subproblem.
                # Cauchy reduction.
                candidate_x = self.cauchy_point(
                    grad,
                    hessian,
                    new_x,
                    problem,
                )
                candidate_solution = self.create_new_solution(
                    tuple(candidate_x), problem
                )
                # Step 3: Compute the ratio.
                # Use r simulated observations to estimate g(x_start\).
                problem.simulate(candidate_solution, n_r)
                self.budget.request(n_r)
                # Find the old objective value and the new objective value.
                g_old = neg_minmax * new_solution.objectives_mean
                g_new = neg_minmax * candidate_solution.objectives_mean
                g_diff = g_old - g_new
                # Construct the polynomial.
                x_diff = candidate_x - new_x
                r_old = g_old
                r_new = g_old + (x_diff @ grad) + 0.5 * ((x_diff @ hessian) @ x_diff)

                r_diff = (r_old - r_new)[0]
                r_diff = make_nonzero(r_diff, "rdiff (stage II)")
                rho = g_diff / r_diff
                # Step 4: Update the trust region size and determine to accept or
                # reject the solution.
                if (rho < eta_0) or (g_diff <= 0) or (r_diff <= 0):
                    # Inner Loop.
                    rr_old = r_old
                    g_b_old = rr_old
                    sub_counter = 1
                    result_solution = new_solution
                    result_x = new_x

                    while np.sum(result_x != new_x) == 0:
                        if self.budget.remaining <= 0:
                            break
                        # A while loop to prevent zero gradient
                        while True:
                            n_r_loop = (sub_counter + 1) * n_r
                            g_var = finite_diff(
                                self,
                                new_solution,
                                bounds_check,
                                problem,
                                self.factors["delta_T"],
                                n_r_loop,
                            )
                            h_var = bfgs_hessian_approx(
                                self, new_solution, bounds_check, problem, n_r_loop
                            )
                            self.budget.request(num_evals * n_r_loop)
                            num_generated_grads += 1
                            if num_generated_grads > 2:
                                # Update n_r and counter after each loop.
                                n_r *= lam
                            # Accept any non-zero gradient, or exit if the budget
                            # is exceeded.
                            if norm(grad) != 0 or self.budget.remaining <= 0:
                                break

                        # Step 2: determine the new inner solution based on the
                        # accumulated design matrix X.
                        try_x = self.cauchy_point(g_var, h_var, new_x, problem)
                        try_solution = self.create_new_solution(tuple(try_x), problem)

                        # Step 3.
                        counter_ceiling = np.ceil(
                            sub_counter ** self.factors["lambda_2"]
                        )
                        counter_lower_ceiling = np.ceil(
                            (sub_counter - 1) ** self.factors["lambda_2"]
                        )
                        # Theoretically these are already integers
                        ceiling_diff = int(counter_ceiling - counter_lower_ceiling)
                        mreps = int(n_r + counter_ceiling)

                        problem.simulate(try_solution, mreps)
                        self.budget.request(mreps)
                        g_b_new = neg_minmax * try_solution.objectives_mean
                        dummy_solution = new_solution
                        problem.simulate(dummy_solution, ceiling_diff)
                        self.budget.request(ceiling_diff)

                        dummy = neg_minmax * dummy_solution.objectives_mean
                        # Update g_old.
                        g_b_old = (
                            g_b_old * (n_r + counter_lower_ceiling)
                            + ceiling_diff * dummy
                        ) / mreps

                        x_diff = try_x - new_x
                        rr_new = (
                            g_b_old
                            + (x_diff @ g_var)
                            + 0.5 * ((x_diff @ h_var) @ x_diff)
                        )

                        rr_old = g_b_old
                        # Set rho to the ratio.
                        g_b_diff = g_b_old - g_b_new
                        rr_diff = (rr_old - rr_new)[0]
                        rr_diff = make_nonzero(rr_diff, "rr_diff")
                        rrho = g_b_diff / rr_diff

                        if (rrho < eta_0) or (g_b_diff <= 0) or (rr_diff <= 0):
                            delta_t = gamma_1 * delta_t
                            result_solution = new_solution
                            result_x = new_x
                        elif (eta_0 <= rrho) and (rrho < eta_1):
                            # Accept the solution and remains the size of trust region.
                            result_solution = try_solution
                            result_x = try_x
                            rr_old = g_b_new
                        else:
                            # Accept the solution and expand the size of trust region.
                            delta_t = gamma_2 * delta_t
                            result_solution = try_solution
                            result_x = try_x
                            rr_old = g_b_new
                        sub_counter += 1
                    new_solution = result_solution
                    # Update incumbent best solution.
                    if (
                        problem.minmax * new_solution.objectives_mean
                        > problem.minmax * best_solution.objectives_mean
                    ):
                        best_solution = new_solution
                        self.recommended_solns.append(new_solution)
                        self.intermediate_budgets.append(self.budget.used)
                else:
                    # The center point moves to the new solution and the trust
                    # region enlarges.
                    if not ((eta_0 <= rho) and (rho < eta_1)):
                        delta_t = gamma_2 * delta_t
                    new_solution = candidate_solution
                    # Update incumbent best solution.
                    if (
                        problem.minmax * new_solution.objectives_mean
                        > problem.minmax * best_solution.objectives_mean
                    ):
                        best_solution = new_solution
                        self.recommended_solns.append(new_solution)
                        self.intermediate_budgets.append(self.budget.used)
                n_r = int(np.ceil(self.factors["lambda_2"] * n_r))

    def cauchy_point(
        self,
        grad: np.ndarray,
        hessian: np.ndarray,
        new_x: np.ndarray,
        problem: Problem,
    ) -> np.ndarray:
        """Find the Cauchy point based on the gradient and Hessian matrix."""
        delta_t = self.factors["delta_T"]
        lower_bound = problem.lower_bounds
        upper_bound = problem.upper_bounds

        val = float(np.dot(grad, hessian @ grad))
        val_dt = delta_t * val
        grad_norm = float(norm(grad))
        grad_norm = make_nonzero(grad_norm, "grad_norm")
        tau = 1 if val <= 0 else min(1, grad_norm**3 / val_dt)
        candidate_x = new_x - tau * delta_t * grad / grad_norm
        return self.check_cons(candidate_x, new_x, lower_bound, upper_bound)

    def check_cons(
        self,
        candidate_x: tuple,
        new_x: tuple | np.ndarray,
        lower_bound: tuple,
        upper_bound: tuple,
    ) -> np.ndarray:
        """Check feasibility of a new point and apply Cauchy point correction if needed.

        This method compares a candidate point to its updated version and enforces
        box constraints defined by lower and upper bounds.

        Args:
            candidate_x (tuple): Current decision variable vector (the Cauchy point).
            new_x (tuple | np.ndarray): Proposed new solution to check and correct.
            lower_bound (tuple): Lower bounds for each decision variable.
            upper_bound (tuple): Upper bounds for each decision variable.

        Returns:
            np.ndarray: The corrected feasible solution, clipped to respect the bounds.
        """
        # Convert the inputs to numpy arrays
        candidate_x_arr = np.array(candidate_x)
        # If new_x is a tuple, convert it to a numpy array
        if isinstance(new_x, tuple):
            new_x = np.array(new_x)
        current_step: np.ndarray = candidate_x_arr - new_x
        lower_bound_arr = np.array(lower_bound)
        upper_bound_arr = np.array(upper_bound)
        # The current step.
        # Form a matrix to determine the possible stepsize.
        min_step = 1.0
        pos_mask = current_step > 0
        if np.any(pos_mask):
            step_diff = (upper_bound_arr[pos_mask] - new_x[pos_mask]) / current_step[
                pos_mask
            ]
            min_step = min(min_step, float(np.min(step_diff).item()))
        neg_mask = current_step < 0
        if np.any(neg_mask):
            step_diff = (lower_bound_arr[neg_mask] - new_x[neg_mask]) / current_step[
                neg_mask
            ]
            min_step = min(min_step, float(np.min(step_diff).item()))
        # Calculate the modified x.
        return new_x + min_step * current_step
