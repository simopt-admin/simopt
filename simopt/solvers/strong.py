"""
Summary
-------
STRONG: A trust-region-based algorithm that fits first- or second-order models through function evaluations taken within
a neighborhood of the incumbent solution.
A detailed description of the solver can be found
`here <https://simopt.readthedocs.io/en/latest/strong.html>`__.
"""

from __future__ import annotations

import math
import sys
from typing import Callable, Literal

import numpy as np
from numpy.linalg import norm

from simopt.base import (
    ConstraintType,
    ObjectiveType,
    Problem,
    Solution,
    Solver,
    VariableType,
)


class STRONG(Solver):
    """
    A trust-region-based algorithm that fits first- or second-order models through function evaluations taken within a neighborhood of the incumbent solution.

    Attributes
    ----------
    name : string
        name of solver
    objective_type : string
        description of objective types:
            "single" or "multi"
    constraint_type : string
        description of constraints types:
            "unconstrained", "box", "deterministic", "stochastic"
    variable_type : string
        description of variable types:
            "discrete", "continuous", "mixed"
    gradient_needed : bool
        indicates if gradient of objective function is needed
    factors : dict
        changeable factors (i.e., parameters) of the solver
    specifications : dict
        details of each factor (for GUI, data validation, and defaults)
    rng_list : list of mrg32k3a.mrg32k3a.MRG32k3a objects
        list of RNGs used for the solver's internal purposes

    Arguments
    ---------
    name : str
        user-specified name for solver
    fixed_factors : dict
        fixed_factors of the solver

    See also
    --------
    base.Solver
    """

    @property
    def objective_type(self) -> ObjectiveType:
        return ObjectiveType.SINGLE

    @property
    def constraint_type(self) -> ConstraintType:
        return ConstraintType.BOX

    @property
    def variable_type(self) -> VariableType:
        return VariableType.CONTINUOUS

    @property
    def gradient_needed(self) -> bool:
        return False

    @property
    def specifications(self) -> dict[str, dict]:
        return {
            "crn_across_solns": {
                "description": "use CRN across solutions?",
                "datatype": bool,
                "default": True,
            },
            "n0": {
                "description": "initial sample size",
                "datatype": int,
                "default": 10,
            },
            "n_r": {
                "description": "number of replications taken at each solution",
                "datatype": int,
                "default": 10,
            },
            "sensitivity": {
                "description": "shrinking scale for VarBds",
                "datatype": float,
                "default": 10 ** (-7),
            },
            "delta_threshold": {
                "description": "maximum value of the radius",
                "datatype": float,
                "default": 1.2,
            },
            "delta_T": {
                "description": "initial size of trust region",
                "datatype": float,
                "default": 2.0,
            },
            "eta_0": {
                "description": "constant for accepting",
                "datatype": float,
                "default": 0.01,
            },
            "eta_1": {
                "description": "constant for more confident accepting",
                "datatype": float,
                "default": 0.3,
            },
            "gamma_1": {
                "description": "constant for shrinking the trust region",
                "datatype": float,
                "default": 0.9,
            },
            "gamma_2": {
                "description": "constant for expanding the trust region",
                "datatype": float,
                "default": 1.11,
            },
            "lambda": {
                "description": "magnifying factor for n_r inside the finite difference function",
                "datatype": int,
                "default": 2,
            },
            "lambda_2": {
                "description": "magnifying factor for n_r in stage I and stage II (>1)",
                "datatype": float,
                "default": 1.01,
            },
        }

    @property
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "crn_across_solns": self.check_crn_across_solns,
            "n0": self.check_n0,
            "n_r": self.check_n_r,
            "sensitivity": self.check_sensitivity,
            "delta_threshold": self.check_delta_threshold,
            "delta_T": self.check_delta_t,
            "eta_0": self.check_eta_0,
            "eta_1": self.check_eta_1,
            "gamma_1": self.check_gamma_1,
            "gamma_2": self.check_gamma_2,
            "lambda": self.check_lambda,
            "lambda_2": self.check_lambda_2,
        }

    def __init__(
        self, name: str = "STRONG", fixed_factors: dict | None = None
    ) -> None:
        # Let the base class handle default arguments.
        super().__init__(name, fixed_factors)

    def check_n0(self) -> None:
        if self.factors["n0"] <= 0:
            raise ValueError("n0 must be greater than 0.")

    def check_n_r(self) -> None:
        if self.factors["n_r"] <= 0:
            raise ValueError(
                "The number of replications taken at each solution must be greater than 0."
            )

    def check_sensitivity(self) -> None:
        if self.factors["sensitivity"] <= 0:
            raise ValueError("sensitivity must be greater than 0.")

    def check_delta_threshold(self) -> None:
        if self.factors["delta_threshold"] <= 0:
            raise ValueError("delta_threshold must be greater than 0.")

    def check_delta_t(self) -> None:
        if self.factors["delta_T"] <= self.factors["delta_threshold"]:
            raise ValueError("delta_T must be greater than delta_threshold")

    def check_eta_0(self) -> None:
        if self.factors["eta_0"] <= 0 or self.factors["eta_0"] >= 1:
            raise ValueError("eta_0 must be between 0 and 1.")

    def check_eta_1(self) -> None:
        if (
            self.factors["eta_1"] >= 1
            or self.factors["eta_1"] <= self.factors["eta_0"]
        ):
            raise ValueError("eta_1 must be between eta_0 and 1.")

    def check_gamma_1(self) -> None:
        if self.factors["gamma_1"] <= 0 or self.factors["gamma_1"] >= 1:
            raise ValueError("gamma_1 must be between 0 and 1.")

    def check_gamma_2(self) -> None:
        if self.factors["gamma_2"] <= 1:
            raise ValueError("gamma_2 must be greater than 1.")

    def check_lambda(self) -> None:
        if self.factors["lambda"] <= 1:
            raise ValueError("lambda must be greater than 1.")

    def check_lambda_2(self) -> None:
        # TODO: Check if this is the correct condition.
        if self.factors["lambda_2"] <= 1:
            raise ValueError("lambda_2 must be greater than 1.")

    def solve(self, problem: Problem) -> tuple[list[Solution], list[int]]:
        """
        Run a single macroreplication of a solver on a problem.

        Arguments
        ---------
        problem : Problem object
            simulation-optimization problem to solve
        crn_across_solns : bool
            indicates if CRN are used when simulating different solutions

        Returns
        -------
        recommended_solns : list of Solution objects
            list of solutions recommended throughout the budget
        intermediate_budgets : list of ints
            list of intermediate budgets when recommended solutions changes
        """
        recommended_solns = []
        intermediate_budgets = []
        expended_budget = 0

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
        problem.simulate(new_solution, n0)
        expended_budget += n0
        best_solution = new_solution
        recommended_solns.append(new_solution)
        intermediate_budgets.append(expended_budget)

        while expended_budget < problem.factors["budget"]:
            new_x = new_solution.x
            # Check variable bounds.
            forward = np.isclose(
                new_x, lower_bound, atol=self.factors["sensitivity"]
            ).astype(int)
            backward = np.isclose(
                new_x, upper_bound, atol=self.factors["sensitivity"]
            ).astype(int)
            # bounds_check: 1 stands for forward, -1 stands for backward, 0 means central diff.
            bounds_check = np.subtract(forward, backward)

            # Stage I.
            if delta_t > delta_threshold:
                # Step 1: Build the linear model.
                num_evals = 2 * problem.dim - np.sum(bounds_check != 0)
                grad, hessian = self.finite_diff(
                    new_solution, bounds_check, 1, problem, n_r
                )
                expended_budget += num_evals * n_r
                # A while loop to prevent zero gradient
                while norm(grad) == 0:
                    if expended_budget > problem.factors["budget"]:
                        break
                    grad, hessian = self.finite_diff(
                        new_solution, bounds_check, 1, problem, n_r
                    )
                    expended_budget += num_evals * n_r
                    # Update n_r and counter after each loop.
                    n_r = int(lam * n_r)

                # Step 2: Solve the subproblem.
                # Cauchy reduction.
                candidate_x = self.cauchy_point(grad, hessian, new_x, problem)
                candidate_solution = self.create_new_solution(
                    tuple(candidate_x), problem
                )

                # Step 3: Compute the ratio.
                # Use n_r simulated observations to estimate g_new.
                problem.simulate(candidate_solution, n_r)
                expended_budget += n_r
                # Find the old objective value and the new objective value.
                g_old = -1 * problem.minmax[0] * new_solution.objectives_mean
                g_new = (
                    -1 * problem.minmax[0] * candidate_solution.objectives_mean
                )
                g_diff = g_old - g_new
                # Construct the polynomial.
                r_old = g_old
                r_new = (
                    g_old
                    + np.matmul(np.subtract(candidate_x, new_x), grad)
                    + 0.5
                    * np.matmul(
                        np.matmul(np.subtract(candidate_x, new_x), hessian),
                        np.subtract(candidate_x, new_x),
                    )
                )
                r_diff = r_old - r_new
                if r_diff == 0:
                    print(
                        "Warning: Division by zero in STRONG solver (r_diff == 0 (Step I_3))",
                        file=sys.stderr,
                    )
                    # Follow IEEE 754 standard.
                    if g_diff < 0:
                        rho = -np.inf
                    elif g_diff > 0:
                        rho = np.inf
                    else:
                        rho = np.nan
                else:
                    rho = g_diff / r_diff

                # Step 4: Update the trust region size and determine to accept or reject the solution.
                if (rho < eta_0) | (g_diff <= 0) | (r_diff <= 0):
                    # The solution fails either the RC or SR test, the center point reamins and the trust region shrinks.
                    delta_t = gamma_1 * delta_t
                elif (eta_0 <= rho) & (rho < eta_1):
                    # The center point moves to the new solution and the trust region remains.
                    new_solution = candidate_solution
                    # Update incumbent best solution.
                    if (
                        problem.minmax * new_solution.objectives_mean
                        > problem.minmax * best_solution.objectives_mean
                    ):
                        best_solution = new_solution
                        recommended_solns.append(new_solution)
                        intermediate_budgets.append(expended_budget)
                else:
                    # The center point moves to the new solution and the trust region enlarges.
                    delta_t = gamma_2 * delta_t
                    new_solution = candidate_solution
                    # Update incumbent best solution.
                    if (
                        problem.minmax * new_solution.objectives_mean
                        > problem.minmax * best_solution.objectives_mean
                    ):
                        best_solution = new_solution
                        recommended_solns.append(new_solution)
                        intermediate_budgets.append(expended_budget)
                n_r = int(np.ceil(self.factors["lambda_2"] * n_r))

            # Stage II.
            # When trust region size is very small, use the quadratic design.
            else:
                n_onbound = np.sum(bounds_check != 0)
                if n_onbound <= 1:
                    num_evals = problem.dim**2
                else:
                    # TODO: Check the formula, it seems to be dividing an
                    # integer by a tuple.
                    num_evals = (
                        problem.dim**2
                        + problem.dim
                        - math.factorial(n_onbound)
                        / (math.factorial(2), math.factorial(n_onbound - 2))
                    )
                # Step 1: Build the quadratic model.
                grad, hessian = self.finite_diff(
                    new_solution, bounds_check, 2, problem, n_r
                )
                expended_budget += num_evals * n_r
                # A while loop to prevent zero gradient
                while norm(grad) == 0:
                    if expended_budget > problem.factors["budget"]:
                        break
                    grad, hessian = self.finite_diff(
                        new_solution, bounds_check, 2, problem, n_r
                    )
                    expended_budget += num_evals * n_r
                    # Update n_r and counter after each loop.
                    n_r = int(lam * n_r)
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
                expended_budget += n_r
                # Find the old objective value and the new objective value.
                g_old = -1 * problem.minmax[0] * new_solution.objectives_mean
                g_new = (
                    -1 * problem.minmax[0] * candidate_solution.objectives_mean
                )
                g_diff = g_old - g_new
                # Construct the polynomial.
                r_old = g_old
                r_new = (
                    g_old
                    + np.matmul(np.subtract(candidate_x, new_x), grad)
                    + 0.5
                    * np.matmul(
                        np.matmul(np.subtract(candidate_x, new_x), hessian),
                        np.subtract(candidate_x, new_x),
                    )
                )
                r_diff = r_old - r_new
                if r_diff == 0:
                    print(
                        "Warning: Division by zero in STRONG solver (r_diff == 0 (Step II_3))",
                        file=sys.stderr,
                    )
                    rho = 0
                else:
                    rho = g_diff / r_diff
                # Step 4: Update the trust region size and determine to accept or reject the solution.
                if (rho < eta_0) | (g_diff <= 0) | (r_diff <= 0):
                    # Inner Loop.
                    rr_old = r_old
                    g_b_old = rr_old
                    sub_counter = 1
                    result_solution = new_solution
                    result_x = new_x

                    while np.sum(result_x != new_x) == 0:
                        if expended_budget > problem.factors["budget"]:
                            break
                        # Step1: Build the quadratic model.
                        g_var, h_var = self.finite_diff(
                            new_solution,
                            bounds_check,
                            2,
                            problem,
                            (sub_counter + 1) * n_r,
                        )
                        expended_budget += num_evals * (sub_counter + 1) * n_r
                        # A while loop to prevent zero gradient
                        while norm(g_var) == 0:
                            if expended_budget > problem.factors["budget"]:
                                break
                            g_var, h_var = self.finite_diff(
                                new_solution,
                                bounds_check,
                                2,
                                problem,
                                (sub_counter + 1) * n_r,
                            )
                            expended_budget += (
                                num_evals * (sub_counter + 1) * n_r
                            )
                            # Update n_r and counter after each loop.
                            n_r = int(lam * n_r)

                        # Step 2: determine the new inner solution based on the accumulated design matrix X.
                        try_x = self.cauchy_point(g_var, h_var, new_x, problem)
                        try_solution = self.create_new_solution(
                            tuple(try_x), problem
                        )

                        # Step 3.
                        problem.simulate(
                            try_solution,
                            int(
                                n_r
                                + np.ceil(
                                    sub_counter ** self.factors["lambda_2"]
                                )
                            ),
                        )
                        expended_budget += int(
                            n_r
                            + np.ceil(sub_counter ** self.factors["lambda_2"])
                        )
                        g_b_new = (
                            -1
                            * problem.minmax[0]
                            * try_solution.objectives_mean
                        )
                        dummy_solution = new_solution
                        problem.simulate(
                            dummy_solution,
                            int(
                                np.ceil(sub_counter ** self.factors["lambda_2"])
                                - np.ceil(
                                    (sub_counter - 1)
                                    ** self.factors["lambda_2"]
                                )
                            ),
                        )
                        expended_budget += int(
                            np.ceil(sub_counter ** self.factors["lambda_2"])
                            - np.ceil(
                                (sub_counter - 1) ** self.factors["lambda_2"]
                            )
                        )
                        dummy = (
                            -1
                            * problem.minmax[0]
                            * dummy_solution.objectives_mean
                        )
                        # Update g_old.
                        g_b_old = (
                            g_b_old
                            * (
                                n_r
                                + np.ceil(
                                    (sub_counter - 1)
                                    ** self.factors["lambda_2"]
                                )
                            )
                            + (
                                np.ceil(sub_counter ** self.factors["lambda_2"])
                                - np.ceil(
                                    (sub_counter - 1)
                                    ** self.factors["lambda_2"]
                                )
                            )
                            * dummy
                        ) / (
                            n_r
                            + np.ceil(sub_counter ** self.factors["lambda_2"])
                        )
                        rr_new = (
                            g_b_old
                            + np.matmul(np.subtract(try_x, new_x), g_var)
                            + 0.5
                            * np.matmul(
                                np.matmul(np.subtract(try_x, new_x), h_var),
                                np.subtract(try_x, new_x),
                            )
                        )
                        rr_old = g_b_old
                        # Set rho to the ratio.
                        g_b_diff = g_b_old - g_b_new
                        rr_diff = rr_old - rr_new
                        if rr_diff == 0:
                            print(
                                "Warning: Division by zero in STRONG solver (rr_diff == 0)",
                                file=sys.stderr,
                            )
                            rrho = 0
                        else:
                            rrho = (g_b_diff) / (rr_diff)

                        if (
                            (rrho < eta_0)
                            | ((g_b_diff) <= 0)
                            | ((rr_diff) <= 0)
                        ):
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
                        sub_counter = sub_counter + 1
                    new_solution = result_solution
                    # Update incumbent best solution.
                    if (
                        problem.minmax * new_solution.objectives_mean
                        > problem.minmax * best_solution.objectives_mean
                    ):
                        best_solution = new_solution
                        recommended_solns.append(new_solution)
                        intermediate_budgets.append(expended_budget)
                elif (eta_0 <= rho) and (rho < eta_1):
                    # The center point moves to the new solution and the trust region remains.
                    new_solution = candidate_solution
                    # Update incumbent best solution.
                    if (
                        problem.minmax * new_solution.objectives_mean
                        > problem.minmax * best_solution.objectives_mean
                    ):
                        best_solution = new_solution
                        recommended_solns.append(new_solution)
                        intermediate_budgets.append(expended_budget)
                else:
                    # The center point moves to the new solution and the trust region enlarges.
                    delta_t = gamma_2 * delta_t
                    new_solution = candidate_solution
                    # Update incumbent best solution.
                    if (
                        problem.minmax * new_solution.objectives_mean
                        > problem.minmax * best_solution.objectives_mean
                    ):
                        best_solution = new_solution
                        recommended_solns.append(new_solution)
                        intermediate_budgets.append(expended_budget)
                n_r = int(np.ceil(self.factors["lambda_2"] * n_r))
        # Loop through each budget and convert any numpy int32s to Python ints.
        for i in range(len(intermediate_budgets)):
            intermediate_budgets[i] = int(intermediate_budgets[i])
        return recommended_solns, intermediate_budgets

    def cauchy_point(
        self,
        grad: np.ndarray,
        hessian: np.ndarray,
        new_x: tuple,
        problem: Problem,
    ) -> np.ndarray:
        """
        Find the Cauchy point based on the gradient and Hessian matrix.
        """
        delta_t = self.factors["delta_T"]
        lower_bound = problem.lower_bounds
        upper_bound = problem.upper_bounds
        if np.dot(np.matmul(grad, hessian), grad) <= 0:
            tau = 1
        else:
            tau = min(
                1,
                norm(grad) ** 3
                / (delta_t * np.dot(np.matmul(grad, hessian), grad)),
            )
        grad = np.reshape(grad, (1, problem.dim))[0]
        candidate_x = new_x - tau * delta_t * grad / norm(grad)
        cauchy_x = self.check_cons(candidate_x, new_x, lower_bound, upper_bound)
        return cauchy_x

    def check_cons(
        self,
        candidate_x: tuple,
        new_x: tuple,
        lower_bound: tuple,
        upper_bound: tuple,
    ) -> np.ndarray:
        """
        Check the feasibility of the Cauchy point and update the point accordingly.
        """
        # The current step.
        current_step = np.subtract(candidate_x, new_x)
        # Form a matrix to determine the possible stepsize.
        max_step_matrix = np.ones((2, len(candidate_x)))
        for i in range(0, len(candidate_x)):
            if current_step[i] > 0:
                step_diff = upper_bound[i] - new_x[i]
                max_step_matrix[0, i] = step_diff / current_step[i]
            elif current_step[i] < 0:
                step_diff = lower_bound[i] - new_x[i]
                max_step_matrix[1, i] = step_diff / current_step[i]
        # Find the minimum stepsize.
        t2 = max_step_matrix.min()
        # Calculate the modified x.
        modified_x = new_x + t2 * current_step
        return modified_x

    def finite_diff(
        self,
        new_solution: Solution,
        bounds_check: np.ndarray,
        stage: Literal[1, 2],
        problem: Problem,
        n_r: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Finite difference for calculating gradients and BFGS for calculating Hessian matrix
        """
        delta_t = self.factors["delta_T"]
        lower_bound = problem.lower_bounds
        upper_bound = problem.upper_bounds
        fn = -1 * problem.minmax[0] * new_solution.objectives_mean
        new_x = new_solution.x
        # Store values for each dimension.
        func_diff = np.zeros((problem.dim, 3))
        grad = np.zeros(problem.dim)
        hessian = np.zeros((problem.dim, problem.dim))

        for i in range(problem.dim):
            # Initialization.
            x1 = list(new_x)
            x2 = list(new_x)
            # Forward stepsize.
            steph1 = delta_t
            # Backward stepsize.
            steph2 = delta_t

            # Check variable bounds.
            if x1[i] + steph1 > upper_bound[i]:
                steph1 = np.abs(upper_bound[i] - x1[i])
            if x2[i] - steph2 < lower_bound[i]:
                steph2 = np.abs(x2[i] - lower_bound[i])

            # Decide stepsize.
            # Central diff.
            if bounds_check[i] == 0:
                func_diff[i, 2] = min(steph1, steph2)
                x1[i] = x1[i] + func_diff[i, 2]
                x2[i] = x2[i] - func_diff[i, 2]
            # Forward diff.
            elif bounds_check[i] == 1:
                func_diff[i, 2] = steph1
                x1[i] = x1[i] + func_diff[i, 2]
            # Backward diff
            else:
                func_diff[i, 2] = steph2
                x2[i] = x2[i] - func_diff[i, 2]
            x1_solution = self.create_new_solution(tuple(x1), problem)

            # Run bounds checks.
            if bounds_check[i] != -1:
                problem.simulate_up_to([x1_solution], n_r)
                fn1 = -1 * problem.minmax[0] * x1_solution.objectives_mean
                # First column is f(x+h,y).
                func_diff[i, 0] = fn1[0] if isinstance(fn1, np.ndarray) else fn1
            x2_solution = self.create_new_solution(tuple(x2), problem)
            if bounds_check[i] != 1:
                problem.simulate_up_to([x2_solution], n_r)
                fn2 = -1 * problem.minmax[0] * x2_solution.objectives_mean
                # Second column is f(x-h,y).
                func_diff[i, 1] = fn2[0] if isinstance(fn2, np.ndarray) else fn2

            # Calculate gradient.
            fn_divisor = (
                func_diff[i, 2][0]
                if isinstance(func_diff[i, 2], np.ndarray)
                else func_diff[i, 2]
            )
            if bounds_check[i] == 0:
                fn_diff = fn1 - fn2  # type: ignore
                fn_divisor = 2 * fn_divisor
                if isinstance(fn_diff, np.ndarray):
                    grad[i] = fn_diff[0] / fn_divisor
                else:
                    grad[i] = fn_diff / fn_divisor
            elif bounds_check[i] == 1:
                fn_diff = fn1 - fn  # type: ignore
                if isinstance(fn_diff, np.ndarray):
                    grad[i] = fn_diff[0] / fn_divisor
                else:
                    grad[i] = fn_diff / fn_divisor
            elif bounds_check[i] == -1:
                fn_diff = fn - fn2  # type: ignore
                if isinstance(fn_diff, np.ndarray):
                    grad[i] = fn_diff[0] / fn_divisor
                else:
                    grad[i] = fn_diff / fn_divisor

        if stage == 2:
            # Diagonal in Hessian.
            for i in range(problem.dim):
                fn_1 = (
                    func_diff[i, 1][0]
                    if isinstance(func_diff[i, 1], np.ndarray)
                    else func_diff[i, 1]
                )
                fn_2 = (
                    func_diff[i, 2][0]
                    if isinstance(func_diff[i, 2], np.ndarray)
                    else func_diff[i, 2]
                )
                if bounds_check[i] == 0:
                    fn_0 = (
                        func_diff[i, 0][0]
                        if isinstance(func_diff[i, 0], np.ndarray)
                        else func_diff[i, 0]
                    )
                    hessian[i, i] = (fn_0 - 2 * fn[0] + fn_1) / (fn_2**2)
                elif bounds_check[i] == 1:
                    x3 = list(new_x)
                    x3[i] = x3[i] + func_diff[i, 2] / 2
                    x3_solution = self.create_new_solution(tuple(x3), problem)
                    # Check budget.
                    problem.simulate_up_to([x3_solution], n_r)
                    fn3 = -1 * problem.minmax[0] * x3_solution.objectives_mean
                    hessian[i, i] = 4 * (fn_1 - 2 * fn3[0] + fn[0]) / (fn_2**2)
                elif bounds_check[i] == -1:
                    x4 = list(new_x)
                    x4[i] = x4[i] - func_diff[i, 2] / 2
                    x4_solution = self.create_new_solution(tuple(x4), problem)
                    # Check budget.
                    problem.simulate_up_to([x4_solution], n_r)
                    fn4 = -1 * problem.minmax[0] * x4_solution.objectives_mean
                    hessian[i, i] = 4 * (fn[0] - 2 * fn4[0] + fn_1) / (fn_2**2)

                # Upper triangle in Hessian
                for j in range(i + 1, problem.dim):
                    # Neither x nor y on boundary.
                    if bounds_check[i] ** 2 + bounds_check[j] ** 2 == 0:
                        # Represent f(x+h,y+k).
                        x5 = list(new_x)
                        x5[i] = x5[i] + func_diff[i, 2]
                        x5[j] = x5[j] + func_diff[j, 2]
                        x5_solution = self.create_new_solution(
                            tuple(x5), problem
                        )
                        # Check budget.
                        problem.simulate_up_to([x5_solution], n_r)
                        fn5 = (
                            -1 * problem.minmax[0] * x5_solution.objectives_mean
                        )
                        # Represent f(x-h,y-k).
                        x6 = list(new_x)
                        x6[i] = x6[i] - func_diff[i, 2]
                        x6[j] = x6[j] - func_diff[j, 2]
                        x6_solution = self.create_new_solution(
                            tuple(x5), problem
                        )
                        # Check budget.
                        problem.simulate_up_to([x6_solution], n_r)
                        fn6 = (
                            -1 * problem.minmax[0] * x6_solution.objectives_mean
                        )
                        # Compute second order gradient.
                        fn_i0 = (
                            func_diff[i, 0][0]
                            if isinstance(func_diff[i, 0], np.ndarray)
                            else func_diff[i, 0]
                        )
                        fn_j0 = (
                            func_diff[j, 0][0]
                            if isinstance(func_diff[j, 0], np.ndarray)
                            else func_diff[j, 0]
                        )
                        fn_i1 = (
                            func_diff[i, 1][0]
                            if isinstance(func_diff[i, 1], np.ndarray)
                            else func_diff[i, 1]
                        )
                        fn_j1 = (
                            func_diff[j, 1][0]
                            if isinstance(func_diff[j, 1], np.ndarray)
                            else func_diff[j, 1]
                        )
                        fn_i2 = (
                            func_diff[i, 2][0]
                            if isinstance(func_diff[i, 2], np.ndarray)
                            else func_diff[i, 2]
                        )
                        fn_j2 = (
                            func_diff[j, 2][0]
                            if isinstance(func_diff[j, 2], np.ndarray)
                            else func_diff[j, 2]
                        )
                        hessian[i, j] = (
                            fn5[0]
                            - fn_i0
                            - fn_j0
                            + 2 * fn[0]
                            - fn_i1
                            - fn_j1
                            + fn6[0]
                        ) / (2 * fn_i2 * fn_j2)
                        hessian[j, i] = hessian[i, j]
                    # When x on boundary, y not.
                    elif bounds_check[j] == 0:
                        # Represent f(x+/-h,y+k).
                        x5 = list(new_x)
                        x5[i] = x5[i] + bounds_check[i] * func_diff[i, 2]
                        x5[j] = x5[j] + func_diff[j, 2]
                        x5_solution = self.create_new_solution(
                            tuple(x5), problem
                        )
                        # Check budget.
                        problem.simulate_up_to([x5_solution], n_r)
                        fn5 = (
                            -1 * problem.minmax[0] * x5_solution.objectives_mean
                        )
                        # Represent f(x+/-h,y-k).
                        x6 = list(new_x)
                        x6[i] = x6[i] + bounds_check[i] * func_diff[i, 2]
                        x6[j] = x6[j] - func_diff[j, 2]
                        x6_solution = self.create_new_solution(
                            tuple(x6), problem
                        )
                        # Check budget.
                        problem.simulate_up_to([x6_solution], n_r)
                        fn6 = (
                            -1 * problem.minmax[0] * x6_solution.objectives_mean
                        )
                        # Compute second order gradient.
                        hessian[i, j] = (
                            fn5 - func_diff[j, 0] - fn6 + func_diff[j, 1]
                        ) / (
                            2
                            * func_diff[i, 2]
                            * func_diff[j, 2]
                            * bounds_check[i]
                        )
                        hessian[j, i] = hessian[i, j]
                    # When y on boundary, x not.
                    elif bounds_check[i] == 0:
                        # Represent f(x+h,y+/-k).
                        x5 = list(new_x)
                        x5[i] = x5[i] + func_diff[i, 2]
                        x5[j] = x5[j] + bounds_check[j] * func_diff[j, 2]
                        x5_solution = self.create_new_solution(
                            tuple(x5), problem
                        )
                        # Check budget.
                        problem.simulate_up_to([x5_solution], n_r)
                        fn5 = (
                            -1 * problem.minmax[0] * x5_solution.objectives_mean
                        )
                        # Represent f(x-h,y+/-k).
                        x6 = list(new_x)
                        x6[i] = x6[i] + func_diff[i, 2]
                        x6[j] = x6[j] + bounds_check[j] * func_diff[j, 2]
                        x6_solution = self.create_new_solution(
                            tuple(x6), problem
                        )
                        # Check budget.
                        problem.simulate_up_to([x6_solution], n_r)
                        fn6 = (
                            -1 * problem.minmax[0] * x6_solution.objectives_mean
                        )
                        # Compute second order gradient.
                        hessian[i, j] = (
                            fn5 - func_diff[i, 0] - fn6 + func_diff[i, 1]
                        ) / (
                            2
                            * func_diff[i, 2]
                            * func_diff[j, 2]
                            * bounds_check[j]
                        )
                        hessian[j, i] = hessian[i, j]
                    elif bounds_check[i] == 1:
                        if bounds_check[j] == 1:
                            # Represent f(x+h,y+k).
                            x5 = list(new_x)
                            x5[i] = x5[i] + func_diff[i, 2]
                            x5[j] = x5[j] + func_diff[j, 2]
                            x5_solution = self.create_new_solution(
                                tuple(x5), problem
                            )
                            # Check budget.
                            problem.simulate_up_to([x5_solution], n_r)
                            fn5 = (
                                -1
                                * problem.minmax[0]
                                * x5_solution.objectives_mean
                            )
                            # Compute second order gradient.
                            hessian[i, j] = (
                                fn5 - func_diff[i, 0] - func_diff[j, 0] + fn
                            ) / (func_diff[i, 2] * func_diff[j, 2])
                            hessian[j, i] = hessian[i, j]
                        else:
                            # Represent f(x+h,y-k).
                            x5 = list(new_x)
                            x5[i] = x5[i] + func_diff[i, 2]
                            x5[j] = x5[j] - func_diff[j, 2]
                            x5_solution = self.create_new_solution(
                                tuple(x5), problem
                            )
                            # Check budget.
                            problem.simulate_up_to([x5_solution], n_r)
                            fn5 = (
                                -1
                                * problem.minmax[0]
                                * x5_solution.objectives_mean
                            )
                            # Compute second order gradient.
                            hessian[i, j] = (
                                func_diff[i, 0] - fn5 - fn + func_diff[j, 1]
                            ) / (func_diff[i, 2] * func_diff[j, 2])
                            hessian[j, i] = hessian[i, j]
                    elif bounds_check[i] == -1:
                        if bounds_check[j] == 1:
                            # Represent f(x-h,y+k).
                            x5 = list(new_x)
                            x5[i] = x5[i] - func_diff[i, 2]
                            x5[j] = x5[j] + func_diff[j, 2]
                            x5_solution = self.create_new_solution(
                                tuple(x5), problem
                            )
                            # Check budget
                            problem.simulate_up_to([x5_solution], n_r)
                            fn5 = (
                                -1
                                * problem.minmax[0]
                                * x5_solution.objectives_mean
                            )
                            # Compute second order gradient.
                            hessian[i, j] = (
                                func_diff[j, 0] - fn - fn5 + func_diff[i, 1]
                            ) / (func_diff[i, 2] * func_diff[j, 2])
                            hessian[j, i] = hessian[i, j]
                        else:
                            # Represent f(x-h,y-k).
                            x5 = list(new_x)
                            x5[i] = x5[i] - func_diff[i, 2]
                            x5[j] = x5[j] - func_diff[j, 2]
                            x5_solution = self.create_new_solution(
                                tuple(x5), problem
                            )
                            # Check budget.
                            problem.simulate_up_to([x5_solution], n_r)
                            fn5 = (
                                -1
                                * problem.minmax[0]
                                * x5_solution.objectives_mean
                            )
                            # Compute second order gradient.
                            hessian[i, j] = (
                                fn - func_diff[j, 1] - func_diff[i, 1] + fn5
                            ) / (func_diff[i, 2] * func_diff[j, 2])
                            hessian[j, i] = hessian[i, j]
        return grad, hessian
