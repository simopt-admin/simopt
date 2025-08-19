"""STRONG Solver.

STRONG: A trust-region-based algorithm that fits first- or second-order models through
function evaluations taken within a neighborhood of the incumbent solution.
A detailed description of the solver can be found
`here <https://simopt.readthedocs.io/en/latest/strong.html>`__.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
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
from simopt.utils import classproperty, make_nonzero, override


class STRONG(Solver):
    """STRONG Solver.

    A trust-region-based algorithm that fits first- or second-order models through
    function evaluations taken within a neighborhood of the incumbent solution.
    """

    @classproperty
    @override
    def objective_type(cls) -> ObjectiveType:
        return ObjectiveType.SINGLE

    @classproperty
    @override
    def constraint_type(cls) -> ConstraintType:
        return ConstraintType.BOX

    @classproperty
    @override
    def variable_type(cls) -> VariableType:
        return VariableType.CONTINUOUS

    @classproperty
    @override
    def gradient_needed(cls) -> bool:
        return False

    @classproperty
    @override
    def specifications(cls) -> dict[str, dict]:
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
                "description": (
                    "magnifying factor for n_r inside the finite difference function"
                ),
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
    @override
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "crn_across_solns": self.check_crn_across_solns,
            "n0": self._check_n0,
            "n_r": self._check_n_r,
            "sensitivity": self._check_sensitivity,
            "delta_threshold": self._check_delta_threshold,
            "delta_T": self._check_delta_t,
            "eta_0": self._check_eta_0,
            "eta_1": self._check_eta_1,
            "gamma_1": self._check_gamma_1,
            "gamma_2": self._check_gamma_2,
            "lambda": self._check_lambda,
            "lambda_2": self._check_lambda_2,
        }

    def __init__(self, name: str = "STRONG", fixed_factors: dict | None = None) -> None:
        """Initialize STRONG solver.

        Args:
            name (str): name of the solver.
            fixed_factors (dict, optional): fixed factors of the solver.
                Defaults to None.
        """
        # Let the base class handle default arguments.
        super().__init__(name, fixed_factors)

    def _check_n0(self) -> None:
        if self.factors["n0"] <= 0:
            raise ValueError("n0 must be greater than 0.")

    def _check_n_r(self) -> None:
        if self.factors["n_r"] <= 0:
            raise ValueError(
                "The number of replications taken at each solution must be greater "
                "than 0."
            )

    def _check_sensitivity(self) -> None:
        if self.factors["sensitivity"] <= 0:
            raise ValueError("sensitivity must be greater than 0.")

    def _check_delta_threshold(self) -> None:
        if self.factors["delta_threshold"] <= 0:
            raise ValueError("delta_threshold must be greater than 0.")

    def _check_delta_t(self) -> None:
        if self.factors["delta_T"] <= self.factors["delta_threshold"]:
            raise ValueError("delta_T must be greater than delta_threshold")

    def _check_eta_0(self) -> None:
        if self.factors["eta_0"] <= 0 or self.factors["eta_0"] >= 1:
            raise ValueError("eta_0 must be between 0 and 1.")

    def _check_eta_1(self) -> None:
        if self.factors["eta_1"] >= 1 or self.factors["eta_1"] <= self.factors["eta_0"]:
            raise ValueError("eta_1 must be between eta_0 and 1.")

    def _check_gamma_1(self) -> None:
        if self.factors["gamma_1"] <= 0 or self.factors["gamma_1"] >= 1:
            raise ValueError("gamma_1 must be between 0 and 1.")

    def _check_gamma_2(self) -> None:
        if self.factors["gamma_2"] <= 1:
            raise ValueError("gamma_2 must be greater than 1.")

    def _check_lambda(self) -> None:
        if self.factors["lambda"] <= 1:
            raise ValueError("lambda must be greater than 1.")

    def _check_lambda_2(self) -> None:
        # TODO: Check if this is the correct condition.
        if self.factors["lambda_2"] <= 1:
            raise ValueError("lambda_2 must be greater than 1.")

    @override
    def solve(self, problem: Problem) -> None:
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
            ).astype(int)
            backward = np.isclose(
                new_x, upper_bound, atol=self.factors["sensitivity"]
            ).astype(int)
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
                    grad, hessian = self.finite_diff(
                        new_solution, bounds_check, 1, problem, n_r
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
                    grad, hessian = self.finite_diff(
                        new_solution, bounds_check, 2, problem, n_r
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
                            g_var, h_var = self.finite_diff(
                                new_solution,
                                bounds_check,
                                2,
                                problem,
                                n_r_loop,
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
        # Loop through each budget and convert any numpy int32s to Python ints.
        self.intermediate_budgets = [int(i) for i in self.intermediate_budgets]

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
        min_step = 1
        pos_mask = current_step > 0
        if np.any(pos_mask):
            step_diff = (upper_bound_arr[pos_mask] - new_x[pos_mask]) / current_step[
                pos_mask
            ]
            min_step = min(min_step, float(np.min(step_diff)))
        neg_mask = current_step < 0
        if np.any(neg_mask):
            step_diff = (lower_bound_arr[neg_mask] - new_x[neg_mask]) / current_step[
                neg_mask
            ]
            min_step = min(min_step, float(np.min(step_diff)))
        # Calculate the modified x.
        return new_x + min_step * current_step

    def finite_diff(
        self,
        new_solution: Solution,
        bounds_check: np.ndarray,
        stage: Literal[1, 2],
        problem: Problem,
        n_r: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate gradients and approximate Hessian using finite differences and BFGS.

        This method uses finite differencing to compute gradients of the objective,
        and applies BFGS updates to build or refine a Hessian approximation.

        Args:
            new_solution (Solution): The solution at which derivatives are computed.
            bounds_check (np.ndarray): Boolean mask indicating which variables are
                within bounds and eligible for perturbation.
            stage (Literal[1, 2]): Indicates the optimization stage
                (e.g., 1 for initial approximation, 2 for refinement).
            problem (Problem): The simulation-optimization problem being solved.
            n_r (int): Number of replications used when estimating gradients.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing:
                - Gradient estimate as a NumPy array.
                - Hessian approximation (updated via BFGS) as a NumPy array.
        """
        neg_minmax = -np.array(problem.minmax)
        fn = (neg_minmax * new_solution.objectives_mean)[0]
        new_x = np.array(new_solution.x, dtype=float)
        # Store values for each dimension.
        f_x_minus_h = np.zeros(problem.dim)  # f(x - h)
        f_x_plus_h = np.zeros(problem.dim)  # f(x + h)

        def get_fn_x(x: Iterable) -> float:
            """Helper to simulate the function at a given x."""
            x_solution = self.create_new_solution(tuple(x), problem)
            problem.simulate_up_to([x_solution], n_r)
            return (neg_minmax * x_solution.objectives_mean)[0]

        # Initialize step sizes.
        delta_t: float = self.factors["delta_T"]
        ub_steps = np.minimum(delta_t, np.array(problem.upper_bounds) - new_x)
        lb_steps = np.minimum(delta_t, new_x - np.array(problem.lower_bounds))

        # Create independent fresh copies for each dimension
        # Tiling creates a 2D array, each row is a copy of new_x
        x1 = np.tile(new_x, (problem.dim, 1))
        x2 = np.tile(new_x, (problem.dim, 1))

        # Compute masks for numpy vectorization
        bounds_neg = bounds_check == -1
        bounds_zero = bounds_check == 0
        bounds_pos = bounds_check == 1
        bounds_non_neg = bounds_zero | bounds_pos
        bounds_non_zero = bounds_neg | bounds_pos
        bounds_non_pos = bounds_zero | bounds_neg

        steps = np.minimum(ub_steps, lb_steps)
        # Apply step modifications per bounds_check conditions
        steps = np.where(bounds_neg, lb_steps, steps)
        steps = np.where(bounds_pos, ub_steps, steps)

        # Modify x1 and x2 based on step sizes
        x1[np.arange(problem.dim), bounds_non_neg] += steps[bounds_non_neg]
        x2[np.arange(problem.dim), bounds_non_pos] -= steps[bounds_non_pos]

        # Compute function values
        non_neg_indices = np.where(bounds_non_neg)[0]
        non_pos_indices = np.where(bounds_non_pos)[0]
        f_x_minus_h[non_neg_indices] = np.array(
            list(map(get_fn_x, x1[non_neg_indices]))
        )
        f_x_plus_h[non_pos_indices] = np.array(list(map(get_fn_x, x2[non_pos_indices])))

        # Compute gradients
        grad = np.zeros(problem.dim)
        grad[bounds_neg] = (fn - f_x_plus_h[bounds_neg]) / steps[bounds_neg]
        grad[bounds_zero] = (f_x_minus_h[bounds_zero] - f_x_plus_h[bounds_zero]) / (
            2 * steps[bounds_zero]
        )
        grad[bounds_pos] = (f_x_minus_h[bounds_pos] - fn) / steps[bounds_pos]

        hessian = np.zeros((problem.dim, problem.dim))
        # If stage 1, exit without calculating the Hessian.
        if stage == 1:
            return grad, hessian

        # Initialize the diagonal of the Hessian matrix.
        hessian_diag = np.zeros(problem.dim)

        # Case where bounds_check[i] == 0 (Central Difference)
        if np.any(bounds_zero):
            hessian_diag[bounds_zero] = (
                f_x_minus_h[bounds_zero] - 2 * fn + f_x_plus_h[bounds_zero]
            ) / (steps[bounds_zero] ** 2)

        # Case where bounds_check[i] != 0 (One-Sided Difference)
        if np.any(bounds_non_zero):
            x = new_x.copy()
            x[bounds_non_zero] += (steps[bounds_non_zero] / 2) * bounds_check[
                bounds_non_zero
            ]  # Apply h shift
            fn_x = np.array(
                [get_fn_x(x) for _ in range(np.sum(bounds_non_zero))]
            )  # Simulate function evaluations
            hessian_diag[bounds_non_zero] = (
                4
                * (fn - 2 * fn_x + f_x_plus_h[bounds_non_zero])
                / (steps[bounds_non_zero] ** 2)
            )

        # Fill the diagonal of the Hessian matrix.
        np.fill_diagonal(hessian, hessian_diag)

        # Fill the upper triangle of the Hessian matrix.
        for i in range(problem.dim):
            f_i_minus_h = f_x_minus_h[i]
            f_i_plus_h = f_x_plus_h[i]
            h = steps[i]  # h step size
            # Upper triangle in Hessian
            for j in range(i + 1, problem.dim):
                f_j_minus_k = f_x_minus_h[j]
                f_j_plus_k = f_x_plus_h[j]
                k = steps[j]  # k step size

                x5 = new_x.copy()
                # Neither x nor y on boundary.
                if bounds_check[i] == 0 and bounds_check[j] == 0:
                    # Represent f(x+h,y+k).
                    x5[i] += h
                    x5[j] += k
                    fn5 = get_fn_x(x5)
                    # Represent f(x-h,y-k).
                    x6 = new_x.copy()
                    x6[i] -= h
                    x6[j] -= k
                    fn6 = get_fn_x(x6)
                    # Compute second order gradient.
                    hessian[i, j] = (
                        2 * fn
                        + (fn5 - f_i_minus_h - f_j_minus_k)
                        + (fn6 - f_i_plus_h - f_j_plus_k)
                    ) / (2 * h * k)
                # When x on boundary, y not.
                elif bounds_check[j] == 0:
                    i_plus_minus = bounds_check[i] * h
                    # Represent f(x+/-h,y+k).
                    x5[i] += i_plus_minus
                    x5[j] += k
                    fn5 = get_fn_x(x5)
                    # Represent f(x+/-h,y-k).
                    x6 = new_x.copy()
                    x6[i] += i_plus_minus
                    x6[j] -= k
                    fn6 = get_fn_x(x6)
                    # Compute second order gradient.
                    hessian[i, j] = (
                        (fn5 - f_j_minus_k - fn6 + f_j_plus_k)
                        / (2 * h * k)
                        * bounds_check[i]
                    )
                # When y on boundary, x not.
                elif bounds_check[i] == 0:
                    k_plus_minus = bounds_check[j] * k
                    # Represent f(x+h,y+/-k).
                    x5[i] += h
                    x5[j] += k_plus_minus
                    fn5 = get_fn_x(x5)
                    # Represent f(x-h,y+/-k).
                    x6 = new_x.copy()
                    x6[i] -= h
                    x6[j] += k_plus_minus
                    fn6 = get_fn_x(x6)
                    # Compute second order gradient.
                    hessian[i, j] = (
                        (fn5 - f_i_minus_h - fn6 + f_i_plus_h)
                        / (2 * h * k)
                        * bounds_check[j]
                    )
                # If only using one side
                else:
                    x5[i] += h * bounds_check[i]
                    x5[j] += k * bounds_check[j]
                    # TODO: verify the i and j mappings are inverted
                    fd_ix = f_i_minus_h if bounds_check[i] == -1 else f_i_plus_h
                    fd_jx = f_j_minus_k if bounds_check[j] == -1 else f_j_plus_k
                    fn5 = get_fn_x(x5)
                    hessian[i, j] = (
                        ((fn + fn5) - (fd_jx + fd_ix)) / (h * k) * bounds_check[j]
                    )
                # Since we're only computing the upper half the matrix, we
                # need to copy the value to the lower triangle.
                hessian[j, i] = hessian[i, j]
        return grad, hessian
