"""Utility functions for optimization solvers."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from simopt.base import (
    Problem,
    Solution,
    Solver,
)


def finite_diff(
    solver: Solver,
    new_solution: Solution,
    bounds_check: np.ndarray,
    problem: Problem,
    stepsize: float,
    r: int,
) -> np.ndarray:
    """Compute the finite difference approximation of the gradient for a solution.

    Args:
        solver (Solver): The solver instance used to create new solutions.
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
    new_x = np.array(new_solution.x, dtype=float)
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
        x1_solution = solver.create_new_solution(tuple(x1[i]), problem)
        problem.simulate_up_to([x1_solution], r)
        fn1 = -problem.minmax[0] * x1_solution.objectives_mean
        function_diff[i, 0] = fn1[0] if isinstance(fn1, np.ndarray) else fn1

    for i in x2_indices:
        x2_solution = solver.create_new_solution(tuple(x2[i]), problem)
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


def bfgs_hessian_approx(
    solver: Solver,
    new_solution: Solution,
    bounds_check: np.ndarray,
    problem: Problem,
    r: int,
) -> np.ndarray:
    """Approximate Hessian using BFGS.

    This method applies BFGS updates to build or refine a Hessian approximation.

    Args:
        solver (Solver): The solver instance used to create new solutions.
        new_solution (Solution): The solution at which derivatives are computed.
        bounds_check (np.ndarray): Boolean mask indicating which variables are
            within bounds and eligible for perturbation.
        problem (Problem): The simulation-optimization problem being solved.
        r (int): Number of replications used when estimating gradients.

    Returns:
        np.ndarray: Hessian approximation (updated via BFGS).
    """
    neg_minmax = -np.array(problem.minmax)
    fn = (neg_minmax * new_solution.objectives_mean)[0]
    new_x = np.array(new_solution.x, dtype=float)

    # Initialize step sizes.
    delta_t: float = solver.factors["delta_T"]
    ub_steps = np.minimum(delta_t, np.array(problem.upper_bounds) - new_x)
    lb_steps = np.minimum(delta_t, new_x - np.array(problem.lower_bounds))

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

    # Create independent fresh copies for each dimension
    # Tiling creates a 2D array, each row is a copy of new_x
    x1 = np.tile(new_x, (problem.dim, 1))
    x2 = np.tile(new_x, (problem.dim, 1))
    # Modify x1 and x2 based on step sizes
    x1[np.arange(problem.dim), bounds_non_neg] += steps[bounds_non_neg]
    x2[np.arange(problem.dim), bounds_non_pos] -= steps[bounds_non_pos]

    def get_fn_x(x: Iterable) -> float:
        """Helper to simulate the function at a given x."""
        x_solution = solver.create_new_solution(tuple(x), problem)
        problem.simulate_up_to([x_solution], r)
        return (neg_minmax * x_solution.objectives_mean)[0]

    # Compute function values
    f_x_minus_h = np.zeros(problem.dim)  # f(x - h)
    non_neg_indices = np.where(bounds_non_neg)[0]
    f_x_minus_h[non_neg_indices] = np.array(list(map(get_fn_x, x1[non_neg_indices])))

    f_x_plus_h = np.zeros(problem.dim)  # f(x + h)
    non_pos_indices = np.where(bounds_non_pos)[0]
    f_x_plus_h[non_pos_indices] = np.array(list(map(get_fn_x, x2[non_pos_indices])))

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
    hessian = np.zeros((problem.dim, problem.dim))
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
    return hessian
