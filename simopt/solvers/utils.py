"""Utility functions for optimization solvers."""

from __future__ import annotations

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
