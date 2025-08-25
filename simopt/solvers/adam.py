"""First-order gradient-based optimization of stochastic objective functions.

An algorithm for first-order gradient-based optimization of
stochastic objective functions, based on adaptive estimates of lower-order moments.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from simopt.base import (
    ConstraintType,
    ObjectiveType,
    Problem,
    Solution,
    Solver,
    VariableType,
)
from simopt.utils import classproperty, override


class ADAM(Solver):
    """First-order gradient-based optimization of stochastic objective functions.

    An algorithm for first-order gradient-based optimization of
    stochastic objective functions, based on adaptive estimates of lower-order moments.
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
            "r": {
                "description": "number of replications taken at each solution",
                "datatype": int,
                "default": 30,
            },
            "beta_1": {
                "description": (
                    "exponential decay of the rate for the first moment estimates"
                ),
                "datatype": float,
                "default": 0.9,
            },
            "beta_2": {
                "description": "exponential decay rate for the second-moment estimates",
                "datatype": float,
                "default": 0.999,
            },
            "alpha": {
                "description": "step size",
                "datatype": float,
                "default": 0.5,  # Changing the step size matters a lot.
            },
            "epsilon": {
                "description": "a small value to prevent zero-division",
                "datatype": float,
                "default": 10 ** (-8),
            },
            "sensitivity": {
                "description": "shrinking scale for variable bounds",
                "datatype": float,
                "default": 10 ** (-7),
            },
        }

    @property
    @override
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "crn_across_solns": self.check_crn_across_solns,
            "r": self._check_r,
            "beta_1": self._check_beta_1,
            "beta_2": self._check_beta_2,
            "alpha": self._check_alpha,
            "epsilon": self._check_epsilon,
            "sensitivity": self._check_sensitivity,
        }

    def __init__(self, name: str = "ADAM", fixed_factors: dict | None = None) -> None:
        """Initialize the ADAM solver.

        Args:
            name (str, optional): The name of the solver. Defaults to "ADAM".
            fixed_factors (dict, optional): A dictionary of fixed factors.
                Defaults to None.
        """
        # Let the base class handle default arguments.
        super().__init__(name, fixed_factors)

    def _check_r(self) -> None:
        if self.factors["r"] <= 0:
            raise ValueError(
                "The number of replications taken at each solution must be greater "
                "than 0."
            )

    def _check_beta_1(self) -> None:
        if self.factors["beta_1"] <= 0 or self.factors["beta_1"] >= 1:
            raise ValueError("Beta 1 must be between 0 and 1.")

    def _check_beta_2(self) -> None:
        if self.factors["beta_2"] > 0 and self.factors["beta_2"] >= 1:
            raise ValueError("Beta 2 must be less than 1.")

    def _check_alpha(self) -> None:
        if self.factors["alpha"] <= 0:
            raise ValueError("Alpha must be greater than 0.")

    def _check_epsilon(self) -> None:
        if self.factors["epsilon"] <= 0:
            raise ValueError("Epsilon must be greater than 0.")

    def _check_sensitivity(self) -> None:
        if self.factors["sensitivity"] <= 0:
            raise ValueError("Sensitivity must be greater than 0.")

    @override
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
        new_solution = self.create_new_solution(
            problem.factors["initial_solution"], problem
        )
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
            forward = np.isclose(
                new_solution.x, lower_bound, atol=self.factors["sensitivity"]
            ).astype(int)
            backward = np.isclose(
                new_solution.x, upper_bound, atol=self.factors["sensitivity"]
            ).astype(int)
            # 1 stands for forward, -1 stands for backward, 0 means central diff.
            bounds_check = np.subtract(forward, backward)
            if problem.gradient_available:
                # Use IPA gradient if available.
                grad = -problem.minmax[0] * new_solution.objectives_gradients_mean[0]
            else:
                # Use finite difference to estimate gradient if IPA gradient is
                # not available.
                finite_diff_budget = (
                    2 * problem.dim - np.count_nonzero(bounds_check)
                ) * r
                self.budget.request(finite_diff_budget)
                grad = self._finite_diff(new_solution, bounds_check, problem)

            # Update biased first moment estimate.
            m = beta_1 * m + (1 - beta_1) * grad
            # Update biased second raw moment estimate.
            v = beta_2 * v + (1 - beta_2) * grad**2
            # Compute bias-corrected first moment estimate.
            mhat = m / (1 - beta_1**t)
            # Compute bias-corrected second raw moment estimate.
            vhat = v / (1 - beta_2**t)
            # Update new_x (vectorized) and apply box constraints
            new_x = new_solution.x - alpha * mhat / (np.sqrt(vhat) + epsilon)
            new_x = np.clip(new_x, lower_bound, upper_bound)

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

    def _finite_diff(
        self,
        new_solution: Solution,
        bounds_check: np.ndarray,
        problem: Problem,
    ) -> np.ndarray:
        """Compute the finite difference approximation of the gradient for a solution.

        Args:
            new_solution (Solution): The current solution to perturb.
            bounds_check (np.ndarray): Array indicating which perturbation method to
                use per dimension.
            problem (Problem): The problem instance providing bounds and function
                evaluations.

        Returns:
            np.ndarray: The approximated gradient of the function at the given solution.
        """
        r = self.factors["r"]
        alpha = self.factors["alpha"]
        lower_bound = problem.lower_bounds
        upper_bound = problem.upper_bounds
        fn = -problem.minmax[0] * new_solution.objectives_mean
        new_x = np.array(new_solution.x, dtype=np.float64)

        function_diff = np.zeros((problem.dim, 3))

        # Compute step sizes
        step_size = np.full(problem.dim, alpha)
        # Compute step sizes for forward and backward differences
        step_forward = np.minimum(step_size, upper_bound - new_x)
        step_backward = np.minimum(step_size, new_x - lower_bound)

        # Create perturbed variables
        x1 = np.repeat(new_x[:, np.newaxis], problem.dim, axis=1)
        x2 = np.repeat(new_x[:, np.newaxis], problem.dim, axis=1)

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
        x1[forward_mask, :] += function_diff[forward_mask, 2][:, np.newaxis]
        x2[backward_mask, :] -= function_diff[backward_mask, 2][:, np.newaxis]

        # TODO: combine this with the version in ALOE. Test results might need
        # regenerated since the ALOE algorithm only makes a subset of solutions.

        # Simulate perturbed solutions per dimension
        for i in range(problem.dim):
            x1_solution = self.create_new_solution(tuple(x1[:, i]), problem)
            x2_solution = self.create_new_solution(tuple(x2[:, i]), problem)
            problem.simulate_up_to([x1_solution, x2_solution], r)

            fn1 = -problem.minmax[0] * x1_solution.objectives_mean
            fn2 = -problem.minmax[0] * x2_solution.objectives_mean

            function_diff[i, 0] = fn1
            function_diff[i, 1] = fn2

        # Compute gradient
        fn_divisor = function_diff[:, 2].copy()  # Extract step sizes
        fn_divisor[central_mask] *= 2  # Double for central difference

        fn_diff = np.zeros(problem.dim)
        if np.any(central_mask):
            fn_diff[central_mask] = function_diff[:, 0] - function_diff[:, 1]
        if np.any(forward_mask):
            fn_diff[forward_mask] = function_diff[forward_mask, 0] - fn
        if np.any(backward_mask):
            fn_diff[backward_mask] = fn - function_diff[backward_mask, 1]

        return fn_diff / fn_divisor
