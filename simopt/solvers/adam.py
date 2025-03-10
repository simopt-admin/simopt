"""
Summary
-------
ADAM
An algorithm for first-order gradient-based optimization of
stochastic objective functions, based on adaptive estimates of lower-order moments.
A detailed description of the solver can be found `here <https://simopt.readthedocs.io/en/latest/adam.html>`__.
"""

from __future__ import annotations
from simopt.utils import classproperty

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


class ADAM(Solver):
    """
    An algorithm for first-order gradient-based optimization of
    stochastic objective functions, based on adaptive estimates of lower-order moments.

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

    @classproperty
    def objective_type(cls) -> ObjectiveType:
        return ObjectiveType.SINGLE

    @classproperty
    def constraint_type(cls) -> ConstraintType:
        return ConstraintType.BOX

    @classproperty
    def variable_type(cls) -> VariableType:
        return VariableType.CONTINUOUS

    @classproperty
    def gradient_needed(cls) -> bool:
        return False

    @classproperty
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
                "description": "exponential decay of the rate for the first moment estimates",
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
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "crn_across_solns": self.check_crn_across_solns,
            "r": self.check_r,
            "beta_1": self.check_beta_1,
            "beta_2": self.check_beta_2,
            "alpha": self.check_alpha,
            "epsilon": self.check_epsilon,
            "sensitivity": self.check_sensitivity,
        }

    def __init__(
        self, name: str = "ADAM", fixed_factors: dict | None = None
    ) -> None:
        # Let the base class handle default arguments.
        super().__init__(name, fixed_factors)

    def check_r(self) -> None:
        if self.factors["r"] <= 0:
            raise ValueError(
                "The number of replications taken at each solution must be greater than 0."
            )

    def check_beta_1(self) -> None:
        if self.factors["beta_1"] <= 0 or self.factors["beta_1"] >= 1:
            raise ValueError("Beta 1 must be between 0 and 1.")

    def check_beta_2(self) -> None:
        if self.factors["beta_2"] > 0 and self.factors["beta_2"] >= 1:
            raise ValueError("Beta 2 must be less than 1.")

    def check_alpha(self) -> None:
        if self.factors["alpha"] <= 0:
            raise ValueError("Alpha must be greater than 0.")

    def check_epsilon(self) -> None:
        if self.factors["epsilon"] <= 0:
            raise ValueError("Epsilon must be greater than 0.")

    def check_sensitivity(self) -> None:
        if self.factors["sensitivity"] <= 0:
            raise ValueError("Sensitivity must be greater than 0.")

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
        recommended_solns.append(new_solution)
        intermediate_budgets.append(expended_budget)
        problem.simulate(new_solution, r)
        expended_budget += r
        best_solution = new_solution

        # Initialize the first moment vector, the second moment vector, and the timestep.
        m = np.zeros(problem.dim)
        v = np.zeros(problem.dim)
        t = 0

        while expended_budget < problem.factors["budget"]:
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
                grad = (
                    -problem.minmax[0]
                    * new_solution.objectives_gradients_mean[0]
                )
            else:
                # Use finite difference to estimate gradient if IPA gradient is not available.
                grad = self.finite_diff(new_solution, bounds_check, problem)
                expended_budget += (
                    2 * problem.dim - np.count_nonzero(bounds_check)
                ) * r

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
            problem.simulate(new_solution, r)
            expended_budget += r
            if (
                new_solution.objectives_mean > best_solution.objectives_mean
            ) ^ (problem.minmax[0] < 0):
                best_solution = new_solution
                recommended_solns.append(new_solution)
                intermediate_budgets.append(expended_budget)

        return recommended_solns, intermediate_budgets

    # Finite difference for approximating gradients.
    def finite_diff(
        self, new_solution: Solution, bounds_check: np.ndarray, problem: Problem
    ) -> np.ndarray:
        r = self.factors["r"]
        alpha = self.factors["alpha"]
        lower_bound = problem.lower_bounds
        upper_bound = problem.upper_bounds
        fn = -1 * problem.minmax[0] * new_solution.objectives_mean
        new_x = new_solution.x
        # Store values for each dimension.
        function_diff = np.zeros((problem.dim, 3))
        grad = np.zeros(problem.dim)

        for i in range(problem.dim):
            # Initialization.
            x1 = list(new_x)
            x2 = list(new_x)
            # Forward stepsize.
            steph1 = alpha
            # Backward stepsize.
            steph2 = alpha

            # Check variable bounds.
            if x1[i] + steph1 > upper_bound[i]:
                steph1 = np.abs(upper_bound[i] - x1[i])
            if x2[i] - steph2 < lower_bound[i]:
                steph2 = np.abs(x2[i] - lower_bound[i])

            # Decide stepsize.
            # Central diff.
            if bounds_check[i] == 0:
                function_diff[i, 2] = min(steph1, steph2)
                x1[i] += function_diff[i, 2]
                x2[i] -= function_diff[i, 2]
            # Forward diff.
            elif bounds_check[i] == 1:
                function_diff[i, 2] = steph1
                x1[i] += function_diff[i, 2]
            # Backward diff.
            else:
                function_diff[i, 2] = steph2
                x2[i] -= function_diff[i, 2]
            x1_solution = self.create_new_solution(tuple(x1), problem)
            if bounds_check[i] != -1:
                problem.simulate_up_to([x1_solution], r)
                fn1 = -1 * problem.minmax[0] * x1_solution.objectives_mean
                # First column is f(x+h,y).
                function_diff[i, 0] = (
                    fn1[0] if isinstance(fn1, np.ndarray) else fn1
                )
            x2_solution = self.create_new_solution(tuple(x2), problem)
            if bounds_check[i] != 1:
                problem.simulate_up_to([x2_solution], r)
                fn2 = -1 * problem.minmax[0] * x2_solution.objectives_mean
                # Second column is f(x-h,y).
                function_diff[i, 1] = (
                    fn2[0] if isinstance(fn2, np.ndarray) else fn2
                )

            # Calculate gradient.
            fn_divisor = (
                function_diff[i, 2][0]
                if isinstance(function_diff[i, 2], np.ndarray)
                else function_diff[i, 2]
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
        return grad
