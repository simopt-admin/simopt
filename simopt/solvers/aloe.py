"""Stochastic line search algorithm with gradient estimation.

The solver is a stochastic line search algorithm  with the gradient estimate recomputed
in each iteration, whether or not a step is accepted. The algorithm includes the
relaxation of the Armijo condition by an additive constant. A detailed description of
the solver can be found `here <https://simopt.readthedocs.io/en/latest/aloe.html>`__.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from simopt.base import (
    ConstraintType,
    ObjectiveType,
    Problem,
    Solver,
    VariableType,
)
from simopt.solvers.utils import finite_diff
from simopt.utils import classproperty, override


class ALOE(Solver):
    """Adaptive Line-search with Oracle Estimations."""

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
            "theta": {
                "description": "constant in the Armijo condition",
                "datatype": float,
                "default": 0.2,
            },
            "gamma": {
                "description": "constant for shrinking the step size",
                "datatype": float,
                "default": 0.8,
            },
            "alpha_max": {
                "description": "maximum step size",
                "datatype": int,
                "default": 10,
            },
            "alpha_0": {
                "description": "initial step size",
                "datatype": int,
                "default": 1,
            },
            # In the paper, this value is estimated for every epoch but a value > 0
            # is justified in practice.
            "epsilon_f": {
                "description": "additive constant in the Armijo condition",
                "datatype": int,
                "default": 1,
            },
            "sensitivity": {
                "description": "shrinking scale for variable bounds",
                "datatype": float,
                "default": 10 ** (-7),
            },
            "lambda": {
                "description": (
                    "magnifying factor for n_r inside the finite difference function"
                ),
                "datatype": int,
                "default": 2,
            },
        }

    @property
    @override
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "crn_across_solns": self.check_crn_across_solns,
            "r": self._check_r,
            "theta": self._check_theta,
            "gamma": self._check_gamma,
            "alpha_max": self._check_alpha_max,
            "alpha_0": self._check_alpha_0,
            "epsilon_f": self._check_epsilon_f,
            "sensitivity": self._check_sensitivity,
            "lambda": self._check_lambda,
        }

    def __init__(self, name: str = "ALOE", fixed_factors: dict | None = None) -> None:
        """Initialize the ALOE solver.

        Args:
            name (str): The name of the solver.
            fixed_factors (dict, optional): Fixed factors for the solver.
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

    def _check_theta(self) -> None:
        if self.factors["theta"] <= 0 or self.factors["theta"] >= 1:
            raise ValueError("Theta must be between 0 and 1.")

    def _check_gamma(self) -> None:
        if self.factors["gamma"] <= 0 or self.factors["gamma"] >= 1:
            raise ValueError("Gamma must be between 0 and 1.")

    def _check_alpha_max(self) -> None:
        if self.factors["alpha_max"] <= 0:
            raise ValueError("The maximum step size must be greater than 0.")

    def _check_alpha_0(self) -> None:
        if self.factors["alpha_0"] <= 0:
            raise ValueError("The initial step size must be greater than 0.")

    def _check_epsilon_f(self) -> None:
        if self.factors["epsilon_f"] <= 0:
            raise ValueError("epsilon_f must be greater than 0.")

    def _check_sensitivity(self) -> None:
        if self.factors["sensitivity"] <= 0:
            raise ValueError("Sensitivity must be greater than 0.")

    def _check_lambda(self) -> None:
        if self.factors["lambda"] <= 0:
            raise ValueError("Lambda must be greater than 0.")

    @override
    def solve(self, problem: Problem) -> None:
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
            ).astype(int)
            backward = np.isclose(
                new_x, upper_bound, atol=self.factors["sensitivity"]
            ).astype(int)
            bounds_check = forward - backward

            if problem.gradient_available:
                grad = -problem.minmax[0] * new_solution.objectives_gradients_mean[0]
            else:
                finite_diff_budget = (
                    2 * problem.dim - np.count_nonzero(bounds_check)
                ) * r
                self.budget.request(finite_diff_budget)
                grad = finite_diff(self, new_solution, bounds_check, problem, alpha, r)

                while np.all(grad == 0):
                    finite_diff_budget = (
                        2 * problem.dim - np.count_nonzero(bounds_check)
                    ) * r
                    self.budget.request(finite_diff_budget)
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
