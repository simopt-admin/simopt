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
    Solution,
    Solver,
    VariableType,
)
from simopt.utils import classproperty


class ALOE(Solver):
    """Adaptive Line-search with Oracle Estimations.

    Attributes:
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

    Arguments:
    ---------
    name : str
        user-specified name for solver
    fixed_factors : dict
        fixed_factors of the solver

    See Also:
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
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "crn_across_solns": self.check_crn_across_solns,
            "r": self.check_r,
            "theta": self.check_theta,
            "gamma": self.check_gamma,
            "alpha_max": self.check_alpha_max,
            "alpha_0": self.check_alpha_0,
            "epsilon_f": self.check_epsilon_f,
            "sensitivity": self.check_sensitivity,
            "lambda": self.check_lambda,
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

    def check_r(self) -> None:
        if self.factors["r"] <= 0:
            raise ValueError(
                "The number of replications taken at each solution must be greater "
                "than 0."
            )

    def check_theta(self) -> None:
        if self.factors["theta"] <= 0 or self.factors["theta"] >= 1:
            raise ValueError("Theta must be between 0 and 1.")

    def check_gamma(self) -> None:
        if self.factors["gamma"] <= 0 or self.factors["gamma"] >= 1:
            raise ValueError("Gamma must be between 0 and 1.")

    def check_alpha_max(self) -> None:
        if self.factors["alpha_max"] <= 0:
            raise ValueError("The maximum step size must be greater than 0.")

    def check_alpha_0(self) -> None:
        if self.factors["alpha_0"] <= 0:
            raise ValueError("The initial step size must be greater than 0.")

    def check_epsilon_f(self) -> None:
        if self.factors["epsilon_f"] <= 0:
            raise ValueError("epsilon_f must be greater than 0.")

    def check_sensitivity(self) -> None:
        if self.factors["sensitivity"] <= 0:
            raise ValueError("Sensitivity must be greater than 0.")

    def check_lambda(self) -> None:
        if self.factors["lambda"] <= 0:
            raise ValueError("Lambda must be greater than 0.")

    def solve(self, problem: Problem) -> tuple[list[Solution], list[int]]:
        """Run a single macroreplication of the ALOE solver on a problem.

        Arguments:
        ---------
        problem : Problem
            The simulation-optimization problem to solve.

        Returns:
        -------
        list[Solution]
            List of solutions recommended throughout the budget.
        list[int]
            List of intermediate budgets when recommended solutions change.
        """
        recommended_solns = []
        intermediate_budgets = []
        expended_budget = 0

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
        recommended_solns.append(new_solution)
        intermediate_budgets.append(expended_budget)
        problem.simulate(new_solution, r)
        expended_budget += r
        best_solution = new_solution

        while expended_budget < problem.factors["budget"]:
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
                grad = self._finite_diff(new_solution, bounds_check, problem, alpha, r)
                expended_budget += (
                    2 * problem.dim - np.count_nonzero(bounds_check)
                ) * r
                while (
                    np.all(grad == 0) and expended_budget <= problem.factors["budget"]
                ):
                    grad = self._finite_diff(
                        new_solution, bounds_check, problem, alpha, r
                    )
                    expended_budget += (
                        2 * problem.dim - np.count_nonzero(bounds_check)
                    ) * r
                    r = int(self.factors["lambda"] * r)  # Update sample size

            # Compute candidate solution and apply box constraints (vectorized).
            candidate_x = np.clip(new_x - alpha * grad, lower_bound, upper_bound)
            candidate_solution = self.create_new_solution(tuple(candidate_x), problem)

            problem.simulate(candidate_solution, r)
            expended_budget += r

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
                recommended_solns.append(new_solution)
                intermediate_budgets.append(expended_budget)

        return recommended_solns, intermediate_budgets

    def _finite_diff(
        self,
        new_solution: Solution,
        bounds_check: np.ndarray,
        problem: Problem,
        stepsize: float,
        r: int,
    ) -> np.ndarray:
        """Compute the finite difference approximation of the gradient for a solution.

        Arguments:
        ---------
        new_solution : Solution
            The current solution to perturb.
        bounds_check : np.ndarray
            Array indicating which perturbation method to use per dimension.
        problem : Problem
            The problem instance providing bounds and function evaluations.
        stepsize : float
            The step size used for finite difference calculations.
        r : int
            The number of replications used for each function evaluation.

        Returns:
        -------
        np.ndarray
            The approximated gradient of the function at the given solution.
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
