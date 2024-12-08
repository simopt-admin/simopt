"""
Summary
-------
Simultaneous perturbation stochastic approximation (SPSA) is an algorithm for optimizing systems with multiple unknown parameters.
"""

from __future__ import annotations

import sys
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


class SPSA(Solver):
    """
    Simultaneous perturbation stochastic approximation (SPSA) is an algorithm for optimizing systems with multiple unknown parameters.

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

    Parameters
    ----------
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
            "alpha": {
                "description": "non-negative coefficient in the SPSA gain sequecence ak",
                "datatype": float,
                "default": 0.602,
            },
            "gamma": {
                "description": "non-negative coefficient in the SPSA gain sequence ck",
                "datatype": float,
                "default": 0.101,
            },
            "step": {
                "description": "initial desired magnitude of change in the theta elements",
                "datatype": float,
                "default": 0.1,
            },
            "gavg": {
                "description": "averaged SP gradients used per iteration",
                "datatype": int,
                "default": 1,
            },
            "n_reps": {
                "description": "number of replications takes at each solution",
                "datatype": int,
                "default": 30,
            },
            "n_loss": {
                "description": "number of loss function evaluations used in this gain calculation",
                "datatype": int,
                "default": 2,
            },
            "eval_pct": {
                "description": "percentage of the expected number of loss evaluations per run",
                "datatype": float,
                "default": 2 / 3,
            },
            "iter_pct": {
                "description": "percentage of the maximum expected number of iterations",
                "datatype": float,
                "default": 0.1,
            },
        }

    @property
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "crn_across_solns": self.check_crn_across_solns,
            "alpha": self.check_alpha,
            "gamma": self.check_gamma,
            "step": self.check_step,
            "gavg": self.check_gavg,
            "n_reps": self.check_n_reps,
            "n_loss": self.check_n_loss,
            "eval_pct": self.check_eval_pct,
            "iter_pct": self.check_iter_pct,
        }

    def __init__(
        self, name: str = "SPSA", fixed_factors: dict | None = None
    ) -> None:
        # Let the base class handle default arguments.
        super().__init__(name, fixed_factors)

    def check_alpha(self) -> None:
        if self.factors["alpha"] <= 0:
            raise ValueError("Alpha must be greater than 0.")

    def check_gamma(self) -> None:
        if self.factors["gamma"] <= 0:
            raise ValueError("Gamma must be greater than 0.")

    def check_step(self) -> None:
        if self.factors["step"] <= 0:
            raise ValueError("Step must be greater than 0.")

    def check_gavg(self) -> None:
        if self.factors["gavg"] <= 0:
            raise ValueError("gavg must be greater than 0.")

    def check_n_reps(self) -> None:
        if self.factors["n_reps"] <= 0:
            raise ValueError(
                "The number of replications taken at each solution must be greater than 0."
            )

    def check_n_loss(self) -> None:
        if self.factors["n_loss"] <= 0:
            raise ValueError("n_loss must be greater than 0.")

    def check_eval_pct(self) -> None:
        if 0 >= self.factors["eval_pct"] or self.factors["eval_pct"] > 1:
            raise ValueError("eval_pct must be between 0 and 1.")

    def check_iter_pct(self) -> None:
        if 0 >= self.factors["iter_pct"] or self.factors["iter_pct"] > 1:
            raise ValueError("iter_pct must be between 0 and 1.")

    def check_problem_factors(self) -> bool:
        # Check divisibility for the for loop.
        return self.factors["n_loss"] % (2 * self.factors["gavg"]) == 0

    def gen_simul_pert_vec(self, dim: int) -> list[int]:
        """
        Generate a new simulatanious pertubation vector with a 50/50 probability
        discrete distribution, with values of -1 and 1. The vector size is the
        problem's dimension. The vector components are independent from each other.

        Parameters
        ----------
        dim : int
            Length of the vector.

        Returns
        -------
        list
            Vector of -1's and 1's.
        """
        prob_list = self.rng_list[2].choices([-1, 1], [0.5, 0.5], k=dim)
        return prob_list

    def solve(self, problem: Problem) -> tuple[list[Solution], list[int]]:
        """
        Run a single macroreplication of a solver on a problem.

        Parameters
        ----------
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
        # problem.minmax = [int(i) for i in problem.minmax]
        # Start at initial solution and record as best.
        theta = problem.factors["initial_solution"]
        theta_sol = self.create_new_solution(tuple(theta), problem)
        recommended_solns.append(theta_sol)
        intermediate_budgets.append(expended_budget)
        # Simulate initial solution.
        problem.simulate(theta_sol, self.factors["n_reps"])
        expended_budget = self.factors["n_reps"]
        # Determine initial value for the parameters c, a, and A (Aalg) (according to Section III.B of Spall (1998)).
        if len(theta_sol.objectives_var) == 1:
            # There's only one index, so we can safely index it.
            c_calculated = np.sqrt(
                theta_sol.objectives_var[0] / self.factors["gavg"]
            )
            c = float(max(c_calculated, 0.0001))
        else:
            # Raise the error to the user
            with np.errstate(all="raise"):
                c_calculated = np.sqrt(
                    max(theta_sol.objectives_var) / self.factors["gavg"]
                )
                c = float(max(c_calculated, 0.0001))
        # Calculating the maximum expected number of loss evaluations per run.
        num_evals = round(
            (problem.factors["budget"] / self.factors["n_reps"])
            * self.factors["eval_pct"]
        )
        aalg = self.factors["iter_pct"] * num_evals / (2 * self.factors["gavg"])
        gbar = np.zeros((1, problem.dim))
        for _ in range(
            int(self.factors["n_loss"] / (2 * self.factors["gavg"]))
        ):
            ghat = np.zeros((1, problem.dim))
            for _ in range(self.factors["gavg"]):
                # Generate a random random direction (delta).
                delta = self.gen_simul_pert_vec(problem.dim)
                # Determine points forward/backward relative to random direction.
                thetaplus = np.add(theta, np.dot(c, delta))
                thetaminus = np.subtract(theta, np.dot(c, delta))
                thetaplus, step_weight_plus = check_cons(
                    thetaplus, theta, problem.lower_bounds, problem.upper_bounds
                )
                thetaminus, step_weight_minus = check_cons(
                    thetaminus,
                    theta,
                    problem.lower_bounds,
                    problem.upper_bounds,
                )
                thetaplus_sol = self.create_new_solution(
                    tuple(thetaplus), problem
                )
                thetaminus_sol = self.create_new_solution(
                    tuple(thetaminus), problem
                )
                # Evaluate two points and update budget spent.
                problem.simulate(thetaplus_sol, self.factors["n_reps"])
                problem.simulate(thetaminus_sol, self.factors["n_reps"])
                expended_budget += 2 * self.factors["n_reps"]
                # Estimate gradient.
                # (-minmax is needed to cast this as a minimization problem,
                # but is not essential here because of the absolute value taken.)
                ghat += np.dot(-1, problem.minmax) * np.divide(
                    (
                        thetaplus_sol.objectives_mean
                        - thetaminus_sol.objectives_mean
                    )
                    / ((step_weight_plus + step_weight_minus) * c),
                    delta,
                )
            gbar += np.abs(np.divide(ghat, self.factors["gavg"]))
        meangbar = np.mean(gbar) / (
            self.factors["n_loss"] / (2 * self.factors["gavg"])
        )

        a_leftside = self.factors["step"] * (
            (aalg + 1) ** self.factors["alpha"]
        )
        if meangbar == 0:
            print(
                "Warning: Division by zero in SPSA solver (meangbar == 0)",
                file=sys.stderr,
            )
            # Follow IEEE 754 standard.
            if a_leftside < 0:
                a = -np.inf
            elif a_leftside > 0:
                a = np.inf
            else:
                a = np.nan
        else:
            a = a_leftside / meangbar
        # Run the main algorithm.
        # Initiate iteration counter.
        k = 0
        ftheta_best = None
        while expended_budget < problem.factors["budget"]:
            k += 1
            # Calculate the gain sequences ak and ck.
            ak = a / (k + aalg) ** self.factors["alpha"]
            ck = c / (k ** self.factors["gamma"])
            # Generate random direction (delta).
            delta = self.gen_simul_pert_vec(problem.dim)
            # Determine points forward/backward relative to random direction.
            thetaplus = np.add(theta, np.dot(ck, delta))
            thetaminus = np.subtract(theta, np.dot(ck, delta))
            thetaplus, step_weight_plus = check_cons(
                thetaplus, theta, problem.lower_bounds, problem.upper_bounds
            )
            thetaminus, step_weight_minus = check_cons(
                thetaminus, theta, problem.lower_bounds, problem.upper_bounds
            )
            thetaplus_sol = self.create_new_solution(tuple(thetaplus), problem)
            thetaminus_sol = self.create_new_solution(
                tuple(thetaminus), problem
            )
            # Evaluate two points and update budget spent.
            problem.simulate(thetaplus_sol, self.factors["n_reps"])
            problem.simulate(thetaminus_sol, self.factors["n_reps"])
            expended_budget += 2 * self.factors["n_reps"]
            # Estimate current solution's objective funtion value by weighted average.
            mean_minus = thetaplus_sol.objectives_mean * step_weight_minus
            mean_plus = thetaminus_sol.objectives_mean * step_weight_plus
            mean_net = mean_minus + mean_plus
            net_step_weight = step_weight_plus + step_weight_minus
            if net_step_weight == 0:
                print(
                    "Warning: Division by zero in SPSA solver (step_weight_minus = step_weight_plus)",
                    file=sys.stderr,
                )
                # Follow IEEE 754 standard.
                if mean_net < 0:
                    ftheta = -np.inf
                elif mean_net > 0:
                    ftheta = np.inf
                else:
                    ftheta = np.nan
            else:
                ftheta = mean_net / net_step_weight
            # If on the first iteration, record the initial solution as best estimated objective.
            if k == 1:
                ftheta_best = ftheta
            # Check if new solution is better than the best recorded and update accordingly.
            if (
                ftheta_best is not None
                and np.dot(-1, problem.minmax) * ftheta
                < np.dot(-1, problem.minmax) * ftheta_best
            ):
                ftheta_best = ftheta
                # Record data from the new best solution.
                recommended_solns.append(theta_sol)
                intermediate_budgets.append(expended_budget)
            # Estimate gradient. (-minmax is needed to cast this as a minimization problem.)
            ghat = np.dot(-1, problem.minmax) * np.divide(
                (thetaplus_sol.objectives_mean - thetaminus_sol.objectives_mean)
                / ((step_weight_plus + step_weight_minus) * c),
                delta,
            )
            # Take step and check feasibility.
            theta_next = np.subtract(theta, np.dot(ak, ghat))
            theta, _ = check_cons(
                theta_next, theta, problem.lower_bounds, problem.upper_bounds
            )
            theta_sol = self.create_new_solution(tuple(theta), problem)
        return recommended_solns, intermediate_budgets


def check_cons(
    candidate_x: np.ndarray,
    new_x: np.ndarray,
    lower_bound: tuple,
    upper_bound: tuple,
) -> tuple[np.ndarray, float]:
    """Evaluates the distance from the new vector (candiate_x) compared to the current vector (new_x) respecting the vector's boundaries of feasibility.
    Returns the evaluated vector (modified_x) and the weight (t2 - how much of a full step took) of the new vector.
    The weight (t2) is used to calculate the weigthed average in the ftheta calculation."""
    # The current step.
    current_step = np.subtract(candidate_x, new_x)
    # Form a matrix to determine the possible stepsize.
    step_size_matrix = np.ones((2, len(candidate_x)))
    for i in range(0, len(candidate_x)):
        if current_step[i] > 0:
            diff = upper_bound[i] - new_x[i]
            if current_step[i] == np.inf:
                print(
                    "Warning: Division by +inf in SPSA solver", file=sys.stderr
                )
                # IEEE 754 standard.
                step_size_matrix[0, i] = 0
            else:
                step_size_matrix[0, i] = diff / current_step[i]
        elif current_step[i] < 0:
            diff = lower_bound[i] - new_x[i]
            if current_step[i] == -np.inf:
                print(
                    "Warning: Division by -inf in SPSA solver", file=sys.stderr
                )
                # IEEE 754 standard.
                step_size_matrix[1, i] = 0
            else:
                step_size_matrix[1, i] = diff / current_step[i]
    # Find the minimum stepsize.
    min_step_size = step_size_matrix.min()
    # Calculate the modified x.
    if min_step_size == 0:
        # If t2 is 0, then there shouldn't be any change.
        modified_x = new_x
    else:
        modified_x = new_x + min_step_size * current_step
    return modified_x, min_step_size
