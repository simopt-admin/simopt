"""SPSA solver for simulation-optimization problems.

Simultaneous perturbation stochastic approximation (SPSA) is an algorithm for optimizing systems with multiple unknown parameters.
"""

from __future__ import annotations

import numpy as np
from simopt.base import Solver, Problem, Solution


class SPSA(Solver):
    """Simultaneous perturbation stochastic approximation (SPSA) is an algorithm for optimizing systems with multiple unknown parameters.

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

    See Also
    --------
    base.Solver

    """

    @property
    def objective_type(self) -> str:
        """The description of objective types."""
        return "single"

    @property
    def constraint_type(self) -> str:
        """The description of constraints types."""
        return "box"

    @property
    def variable_type(self) -> str:
        """The description of variable types."""
        return "continuous"

    @property
    def gradient_needed(self) -> bool:
        """If gradient of objective function is needed."""
        return False

    def __init__(
        self, name: str = "SPSA", fixed_factors: dict | None = None
    ) -> None:
        """Initialize SPSA solver with default values for factors."""
        if fixed_factors is None:
            fixed_factors = {}

        self.name = name
        self.specifications = {
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
        self.check_factor_list = {
            "alpha": self.check_alpha,
            "gamma": self.check_gamma,
            "step": self.check_step,
            "gavg": self.check_gavg,
            "n_reps": self.check_n_reps,
            "n_loss": self.check_n_loss,
            "eval_pct": self.check_eval_pct,
            "iter_pct": self.check_iter_pct,
        }
        super().__init__(fixed_factors)

    def check_alpha(self) -> bool:
        """Check if alpha is positive."""
        return self.factors["alpha"] > 0

    def check_gamma(self) -> bool:
        """Check if gamma is positive."""
        return self.factors["gamma"] > 0

    def check_step(self) -> bool:
        """Check if step is positive."""
        return self.factors["step"] > 0

    def check_gavg(self) -> bool:
        """Check if gavg is positive."""
        return self.factors["gavg"] > 0

    def check_n_reps(self) -> bool:
        """Check if n_reps is positive."""
        return self.factors["n_reps"] > 0

    def check_n_loss(self) -> bool:
        """Check if n_loss is positive."""
        return self.factors["n_loss"] > 0

    def check_eval_pct(self) -> bool:
        """Check if eval_pct is between 0 and 1."""
        return 0 < self.factors["eval_pct"] <= 1

    def check_iter_pct(self) -> bool:
        """Check if iter_pct is between 0 and 1."""
        return 0 < self.factors["iter_pct"] <= 1

    def check_problem_factors(self) -> bool:
        """Check if the problem factors are divisible by the for loop."""
        # Check divisibility for the for loop.
        return self.factors["n_loss"] % (2 * self.factors["gavg"]) == 0

    def gen_simul_pert_vec(self, dim: int) -> list[int]:
        """Generate a new simulatanious pertubation vector with a 50/50 probability discrete distribution, with values of -1 and 1. The vector size is the problem's dimension. The vector components are independent from each other.

        Parameters
        ----------
        dim : int
            Length of the vector.

        Returns
        -------
        list[int]
            Vector of -1's and 1's.

        """
        sp_vect = self.rng_list[2].choices([-1, 1], [0.5, 0.5], k=dim)
        return sp_vect

    def solve(self, problem: Problem) -> tuple[list[Solution], list[int]]:
        """Run a single macroreplication of a solver on a problem.

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
        c = float(
            max(
                (theta_sol.objectives_var / self.factors["gavg"]) ** 0.5, 0.0001
            )
        )
        # Calculating the maximum expected number of loss evaluations per run.
        n_evals = round(
            (problem.factors["budget"] / self.factors["n_reps"])
            * self.factors["eval_pct"]
        )
        a_alg = self.factors["iter_pct"] * n_evals / (2 * self.factors["gavg"])
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
        a = (
            self.factors["step"]
            * ((a_alg + 1) ** self.factors["alpha"])
            / meangbar
        )
        # Run the main algorithm.
        # Initiate iteration counter.
        k = 0
        while expended_budget < problem.factors["budget"]:
            k += 1
            # Calculate the gain sequences ak and ck.
            ak = a / (k + a_alg) ** self.factors["alpha"]
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
            ftheta = (
                (thetaplus_sol.objectives_mean * step_weight_minus)
                + (thetaminus_sol.objectives_mean * step_weight_plus)
            ) / (step_weight_plus + step_weight_minus)
            # If on the first iteration, record the initial solution as best estimated objective.
            if k == 1:
                ftheta_best = ftheta
            # Check if new solution is better than the best recorded and update accordingly.
            if (
                np.dot(-1, problem.minmax) * ftheta
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
    candidate_x: np.array,
    new_x: np.array,
    lower_bound: np.array,
    upper_bound: np.array,
) -> tuple[np.array, float]:
    """Evaluate the distance from the new vector (candiate_x) compared to the current vector (new_x) respecting the vector's boundaries of feasibility.

    Returns the evaluated vector (modified_x) and the weight (t2 - how much of a full step took) of the new vector.
    The weight (t2) is used to calculate the weigthed average in the ftheta calculation.

    Parameters
    ----------
    candidate_x : np.array
        candidate vector
    new_x : np.array
        current vector
    lower_bound : np.array
        lower boundary of the vector
    upper_bound : np.array
        upper boundary of the vector

    Returns
    -------
    np.array
        modified vector
    float
        weight of the new vector

    """
    # The current step.
    step_v = np.subtract(candidate_x, new_x)
    # Form a matrix to determine the possible stepsize.
    tmax_v = np.ones((2, len(candidate_x)))
    for i in range(0, len(candidate_x)):
        if step_v[i] > 0:
            tmax_v[0, i] = (upper_bound[i] - new_x[i]) / step_v[i]
        elif step_v[i] < 0:
            tmax_v[1, i] = (lower_bound[i] - new_x[i]) / step_v[i]
    # Find the minimum stepsize.
    t2 = tmax_v.min()
    # Calculate the modified x.
    modified_x = new_x + t2 * step_v
    return modified_x, t2
