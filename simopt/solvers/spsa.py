"""Simultaneous Perturbation Stochastic Approximation (SPSA) Solver.

Simultaneous perturbation stochastic approximation (SPSA) is an algorithm for
optimizing systems with multiple unknown parameters.
"""

from __future__ import annotations

from typing import Annotated, ClassVar, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, model_validator

from simopt.base import (
    ConstraintType,
    ObjectiveType,
    Problem,
    Solver,
    SolverConfig,
    VariableType,
)
from simopt.utils import make_nonzero


class SPSAConfig(SolverConfig):
    """Configuration for SPSA solver."""

    alpha: Annotated[
        float,
        Field(
            default=0.602,
            gt=0,
            description="non-negative coefficient in the SPSA gain sequence ak",
        ),
    ]
    gamma: Annotated[
        float,
        Field(
            default=0.101,
            gt=0,
            description="non-negative coefficient in the SPSA gain sequence ck",
        ),
    ]
    step: Annotated[
        float,
        Field(
            default=0.1,
            gt=0,
            description="initial desired magnitude of change in the theta elements",
        ),
    ]
    gavg: Annotated[
        int,
        Field(default=1, gt=0, description="averaged SP gradients used per iteration"),
    ]
    n_reps: Annotated[
        int,
        Field(
            default=30,
            gt=0,
            description="number of replications takes at each solution",
        ),
    ]
    n_loss: Annotated[
        int,
        Field(
            default=2,
            gt=0,
            description="number of loss function evaluations used in gain calculation",
        ),
    ]
    eval_pct: Annotated[
        float,
        Field(
            default=2 / 3,
            gt=0,
            le=1,
            description="percentage of the expected number of loss evaluations per run",
        ),
    ]
    iter_pct: Annotated[
        float,
        Field(
            default=0.1,
            gt=0,
            le=1,
            description="percentage of the maximum expected number of iterations",
        ),
    ]

    @model_validator(mode="after")
    def _validate_n_loss_and_gavg(self) -> Self:
        if self.n_loss % (2 * self.gavg) != 0:
            raise ValueError("n_loss must be a multiple of 2 * gavg.")
        return self


class SPSA(Solver):
    """Simultaneous Perturbation Stochastic Approximation (SPSA) Solver.

    Simultaneous perturbation stochastic approximation (SPSA) is an algorithm for
    optimizing systems with multiple unknown parameters.
    """

    name: str = "SPSA"
    config_class: ClassVar[type[SolverConfig]] = SPSAConfig
    class_name_abbr: ClassVar[str] = "SPSA"
    class_name: ClassVar[str] = "SPSA"
    objective_type: ClassVar[ObjectiveType] = ObjectiveType.SINGLE
    constraint_type: ClassVar[ConstraintType] = ConstraintType.BOX
    variable_type: ClassVar[VariableType] = VariableType.CONTINUOUS
    gradient_needed: ClassVar[bool] = False

    def check_problem_factors(self) -> bool:
        """Determine if the joint settings of problem factors are permissible.

        Returns:
            bool: True if problem factors are permissible; False otherwise.
        """
        # Check divisibility for the for loop.
        return self.factors["n_loss"] % (2 * self.factors["gavg"]) == 0

    def _gen_simul_pert_vec(self, dim: int) -> NDArray[np.int_]:
        """Generate a random perturbation vector.

        Generate a new simulatanious pertubation vector with a 50/50 probability
        discrete distribution, with values of -1 and 1. The vector size is the
        problem's dimension. The vector components are independent from each other.

        Args:
            dim (int): The length of the vector.

        Returns:
            NDArray[np.int_]: A random vector of -1's and 1's.
        """
        return np.array(self.rng_list[2].choices([-1, 1], [0.5, 0.5], k=dim))

    def solve(self, problem: Problem) -> None:  # noqa: D102
        # -minmax is needed to cast this as a minimization problem
        neg_minmax = -np.array(problem.minmax)
        lower_bound = np.array(problem.lower_bounds)
        upper_bound = np.array(problem.upper_bounds)

        # Start at initial solution and record as best.
        theta = problem.factors["initial_solution"]
        theta_sol = self.create_new_solution(tuple(theta), problem)
        self.recommended_solns.append(theta_sol)
        self.intermediate_budgets.append(self.budget.used)

        # Simulate initial solution.
        self.budget.request(self.factors["n_reps"])
        problem.simulate(theta_sol, self.factors["n_reps"])

        # Determine initial value for the parameters c, a, and A (Aalg)
        # (according to Section III.B of Spall (1998)).
        objective_var = max(theta_sol.objectives_var)
        c: float = max(np.sqrt(objective_var / self.factors["gavg"]), 1e-4)

        # Calculating the maximum expected number of loss evaluations per run.
        num_evals = round(
            (self.budget.total / self.factors["n_reps"]) * self.factors["eval_pct"]
        )
        aalg = self.factors["iter_pct"] * num_evals / (2 * self.factors["gavg"])
        gbar = np.zeros((1, problem.dim))

        for _ in range(int(self.factors["n_loss"] / (2 * self.factors["gavg"]))):
            ghat = np.zeros((1, problem.dim))
            for _ in range(self.factors["gavg"]):
                # Generate random direction (delta).
                delta = self._gen_simul_pert_vec(problem.dim)
                c_delta = c * delta
                # Determine points forward/backward relative to random direction.
                theta_forward = theta + c_delta
                theta_backward = theta - c_delta
                theta_forward, step_weight_plus = _check_cons(
                    theta_forward, theta, lower_bound, upper_bound
                )
                theta_backward, step_weight_minus = _check_cons(
                    theta_backward,
                    theta,
                    lower_bound,
                    upper_bound,
                )
                thetaplus_sol = self.create_new_solution(tuple(theta_forward), problem)
                thetaminus_sol = self.create_new_solution(
                    tuple(theta_backward), problem
                )
                # Evaluate two points and update budget spent.
                self.budget.request(2 * self.factors["n_reps"])
                problem.simulate(thetaplus_sol, self.factors["n_reps"])
                problem.simulate(thetaminus_sol, self.factors["n_reps"])
                # Estimate gradient.
                # (-minmax is needed to cast this as a minimization problem,
                # but is not essential here because of the absolute value taken.)
                step_weight_net = step_weight_plus + step_weight_minus
                step_weight_net = make_nonzero(step_weight_net, "net_step_weight")
                theta_mean_diff = (
                    thetaplus_sol.objectives_mean - thetaminus_sol.objectives_mean
                )
                ghat += (neg_minmax * theta_mean_diff) / (step_weight_net * c * delta)
            gbar += np.abs(ghat / self.factors["gavg"])

        a_leftside = self.factors["step"] * ((aalg + 1) ** self.factors["alpha"])
        meangbar = np.mean(gbar) / (self.factors["n_loss"] / (2 * self.factors["gavg"]))
        meangbar = make_nonzero(meangbar, "meangbar")
        a = a_leftside / meangbar
        # Run the main algorithm.
        # Initiate iteration counter.
        k = 0
        best_solution_value: float | None = None
        while True:
            k += 1
            # Calculate the gain sequences ak and ck.
            ak = a / (k + aalg) ** self.factors["alpha"]
            ck = c / (k ** self.factors["gamma"])
            # Generate random direction (delta).
            delta = self._gen_simul_pert_vec(problem.dim)
            ck_delta = ck * delta
            # Determine points forward/backward relative to random direction.
            theta_forward = theta + ck_delta
            theta_backward = theta - ck_delta
            theta_forward, step_weight_plus = _check_cons(
                theta_forward, theta, lower_bound, upper_bound
            )
            theta_backward, step_weight_minus = _check_cons(
                theta_backward, theta, lower_bound, upper_bound
            )
            thetaplus_sol = self.create_new_solution(tuple(theta_forward), problem)
            thetaminus_sol = self.create_new_solution(tuple(theta_backward), problem)
            # Evaluate two points and update budget spent.
            self.budget.request(2 * self.factors["n_reps"])
            problem.simulate(thetaplus_sol, self.factors["n_reps"])
            problem.simulate(thetaminus_sol, self.factors["n_reps"])
            # Estimate current solution's objective funtion value by weighted average.
            mean_minus = thetaplus_sol.objectives_mean * step_weight_minus
            mean_plus = thetaminus_sol.objectives_mean * step_weight_plus
            mean_net = mean_minus + mean_plus
            step_weight_net = step_weight_plus + step_weight_minus
            step_weight_net = make_nonzero(step_weight_net, "net_step_weight")
            solution_value = float((mean_net / step_weight_net) * neg_minmax)
            if best_solution_value is None:
                # Record data from the initial solution.
                best_solution_value = solution_value
            # Check if new solution is better than the best recorded and update
            # accordingly.
            if solution_value < best_solution_value:
                best_solution_value = solution_value
                # Record data from the new best solution.
                self.recommended_solns.append(theta_sol)
                self.intermediate_budgets.append(self.budget.used)
            # Estimate gradient.
            theta_mean_diff = (
                thetaplus_sol.objectives_mean - thetaminus_sol.objectives_mean
            )
            ghat = (neg_minmax * theta_mean_diff * delta) / (step_weight_net * c)
            # Take step and check feasibility.
            theta_next = theta - (ak * ghat)
            theta, _ = _check_cons(theta_next, theta, lower_bound, upper_bound)
            theta_sol = self.create_new_solution(tuple(theta), problem)


def _check_cons(
    candidate_x: np.ndarray,
    new_x: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Checks the feasibility of a new solution.

    Evaluates the distance from the new vector (candiate_x) compared to the current
    vector (new_x) respecting the vector's boundaries of feasibility.
    Returns the evaluated vector (modified_x) and the weight
    (t2 - how much of a full step took) of the new vector.
    The weight (t2) is used to calculate the weigthed average in the ftheta calculation.
    """
    # Compute step direction
    current_step = candidate_x - new_x

    # Initialize minimum step size
    # TODO: figure out if this should be greater than 1
    min_step_size = 1

    # Check positive steps for a minimum
    pos_mask = current_step > 0
    if np.any(pos_mask):
        # Make sure there aren't any infinite steps
        inf_mask = np.isinf(current_step)
        if np.any(inf_mask):
            current_step[inf_mask] = 1e15

        diff = upper_bound - new_x
        step_size = diff[pos_mask] / current_step[pos_mask]
        min_step_size = min(float(min_step_size), float(np.min(step_size)))

    # Check negative steps for a minimum
    neg_mask = current_step < 0
    if np.any(neg_mask):
        diff = lower_bound - new_x
        step_size = diff[neg_mask] / current_step[neg_mask]
        min_step_size = min(float(min_step_size), float(np.min(step_size)))

    # Calculate the modified x.
    modified_x = new_x + min_step_size * current_step
    return modified_x, min_step_size
