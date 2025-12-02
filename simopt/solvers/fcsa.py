"""Fully Cooperative Stochastic Approximation (FCSA) solver.

This solver is based on the paper 'Diagnostic Tools for Evaluating Solvers for
Stochastically Constrained Simulation Optimization Problems' by Felice, N.,
D. J. Eckman, S. G. Henderson, and S. Shashaani
"""

from typing import Annotated, ClassVar

import cvxpy as cp
import numpy as np
from pydantic import Field

from simopt.base import (
    ConstraintType,
    ObjectiveType,
    Problem,
    Solution,
    Solver,
    SolverConfig,
    VariableType,
)

EPSILON = np.finfo(float).eps


class FCSAConfig(SolverConfig):
    """Configuration for the FCSA solver."""

    r: Annotated[
        int,
        Field(
            default=30,
            gt=0,
            description="number of replications taken at each solution",
        ),
    ]
    h: Annotated[
        float,
        Field(
            default=0.1,
            gt=0,
            description="difference in finite difference gradient",
        ),
    ]
    step_type: Annotated[
        # TODO: change back when the old GUI is removed
        # Literal["const", "decay"],
        str,
        Field(default="const", description="constant or decaying step size?"),
    ]
    step_mult: Annotated[
        float,
        Field(
            default=0.1,
            gt=0,
            description="value of const step size or multiplier of k for decay",
        ),
    ]
    tolerance: Annotated[
        float,
        Field(default=0.01, ge=0, description="tolerance function"),
    ]
    search_direction: Annotated[
        # TODO: change back when the old GUI is removed
        # Literal["FCSA", "CSA-N", "CSA"],
        str,
        Field(
            default="FCSA",
            description=(
                "determines how solver finds the search direction for the next "
                "iteration. Can be FCSA, CSA-N, or CSA"
            ),
        ),
    ]
    normalize_grads: Annotated[
        bool,
        Field(
            default=True, description="normalize gradients used for search direction?"
        ),
    ]
    feas_const: Annotated[
        float,
        Field(
            default=0.0,
            ge=0,
            description="feasibility constant to relax objective constraint",
        ),
    ]
    feas_score: Annotated[
        int,
        Field(
            default=2,
            gt=0,
            description="degree of feasibility score to relax objective constraint",
        ),
    ]
    report_all_solns: Annotated[
        bool,
        Field(default=False, description="report all incumbent solutions?"),
    ]


class FCSA(Solver):
    """Fully Cooperative Stochastic Approximation (FCSA) solver."""

    name: str = "FCSA"
    config_class: ClassVar[type[FCSAConfig]] = FCSAConfig
    class_name_abbr: ClassVar[str] = "FCSA"
    class_name: ClassVar[str] = "FCSA"
    objective_type: ClassVar[ObjectiveType] = ObjectiveType.SINGLE
    constraint_type: ClassVar[ConstraintType] = ConstraintType.STOCHASTIC
    variable_type: ClassVar[VariableType] = VariableType.CONTINUOUS
    gradient_needed: ClassVar[bool] = False

    def _step_fn(self, k: int) -> float:
        mult = self.factors["step_mult"]
        return mult if self.factors["step_type"] == "const" else 1 / (mult * (k + 1))

    def _objective_at(self, problem: Problem, x: np.ndarray, r: int) -> float:
        solution = self.create_new_solution(tuple(x), problem)
        self.budget.request(r)
        problem.simulate(solution, r)
        return -problem.minmax[0] * solution.objectives_mean[0]

    def _finite_difference(
        self, problem: Problem, x: np.ndarray, h: float, r: int
    ) -> np.ndarray:
        d = len(x)

        grad = np.zeros(d)
        for i in range(d):
            x1 = x.copy()
            x1[i] += h / 2
            f1 = self._objective_at(problem, x1, r)

            x2 = x.copy()
            x2[i] -= h / 2
            f2 = self._objective_at(problem, x2, r)

            grad[i] = (f1 - f2) / h

        return grad

    def _objective_grad(
        self, problem: Problem, solution: Solution, normalize: bool
    ) -> np.ndarray:
        if problem.gradient_available:
            grad = -problem.minmax[0] * solution.objectives_gradients_mean[0]
        else:
            h = self.factors["h"]
            r = self.factors["r"]
            grad = self._finite_difference(problem, np.array(solution.x), h, r)
        if normalize:
            norm = np.linalg.norm(grad)
            if norm == 0:
                norm = EPSILON
            grad /= norm

        return grad

    def _violated_constraint_values_and_grads(
        self, solution: Solution, normalize: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        values = solution.stoch_constraints_mean
        grads = np.array(solution.stoch_constraints_gradients_mean)
        gt_tolerance = values > self.factors["tolerance"]
        values, grads = values[gt_tolerance], grads[gt_tolerance]
        if normalize:
            norms = np.linalg.norm(grads, axis=1, keepdims=True)
            norms[norms == 0] = EPSILON
            grads /= norms
        return values, grads

    def _direction_csa(self, solution: Solution) -> np.ndarray:
        violated_values = solution.stoch_constraints_mean
        violated_index = np.argmax(violated_values)
        grad = solution.stoch_constraints_gradients_mean[violated_index]
        norm = np.linalg.norm(grad)
        if norm == 0:
            norm = EPSILON
        return grad / norm if self.factors["normalize_grads"] else grad

    def _direction_csa_n(self, solution: Solution) -> np.ndarray:
        normalize = self.factors["normalize_grads"]
        _, violated_grads = self._violated_constraint_values_and_grads(
            solution, normalize
        )

        # LP formulation
        direction = cp.Variable(violated_grads.shape[1])
        theta = cp.Variable()
        objective = cp.Maximize(theta)
        constraints = []
        constraints.append(cp.norm(direction, 2) <= 1)  # pyrefly: ignore
        for grad in violated_grads:
            constraints.append(-grad @ direction >= theta)
        problem = cp.Problem(objective, constraints)
        problem.solve()

        assert direction.value is not None
        return -direction.value

    def _direction_fcsa(self, problem: Problem, solution: Solution) -> np.ndarray:
        normalize = self.factors["normalize_grads"]

        obj_grad = self._objective_grad(problem, solution, normalize)
        violated_values, violated_grads = self._violated_constraint_values_and_grads(
            solution, normalize
        )

        # NLP formulation
        direction = cp.Variable(violated_grads.shape[1])
        theta = cp.Variable()
        objective = cp.Maximize(theta)

        constraints = []
        feas_score = np.linalg.norm(violated_values, ord=self.factors["feas_score"])
        feas_constant = self.factors["feas_const"]
        constraints.append(-obj_grad @ direction >= theta - feas_constant * feas_score)
        for grad in violated_grads:
            constraints.append(-grad @ direction >= theta)
        constraints.append(cp.norm(direction, 2) <= 1)  # pyrefly: ignore

        p = cp.Problem(objective, constraints)
        p.solve()

        assert direction.value is not None
        return -direction.value

    def _direction_no_violation(
        self, problem: Problem, solution: Solution
    ) -> np.ndarray:
        normalize = self.factors["normalize_grads"]
        return self._objective_grad(problem, solution, normalize)

    def _direction(self, solution: Solution, problem: Problem) -> np.ndarray:
        method = self.factors["search_direction"]

        violated_values = solution.stoch_constraints_mean
        is_violated = bool(np.max(violated_values) > self.factors["tolerance"])

        if not is_violated:
            return self._direction_no_violation(problem, solution)
        if method == "CSA":
            return self._direction_csa(solution)
        if method == "CSA-N":
            return self._direction_csa_n(solution)
        if method == "FCSA":
            return self._direction_fcsa(problem, solution)

        raise ValueError(f"unknown search direction method: {method}")

    def _prox_fn(
        self,
        a: np.ndarray,
        x: np.ndarray,
        lower: np.ndarray | None,
        upper: np.ndarray | None,
        ieq_lhs: np.ndarray | None,
        ieq_rhs: np.ndarray | None,
        eq_lhs: np.ndarray | None,
        eq_rhs: np.ndarray | None,
    ) -> np.ndarray:
        n = len(x)
        z = cp.Variable(n)

        objective = cp.Minimize(a @ z + 0.5 * (cp.norm(x - z) ** 2))  # pyrefly: ignore
        constraints = []

        if lower is not None and (lower > -np.inf).all():
            constraints.append(z >= lower)
        if upper is not None and (upper < np.inf).all():
            constraints.append(z <= upper)

        if ieq_lhs is not None and ieq_rhs is not None:
            constraints.append(ieq_lhs @ z <= ieq_rhs)
        if eq_lhs is not None and eq_rhs is not None:
            constraints.append(eq_lhs @ z == eq_rhs)

        problem = cp.Problem(objective, constraints)  # pyrefly: ignore
        problem.solve()

        assert z.value is not None
        return z.value

    def _new_solution_found(
        self, problem: Problem, x: tuple, check_feasibility: bool
    ) -> Solution:
        r = self.factors["r"]
        solution = self.create_new_solution(x, problem)
        self.budget.request(r)
        problem.simulate(solution, r)

        report_all_solutions = self.factors["report_all_solns"]

        # Test whether a feasible solution has been found and improved.
        feasible_found_and_improved = False
        if check_feasibility:
            violated_values = solution.stoch_constraints_mean
            is_violated = max(violated_values) > self.factors["tolerance"]
            self._feasible_found = self._feasible_found or (not is_violated)
            feasible_found_and_improved = (
                not is_violated
                and self._best_solution is not None
                and (
                    problem.minmax[0] * solution.objectives_mean
                    < problem.minmax[0] * self._best_solution.objectives_mean
                )
            )

        if (
            not check_feasibility
            or report_all_solutions
            or not self._feasible_found
            or feasible_found_and_improved
        ):
            self.recommended_solns.append(solution)
            self.intermediate_budgets.append(self.budget.used)
            self._best_solution = solution

        return solution

    def solve(self, problem: Problem) -> None:  # noqa: D102
        assert problem.n_stochastic_constraints > 0

        lower = np.array(problem.lower_bounds)
        upper = np.array(problem.upper_bounds)

        # NOTE: currently not used
        ieq_lhs = None
        ieq_rhs = None
        eq_lhs = None
        eq_rhs = None

        self._best_solution: Solution | None = None  # pyrefly: ignore
        self._feasible_found = False

        # Start with the initial solution.
        x = problem.factors["initial_solution"]
        solution = self._new_solution_found(problem, x, False)

        k = 0
        while True:
            step = self._step_fn(k)
            direction = self._direction(solution, problem)
            x = self._prox_fn(
                step * direction,
                np.array(solution.x),
                lower,
                upper,
                ieq_lhs,
                ieq_rhs,
                eq_lhs,
                eq_rhs,
            )
            solution = self._new_solution_found(problem, tuple(x), True)
            k += 1
