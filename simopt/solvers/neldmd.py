"""Nelder-Mead Algorithm.

Nelder-Mead: An algorithm that maintains a simplex of points that moves around the
feasible region according to certain geometric operations: reflection, expansion,
contraction, and shrinking. A detailed description of the solver can be found
`here <https://simopt.readthedocs.io/en/latest/neldmd.html>`__.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Annotated, ClassVar

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


class NelderMeadConfig(SolverConfig):
    """Configuration for Nelder-Mead solver."""

    r: Annotated[
        int,
        Field(
            default=30,
            gt=0,
            description="number of replications taken at each solution",
        ),
    ]
    alpha: Annotated[
        float, Field(default=1.0, gt=0, description="reflection coefficient > 0")
    ]
    gammap: Annotated[
        float, Field(default=2.0, gt=1, description="expansion coefficient > 1")
    ]
    betap: Annotated[
        float,
        Field(
            default=0.5,
            gt=0,
            lt=1,
            description="contraction coefficient > 0, < 1",
        ),
    ]
    delta: Annotated[
        float,
        Field(default=0.5, gt=0, lt=1, description="shrink factor > 0, < 1"),
    ]
    sensitivity: Annotated[
        float,
        Field(default=1e-7, gt=0, description="shrinking scale for bounds"),
    ]
    initial_spread: Annotated[
        float,
        Field(
            default=0.1,
            gt=0,
            description="fraction of distance between bounds used for initial points",
        ),
    ]


class NelderMead(Solver):
    """Nelder-Mead Algorithm.

    The Nelder-Mead algorithm, which maintains a simplex of points that moves around
    the feasible region according to certain geometric operations: reflection,
    expansion, contraction, and shrinking.
    """

    name: str = "NELDMD"
    config_class: ClassVar[type[SolverConfig]] = NelderMeadConfig
    class_name_abbr: ClassVar[str] = "NELDMD"
    class_name: ClassVar[str] = "Nelder-Mead"
    objective_type: ClassVar[ObjectiveType] = ObjectiveType.SINGLE
    constraint_type: ClassVar[ConstraintType] = ConstraintType.BOX
    variable_type: ClassVar[VariableType] = VariableType.CONTINUOUS
    gradient_needed: ClassVar[bool] = False

    # FIXME: fix typing on `sort_sol`
    def solve(self, problem: Problem) -> None:  # noqa: D102
        # Designate random number generator for random sampling.
        get_rand_soln_rng = self.rng_list[1]
        n_pts = problem.dim + 1

        # Check for sufficiently large budget.
        if self.budget.total < self.factors["r"] * n_pts:
            err_msg = "Budget is too small for a good quality run of Nelder-Mead."
            raise ValueError(err_msg)

        # Shrink variable bounds to avoid floating errors.
        if problem.lower_bounds and not np.all(np.isneginf(problem.lower_bounds)):
            self.lower_bounds = (
                np.array(problem.lower_bounds) + self.factors["sensitivity"]
            )
        else:
            self.lower_bounds = None

        if problem.upper_bounds and not np.all(np.isposinf(problem.upper_bounds)):
            self.upper_bounds = (
                np.array(problem.upper_bounds) - self.factors["sensitivity"]
            )
        else:
            self.upper_bounds = None

        # Initial dim + 1 points.
        sol = [self.create_new_solution(problem.factors["initial_solution"], problem)]

        if self.lower_bounds is None or self.upper_bounds is None:
            sol.extend(
                self.create_new_solution(
                    problem.get_random_solution(get_rand_soln_rng), problem
                )
                for _ in range(1, n_pts)
            )
        else:  # Restrict starting shape/location.
            initial_solution = np.array(
                problem.factors["initial_solution"], dtype=float
            )
            distances = (self.upper_bounds - self.lower_bounds) * self.factors[
                "initial_spread"
            ]

            # Generate new points
            new_pts = np.tile(initial_solution, (problem.dim, 1))
            new_pts[np.arange(problem.dim), np.arange(problem.dim)] += distances

            # Apply boundary conditions
            out_of_bounds = (new_pts > self.upper_bounds) | (
                new_pts < self.lower_bounds
            )
            if np.any(out_of_bounds):
                new_pts[out_of_bounds] -= 2 * distances

            # If still out of bounds, set to nearest bound
            out_of_bounds = (new_pts > self.upper_bounds) | (
                new_pts < self.lower_bounds
            )
            if np.any(out_of_bounds):
                new_pts[out_of_bounds] = np.where(
                    np.array(problem.minmax)[np.newaxis, :] == -1,
                    self.lower_bounds,
                    self.upper_bounds,
                )

            sol.extend(self.create_new_solution(pt, problem) for pt in new_pts)

        r = self.factors["r"]  # For increasing replications.

        # Start Solving.
        # Evaluate solutions in initial structure.
        for solution in sol:
            self.budget.request(r)
            problem.simulate(solution, r)

        # Record initial solution data.
        # FIXME: I think this might be wrong
        self.intermediate_budgets.append(0)
        self.recommended_solns.append(sol[0])
        # Sort solutions by obj function estimate.
        sort_sol = self._sort_and_end_update(problem, sol)

        # Maximization problem is converted to minimization by using minmax.
        while True:
            # Shrink towards best if out of bounds.
            while True:
                # Reflect worst and update sort_sol.
                p_high = sort_sol[-1]  # Current worst point. # pyrefly: ignore
                p_high_x = np.array(p_high.x)
                p_cent = np.mean(
                    [s.x for s in sort_sol[:-1]],  # pyrefly: ignore
                    axis=0,
                )
                p_refl = np.array(
                    (1 + self.factors["alpha"]) * p_cent
                    - self.factors["alpha"] * p_high_x
                )

                # Check if reflection point is within bounds.
                if np.equal(p_refl, self._check_const(p_refl, p_high_x)).all():
                    break

                sol_0_x = np.array(sort_sol[0].x)  # pyrefly: ignore
                for i in range(1, len(sort_sol)):
                    p_new = (
                        self.factors["delta"]
                        * np.array(sort_sol[i].x)  # pyrefly: ignore
                        + (1 - self.factors["delta"]) * sol_0_x
                    )
                    p_new = self._check_const(p_new, sol_0_x)
                    p_new = Solution(p_new, problem)
                    p_new.attach_rngs(rng_list=self.solution_progenitor_rngs, copy=True)
                    self.budget.request(r)
                    problem.simulate(p_new, r)

                    # Update sort_sol.
                    sort_sol[i] = p_new  # p_new replaces pi.

                # Sort & end updating.
                sort_sol = self._sort_and_end_update(
                    problem,
                    sort_sol,  # pyrefly: ignore
                )

            # Evaluate reflected point.
            p_refl = tuple(p_refl.tolist())
            p_refl = Solution(p_refl, problem)
            p_refl.attach_rngs(rng_list=self.solution_progenitor_rngs, copy=True)
            self.budget.request(r)
            problem.simulate(p_refl, r)
            np_minmax = np.array(problem.minmax)
            refl_fn_val = np_minmax * -p_refl.objectives_mean

            # Track best, worst, and second worst points.
            p_low = sort_sol[0]  # Current best pt. # pyrefly: ignore
            inv_minmax = np_minmax * -1
            fn_low = inv_minmax * sort_sol[0].objectives_mean  # pyrefly: ignore
            fn_sec = inv_minmax * sort_sol[-2].objectives_mean  # pyrefly: ignore
            fn_high = inv_minmax * sort_sol[-1].objectives_mean  # pyrefly: ignore

            # Check if accept reflection.
            if fn_low <= refl_fn_val and refl_fn_val <= fn_sec:
                # The new point replaces the previous worst.
                sort_sol[-1] = p_refl  # pyrefly: ignore
                # Sort & end updating.
                sort_sol = self._sort_and_end_update(
                    problem,
                    sort_sol,  # pyrefly: ignore
                )
                # Best solution remains the same, so no reporting.

            # Check if accept expansion (of reflection in the same direction).
            elif refl_fn_val < fn_low:
                p_exp = self.factors["gammap"] * np.array(p_refl.x) + (
                    1 - self.factors["gammap"]
                ) * np.array(p_cent)
                p_exp = self._check_const(p_exp, p_refl.x)

                # Evaluate expansion point.
                p_exp = Solution(p_exp, problem)
                p_exp.attach_rngs(rng_list=self.solution_progenitor_rngs, copy=True)
                self.budget.request(r)
                problem.simulate(p_exp, r)
                exp_fn_val = inv_minmax * p_exp.objectives_mean

                # Check if expansion point is an improvement relative to simplex.
                sort_sol[-1] = (  # pyrefly: ignore
                    p_exp if exp_fn_val < fn_low else p_refl
                )

                # Sort & end updating.
                sort_sol = self._sort_and_end_update(
                    problem,
                    sort_sol,  # pyrefly: ignore
                )

                # Record data if within budget.
                self.intermediate_budgets.append(self.budget.used)
                self.recommended_solns.append(p_exp if exp_fn_val < fn_low else p_refl)

            # Check if accept contraction or shrink.
            elif refl_fn_val > fn_sec:
                if refl_fn_val <= fn_high:
                    p_high = p_refl  # p_refl replaces p_high.
                    fn_high = refl_fn_val  # Replace fn_high.

                # Attempt contraction or shrinking.
                p_cont2 = p_high
                p_cont = self.factors["betap"] * np.array(p_high.x) + (
                    1 - self.factors["betap"]
                ) * np.array(p_cent)
                p_cont = self._check_const(p_cont, p_cont2.x)

                # Evaluate contraction point.
                p_cont = Solution(p_cont, problem)
                p_cont.attach_rngs(rng_list=self.solution_progenitor_rngs, copy=True)
                self.budget.request(r)
                problem.simulate(p_cont, r)
                cont_fn_val = inv_minmax * p_cont.objectives_mean

                # Accept contraction.
                if cont_fn_val <= fn_high:
                    sort_sol[-1] = p_cont  # p_cont replaces p_high. # pyrefly: ignore

                    # Sort & end updating.
                    sort_sol = self._sort_and_end_update(
                        problem,
                        sort_sol,  # pyrefly: ignore
                    )

                    # Check if contraction point is new best.
                    if cont_fn_val < fn_low:
                        # Record data from contraction point (new best).
                        self.intermediate_budgets.append(self.budget.used)
                        self.recommended_solns.append(p_cont)
                # Contraction fails -> simplex shrinks by delta with p_low fixed.
                else:
                    # Set pre-loop variables
                    sort_sol[-1] = p_high  # Replaced by p_refl. # pyrefly: ignore
                    is_new_best = False
                    p_low_x = np.array(p_low.x)
                    for i in range(1, len(sort_sol)):  # pyrefly: ignore
                        p_new = (
                            self.factors["delta"] * np.array(sort_sol[i].x)
                            + (1 - self.factors["delta"]) * p_low_x
                        )
                        p_new = self._check_const(p_new, p_low.x)

                        p_new = Solution(p_new, problem)
                        p_new.attach_rngs(
                            rng_list=self.solution_progenitor_rngs, copy=True
                        )
                        self.budget.request(r)
                        problem.simulate(p_new, r)
                        new_fn_val = inv_minmax * p_new.objectives_mean

                        # Check for new best.
                        if new_fn_val <= fn_low:
                            is_new_best = True

                        # Update sort_sol.
                        sort_sol[i] = p_new  # p_new replaces pi.

                    # Sort & end updating.
                    sort_sol = self._sort_and_end_update(
                        problem,
                        sort_sol,  # pyrefly: ignore
                    )

                    # Record data if there is a new best solution in the contraction.
                    if is_new_best:
                        self.intermediate_budgets.append(self.budget.used)
                        self.recommended_solns.append(sort_sol[0])

    def _sort_and_end_update(
        self, problem: Problem, sol: Iterable[Solution]
    ) -> list[Solution]:
        """Sort solutions by objective values, accounting for minimization/maximization.

        Args:
            problem (Problem): The simulation-optimization problem defining the
                objective direction (minimize or maximize).
            sol (Iterable[Solution]): An iterable of solutions to be sorted.

        Returns:
            list[Solution]: A list of solutions sorted according to their
                objective values.
        """
        minmax_array = np.array(problem.minmax)
        return sorted(
            sol,
            key=lambda s: np.dot(minmax_array, s.objectives_mean),
            reverse=True,
        )

    def _check_const(
        self, new_point: Iterable[float], reference_point: Iterable[float]
    ) -> tuple:
        """Adjust a point to ensure it remains within the specified bounds.

        Args:
            new_point (Iterable[float]): The proposed new point to be checked and
                adjusted if necessary.
            reference_point (Iterable[float]): The original reference point used to
                compute movement direction.

        Returns:
            tuple: The modified point that adheres to the given bounds.
        """
        # Make sure everything is a NumPy array
        new_point = np.array(new_point)
        reference_point = np.array(reference_point)
        # Create or compute the other variables we need
        step = new_point - reference_point
        tmin = 1

        # Apply bounding constraints using NumPy masks
        if self.upper_bounds is not None:
            mask = step > 0
            if np.any(mask):
                tmin = min(
                    tmin,
                    np.min(
                        (self.upper_bounds[mask] - reference_point[mask]) / step[mask]
                    ),
                )

        if self.lower_bounds is not None:
            mask = step < 0
            if np.any(mask):
                tmin = min(
                    tmin,
                    np.min(
                        (self.lower_bounds[mask] - reference_point[mask]) / step[mask]
                    ),
                )

        # Compute the modified point
        adjusted_point = reference_point + tmin * step

        # Remove rounding errors
        adjusted_point[np.abs(adjusted_point) < self.factors["sensitivity"]] = 0

        return tuple(adjusted_point.tolist())
