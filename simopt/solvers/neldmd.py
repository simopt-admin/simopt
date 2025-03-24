"""
Summary
-------
Nelder-Mead: An algorithm that maintains a simplex of points that moves around the feasible
region according to certain geometric operations: reflection, expansion,
contraction, and shrinking.
A detailed description of the solver can be found
`here <https://simopt.readthedocs.io/en/latest/neldmd.html>`__.
"""

from __future__ import annotations

from collections.abc import Iterable
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


class NelderMead(Solver):
    """The Nelder-Mead algorithm, which maintains a simplex of points that moves around the feasible
    region according to certain geometric operations: reflection, expansion,
    contraction, and shrinking.

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
    def class_name_abbr(cls) -> str:
        return "NELDMD"

    @classproperty
    def class_name(cls) -> str:
        return "Nelder-Mead"

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
            "alpha": {
                "description": "reflection coefficient > 0",
                "datatype": float,
                "default": 1.0,
            },
            "gammap": {
                "description": "expansion coefficient > 1",
                "datatype": float,
                "default": 2.0,
            },
            "betap": {
                "description": "contraction coefficient > 0, < 1",
                "datatype": float,
                "default": 0.5,
            },
            "delta": {
                "description": "shrink factor > 0, < 1",
                "datatype": float,
                "default": 0.5,
            },
            "sensitivity": {
                "description": "shrinking scale for bounds",
                "datatype": float,
                "default": 10 ** (-7),
            },
            "initial_spread": {
                "description": "fraction of the distance between bounds used to select initial points",
                "datatype": float,
                "default": 1 / 10,
            },
        }

    @property
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "crn_across_solns": self.check_crn_across_solns,
            "r": self.check_r,
            "alpha": self.check_alpha,
            "gammap": self.check_gammap,
            "betap": self.check_betap,
            "delta": self.check_delta,
            "sensitivity": self.check_sensitivity,
            "initial_spread": self.check_initial_spread,
        }

    def __init__(
        self, name: str = "NELDMD", fixed_factors: dict | None = None
    ) -> None:
        # Let the base class handle default arguments.
        super().__init__(name, fixed_factors)

    def check_r(self) -> None:
        if self.factors["r"] <= 0:
            raise ValueError(
                "The number of replications taken at each solution must be greater than 0."
            )

    def check_alpha(self) -> None:
        if self.factors["alpha"] <= 0:
            raise ValueError("Alpha must be greater than 0.")

    def check_gammap(self) -> None:
        if self.factors["gammap"] <= 1:
            raise ValueError("Gammap must be greater than 1.")

    def check_betap(self) -> None:
        if (self.factors["betap"] <= 0) or (self.factors["betap"] >= 1):
            raise ValueError("betap must be between 0 and 1.")

    def check_delta(self) -> None:
        if (self.factors["delta"] <= 0) or (self.factors["delta"] >= 1):
            raise ValueError("Delta must be between 0 and 1.")

    def check_sensitivity(self) -> None:
        if self.factors["sensitivity"] <= 0:
            raise ValueError("Sensitivity must be greater than 0.")

    def check_initial_spread(self) -> None:
        if self.factors["initial_spread"] <= 0:
            raise ValueError("Initial spread must be greater than 0.")

    def solve(self, problem: Problem) -> tuple[list[Solution], list[int]]:
        """
        Run a single macroreplication of a solver on a problem.

        Arguments
        ---------
        problem : Problem object
            simulation-optimization problem to solve

        Returns
        -------
        list[Solution]
            list of solutions recommended throughout the budget
        list[int]
            list of intermediate budgets when recommended solutions changes
        """
        # Designate random number generator for random sampling.
        get_rand_soln_rng = self.rng_list[1]
        n_pts = problem.dim + 1

        # Check for sufficiently large budget.
        if problem.factors["budget"] < self.factors["r"] * n_pts:
            err_msg = (
                "Budget is too small for a good quality run of Nelder-Mead."
            )
            raise ValueError(err_msg)

        # Shrink variable bounds to avoid floating errors.
        if problem.lower_bounds and not np.all(
            np.isneginf(problem.lower_bounds)
        ):
            self.lower_bounds = (
                np.array(problem.lower_bounds) + self.factors["sensitivity"]
            )
        else:
            self.lower_bounds = None

        if problem.upper_bounds and not np.all(
            np.isposinf(problem.upper_bounds)
        ):
            self.upper_bounds = (
                np.array(problem.upper_bounds) - self.factors["sensitivity"]
            )
        else:
            self.upper_bounds = None

        # Initial dim + 1 points.
        sol = [
            self.create_new_solution(
                problem.factors["initial_solution"], problem
            )
        ]

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
                    problem.minmax[np.newaxis, :] == -1,
                    self.lower_bounds,
                    self.upper_bounds,
                )

            sol.extend(self.create_new_solution(pt, problem) for pt in new_pts)

        # Initialize lists to track budget and best solutions.
        intermediate_budgets = []
        recommended_solns = []
        # Track overall budget spent.
        budget_spent = 0
        r = self.factors["r"]  # For increasing replications.

        # Start Solving.
        # Evaluate solutions in initial structure.
        for solution in sol:
            problem.simulate(solution, r)
            budget_spent += r
        # Record initial solution data.
        intermediate_budgets.append(0)
        recommended_solns.append(sol[0])
        # Sort solutions by obj function estimate.
        sort_sol = self._sort_and_end_update(problem, sol)

        # Maximization problem is converted to minimization by using minmax.
        while budget_spent <= problem.factors["budget"]:
            # Shrink towards best if out of bounds.
            while True:
                # Reflect worst and update sort_sol.
                p_high = sort_sol[-1]  # Current worst point.
                p_high_x = np.array(p_high.x)
                p_cent = np.mean([s.x for s in sort_sol[:-1]], axis=0)
                p_refl = np.array(
                    (1 + self.factors["alpha"]) * p_cent
                    - self.factors["alpha"] * p_high_x
                )

                # Check if reflection point is within bounds.
                if np.equal(p_refl, self._check_const(p_refl, p_high_x)).all():
                    break

                sol_0_x = np.array(sort_sol[0].x)
                for i in range(1, len(sort_sol)):
                    p_new = (
                        self.factors["delta"] * np.array(sort_sol[i].x)
                        + (1 - self.factors["delta"]) * sol_0_x
                    )
                    p_new = self._check_const(p_new, sol_0_x)
                    p_new = Solution(p_new, problem)
                    p_new.attach_rngs(
                        rng_list=self.solution_progenitor_rngs, copy=True
                    )
                    problem.simulate(p_new, r)
                    budget_spent += r

                    # Update sort_sol.
                    sort_sol[i] = p_new  # p_new replaces pi.

                # Sort & end updating.
                sort_sol = self._sort_and_end_update(problem, sort_sol)

            # Evaluate reflected point.
            p_refl = tuple(p_refl.tolist())
            p_refl = Solution(p_refl, problem)
            p_refl.attach_rngs(
                rng_list=self.solution_progenitor_rngs, copy=True
            )
            problem.simulate(p_refl, r)
            budget_spent += r
            np_minmax = np.array(problem.minmax)
            refl_fn_val = np_minmax * -p_refl.objectives_mean

            # Track best, worst, and second worst points.
            p_low = sort_sol[0]  # Current best pt.
            inv_minmax = np_minmax * -1
            fn_low = inv_minmax * sort_sol[0].objectives_mean
            fn_sec = inv_minmax * sort_sol[-2].objectives_mean
            fn_high = inv_minmax * sort_sol[-1].objectives_mean

            # Check if accept reflection.
            if fn_low <= refl_fn_val and refl_fn_val <= fn_sec:
                # The new point replaces the previous worst.
                sort_sol[-1] = p_refl
                # Sort & end updating.
                sort_sol = self._sort_and_end_update(problem, sort_sol)
                # Best solution remains the same, so no reporting.

            # Check if accept expansion (of reflection in the same direction).
            elif refl_fn_val < fn_low:
                p_exp = self.factors["gammap"] * np.array(p_refl.x) + (
                    1 - self.factors["gammap"]
                ) * np.array(p_cent)
                p_exp = self._check_const(p_exp, p_refl.x)

                # Evaluate expansion point.
                p_exp = Solution(p_exp, problem)
                p_exp.attach_rngs(
                    rng_list=self.solution_progenitor_rngs, copy=True
                )
                problem.simulate(p_exp, r)
                budget_spent += r
                exp_fn_val = inv_minmax * p_exp.objectives_mean

                # Check if expansion point is an improvement relative to simplex.
                sort_sol[-1] = p_exp if exp_fn_val < fn_low else p_refl

                # Sort & end updating.
                sort_sol = self._sort_and_end_update(problem, sort_sol)

                # Record data if within budget.
                if budget_spent <= problem.factors["budget"]:
                    intermediate_budgets.append(budget_spent)
                    recommended_solns.append(
                        p_exp if exp_fn_val < fn_low else p_refl
                    )

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
                p_cont.attach_rngs(
                    rng_list=self.solution_progenitor_rngs, copy=True
                )
                problem.simulate(p_cont, r)
                budget_spent += r
                cont_fn_val = inv_minmax * p_cont.objectives_mean

                # Accept contraction.
                if cont_fn_val <= fn_high:
                    sort_sol[-1] = p_cont  # p_cont replaces p_high.

                    # Sort & end updating.
                    sort_sol = self._sort_and_end_update(problem, sort_sol)

                    # Check if contraction point is new best.
                    if cont_fn_val < fn_low:
                        # Record data from contraction point (new best).
                        if budget_spent <= problem.factors["budget"]:
                            intermediate_budgets.append(budget_spent)
                            recommended_solns.append(p_cont)
                # Contraction fails -> simplex shrinks by delta with p_low fixed.
                else:
                    # Set pre-loop variables
                    sort_sol[-1] = p_high  # Replaced by p_refl.
                    is_new_best = False
                    p_low_x = np.array(p_low.x)
                    for i in range(1, len(sort_sol)):
                        p_new = (
                            self.factors["delta"] * np.array(sort_sol[i].x)
                            + (1 - self.factors["delta"]) * p_low_x
                        )
                        p_new = self._check_const(p_new, p_low.x)

                        p_new = Solution(p_new, problem)
                        p_new.attach_rngs(
                            rng_list=self.solution_progenitor_rngs, copy=True
                        )
                        problem.simulate(p_new, r)
                        budget_spent += r
                        new_fn_val = inv_minmax * p_new.objectives_mean

                        # Check for new best.
                        if new_fn_val <= fn_low:
                            is_new_best = True

                        # Update sort_sol.
                        sort_sol[i] = p_new  # p_new replaces pi.

                    # Sort & end updating.
                    sort_sol = self._sort_and_end_update(problem, sort_sol)

                    # Record data if there is a new best solution in the contraction.
                    if (
                        is_new_best
                        and budget_spent <= problem.factors["budget"]
                    ):
                        intermediate_budgets.append(budget_spent)
                        recommended_solns.append(sort_sol[0])

        return recommended_solns, intermediate_budgets

    def _sort_and_end_update(
        self, problem: Problem, sol: Iterable[Solution]
    ) -> list[Solution]:
        """
        Sorts solutions based on their objectives while considering the problem's min/max direction.

        Arguments
        ---------
        problem : Problem
            The simulation-optimization problem containing the objective direction (min/max).
        sol : Iterable[Solution]
            Iterable of solutions to be sorted.

        Returns
        -------
        list[Solution]
            The sorted list of solutions.
        """
        minmax_array = np.array(problem.minmax)
        sort_sol = sorted(
            sol, key=lambda s: minmax_array * s.objectives_mean, reverse=True
        )
        return sort_sol

    def _check_const(
        self, new_point: Iterable[float], reference_point: Iterable[float]
    ) -> tuple:
        """
        Adjust a point to ensure it remains within the specified bounds.

        new_point : Iterable[float]
            The proposed new point to be checked and adjusted if necessary.
        reference_point : Iterable[float]
            The original reference point used to compute movement direction.

        Returns
        -------
        tuple
            The modified point that adheres to the given bounds.
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
                        (self.upper_bounds[mask] - reference_point[mask])
                        / step[mask]
                    ),
                )

        if self.lower_bounds is not None:
            mask = step < 0
            if np.any(mask):
                tmin = min(
                    tmin,
                    np.min(
                        (self.lower_bounds[mask] - reference_point[mask])
                        / step[mask]
                    ),
                )

        # Compute the modified point
        adjusted_point = reference_point + tmin * step

        # Remove rounding errors
        adjusted_point[np.abs(adjusted_point) < self.factors["sensitivity"]] = 0

        adjusted_point_tuple = tuple(adjusted_point.tolist())

        return adjusted_point_tuple
