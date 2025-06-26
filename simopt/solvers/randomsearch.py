"""Random Search Solver.

Randomly sample solutions from the feasible region.
Can handle stochastic constraints.
A detailed description of the solver can be found `here <https://simopt.readthedocs.io/en/latest/randomsearch.html>`__.
"""

from __future__ import annotations

from typing import Callable

from simopt.base import (
    ConstraintType,
    ObjectiveType,
    Problem,
    Solution,
    Solver,
    VariableType,
)
from simopt.utils import classproperty, override


class RandomSearch(Solver):
    """Random Search Solver.

    A solver that randomly samples solutions from the feasible region.
    Take a fixed number of replications at each solution.
    """

    @classproperty
    @override
    def class_name_abbr(cls) -> str:
        return "RNDSRCH"

    @classproperty
    @override
    def class_name(cls) -> str:
        return "Random Search"

    @classproperty
    @override
    def objective_type(cls) -> ObjectiveType:
        return ObjectiveType.SINGLE

    @classproperty
    @override
    def constraint_type(cls) -> ConstraintType:
        return ConstraintType.STOCHASTIC

    @classproperty
    @override
    def variable_type(cls) -> VariableType:
        return VariableType.MIXED

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
            "sample_size": {
                "description": "sample size per solution",
                "datatype": int,
                "default": 10,
            },
        }

    @property
    @override
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "crn_across_solns": self.check_crn_across_solns,
            "sample_size": self._check_sample_size,
        }

    def __init__(
        self, name: str = "RNDSRCH", fixed_factors: dict | None = None
    ) -> None:
        """Initialize Random Search solver.

        Args:
            name (str): user-specified name for solver
            fixed_factors (dict, optional): fixed_factors of the solver.
                Defaults to None.
        """
        # Let the base class handle default arguments.
        super().__init__(name, fixed_factors)

    def _check_sample_size(self) -> None:
        if self.factors["sample_size"] <= 0:
            raise ValueError("Sample size must be greater than 0.")

    @override
    def solve(self, problem: Problem) -> tuple[list[Solution], list[int]]:
        # Designate random number generator for random sampling.
        find_next_soln_rng = self.rng_list[1]
        # Start at initial solution and record as best.
        new_x = problem.factors["initial_solution"]
        new_solution = self.create_new_solution(new_x, problem)
        best_solution = new_solution
        recommended_solns = [new_solution]
        # Initialize budget and record initial expenditure.
        expended_budget = 0
        intermediate_budgets = [expended_budget]
        # Prepare other variables in the loop.
        sample_size = self.factors["sample_size"]
        stoch_constraint_range = range(problem.n_stochastic_constraints)
        # Sequentially generate random solutions and simulate them.
        while True:
            # Simulate new solution and update budget.
            problem.simulate(new_solution, sample_size)
            expended_budget += sample_size
            # Check for improvement relative to incumbent best solution.
            # Also check for feasibility w.r.t. stochastic constraints.
            mean_diff = new_solution.objectives_mean - best_solution.objectives_mean
            if all(problem.minmax * mean_diff > 0) and all(
                new_solution.stoch_constraints_mean[idx] <= 0
                for idx in stoch_constraint_range
            ):
                # If better, record incumbent solution as best.
                best_solution = new_solution
                recommended_solns.append(new_solution)
                intermediate_budgets.append(expended_budget)

            # Check if budget is exceeded.
            if expended_budget >= problem.factors["budget"]:
                return recommended_solns, intermediate_budgets

            # Identify new solution to simulate for next iteration.
            new_x = problem.get_random_solution(find_next_soln_rng)
            new_solution = self.create_new_solution(new_x, problem)
