"""
Summary
-------
Randomly sample solutions from the feasible region.
Can handle stochastic constraints.
A detailed description of the solver can be found `here <https://simopt.readthedocs.io/en/latest/randomsearch.html>`__.
"""

from __future__ import annotations

from typing import Callable
from simopt.utils import classproperty

from simopt.base import (
    ConstraintType,
    ObjectiveType,
    Problem,
    Solution,
    Solver,
    VariableType,
)


class RandomSearch(Solver):
    """
    A solver that randomly samples solutions from the feasible region.
    Take a fixed number of replications at each solution.

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
        return "RNDSRCH"

    @classproperty
    def class_name(cls) -> str:
        return "Random Search"

    @classproperty
    def objective_type(cls) -> ObjectiveType:
        return ObjectiveType.SINGLE

    @classproperty
    def constraint_type(cls) -> ConstraintType:
        return ConstraintType.STOCHASTIC

    @classproperty
    def variable_type(cls) -> VariableType:
        return VariableType.MIXED

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
            "sample_size": {
                "description": "sample size per solution",
                "datatype": int,
                "default": 10,
            },
        }

    @property
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "crn_across_solns": self.check_crn_across_solns,
            "sample_size": self.check_sample_size,
        }

    def __init__(
        self, name: str = "RNDSRCH", fixed_factors: dict | None = None
    ) -> None:
        # Let the base class handle default arguments.
        super().__init__(name, fixed_factors)

    def check_sample_size(self) -> None:
        if self.factors["sample_size"] <= 0:
            raise ValueError("Sample size must be greater than 0.")

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
        # Designate random number generator for random sampling.
        find_next_soln_rng = self.rng_list[1]
        # Sequentially generate random solutions and simulate them.
        best_solution = None
        while expended_budget < problem.factors["budget"]:
            if expended_budget == 0:
                # Start at initial solution and record as best.
                new_x = problem.factors["initial_solution"]
                new_solution = self.create_new_solution(new_x, problem)
                best_solution = new_solution
                recommended_solns.append(new_solution)
                intermediate_budgets.append(expended_budget)
            else:
                # Identify new solution to simulate.
                new_x = problem.get_random_solution(find_next_soln_rng)
                new_solution = self.create_new_solution(new_x, problem)
            # Simulate new solution and update budget.
            problem.simulate(new_solution, self.factors["sample_size"])
            expended_budget += self.factors["sample_size"]
            # Check for improvement relative to incumbent best solution.
            # Also check for feasibility w.r.t. stochastic constraints.
            if (
                best_solution is not None
                and problem.minmax * new_solution.objectives_mean
                > problem.minmax * best_solution.objectives_mean
                and all(
                    new_solution.stoch_constraints_mean[idx] <= 0
                    for idx in range(problem.n_stochastic_constraints)
                )
            ):
                # If better, record incumbent solution as best.
                best_solution = new_solution
                recommended_solns.append(new_solution)
                intermediate_budgets.append(expended_budget)
        return recommended_solns, intermediate_budgets
