"""Random Search Solver.

Randomly sample solutions from the feasible region.
Can handle stochastic constraints.
A detailed description of the solver can be found `here <https://simopt.readthedocs.io/en/latest/randomsearch.html>`__.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

from simopt.base import (
    ConstraintType,
    ObjectiveType,
    Problem,
    Solver,
    VariableType,
)
from simopt.utils import override


class RandomSearchConfig(BaseModel):
    """Configuration for Random Search solver."""

    crn_across_solns: Annotated[
        bool, Field(default=True, description="use CRN across solutions?")
    ]
    sample_size: Annotated[
        int, Field(default=10, gt=0, description="sample size per solution")
    ]


class RandomSearch(Solver):
    """Random Search Solver.

    A solver that randomly samples solutions from the feasible region.
    Take a fixed number of replications at each solution.
    """

    name: str = "RNDSRCH"
    config_class: type[BaseModel] = RandomSearchConfig
    class_name_abbr: str = "RNDSRCH"
    class_name: str = "Random Search"
    objective_type: ObjectiveType = ObjectiveType.SINGLE
    constraint_type: ConstraintType = ConstraintType.STOCHASTIC
    variable_type: VariableType = VariableType.MIXED
    gradient_needed: bool = False

    @override
    def solve(self, problem: Problem) -> None:
        # Designate random number generator for random sampling.
        find_next_soln_rng = self.rng_list[1]
        # Start at initial solution and record as best.
        new_x = problem.factors["initial_solution"]
        new_solution = self.create_new_solution(new_x, problem)
        best_solution = new_solution
        self.recommended_solns.append(new_solution)
        self.intermediate_budgets.append(self.budget.used)

        # Prepare other variables in the loop.
        sample_size = self.factors["sample_size"]
        stoch_constraint_range = range(problem.n_stochastic_constraints)

        # Sequentially generate random solutions and simulate them.
        while True:
            # Request budget first, then simulate new solution.
            self.budget.request(sample_size)
            problem.simulate(new_solution, sample_size)

            # Check for improvement relative to incumbent best solution.
            # Also check for feasibility w.r.t. stochastic constraints.
            mean_diff = new_solution.objectives_mean - best_solution.objectives_mean
            if all(problem.minmax * mean_diff > 0) and all(
                new_solution.stoch_constraints_mean[idx] <= 0
                for idx in stoch_constraint_range
            ):
                # If better, record incumbent solution as best.
                best_solution = new_solution
                self.recommended_solns.append(new_solution)
                self.intermediate_budgets.append(self.budget.used)

            # Identify new solution to simulate for next iteration.
            new_x = problem.get_random_solution(find_next_soln_rng)
            new_solution = self.create_new_solution(new_x, problem)
