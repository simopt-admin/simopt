
from __future__ import annotations

from typing import Annotated, ClassVar

import numpy as np
from pydantic import Field
from scipy.optimize import minimize, NonlinearConstraint, Bounds

from simopt.base import (
    ConstraintType,
    ObjectiveType,
    Problem,
    Solver,
    SolverConfig,
    VariableType,
)


class COBYLAConfig(SolverConfig):
    """Configuration for Random Search solver."""

    sample_size: Annotated[
        int, Field(default=10, gt=0, description="sample size per solution")
    ]


class COBYLA(Solver):
    """Random Search Solver.

    A solver that randomly samples solutions from the feasible region.
    Take a fixed number of replications at each solution.
    """

    name: str = "COBYLA"
    config_class: ClassVar[type[SolverConfig]] = COBYLAConfig
    class_name_abbr: ClassVar[str] = "COBYLA"
    class_name: ClassVar[str] = "Constrained Optimization by Linear Approximation"
    objective_type: ClassVar[ObjectiveType] = ObjectiveType.SINGLE
    constraint_type: ClassVar[ConstraintType] = ConstraintType.DETERMINISTIC
    variable_type: ClassVar[VariableType] = VariableType.CONTINUOUS
    gradient_needed: ClassVar[bool] = False
    
    def simulate(self, x):
        sample_size = self.factors["sample_size"]
        self.budget.request(sample_size)
        new_solution = self.create_new_solution(x, self.problem)
        self.problem.simulate(new_solution, sample_size)
        return float(new_solution.objectives_mean[0])
    
    def get_constraints(self, x):
        return np.asarray(self.problem.get_deterministic_constraints(tuple(x)), dtype=float)
    
    def callback(self, res):
        int_x = tuple(res)
        new_solution = self.create_new_solution(int_x, self.problem)
        self.recommended_solns.append(new_solution)
        self.intermediate_budgets.append(self.budget.used)
        print('int:', int_x)
        

    def solve(self, problem: Problem) -> None:  # noqa: D102
        # Designate random number generator for random sampling.
        find_next_soln_rng = self.rng_list[1]
        self.problem = problem
        # Start at initial solution and record as best.
        new_x = problem.factors["initial_solution"]
        new_solution = self.create_new_solution(new_x, problem)
        best_solution = new_solution
        self.recommended_solns.append(new_solution)
        self.intermediate_budgets.append(self.budget.used)
        x0 = new_solution.x
        remaining_budget = problem.factors["budget"] - self.budget.used
        maxfev = int(remaining_budget//self.factors['sample_size'])

        # Prepare other variables in the loop.
        #sample_size = self.factors["sample_size"]
        #stoch_constraint_range = range(problem.n_stochastic_constraints)
        
        #determine number of constraints
        m = len(np.atleast_1d(self.problem.get_deterministic_constraints(tuple(new_x))))
        
        #get constraints
        c = NonlinearConstraint(
            self.get_constraints,
            lb = np.zeros(m),
            ub = np.zeros(m)
            )
        bounds = Bounds(problem.lower_bounds, problem.upper_bounds)
        print('test')

        # Sequentially generate random solutions and simulate them.
        #while True:
        # use COBYLA to solve 
        res = minimize(self.simulate, 
                       x0,
                       method = "COBYQA", 
                       constraints = [c], 
                       bounds = bounds,
                       options={"maxfev": maxfev},
                       callback= self.callback)
        # new_x = res.x
        # print(new_x)
        # new_solution = self.create_new_solution(new_x, problem)
        # self.recommended_solns.append(new_solution)
        # self.intermediate_budgets.append(self.budget.used)
        
        # mean_diff = new_solution.objectives_mean - best_solution.objectives_mean
        # if all(np.array(problem.minmax) * mean_diff > 0) and all(
        #     new_solution.stoch_constraints_mean[idx] <= 0
        #     for idx in stoch_constraint_range
        # ):
        #     # If better, record incumbent solution as best.
        #     best_solution = new_solution
        #     self.recommended_solns.append(new_solution)
        #     self.intermediate_budgets.append(self.budget.used)

        # # Identify new solution to simulate for next iteration.
        # new_x = problem.get_random_solution(find_next_soln_rng)
        # new_solution = self.create_new_solution(new_x, problem)
