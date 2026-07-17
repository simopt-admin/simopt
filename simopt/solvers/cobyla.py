
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
    feas_tol: Annotated[
        float, Field(default=1e-8, gt=0, description="sample size per solution")
    ]


class COBYLA(Solver):
    """Random Search Solver.

    A solver that randomly samples solutions from the feasible region.
    Take a fixed number of replications at each solution.
    """

    name: str = "COBQLA"
    config_class: ClassVar[type[SolverConfig]] = COBYLAConfig
    class_name_abbr: ClassVar[str] = "COBYQA"
    class_name: ClassVar[str] = "Constrained Optimization by Linear Approximation"
    objective_type: ClassVar[ObjectiveType] = ObjectiveType.SINGLE
    constraint_type: ClassVar[ConstraintType] = ConstraintType.DETERMINISTIC
    variable_type: ClassVar[VariableType] = VariableType.CONTINUOUS
    gradient_needed: ClassVar[bool] = False
    
    def simulate(self, x):
        sample_size = self.factors["sample_size"]
        x_arr = np.asarray(x, dtype=float)
        x_tuple = tuple(x_arr)

        self.budget.request(sample_size)

        solution = self.create_new_solution(x_tuple, self.problem)
        self.problem.simulate(solution, sample_size)
        
        # save solution key for callback
        key = self._x_key(x_arr)
        self._evaluated_solutions[key] = solution

        objective = float(solution.objectives_mean[0])
        return -self.problem.minmax[0] * objective
    
    def get_equality_constraints(self, x):
        return np.asarray(self.problem.get_deterministic_equality_constraints(tuple(x)), dtype=float).reshape(-1)
    def get_inequality_constraints(self, x):
        return np.asarray(self.problem.get_deterministic_inequality_constraints(tuple(x)), dtype=float).reshape(-1)
    
    def _x_key(self, x):
        return tuple(np.round(np.asarray(x, dtype=float), decimals=12))

    def callback(self, xk):
        key = self._x_key(xk)

        solution = self._evaluated_solutions.get(key)
        if solution is None:
            return

        if key != self._last_recommended_key:
            self.recommended_solns.append(solution)
            self.intermediate_budgets.append(self.budget.used)
            self._last_recommended_key = key

    def solve(self, problem: Problem) -> None:  # noqa: D102
        feas_tol = self.factors["feas_tol"]
        # Designate random number generator for random sampling.
        find_next_soln_rng = self.rng_list[1]
        self.problem = problem
        self._evaluated_solutions = {}
        self._last_recommended_key = None
        # Start at initial solution and record as best.
        new_x = problem.factors["initial_solution"]
        new_solution = self.create_new_solution(new_x, problem)
        
        best_solution = new_solution
        x0 = new_solution.x
        remaining_budget = problem.factors["budget"] - self.budget.used
        maxfev = int(remaining_budget//self.factors['sample_size'])

        # Prepare other variables in the loop.
        #sample_size = self.factors["sample_size"]
        #stoch_constraint_range = range(problem.n_stochastic_constraints)
        
        c = []
        
        #get constraints
        if self.problem.get_deterministic_equality_constraints(tuple(new_x)) != None:
            n_eq = len(np.atleast_1d(self.problem.get_deterministic_equality_constraints(tuple(new_x))))
            c_eq = NonlinearConstraint(
                self.get_equality_constraints,
                lb = np.zeros(n_eq),
                ub = np.zeros(n_eq)
                )
            c.append(c_eq)
        if self.problem.get_deterministic_inequality_constraints(tuple(new_x)) != None:
            n_ineq = len(np.atleast_1d(self.problem.get_deterministic_inequality_constraints(tuple(new_x))))
            c_ineq = NonlinearConstraint(
                self.get_inequality_constraints,
                lb = -np.inf,
                ub = np.zeros(n_ineq)
                )
            c.append(c_ineq)
            
            
        bounds = Bounds(problem.lower_bounds, problem.upper_bounds)


        # Sequentially generate random solutions and simulate them.
        #while True:
        # use COBYLA to solve 
        res = minimize(self.simulate, 
                       x0,
                       method = "COBYQA", 
                       constraints = c, 
                       bounds = bounds,
                       options={"maxfev": maxfev,"feasibility_tol": feas_tol},
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
