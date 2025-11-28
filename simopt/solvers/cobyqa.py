"""Fully Cooperative Stochastic Approximation (FCSA) solver.

This solver is based on the paper 'Diagnostic Tools for Evaluating Solvers for
Stochastically Constrained Simulation Optimization Problems' by Felice, N.,
D. J. Eckman, S. G. Henderson, and S. Shashaani
"""

from __future__ import annotations

from typing import Annotated, ClassVar

import numpy as np
from pydantic import Field



import cobyqa as cq

from simopt.base import (
    ConstraintType,
    ObjectiveType,
    Problem,
    Solution,
    Solver,
    SolverConfig,
    VariableType,
)


from scipy.optimize import (
    Bounds,
    LinearConstraint,
    NonlinearConstraint,
    OptimizeResult,
)

EPSILON = np.finfo(float).eps


class COBYQAConfig(SolverConfig):
    """Configuration for the COBYQA solver."""


    report_all_solns: Annotated[
        bool,
        Field(default=False, description="report all incumbent solutions?"),
    ]



class COBYQA(Solver):
    "COBYQA"

    name: str = "COBYQA"
    config_class: ClassVar[type[COBYQAConfig]] = COBYQAConfig
    class_name_abbr: ClassVar[str] = "COBYQA"
    class_name: ClassVar[str] = "COBYQA"
    objective_type: ClassVar[ObjectiveType] = ObjectiveType.SINGLE
    constraint_type: ClassVar[ConstraintType] = ConstraintType.DETERMINISTIC
    variable_type: ClassVar[VariableType] = VariableType.CONTINUOUS
    gradient_needed: ClassVar[bool] = False
    
    def simulate(self, x, problem: Problem):
        # simulate problem at x to obtain estimate of the objective (single objective only)
        x = np.array(x, dtype=float)
        solution = self.create_new_solution(x, problem)
        self.budget.request(10)
        problem.simulate(solution, 10)
        
        obj = -problem.minmax[0] * solution.objectives_mean[0]
        print("calculated objective", obj)
        print('current x', x)

        return obj        
    
    def solve(self, problem: Problem) -> None:  # noqa: D102
        
        # temp hard code for network 
        print('test print 1')
        x0 = problem.config.initial_solution
        solution = self.create_new_solution(x0, problem)
        self.recommended_solns.append(solution)
        self.intermediate_budgets.append(self.budget.used)
        constraints = [
            LinearConstraint(problem.sum_constraint(), 1.0, 1.0),
        ]

        bounds = Bounds((EPSILON,)*problem.dim, problem.upper_bounds)
        options = {
            "maxfev": 1000/10
        }
        res = cq.minimize(self.simulate, x0, problem,bounds = bounds, constraints=constraints, options=options)
        print('optimal x',res.x)
        self.recommended_solns.append(solution)
        self.intermediate_budgets.append(self.budget.used)
        print('optimal x',res.x)
        print('test print')
    
        

