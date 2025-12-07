"""Fully Cooperative Stochastic Approximation (FCSA) solver.

This solver is based on the paper 'Diagnostic Tools for Evaluating Solvers for
Stochastically Constrained Simulation Optimization Problems' by Felice, N.,
D. J. Eckman, S. G. Henderson, and S. Shashaani
"""

from __future__ import annotations

from typing import Annotated, ClassVar

import numpy as np
from pydantic import Field

import os
import pandas as pd

import cobyqa as cq

from simopt.solver_utils.main import minimize
import matplotlib.pyplot as plt

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


    c: Annotated[
        float,
        Field(default=1, description="const used for adaptive sampling tolerance"),
    ]
    adaptive: Annotated[
        bool,
        Field(default=True, description="use adaptive sampling?"),
    ]
    n: Annotated[
        int,
        Field(default=10, description="initial sample size"),
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
        enough_replications = False
        
        n=self.n_replications
        c = self.c
        self.budget.request(n)
        problem.simulate(solution, n)
        #print("first simulations complete")
        if self.adaptive: 
            while not enough_replications:
                #print("n", n)
                # get objective gradient using finite differencing if not available
                grad = self._objective_grad(problem, solution,n)
                sd = solution.objectives_stderr[0]
                # print("grad", grad)
                # print("sd", sd)
                # test if n big enough
                if (abs(sd)/np.sqrt(n))< c*grad + EPSILON:
                    enough_replications = True
                else:
                    n+= 1
                    # run one replication of simulation
                    self.budget.request(1)
                    problem.simulate(solution, 1)
            #self.n_replications = n
            self.all_n.append(n)
        
        if self.n_eval > 2*10 + 1:    
            self.recommended_solns.append(solution)
            self.intermediate_budgets.append(self.budget.used)
            self.n_replications = n
        self.n_eval +=1
        
        obj = -problem.minmax[0] * solution.objectives_mean[0]
        
        # print("calculated objective", obj)
        # print('n', n)

        return obj   
    
    def _objective_grad(
        self, problem: Problem, solution: Solution,n) -> np.ndarray:
        if problem.gradient_available:
            grad = -problem.minmax[0] * solution.objectives_gradients_mean[0]
        else:
            # h = self.factors["h"]
            # r = self.factors["r"]
            h = .001
            grad = self._finite_difference(problem, np.array(solution.x), h, 10)
        norm = np.linalg.norm(grad)
        if norm == 0:
            norm = EPSILON
        grad /= norm   
        return grad
            
    def _finite_difference(
        self, problem: Problem, x: np.ndarray, h: float, r: int
    ) -> np.ndarray:
        if x.ndim == 1:
            solution = self.create_new_solution(tuple(x), problem)
            self.budget.request(r)
            problem.simulate(solution, r)
            return -problem.minmax[0] * solution.objectives_mean[0]

    # batched case
        values = np.zeros(x.shape[1])
        for j in range(x.shape[1]):
            solution = self.create_new_solution(tuple(x[:, j]), problem)
            self.budget.request(r)
            problem.simulate(solution, r)
            values[j] = -problem.minmax[0] * solution.objectives_mean[0]
        return values


    def _objective_at(self, problem: Problem, x: np.ndarray, r: int) -> float:
        solution = self.create_new_solution(tuple(x), problem)
        self.budget.request(r)
        problem.simulate(solution, r)
        return -problem.minmax[0] * solution.objectives_mean[0]
    
    def solve(self, problem: Problem) -> None:  # noqa: D102
        
        # temp hard code for network 
        #print('test print 1')
        x0 = problem.config.initial_solution
        # initial sample size 
        self.n_replications = self.config.n
        self.all_n = [self.n_replications]
        self.adaptive = self.config.adaptive
        self.c = self.config.c
        solution = self.create_new_solution(x0, problem)
        self.recommended_solns.append(solution)
        self.intermediate_budgets.append(self.budget.used)
        constraints = [
            LinearConstraint(problem.sum_constraint(), 1.0, 1.0),
        ]

        bounds = Bounds((EPSILON,)*problem.dim, problem.upper_bounds)
        options = {
            "radius_init": .05
        }
        self.n_eval = 0
        try:
            res = minimize(self.simulate, x0, problem,bounds = bounds, constraints=constraints, options=options)
            # print('optimal x',res.x)
            # print("after minimize, success =", res.success)
            # set all remaining budgets to found solution
            solution = self.create_new_solution(np.array(res.x), problem)
            self.recommended_solns.append(solution)
            self.intermediate_budgets.append(self.budget.used)
            

            for sol in self.recommended_solns:
                print(sol)
            # print('optimal x',res.x)
            # print(self.recommended_solns[-1].x)
            
            #self.save_results_to_excel(self, filename="fcsa_results.xlsx", sheet_name="run_log")
        except:
            print("Not solved... last solution is")
            print(self.recommended_solns[-1].x)
            
            #self.save_results_to_excel()
            
        
        #temp plotting stuff


        # # Create the plot
        # plt.figure()
        # plt.plot(self.intermediate_budgets, self.all_n)
        
        # # Add labels and title
        # plt.title("n vs budget")
        # plt.xlabel("budget")
        # plt.ylabel("sample size")
        
        # # Show the plot
        # plt.show()
            
    # def save_results_to_excel(self, filename="results.xlsx", sheet_name="Sheet1"):
    #     """
    #     Save self.all_n, self.intermediate_budgets, and self.c to an Excel file.
    #     Adds new columns to an existing file/sheet without overwriting existing columns.
    #     """
    #     len_n = len(self.all_n)
    #     len_b = len(self.intermediate_budgets)

    #     if len_b > len_n and len_n > 0:
    #         last_n = self.all_n[-1]
    #         pad_size = len_b - len_n
    #         self.all_n = self.all_n + [last_n] * pad_size


    #     # Recompute length after padding
    #     length = max(len(self.all_n), len(self.intermediate_budgets))
    #     data = {
    #         "intermediate_budgets": self.intermediate_budgets,
    #         "all_n": self.all_n,
    #         "c": [self.c] * length,  # repeat c so it lines up with rows
    #     }
    #     df_new = pd.DataFrame(data)
    #     print("saving results to file")
    
    #     # Load existing file/sheet if it exists
    #     if os.path.exists(filename):
    #         try:
    #             df_existing = pd.read_excel(filename, sheet_name=sheet_name)
    #         except ValueError:
    #             # File exists but sheet does not
    #             df_existing = pd.DataFrame()
    #     else:
    #         df_existing = pd.DataFrame()
    
    #     # If existing is empty, just write new data
    #     if df_existing.empty:
    #         df_final = df_new
    #     else:
    #         # Make sure we have enough rows to hold both old and new data
    #         max_len = max(len(df_existing), len(df_new))
    #         df_existing = df_existing.reindex(range(max_len))
    #         df_new = df_new.reindex(range(max_len))
    
    #         # Add each new column without overwriting existing ones
    #         for col in df_new.columns:
    #             new_col_name = col
    #             k = 1
    #             # If column name already exists, add suffix _1, _2, ...
    #             while new_col_name in df_existing.columns:
    #                 new_col_name = f"{col}_{k}"
    #                 k += 1
    #             df_existing[new_col_name] = df_new[col]
    
    #         df_final = df_existing
    
    #     # Write back to Excel (overwriting only this sheet with the merged data)
    #     with pd.ExcelWriter(
    #         filename,
    #         engine="openpyxl",
    #         mode="a" if os.path.exists(filename) else "w",
    #         if_sheet_exists="replace" if os.path.exists(filename) else None,
    #     ) as writer:
    #         df_final.to_excel(writer, sheet_name=sheet_name, index=False)

