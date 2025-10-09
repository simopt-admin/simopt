# https://github.com/bodono/apgpy
from __future__ import annotations

import numpy as np
import cvxpy as cp

# import matplotlib.pyplot as plt
from collections.abc import Callable
from simopt.utils import classproperty
# from apgwrapper import NumpyWrapper
# from functools import partial

from simopt.base import (
    ConstraintType,
    ObjectiveType,
    Solver,
    VariableType,
)


class FCSA(Solver):  # noqa: N801
    @classproperty
    def objective_type(cls) -> ObjectiveType:
        return ObjectiveType.SINGLE

    @classproperty
    def constraint_type(cls) -> ConstraintType:
        return ConstraintType.STOCHASTIC

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
            "h": {
                "description": "difference in finite difference gradient",
                "datatype": float,
                "default": 0.1,
            },
            "step_type": {
                "description": "constant or decaying step size?",
                "datatype": str,
                "default": "const",
            },
            "step_mult": {
                "description": "value of const step size or multiplier of k for decaying",
                "datatype": float,
                "default": 0.1,
            },
            "tolerance": {
                "description": "tolerence function",
                "datatype": float,  # TODO: change to Callable
                "default": 0.01,
            },
            
            "search_direction":{
                "description": "determines how solver finds the search direction for the next itterate. Can be FCSA, CSA-M, or CSA",
                "datatype": str,
                "default": "FCSA",
            },
            "normalize_grads":{
                "description": "normalize gradients used for search direction calculations?",
                "datatype": bool,
                "default": True,
            },
            "feas_const":{
                "description": "feasibility constant used to relax objective constraint in the FCSA search problem",
                "datatype": float,
                "default": 0.0,
            },
            "feas_score":{
                "description": "degree of feasibility score used to relax objective constraint in the FCSA search problem",
                "datatype": int,
                "default": 2,
            },
            "report_all_solns":{
                "description": "report all incumbent solutions instead of only reccomended?",
                "datatype": bool,
                "default": False,
            },
            

        }


    # TODO: eliminate this
    def return_true(self) -> bool:
        return True

    @property
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "crn_across_solns": self.check_crn_across_solns,  # type: ignore
            "r": self.check_r,
            "h": self.return_true,
            "step_mult": self.check_step_mult,
            "tolerance": self.return_true,
            "feas_const": self.check_feas_const,
            "search_direction": self.check_search_direction,
            "normalize_grads": self.return_true,
            "feas_score": self.check_feas_score,
            "report_all_solns": self.return_true,
            "step_type": self.check_step_type
        }

    """
    modified Cooperative Stochastic Approximation method by Lan, G. and Zhou Z.
    by improving multiple constraints at the same time
    """

    def __init__(
        self, name: str = "FCSA", fixed_factors: dict | None = None
    ) -> None:

        super().__init__(name, fixed_factors)


    def check_r(self)-> None:
        if self.factors["r"] <= 0:
            raise ValueError("Number of replications must be greater than 0.")
    
    def check_step_type(self)-> None:
        if self.factors["step_type"] not in ("const", "decay"):
            raise ValueError("Step size type not supported. Choose 'const' or 'decay'.")
            
    def check_step_mult(self)-> None:
        if self.factors["step_mult"] <= 0:
            raise ValueError("Step size multiplier must be positive.")

    def check_feas_const(self )-> None:
        if self.factors["feas_const"] < 0:
            raise ValueError("Feasibility constant cannot be negative.")
    
    def check_search_direction(self) -> None:
        if self.factors["search_direction"] not in ("FCSA", "CSA-M", "CSA"):
            raise ValueError("Invalid search direction factor. Must be 'FCSA', 'CSA-M', or 'CSA'.")
        
    def check_feas_score(self)-> None:
        if self.factors["feas_score"] <= 0:
            raise ValueError("Feasibility score must be a positive integer.")

    def get_simulated_values(self, problem, x, value="both"):
        """
        getting either sample path or gradient. The return "value"
        can be specified to "val"|"gradient"|"both"
        """
        r = self.factors["r"]
        sol = self.create_new_solution(tuple(x), problem)
        problem.simulate(sol, r)
        budget = 0

        # getting the function evaluation
        if (value == "both") or (value == "val"):
            budget += r
            Fval = -1 * problem.minmax[0] * sol.objectives_mean

        if (value == "both") or (value == "gradient"):
            if problem.gradient_available:
                # Use IPA gradient if available.
                gradient = (
                    -1 * problem.minmax[0] * sol.objectives_gradients_mean[0]
                )
            else:
                gradient, budget_spent = self.get_FD_grad(
                    x, problem, self.factors["h"], self.factors["r"]
                )
                budget += budget_spent

        if value == "val":
            return Fval, budget
        elif value == "gradient":
            return gradient, budget
        else:
            return Fval, gradient, budget

    def step_f(self, k: int) -> float:
        """
        take in the current iteration k
        """
        mult = self.factors["step_mult"]
        if self.factors["step_type"] == "const":
            step = mult
        else: #decaying step size
            step = 1/(mult*k)
        
        
        return step

    def get_FD_grad(self, x, problem, h, r):
        """
        find a finite difference gradient from the problem at
        the point x
        """
        x = np.array(x)
        d = len(x)

        if d == 1:
            # xsol = self.create_new_solution(tuple(x), problem)
            x1 = x + h / 2
            x2 = x - h / 2

            x1 = self.create_new_solution(tuple(x1), problem)
            problem.simulate(x1, r)
            f1 = -1 * problem.minmax[0] * x1.objectives_mean

            x2 = self.create_new_solution(tuple(x2), problem)
            problem.simulate(x2, r)
            f2 = -1 * problem.minmax[0] * x2.objectives_mean
            grad = (f1 - f2) / h
        else:
            I = np.eye(d)
            grad = 0

            for i in range(d):
                x1 = x + h * I[:, i] / 2
                x2 = x - h * I[:, i] / 2

                x1 = self.create_new_solution(tuple(x1), problem)
                problem.simulate_up_to([x1], r)
                f1 = -1 * problem.minmax[0] * x1.objectives_mean

                x2 = self.create_new_solution(tuple(x2), problem)
                problem.simulate_up_to([x2], r)
                f2 = -1 * problem.minmax[0] * x2.objectives_mean

                grad += ((f1 - f2) / h) * I[:, i]

        return grad, (2 * d * r)

    def get_violated_constraints_grads(self, constraints_results, grads):
        """
        get all violated constraints gradients
        """
        n_cons, n = grads.shape
        violated_cons_grads = []
        violated_cons = [] #lhs of violated constraints

        for i in range(n_cons):
            if constraints_results[i] > self.factors["tolerance"]:
                violated_cons_grads.append(grads[i])
                violated_cons.append(constraints_results[i])

        return np.array(violated_cons_grads), np.array(violated_cons)

    def get_constraints_dir(self, grads, k, obj_grad = None, feas_score=None):
        """
        compute search direction on infeasible itterations one of two ways:
            1: Solve LP of all violated constraint gradients
            2. Solve NLP of all violated constraint gradients plus objective constraint

        solve for d s.t.

        max theta s.t. gi^Td >= theta
        """
        #Option 1 (CSA-M)
        if self.factors["search_direction"] == "CSA-M":
            n_violated_cons, n = grads.shape
            if self.factors["normalize_grads"]:
                con_grads = grads / np.linalg.norm(grads, axis=1).reshape(
                    n_violated_cons, 1
                )
            else:
                con_grads = grads
            
            #solve LP
            direction = cp.Variable(n)
            theta = cp.Variable()
            objective = cp.Maximize(theta)
            constraints = [cp.norm(direction,2) <=1]  #add constraint that direction must be a unit vector
            for grad in con_grads:
                constraints += [-1*grad @ direction >= theta]
            prob = cp.Problem(objective, constraints)
            prob.solve()

            d = direction.value
        
        #Option 2 (FCSA)
        else: 
            n_violated_cons, n = grads.shape
            if self.factors["normalize_grads"]:
                con_grads = grads / np.linalg.norm(grads, axis=1).reshape(
                    n_violated_cons, 1
                )
                obj_grad = obj_grad/np.linalg.norm(obj_grad)
            else:
                con_grads = grads
            feas_constant = self.factors["feas_const"]
            # solve NLP
            direction = cp.Variable(n)
            theta = cp.Variable()
            objective = cp.Maximize(theta)
            constraints = [
                           -1*obj_grad @ direction >= theta - feas_constant*feas_score,
                           cp.norm(direction,2) <=1  #add constraint that direction must be a unit vector
                           ]
            # stochastic constraint constraints
            for grad in con_grads:
                constraints += [-1*grad @ direction >= theta]
            prob = cp.Problem(objective, constraints)
            prob.solve()

            d = direction.value
        
        return d



    def prox_fn(self, a, cur_x, Ci, di, Ce, de, lower, upper):
        """
        prox function for CSA
        'a' is an input, typically use 'step*grad'
        solve the minimization problem (z)

        aTz + 0.5*||z - cur_x||^2

        by default, use the euclidean distance
        """
        n = len(cur_x)
        z = cp.Variable(n)

        objective = cp.Minimize(a @ z + 0.5 * (cp.norm(cur_x - z) ** 2))
        constraints = []

        if (lower is not None) and (lower > -np.inf).all():
            constraints += [z >= lower]
        if (upper is not None) and (upper < np.inf).all():
            constraints += [z <= upper]

        if (Ci is not None) and (di is not None):
            constraints += [Ci @ z <= di]
        if (Ce is not None) and (de is not None):
            constraints += [Ce @ z == de]

        prob = cp.Problem(objective, constraints)
        prob.solve()

        return z.value

    def solve(self, problem):
        r = self.factors["r"]
        
        
        Ci = problem.Ci
        di = problem.di
        Ce = problem.Ce
        de = problem.de

        lower = np.array(problem.lower_bounds)
        upper = np.array(problem.upper_bounds)


        recommended_solns = []
        intermediate_budgets = []
        expended_budget = 0

        k = 0  # iteration

        # Start with the initial solution.
        new_solution = self.create_new_solution(
            problem.factors["initial_solution"], problem
        )

        # Use r simulated observations to estimate the objective value.
        problem.simulate(new_solution, r)
        expended_budget += r
        best_solution = new_solution
        recommended_solns.append(new_solution)
        intermediate_budgets.append(expended_budget)
        feasible_found = False

        while expended_budget < problem.factors["budget"]:
            cur_x = new_solution.x
            #cur_x = best_solution.x

            # check if the constraints are violated
            if problem.n_stochastic_constraints > 0:
                # constraint_results = problem.stoch_constraint(cur_x) #multiple dim of constraints in the form E[Y] <= 0
                constraint_results = new_solution.stoch_constraints_mean
                # print("Constraint Results:", constraint_results)
                # print(constraint_results)
                is_violated = (
                    max(constraint_results) > self.factors["tolerance"]
                )  # 0.01
                # print("constraint violation: ", max(constraint_results))
            else:
                # problems with no stoch constraints
                is_violated = False

            # if the constraints are violated, then improve the feasibility
            if is_violated:
                # get search direction
                if self.factors["search_direction"] == "CSA": # set d to grad of most violated constraint
                    violated_index = np.argmax(constraint_results)
                    grad = new_solution.stoch_constraints_gradients_mean[
                        violated_index
                    ]
                    if self.factors["normalize_grads"]:
                        d = grad/np.linalg.norm(grad)
                    else:
                        d = grad
                elif self.factors["search_direction"] == "CSA-M": # solve search direction LP with all violated constraints
                    grads = np.array(new_solution.stoch_constraints_gradients_mean)
                    violated_grads, violated_cons = self.get_violated_constraints_grads(
                        constraint_results, grads
                    )
                    d = -1*self.get_constraints_dir(violated_grads,k)
                else: #FCSA: solve search direction problem with all violated constraint and objective gradient
                    grads = np.array(new_solution.stoch_constraints_gradients_mean)
                    violated_grads, violated_cons = self.get_violated_constraints_grads(
                        constraint_results, grads
                    )
                    if problem.gradient_available:
                        # Use IPA gradient if available.                        
                        obj_grad = (
                            -1
                            * problem.minmax[0]
                            * new_solution.objectives_gradients_mean[0]
                        )
                    else:
                        # Use finite difference to estimate gradient if IPA gradient is not available.
                        obj_grad, budget_spent = self.get_FD_grad(
                            cur_x, problem, self.factors["h"], self.factors["r"]
                        )
                        expended_budget += budget_spent
                    feas_score =np.linalg.norm(violated_cons, ord=self.factors["feas_score"])
                    d = -1*self.get_constraints_dir(violated_grads,k, obj_grad,feas_score)

            else:

                # if constraints are not violated, then compute gradients
                # computeing the gradients
                if problem.gradient_available:
                    # Use IPA gradient if available.
                    grad = (
                        -1
                        * problem.minmax[0]
                        * new_solution.objectives_gradients_mean[0]
                    )
                    # normalize gradient
                    if self.factors["normalize_grads"]:
                        grad = grad/np.linalg.norm(grad)
                else:
                    # Use finite difference to estimate gradient if IPA gradient is not available.
                    # grad, budget_spent = self.finite_diff(new_solution, problem, r, stepsize = alpha)
                    grad, budget_spent = self.get_FD_grad(
                        cur_x, problem, self.factors["h"], self.factors["r"]
                    )
                    expended_budget += budget_spent
                if self.factors["normalize_grads"]:
                    d = grad/np.linalg.norm(grad)
                else:
                    d = grad
            
            t = self.step_f(k)

            new_x = self.prox_fn(t * d, cur_x, Ci, di, Ce, de, lower, upper)
            candidate_solution = self.create_new_solution(tuple(new_x), problem)
            # Use r simulated observations to estimate the objective value.
            problem.simulate(candidate_solution, r)
            expended_budget += r

            new_solution = candidate_solution
 
            # check feasibility of new solution
            constraint_results = new_solution.stoch_constraints_mean
            # check if the constraints are violated
            if problem.n_stochastic_constraints > 0:
                is_new_violated = (
                    max(constraint_results) > self.factors["tolerance"]
                )
            else:
                is_new_violated = False
            if self.factors["report_all_solns"]:
                best_solution = new_solution
                recommended_solns.append(new_solution)
                intermediate_budgets.append(expended_budget)
            
            else:
                # reccomend all solutions until a feasible solution has been found. Then only reccomend feasible solutions that improve the objective.
                if not is_new_violated:
                    feasible_found = True # feasible solution has been found
    
                if not feasible_found: # reccomend all solutions until a feasible solution has been found
                    best_solution = new_solution
                    recommended_solns.append(new_solution)
                    intermediate_budgets.append(expended_budget)
                else:
                    if not is_new_violated: # reccomended solutions must be feasible
                        if ( problem.minmax[0] * new_solution.objectives_mean
                                    > problem.minmax[0] * best_solution.objectives_mean
                                ):
                            # reccomended solution of objective has improved
                            best_solution = new_solution
                            recommended_solns.append(new_solution)
                            intermediate_budgets.append(expended_budget)
            k += 1




        return recommended_solns, intermediate_budgets
