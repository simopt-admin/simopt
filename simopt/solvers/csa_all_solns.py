# https://github.com/bodono/apgpy
from collections.abc import Callable
import numpy as np
import cvxpy as cp

# import matplotlib.pyplot as plt
from simopt.utils import classproperty
# from apgwrapper import NumpyWrapper
# from functools import partial

from simopt.base import (
    ConstraintType,
    ObjectiveType,
    Solver,
    VariableType,
)


class CSA(Solver):
    """
    Cooperative Stochastic Approximation method by Lan, G. and Zhou Z.
    """

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
                "default": False,
            },
            "LSmethod": {
                "description": "method",
                "datatype": str,
                "default": "backtracking",
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
            "step_f": {
                "description": "step size function",
                "datatype": Callable,
                "default": cls.default_step_f,
            },
            "tolerance": {
                "description": "tolerence function",
                "datatype": float,  # TODO: change to Callable
                "default": 0.01,
            },
            "max_iters": {
                "description": "maximum iterations",
                "datatype": int,
                "default": 300,
            },
        }

    # TODO: eliminate this
    def return_true(self) -> bool:
        return True

    @property
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "crn_across_solns": self.check_crn_across_solns,
            "LSmethod": self.return_true,
            "r": self.check_r,
            "h": self.return_true,
            "step_f": self.return_true,
            "tolerance": self.return_true,
            "max_iters": self.check_max_iters,
        }

    def __init__(
        self, name: str = "CSA", fixed_factors: dict | None = None
    ) -> None:
        if fixed_factors is None:
            fixed_factors = {"max_iters": 300}

        self.name = name

        super().__init__(name, fixed_factors)

    def default_step_f(self, k: int) -> float:
        """
        take in the current iteration k
        """
        if k <= 10:
            d = k+1
        elif k <= 60:
            d = 50
        else:
            d = 70
        
        return .1

    def check_r(self) -> bool:
        return self.factors["r"] > 0

    def check_max_iters(self) -> bool:
        return self.factors["max_iters"] > 0

    def is_feasible(self, x, Ci, di, Ce, de, lower, upper) -> bool:
        """
        Check whether a solution x is feasible to the problem.

        Arguments
        ---------
        x : tuple
            a solution vector
        problem : Problem object
            simulation-optimization problem to solve
        tol: float
            Floating point comparison tolerance
        """
        x = np.asarray(x)

        # Check if x is within bounds
        if lower is not None and np.all(x >= lower):
            return False
        if upper is not None and np.all(x <= upper):
            return False
        # Check if x satisfies the constraints
        if (Ci is not None) and (di is not None) and np.all(Ci @ x <= di):
            return False
        if (
            (Ce is not None)
            and (de is not None)
            and np.allclose(np.dot(Ce, x), de)
        ):
            return False
        # Return true if x is feasible
        return True

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
        max_iters = self.factors["max_iters"]
        r = self.factors["r"]
        #max_gamma = self.factors["max_gamma"]

        # t = 1 #first max step size
        dim = problem.dim

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

        # numiter = 0
        numviolated = 0

        while expended_budget < problem.factors["budget"]:
            cur_x = new_solution.x

            # check if the constraints are violated
            if problem.n_stochastic_constraints > 0:
                # constraint_results = problem.stoch_constraint(cur_x) #multiple dim of constraints in the form E[Y] <= 0
                constraint_results = new_solution.stoch_constraints_mean
                is_violated = (
                    max(constraint_results) > self.factors["tolerance"]
                )  # 0.01
            else:
                # problems with no stoch constraints
                is_violated = 0

            # if the constraints are violated, then improve the feasibility
            if is_violated:
                # find the gradient of the constraints
                #print(f"all constraint means: {constraint_results}")
                violated_index = np.argmax(constraint_results)
                #print(f"constraint {violated_index} most violated with value of {constraint_results[violated_index]}")
                grad = new_solution.stoch_constraints_gradients_mean[
                    violated_index
                ]
                #print(grad)
                numviolated += 1
                #temp
                t = self.factors["step_f"](self, k=k)
            else:
                # if constraints are not violated, then conpute gradients
                # computeing the gradients
                if problem.gradient_available:
                    # Use IPA gradient if available.
                    grad = (
                        -1
                        * problem.minmax[0]
                        * new_solution.objectives_gradients_mean[0]
                    )
                    # normalize gradient
                    #print(f"not violated, obj grad:{grad} ")
                else:
                    # Use finite difference to estimate gradient if IPA gradient is not available.
                    # grad, budget_spent = self.finite_diff(new_solution, problem, r, stepsize = alpha)
                    grad, budget_spent = self.get_FD_grad(
                        cur_x, problem, self.factors["h"], self.factors["r"]
                    )
                    expended_budget += budget_spent

            # step-size
            t = self.factors["step_f"](self, k=k)

            # new_x = cur_x + t * direction
            # normalize gradient
            grad = grad/np.linalg.norm(grad)
            #print('grad', grad)
            new_x = self.prox_fn(t * grad, cur_x, Ci, di, Ce, de, lower, upper)

            candidate_solution = self.create_new_solution(tuple(new_x), problem)
            # Use r simulated observations to estimate the objective value.
            problem.simulate(candidate_solution, r)
            expended_budget += r

            new_solution = candidate_solution
            recommended_solns.append(new_solution)
            intermediate_budgets.append(expended_budget)

            #Append new solution.
            if (
                problem.minmax[0] * new_solution.objectives_mean
                > problem.minmax[0] * best_solution.objectives_mean
            ):
                best_solution = new_solution
                # recommended_solns.append(candidate_solution)
                # recommended_solns.append(new_solution)
                # intermediate_budgets.append(expended_budget)
            #print("****New Accepted Solution****")
            #print('Solution:', new_solution.x)
            #print("objective val:" ,new_solution.objectives_mean)
            #print("k:" ,k)
            #print("budget:", expended_budget)
                
                
            # # new criteria for feasiblity and objective
            # if (
            #     best_solution is not None
            #     and problem.minmax * new_solution.objectives_mean
            #     >= problem.minmax * best_solution.objectives_mean
            #     and all(
            #         new_solution.stoch_constraints_mean[idx] <= max(best_solution.stoch_constraints_mean[idx],0)
            #         for idx in range(problem.n_stochastic_constraints)
            #     )
            # ):
            #     best_solution = new_solution
            #     # recommended_solns.append(candidate_solution)
            #     recommended_solns.append(new_solution)
            #     intermediate_budgets.append(expended_budget)
                
            #best_solution = new_solution
            # print("New best solution", best_solution.x)
            # print("New objective", best_solution.objectives_mean)
            # # recommended_solns.append(candidate_solution)
            # recommended_solns.append(new_solution)
            # intermediate_budgets.append(expended_budget)

            k += 1

        return recommended_solns, intermediate_budgets
