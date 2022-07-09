"""
Summary
-------
PGD
projected gradient descent
"""
from tkinter import N
from sklearn.metrics import euclidean_distances
from base import Solver
from numpy.linalg import norm
import numpy as np
import math
import warnings
warnings.filterwarnings("ignore")


class PGD(Solver):
    """
    Description.

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
    rng_list : list of rng.MRG32k3a objects
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
    def __init__(self, name="PGD", fixed_factors={}):
        self.name = name
        self.objective_type = "single"
        self.constraint_type = "box"
        self.variable_type = "continuous"
        self.gradient_needed = False
        self.specifications = {
            "crn_across_solns": {
                "description": "Use CRN across solutions?",
                "datatype": bool,
                "default": True
            },
            "r": {
                "description": "number of replications taken at each solution",
                "datatype": int,
                "default": 30
            },
            "theta": {
                "description": "Constant in the Armijo condition.",
                "datatype": int,
                "default": 0.2
            },
            "gamma": {
                "description": "Constant for shrinking the step size.",
                "datatype": int,
                "default": 0.8
            },
            "alpha_max": {
                "description": "Maximum step size.",
                "datatype": int,
                "default": 10
            },
            "alpha_0": {
                "description": "Initial step size.",
                "datatype": int,
                "default": 1
            },
            "epsilon_f": {
                "description": "Additive constant in the Armijo condition.",
                "datatype": int,
                "default": 1  # In the paper, this value is estimated for every epoch but a value > 0 is justified in practice.
            },
            "sensitivity": {
                "description": "Shrinking scale for variable bounds.",
                "datatype": float,
                "default": 10**(-7)
            },
            "lambda": {
                "description": "magnifying factor for n_r inside the finite difference function",
                "datatype": int,
                "default": 2
            },
        }
        self.check_factor_list = {
            "crn_across_solns": self.check_crn_across_solns,
            "r": self.check_r,
            "theta": self.check_theta,
            "gamma": self.check_gamma,
            "alpha_max": self.check_alpha_max,
            "alpha_0": self.check_alpha_0,
            "epsilon_f": self.check_epsilon_f,
            "sensitivity": self.check_sensitivity,
            "lambda": self.check_lambda
        }
        super().__init__(fixed_factors)

    def check_r(self):
        return self.factors["r"] > 0

    def check_theta(self):
        return self.factors["theta"] > 0 & self.factors["theta"] < 1

    def check_gamma(self):
        return self.factors["gamma"] > 0 & self.factors["gamma"] < 1

    def check_alpha_max(self):
        return self.factors["alpha_max"] > 0

    def check_alpha_0(self):
        return self.factors["alpha_0"] > 0

    def check_epsilon_f(self):
        return self.factors["epsilon_f"] > 0

    def check_sensitivity(self):
        return self.factors["sensitivity"] > 0
    
    def check_lambda(self):
        return self.factors["lambda"] > 0

    def solve(self, problem):
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

        # Default values.
        r = self.factors["r"]
        theta = self.factors["theta"]
        gamma = self.factors["gamma"]
        alpha_max = self.factors["alpha_max"]
        alpha_0 = self.factors["alpha_0"]
        epsilon_f = self.factors["epsilon_f"]

        # Shrink the bounds to prevent floating errors.
        lower_bound = np.array(problem.lower_bounds) + np.array((self.factors['sensitivity'],) * problem.dim)
        upper_bound = np.array(problem.upper_bounds) - np.array((self.factors['sensitivity'],) * problem.dim)

        # Initialize stepsize.
        alpha = alpha_0

        # Start with the initial solution.
        new_solution = self.create_new_solution(problem.factors["initial_solution"], problem)
        problem.simulate(new_solution, r)
        expended_budget += r
        best_solution = new_solution
        recommended_solns.append(new_solution)
        intermediate_budgets.append(expended_budget)

        while expended_budget < problem.factors["budget"]:
            new_x = new_solution.x
            # Check variable bounds.
            forward = [int(new_x[i] == lower_bound[i]) for i in range(problem.dim)]
            backward = [int(new_x[i] == upper_bound[i]) for i in range(problem.dim)]
            # BdsCheck: 1 stands for forward, -1 stands for backward, 0 means central diff.
            BdsCheck = np.subtract(forward, backward)

            if problem.gradient_available:
                # Use IPA gradient if available.
                grad = -1 * problem.minmax[0] * (new_solution.det_objectives_gradients + new_solution.objectives_gradients_mean)[0]
            else:
                # Use finite difference to estimate gradient if IPA gradient is not available.
                grad = self.finite_diff(new_solution, BdsCheck, problem, alpha, r)
                expended_budget += (2 * problem.dim - np.sum(BdsCheck != 0)) * r
                # A while loop to prevent zero gradient
                while np.all((grad == 0)):
                    if expended_budget > problem.factors["budget"]:
                        break
                    grad = self.finite_diff(new_solution, BdsCheck, problem, alpha, r)
                    expended_budget += (2 * problem.dim - np.sum(BdsCheck != 0)) * r
                    # Update r after each iteration.
                    r = int(self.factors["lambda"] * r)

            # Get the projected gradient.
            proj_grad = self.project_grad(grad)

            # Adjust the step size to respect box constraints if necessary.
            temp_steps = list()
            for i in problem.dim:
                temp_x = new_x[i] - alpha * proj_grad[i]
                if temp_x < lower_bound[i]:
                    temp_steps.append((new_x[i] - lower_bound[i]) / proj_grad[i])
                elif temp_x > upper_bound[i]:
                    temp_steps.append((new_x[i] - upper_bound[i]) / proj_grad[i])
            
            # Update alpha to be the maximum stepsize possible.
            alpha = min(temp_steps)

            # Calculate the candidate solution.
            candidate_x = new_x - alpha * proj_grad
            candidate_solution = self.create_new_solution(tuple(candidate_x), problem)

            # Use r simulated observations to estimate the objective value.
            problem.simulate(candidate_solution, r)
            expended_budget += r

            # Check the modified Armijo condition for sufficient decrease.
            if (-1 * problem.minmax[0] * candidate_solution.objectives_mean) <= (-1 * problem.minmax[0] * new_solution.objectives_mean - alpha * theta * norm(proj_grad)**2 + 2 * epsilon_f):
                # Successful step.
                new_solution = candidate_solution
                alpha = min(alpha_max, alpha / gamma)
            else:
                # Unsuccessful step.
                new_solution = candidate_solution
                alpha = gamma * alpha
            
            # Append new solution.
            if (problem.minmax[0] * new_solution.objectives_mean > problem.minmax[0] * best_solution.objectives_mean):
                best_solution = new_solution
                recommended_solns.append(new_solution)
                intermediate_budgets.append(expended_budget)

        return recommended_solns, intermediate_budgets

    # Finite difference for approximating gradients.
    def finite_diff(self, new_solution, BdsCheck, problem, stepsize, r):
        lower_bound = problem.lower_bounds
        upper_bound = problem.upper_bounds
        fn = -1 * problem.minmax[0] * new_solution.objectives_mean
        new_x = new_solution.x
        # Store values for each dimension.
        FnPlusMinus = np.zeros((problem.dim, 3))
        grad = np.zeros(problem.dim)

        for i in range(problem.dim):
            # Initialization.
            x1 = list(new_x)
            x2 = list(new_x)
            # Forward stepsize.
            steph1 = stepsize
            # Backward stepsize.
            steph2 = stepsize

            # Check variable bounds.
            if x1[i] + steph1 > upper_bound[i]:
                steph1 = np.abs(upper_bound[i] - x1[i])
            if x2[i] - steph2 < lower_bound[i]:
                steph2 = np.abs(x2[i] - lower_bound[i])

            # Decide stepsize.
            # Central diff.
            if BdsCheck[i] == 0:
                FnPlusMinus[i, 2] = min(steph1, steph2)
                x1[i] = x1[i] + FnPlusMinus[i, 2]
                x2[i] = x2[i] - FnPlusMinus[i, 2]
            # Forward diff.
            elif BdsCheck[i] == 1:
                FnPlusMinus[i, 2] = steph1
                x1[i] = x1[i] + FnPlusMinus[i, 2]
            # Backward diff.
            else:
                FnPlusMinus[i, 2] = steph2
                x2[i] = x2[i] - FnPlusMinus[i, 2]
            x1_solution = self.create_new_solution(tuple(x1), problem)
            if BdsCheck[i] != -1:
                problem.simulate_up_to([x1_solution], r)
                fn1 = -1 * problem.minmax[0] * x1_solution.objectives_mean
                # First column is f(x+h,y).
                FnPlusMinus[i, 0] = fn1
            x2_solution = self.create_new_solution(tuple(x2), problem)
            if BdsCheck[i] != 1:
                problem.simulate_up_to([x2_solution], r)
                fn2 = -1 * problem.minmax[0] * x2_solution.objectives_mean
                # Second column is f(x-h,y).
                FnPlusMinus[i, 1] = fn2

            # Calculate gradient.
            if BdsCheck[i] == 0:
                grad[i] = (fn1 - fn2) / (2 * FnPlusMinus[i, 2])
            elif BdsCheck[i] == 1:
                grad[i] = (fn1 - fn) / FnPlusMinus[i, 2]
            elif BdsCheck[i] == -1:
                grad[i] = (fn - fn2) / FnPlusMinus[i, 2]

        return grad

    # Euclidean projection of a vector onto the probability simplex.
    # Referencing Wang and Carreira-Perpinan (2013)
    def proj_prob_simplex(self, x, problem):
        # Sort the vector.
        sorted_x = -np.sort(-np.array(x))
        j = problem.dim
        while j >= 1:
            if sorted_x[j - 1] + 1/j * (1 - sum(sorted_x[:j])):
                rho = j
                break
            else:
                j -= 1
        lam = 1 / rho * (1 - sum(sorted_x[:rho]))
        return [max(i + lam, 0) for i in x]
    
    
    def project_grad(self, grad):
        """
        Project the gradient onto the hyperplane H: sum{x_i} = 0.
        """
        n = len(grad)
        # Generate the projection matrix.
        proj_mat = np.identity(n) - 1/n * np.ones((n, n))
        # Perform the projection.
        proj_grad = np.matmul(proj_mat, grad)
        return proj_grad

        

