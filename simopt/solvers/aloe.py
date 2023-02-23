"""
Summary
-------
ALOE
The solver is a stochastic line search algorithm  with the gradient estimate recomputed in each iteration,
whether or not a step is accepted. The algorithm includes the relaxation of the Armijo condition by
an additive constant.
A detailed description of the solver can be found `here <https://simopt.readthedocs.io/en/latest/aloe.html>`_.
"""
from numpy.linalg import norm
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from ..base import Solver


class ALOE(Solver):
    """
    Adaptive Line-search with Oracle Estimations

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
    def __init__(self, name="ALOE", fixed_factors=None):
        if fixed_factors is None:
            fixed_factors = {}
        self.name = name
        self.objective_type = "single"
        self.constraint_type = "box"
        self.variable_type = "continuous"
        self.gradient_needed = False
        self.specifications = {
            "crn_across_solns": {
                "description": "use CRN across solutions?",
                "datatype": bool,
                "default": True
            },
            "r": {
                "description": "number of replications taken at each solution",
                "datatype": int,
                "default": 30
            },
            "theta": {
                "description": "constant in the Armijo condition",
                "datatype": float,
                "default": 0.2
            },
            "gamma": {
                "description": "constant for shrinking the step size",
                "datatype": float,
                "default": 0.8
            },
            "alpha_max": {
                "description": "maximum step size",
                "datatype": int,
                "default": 10
            },
            "alpha_0": {
                "description": "initial step size",
                "datatype": int,
                "default": 1
            },
            "epsilon_f": {
                "description": "additive constant in the Armijo condition",
                "datatype": int,
                "default": 1  # In the paper, this value is estimated for every epoch but a value > 0 is justified in practice.
            },
            "sensitivity": {
                "description": "shrinking scale for variable bounds",
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

        # Upper bound and lower bound.
        lower_bound = np.array(problem.lower_bounds)
        upper_bound = np.array(problem.upper_bounds)

        # Initialize stepsize.
        alpha = alpha_0

        # Start with the initial solution.
        new_solution = self.create_new_solution(problem.factors["initial_solution"], problem)
        recommended_solns.append(new_solution)
        intermediate_budgets.append(expended_budget)
        problem.simulate(new_solution, r)
        expended_budget += r
        best_solution = new_solution

        while expended_budget < problem.factors["budget"]:
            new_x = new_solution.x
            # Check variable bounds.
            forward = np.isclose(new_x, lower_bound, atol = self.factors["sensitivity"]).astype(int)
            backward = np.isclose(new_x, upper_bound, atol = self.factors["sensitivity"]).astype(int)
            # BdsCheck: 1 stands for forward, -1 stands for backward, 0 means central diff.
            BdsCheck = np.subtract(forward, backward)

            if problem.gradient_available:
                # Use IPA gradient if available.
                grad = -1 * problem.minmax[0] * new_solution.objectives_gradients_mean[0]
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
                    # Update sample size after each iteration.
                    r = int(self.factors["lambda"] * r)

            # Calculate the candidate solution and adjust the solution to respect box constraints.
            candidate_x = list()
            for i in range(problem.dim):
                candidate_x.append(min(max((new_x[i] - alpha * grad[i]), lower_bound[i]), upper_bound[i]))
            candidate_solution = self.create_new_solution(tuple(candidate_x), problem)

            # Use r simulated observations to estimate the objective value.
            problem.simulate(candidate_solution, r)
            expended_budget += r

            # Check the modified Armijo condition for sufficient decrease.
            if (-1 * problem.minmax[0] * candidate_solution.objectives_mean) <= (-1 * problem.minmax[0] * new_solution.objectives_mean - alpha * theta * norm(grad)**2 + 2 * epsilon_f):
                # Successful step.
                new_solution = candidate_solution
                alpha = min(alpha_max, alpha / gamma)
            else:
                # Unsuccessful step.
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
