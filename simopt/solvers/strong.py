"""
Summary
-------
STRONG
A trust-region-based algorithm that fits first- or second-order models through function evaluations taken within
a neighborhood of the incumbent solution.
"""
from base import Solver
from numpy.linalg import norm
import numpy as np
import math
import warnings
warnings.filterwarnings("ignore")


class STRONG(Solver):
    """
    A trust-region-based algorithm that fits first- or second-order models through function evaluations taken within a neighborhood of the incumbent solution.

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
    def __init__(self, name="STRONG", fixed_factors={}):
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
            "n0": {
                "description": "Initial sample size",
                "datatype": int,
                "default": 10
            },
            "n_r": {
                "description": "Number of replications taken at each solution",
                "datatype": int,
                "default": 10
            },
            "sensitivity": {
                "description": "shrinking scale for VarBds",
                "datatype": float,
                "default": 10**(-7)
            },
            "delta_threshold": {
                "description": "maximum value of the radius",
                "datatype": float,
                "default": 1.2
            },
            "delta_T": {
                "description": "initial size of trust region",
                "datatype": float,
                "default": 2
            },
            "eta_0": {
                "description": "the constant of accepting",
                "datatype": float,
                "default": 0.01
            },
            "eta_1": {
                "description": "the constant of more confident accepting",
                "datatype": float,
                "default": 0.3
            },
            "gamma_1": {
                "description": "the constant of shrinking the trust region",
                "datatype": float,
                "default": 0.9
            },
            "gamma_2": {
                "description": "the constant of expanding the trust region",
                "datatype": float,
                "default": 1.11
            },
            "lambda": {
                "description": "magnifying factor for n_r inside the finite difference function",
                "datatype": int,
                "default": 2
            }
        }
        self.check_factor_list = {
            "crn_across_solns": self.check_crn_across_solns,
            "n_r": self.check_n_r,
            "sensitivity": self.check_sensitivity,
            "delta_threshold": self.check_delta_threshold,
            "delta_T": self.check_delta_T,
            "eta_0": self.check_eta_0,
            "eta_1": self.check_eta_1,
            "gamma_1": self.check_gamma_1,
            "gamma_2": self.check_gamma_2,
            "lambda": self.check_lambda
        }
        super().__init__(fixed_factors)

    def check_n_r(self):
        return self.factors["n_r"] > 0

    def check_sensitivity(self):
        return self.factors["sensitivity"] > 0

    def check_delta_threshold(self):
        return self.factors["delta_threshold"] > 0

    def check_delta_T(self):
        return self.factors["delta_T"] > self.factors["delta_threshold"]

    def check_eta_0(self):
        return self.factors["eta_0"] > 0 and self.factors["eta_0"] < 1

    def check_eta_1(self):
        return self.factors["eta_1"] < 1 and self.factors["eta_1"] > self.factors["eta_0"]

    def check_gamma_1(self):
        return self.factors["gamma_1"] > 0 and self.factors["gamma_1"] < 1

    def check_gamma_2(self):
        return self.factors["gamma_2"] > 1

    def check_lambda(self):
        return self.factors["lambda"] > 1

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
        n0 = self.factors["n0"]
        n_r = self.factors["n_r"]
        delta_threshold = self.factors["delta_threshold"]
        delta_T = self.factors["delta_T"]
        eta_0 = self.factors["eta_0"]
        eta_1 = self.factors["eta_1"]
        gamma_1 = self.factors["gamma_1"]
        gamma_2 = self.factors["gamma_2"]
        lam = self.factors["lambda"]

        # Shrink the bounds to prevent floating errors.
        lower_bound = np.array(problem.lower_bounds) + np.array((self.factors['sensitivity'],) * problem.dim)
        upper_bound = np.array(problem.upper_bounds) - np.array((self.factors['sensitivity'],) * problem.dim)

        # Start with the initial solution.
        new_solution = self.create_new_solution(problem.factors["initial_solution"], problem)
        problem.simulate(new_solution, n0)
        expended_budget += n0
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

            # Stage I.
            if delta_T > delta_threshold:
                # Step 1: Build the linear model.
                NumOfEval = 2 * problem.dim - np.sum(BdsCheck != 0)
                grad, Hessian, cnt = self.finite_diff(new_solution, BdsCheck, 1, problem, n_r)
                expended_budget += NumOfEval * n_r * (sum(np.power(lam, i) for i in range(cnt)) - cnt + 1)

                # Step 2: Solve the subproblem.
                # Cauchy reduction.
                candidate_x = self.cauchy_point(grad, Hessian, new_x, problem)
                candidate_solution = self.create_new_solution(tuple(candidate_x), problem)

                # Step 3: Compute the ratio.
                # Use n_r simulated observations to estimate g_new.
                problem.simulate(candidate_solution, n_r)
                expended_budget += n_r
                # Find the old objective value and the new objective value.
                g_old = -1 * problem.minmax[0] * new_solution.objectives_mean
                g_new = -1 * problem.minmax[0] * candidate_solution.objectives_mean
                # Construct the polynomial.
                r_old = g_old
                r_new = g_old + np.matmul(np.subtract(candidate_x, new_x), grad) + 0.5 * np.matmul(np.matmul(np.subtract(candidate_x, new_x), Hessian), np.subtract(candidate_x, new_x))
                rho = (g_old - g_new) / (r_old - r_new)

                # Step 4: Update the trust region size and determine to accept or reject the solution.
                if (rho < eta_0) or ((g_old - g_new) <= 0) or ((r_old - r_new) <= 0):
                    # The solution fails either the RC or SR test, the center point reamins and the trust region shrinks.
                    delta_T = gamma_1 * delta_T
                elif (eta_0 <= rho) and (rho < eta_1):
                    # The center point moves to the new solution and the trust region remains.
                    new_solution = candidate_solution
                    # Update incumbent best solution.
                    if (problem.minmax * new_solution.objectives_mean > problem.minmax * best_solution.objectives_mean):
                        best_solution = new_solution
                        recommended_solns.append(new_solution)
                        intermediate_budgets.append(expended_budget)
                else:
                    # The center point moves to the new solution and the trust region enlarges.
                    delta_T = gamma_2 * delta_T
                    new_solution = candidate_solution
                    # Update incumbent best solution.
                    if (problem.minmax * new_solution.objectives_mean > problem.minmax * best_solution.objectives_mean):
                        best_solution = new_solution
                        recommended_solns.append(new_solution)
                        intermediate_budgets.append(expended_budget)
                n_r = int(np.ceil(1.01 * n_r))

            # Stage II.
            # When trust region size is very small, use the quadratic design.
            else:
                n_onbound = np.sum(BdsCheck != 0)
                if n_onbound <= 1:
                    NumOfEval = problem.dim ** 2
                else:
                    NumOfEval = problem.dim ** 2 + problem.dim - math.factorial(n_onbound) / (math.factorial(2), math.factorial(n_onbound - 2))
                # Step1: Build the quadratic model.
                grad, Hessian, cnt = self.finite_diff(new_solution, BdsCheck, 2, problem, n_r)
                expended_budget += NumOfEval * n_r * (sum(np.power(lam, i) for i in range(cnt)) - cnt + 1)
                # Step2: Solve the subproblem.
                # Cauchy reduction.
                candidate_x = self.cauchy_point(grad, Hessian, new_x, problem,)
                candidate_solution = self.create_new_solution(tuple(candidate_x), problem)
                # Step 3: Compute the ratio.
                # Use r simulated observations to estimate g(x_start\).
                problem.simulate(candidate_solution, n_r)
                expended_budget += n_r
                # Find the old objective value and the new objective value.
                g_old = -1 * problem.minmax[0] * new_solution.objectives_mean
                g_new = -1 * problem.minmax[0] * candidate_solution.objectives_mean
                # Construct the polynomial.
                r_old = g_old
                r_new = g_old + np.matmul(np.subtract(candidate_x, new_x), grad) + 0.5 * np.matmul(np.matmul(np.subtract(candidate_x, new_x), Hessian), np.subtract(candidate_x, new_x))
                rho = (g_old - g_new) / (r_old - r_new)
                # Step4: Update the trust region size and determine to accept or reject the solution.
                if (rho < eta_0) or ((g_old - g_new) <= 0) or ((r_old - r_new) <= 0):
                    # Inner Loop.
                    rr_old = r_old
                    g_b_old = rr_old
                    sub_counter = 1
                    result_solution = new_solution
                    result_x = new_x

                    while np.sum(result_x != new_x) == 0:
                        if expended_budget > problem.factors["budget"]:
                            break
                        # Step1: Build the quadratic model.
                        G, H, cnt = self.finite_diff(new_solution, BdsCheck, 2, problem, (sub_counter + 1) * n_r)
                        expended_budget += NumOfEval * (sub_counter + 1) * n_r * (sum(np.power(lam, i) for i in range(cnt)) - cnt + 1)

                        # Step2: determine the new inner solution based on the accumulated design matrix X.
                        try_x = self.cauchy_point(G, H, new_x, problem)
                        try_solution = self.create_new_solution(tuple(try_x), problem)

                        # Step 3.
                        problem.simulate(try_solution, int(n_r + np.ceil(sub_counter**1.01)))
                        expended_budget += int(n_r + np.ceil(sub_counter**1.01))
                        g_b_new = -1 * problem.minmax[0] * try_solution.objectives_mean
                        dummy_solution = new_solution
                        problem.simulate(dummy_solution, int(np.ceil(sub_counter**1.01) - np.ceil((sub_counter - 1)**1.01)))
                        expended_budget += int(np.ceil(sub_counter**1.01) - np.ceil((sub_counter - 1)**1.01))
                        dummy = -1 * problem.minmax[0] * dummy_solution.objectives_mean
                        # Update g_old.
                        g_b_old = (g_b_old * (n_r + np.ceil((sub_counter - 1)**1.01)) + (np.ceil(sub_counter**1.01) - np.ceil((sub_counter - 1)**1.01)) * dummy) / (n_r + np.ceil(sub_counter**1.01))
                        rr_new = g_b_old + np.matmul(np.subtract(try_x, new_x), G) + 0.5 * np.matmul(np.matmul(np.subtract(try_x, new_x), H), np.subtract(try_x, new_x))
                        rr_old = g_b_old
                        rrho = (g_b_old - g_b_new) / (rr_old - rr_new)
                        if (rrho < eta_0) or ((g_b_old - g_b_new) <= 0) or ((rr_old - rr_new) <= 0):
                            delta_T = gamma_1 * delta_T
                            result_solution = new_solution
                            result_x = new_x

                        elif (eta_0 <= rrho) and (rrho < eta_1):
                            # Accept the solution and remains the size of trust region.
                            result_solution = try_solution
                            result_x = try_x
                            rr_old = g_b_new
                        else:
                            # Accept the solution and expand the size of trust region.
                            delta_T = gamma_2 * delta_T
                            result_solution = try_solution
                            result_x = try_x
                            rr_old = g_b_new
                        sub_counter = sub_counter + 1
                    new_solution = result_solution
                    # Update incumbent best solution.
                    if (problem.minmax * new_solution.objectives_mean > problem.minmax * best_solution.objectives_mean):
                        best_solution = new_solution
                        recommended_solns.append(new_solution)
                        intermediate_budgets.append(expended_budget)
                elif (eta_0 <= rho) and (rho < eta_1):
                    # The center point moves to the new solution and the trust region remains.
                    new_solution = candidate_solution
                    # Update incumbent best solution.
                    if (problem.minmax * new_solution.objectives_mean > problem.minmax * best_solution.objectives_mean):
                        best_solution = new_solution
                        recommended_solns.append(new_solution)
                        intermediate_budgets.append(expended_budget)
                else:
                    # The center point moves to the new solution and the trust region enlarges.
                    delta_T = gamma_2 * delta_T
                    new_solution = candidate_solution
                    # Update incumbent best solution.
                    if (problem.minmax * new_solution.objectives_mean > problem.minmax * best_solution.objectives_mean):
                        best_solution = new_solution
                        recommended_solns.append(new_solution)
                        intermediate_budgets.append(expended_budget)
                n_r = int(np.ceil(1.01 * n_r))
        return recommended_solns, intermediate_budgets

    # Finding the Cauchy Point.
    def cauchy_point(self, grad, Hessian, new_x, problem):
        delta_T = self.factors['delta_T']
        lower_bound = problem.lower_bounds
        upper_bound = problem.upper_bounds
        if np.dot(np.matmul(grad, Hessian), grad) <= 0:
            tau = 1
        else:
            tau = min(1, norm(grad) ** 3 / (delta_T * np.dot(np.matmul(grad, Hessian), grad)))
        grad = np.reshape(grad, (1, problem.dim))[0]
        candidate_x = new_x - tau * delta_T * grad / norm(grad)
        Cauchy_x = self.check_cons(candidate_x, new_x, lower_bound, upper_bound)
        return Cauchy_x

    # Check the feasibility of the Cauchy point and update the point accordingly.
    def check_cons(self, candidate_x, new_x, lower_bound, upper_bound):
        # The current step.
        stepV = np.subtract(candidate_x, new_x)
        # Form a matrix to determine the possible stepsize.
        tmaxV = np.ones((2, len(candidate_x)))
        for i in range(0, len(candidate_x)):
            if stepV[i] > 0:
                tmaxV[0, i] = (upper_bound[i] - new_x[i]) / stepV[i]
            elif stepV[i] < 0:
                tmaxV[1, i] = (lower_bound[i] - new_x[i]) / stepV[i]
        # Find the minimum stepsize.
        t2 = tmaxV.min()
        # Calculate the modified x.
        modified_x = new_x + t2 * stepV
        return modified_x

    # Finite difference for calculating gradients and BFGS for calculating Hessian matrix.
    def finite_diff(self, new_solution, BdsCheck, stage, problem, n_r):
        delta_T = self.factors['delta_T']
        lower_bound = problem.lower_bounds
        upper_bound = problem.upper_bounds
        fn = -1 * problem.minmax[0] * new_solution.objectives_mean
        new_x = new_solution.x
        # Store values for each dimension.
        FnPlusMinus = np.zeros((problem.dim, 3))
        grad = np.zeros(problem.dim)
        Hessian = np.zeros((problem.dim, problem.dim))
        # Counter of the loop.
        cnt = 0
        # While loop to prevent all zero gradient.
        while (norm(grad) == 0):
            for i in range(problem.dim):
                # Initialization.
                x1 = list(new_x)
                x2 = list(new_x)
                # Forward stepsize.
                steph1 = delta_T
                # Backward stepsize.
                steph2 = delta_T

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
                # Backward diff
                else:
                    FnPlusMinus[i, 2] = steph2
                    x2[i] = x2[i] - FnPlusMinus[i, 2]

                x1_solution = self.create_new_solution(tuple(x1), problem)
                if BdsCheck[i] != -1:
                    problem.simulate_up_to([x1_solution], n_r)
                    fn1 = -1 * problem.minmax[0] * x1_solution.objectives_mean
                    # First column is f(x+h,y).
                    FnPlusMinus[i, 0] = fn1
                x2_solution = self.create_new_solution(tuple(x2), problem)
                if BdsCheck[i] != 1:
                    problem.simulate_up_to([x2_solution], n_r)
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

            if stage == 2:
                # Diagonal in Hessian.
                for i in range(problem.dim):
                    if BdsCheck[i] == 0:
                        Hessian[i, i] = (FnPlusMinus[i, 0] - 2 * fn + FnPlusMinus[i, 1]) / (FnPlusMinus[i, 2]**2)
                    elif BdsCheck[i] == 1:
                        x3 = list(new_x)
                        x3[i] = x3[i] + FnPlusMinus[i, 2] / 2
                        x3_solution = self.create_new_solution(tuple(x3), problem)
                        # Check budget.
                        problem.simulate_up_to([x3_solution], n_r)
                        fn3 = -1 * problem.minmax[0] * x3_solution.objectives_mean
                        Hessian[i, i] = 4 * (FnPlusMinus[i, 1] - 2 * fn3 + fn) / (FnPlusMinus[i, 2]**2)
                    elif BdsCheck[i] == -1:
                        x4 = list(new_x)
                        x4[i] = x4[i] - FnPlusMinus[i, 2] / 2
                        x4_solution = self.create_new_solution(tuple(x4), problem)
                        # Check budget.
                        problem.simulate_up_to([x4_solution], n_r)
                        fn4 = -1 * problem.minmax[0] * x4_solution.objectives_mean
                        Hessian[i, i] = 4 * (fn - 2 * fn4 + FnPlusMinus[i, 1]) / (FnPlusMinus[i, 2]**2)

                    # Upper triangle in Hessian
                    for j in range(i + 1, problem.dim):
                        # Neither x nor y on boundary.
                        if BdsCheck[i]**2 + BdsCheck[j]**2 == 0:
                            # Represent f(x+h,y+k).
                            x5 = list(new_x)
                            x5[i] = x5[i] + FnPlusMinus[i, 2]
                            x5[j] = x5[j] + FnPlusMinus[j, 2]
                            x5_solution = self.create_new_solution(tuple(x5), problem)
                            # Check budget.
                            problem.simulate_up_to([x5_solution], n_r)
                            fn5 = -1 * problem.minmax[0] * x5_solution.objectives_mean
                            # Represent f(x-h,y-k).
                            x6 = list(new_x)
                            x6[i] = x6[i] - FnPlusMinus[i, 2]
                            x6[j] = x6[j] - FnPlusMinus[j, 2]
                            x6_solution = self.create_new_solution(tuple(x5), problem)
                            # Check budget.
                            problem.simulate_up_to([x6_solution], n_r)
                            fn6 = -1 * problem.minmax[0] * x6_solution .objectives_mean
                            # Compute second order gradient.
                            Hessian[i, j] = (fn5 - FnPlusMinus[i, 0] - FnPlusMinus[j, 0] + 2 * fn - FnPlusMinus[i, 1] - FnPlusMinus[j, 1] + fn6) / (2 * FnPlusMinus[i, 2] * FnPlusMinus[j, 2])
                            Hessian[j, i] = Hessian[i, j]
                        # When x on boundary, y not.
                        elif BdsCheck[j] == 0:
                            # Represent f(x+/-h,y+k).
                            x5 = list(new_x)
                            x5[i] = x5[i] + BdsCheck[i] * FnPlusMinus[i, 2]
                            x5[j] = x5[j] + FnPlusMinus[j, 2]
                            x5_solution = self.create_new_solution(tuple(x5), problem)
                            # Check budget.
                            problem.simulate_up_to([x5_solution], n_r)
                            fn5 = -1 * problem.minmax[0] * x5_solution.objectives_mean
                            # Represent f(x+/-h,y-k).
                            x6 = list(new_x)
                            x6[i] = x6[i] + BdsCheck[i] * FnPlusMinus[i, 2]
                            x6[j] = x6[j] - FnPlusMinus[j, 2]
                            x6_solution = self.create_new_solution(tuple(x6), problem)
                            # Check budget.
                            problem.simulate_up_to([x6_solution], n_r)
                            fn6 = -1 * problem.minmax[0] * x6_solution.objectives_mean
                            # Compute second order gradient.
                            Hessian[i, j] = (fn5 - FnPlusMinus[j, 0] - fn6 + FnPlusMinus[j, 1]) / (2 * FnPlusMinus[i, 2] * FnPlusMinus[j, 2] * BdsCheck[i])
                            Hessian[j, i] = Hessian[i, j]
                        # When y on boundary, x not.
                        elif BdsCheck[i] == 0:
                            # Represent f(x+h,y+/-k).
                            x5 = list(new_x)
                            x5[i] = x5[i] + FnPlusMinus[i, 2]
                            x5[j] = x5[j] + BdsCheck[j] * FnPlusMinus[j, 2]
                            x5_solution = self.create_new_solution(tuple(x5), problem)
                            # Check budget.
                            problem.simulate_up_to([x5_solution], n_r)
                            fn5 = -1 * problem.minmax[0] * x5_solution.objectives_mean
                            # Represent f(x-h,y+/-k).
                            x6 = list(new_x)
                            x6[i] = x6[i] + FnPlusMinus[i, 2]
                            x6[j] = x6[j] + BdsCheck[j] * FnPlusMinus[j, 2]
                            x6_solution = self.create_new_solution(tuple(x6), problem)
                            # Check budget.
                            problem.simulate_up_to([x6_solution], n_r)
                            fn6 = -1 * problem.minmax[0] * x6_solution.objectives_mean
                            # Compute second order gradient.
                            Hessian[i, j] = (fn5 - FnPlusMinus[i, 0] - fn6 + FnPlusMinus[i, 1]) / (2 * FnPlusMinus[i, 2] * FnPlusMinus[j, 2] * BdsCheck[j])
                            Hessian[j, i] = Hessian[i, j]
                        elif BdsCheck[i] == 1:
                            if BdsCheck[j] == 1:
                                # Represent f(x+h,y+k).
                                x5 = list(new_x)
                                x5[i] = x5[i] + FnPlusMinus[i, 2]
                                x5[j] = x5[j] + FnPlusMinus[j, 2]
                                x5_solution = self.create_new_solution(tuple(x5), problem)
                                # Check budget.
                                problem.simulate_up_to([x5_solution], n_r)
                                fn5 = -1 * problem.minmax[0] * x5_solution.objectives_mean
                                # Compute second order gradient.
                                Hessian[i, j] = (fn5 - FnPlusMinus[i, 0] - FnPlusMinus[j, 0] + fn) / (FnPlusMinus[i, 2] * FnPlusMinus[j, 2])
                                Hessian[j, i] = Hessian[i, j]
                            else:
                                # Represent f(x+h,y-k).
                                x5 = list(new_x)
                                x5[i] = x5[i] + FnPlusMinus[i, 2]
                                x5[j] = x5[j] - FnPlusMinus[j, 2]
                                x5_solution = self.create_new_solution(tuple(x5), problem)
                                # Check budget.
                                problem.simulate_up_to([x5_solution], n_r)
                                fn5 = -1 * problem.minmax[0] * x5_solution.objectives_mean
                                # Compute second order gradient.
                                Hessian[i, j] = (FnPlusMinus[i, 0] - fn5 - fn + FnPlusMinus[j, 1]) / (FnPlusMinus[i, 2] * FnPlusMinus[j, 2])
                                Hessian[j, i] = Hessian[i, j]
                        elif BdsCheck[i] == -1:
                            if BdsCheck[j] == 1:
                                # Represent f(x-h,y+k).
                                x5 = list(new_x)
                                x5[i] = x5[i] - FnPlusMinus[i, 2]
                                x5[j] = x5[j] + FnPlusMinus[j, 2]
                                x5_solution = self.create_new_solution(tuple(x5), problem)
                                # Check budget
                                problem.simulate_up_to([x5_solution], n_r)
                                fn5 = -1 * problem.minmax[0] * x5_solution.objectives_mean
                                # Compute second order gradient.
                                Hessian[i, j] = (FnPlusMinus[j, 0] - fn - fn5 + FnPlusMinus[i, 1]) / (FnPlusMinus[i, 2] * FnPlusMinus[j, 2])
                                Hessian[j, i] = Hessian[i, j]
                            else:
                                # Represent f(x-h,y-k).
                                x5 = list(new_x)
                                x5[i] = x5[i] - FnPlusMinus[i, 2]
                                x5[j] = x5[j] - FnPlusMinus[j, 2]
                                x5_solution = self.create_new_solution(tuple(x5), problem)
                                # Check budget.
                                problem.simulate_up_to([x5_solution], n_r)
                                fn5 = -1 * problem.minmax[0] * x5_solution.objectives_mean
                                # Compute second order gradient.
                                Hessian[i, j] = (fn - FnPlusMinus[j, 1] - FnPlusMinus[i, 1] + fn5) / (FnPlusMinus[i, 2] * FnPlusMinus[j, 2])
                                Hessian[j, i] = Hessian[i, j]
            # Update n_r and counter after each loop.
            n_r = self.factors['lambda'] * n_r
            cnt += 1
        return grad, Hessian, cnt
