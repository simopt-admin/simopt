"""
Summary
-------
STRONG
A trust-region-based algorithm that fits first- or second-order models through function evaluations taken within
a neighborhood of the incumbent solution.
"""
from base import Solver
from numpy.linalg import inv
from numpy.linalg import norm
import numpy as np
import math
import warnings
warnings.filterwarnings("ignore")

class STRONG(Solver):
    """
    Needed description

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
        self.constraint_type = "deterministic"
        self.variable_type = "continuous"
        self.gradient_needed = False
        self.specifications = {
            "crn_across_solns": {
                "description": "Use CRN across solutions?",
                "datatype": bool,
                "default": True
            },
            "r": {
                "description": "Number of replications taken at each solution",
                "datatype": int,
                "default": 30
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
                "description": "the size of trust region",
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
                "description": "the constant of shrinking the trust regionthe new solution",
                "datatype": float,
                "default": 0.9
            },
            "gamma_2": {
                "description": "the constant of expanding the trust region",
                "datatype": float,
                "default": 1.11
            }
        }
        self.check_factor_list = {
            "crn_across_solns": self.check_crn_across_solns,
            "sample_size": self.check_r
        }
        super().__init__(fixed_factors)
    
    def check_r(self):
        return self.factors["r"] > 0
    '''
    def check_solver_factors(self):
        pass
    '''

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
        delta_threshold = self.factors["delta_threshold"]

        # default values
        r = self.factors["r"]
        delta_T = self.factors["delta_T"]
        eta_0 = self.factors["eta_0"]
        eta_1 = self.factors["eta_1"]
        gamma_1 = self.factors["gamma_1"]
        gamma_2 = self.factors["gamma_2"]
        k = 0

        # Start with the initial solution
        new_x = problem.factors["initial_solution"]
        new_solution = self.create_new_solution(new_x, problem)
        problem.simulate(new_solution, r)
        expended_budget += r
        best_solution = new_solution
        recommended_solns.append(new_solution)
        intermediate_budgets.append(expended_budget)

        while expended_budget < problem.factors["budget"]: 
            k += 1
            # check variable bounds
            forward = [int(new_x[i] == problem.lower_bounds[i]) for i in range(problem.dim)]
            backward = [int(new_x[i] == problem.upper_bounds[i]) for i in range(problem.dim)]
            # BdsCheck: 1 stands for forward, -1 stands for backward, 0 means central diff
            BdsCheck = np.subtract(forward, backward)

            # Stage 1
            if delta_T > delta_threshold:
                # step 1: build the linear model
                NumOfEval = 2 * problem.dim - np.sum(BdsCheck != 0)
                grad, Hessian = self.finite_diff(new_x, new_solution, delta_T, BdsCheck, 1, problem, r)
                expended_budget += NumOfEval * r

                # step 2: solve the subproblem
                # Cauchy reduction
                if np.matmul(np.matmul(grad, Hessian), grad) <= 0:
                    tau = 1
                else:
                    tau = min(1, norm(grad) ** 3 / (delta_T * np.matmul(np.matmul(grad, Hessian), grad)))

                grad = np.reshape(grad, (1, problem.dim))[0]
                candidate_x = new_x - tau * delta_T * grad / norm(grad)

                for i in range(problem.dim):
                    if candidate_x[i] < problem.lower_bounds[i]:
                        candidate_x[i] = problem.lower_bounds[i] + 0.01
                    elif candidate_x[i] > problem.upper_bounds[i]:
                        candidate_x[i] = problem.upper_bounds[i] - 0.01

                candidate_solution = self.create_new_solution(tuple(candidate_x), problem)
                # step 3: compute the ratio
                # Use r simulated observations to estimate g_new
                problem.simulate(candidate_solution, r)
                expended_budget += r
                # Find the old objective value and the new objective value
                g_old = -1 * problem.minmax[0] * new_solution.objectives_mean
                g_new = -1 * problem.minmax[0] * candidate_solution.objectives_mean
                # construct the polynomial 
                r_old = g_old
                r_new = g_old + np.matmul(np.subtract(candidate_x, new_x), grad) + (1/2)* np.matmul(np.matmul(np.subtract(candidate_x, new_x), Hessian), np.subtract(candidate_x, new_x))
                rho = (g_old - g_new)/(r_old - r_new)

                # step 4: update the trust region size and determine to accept or reject the solution
                if (rho < eta_0) | ((g_old  - g_new) <= 0) | ((r_old - r_new) <= 0):
                    # the solution fails either the RC or SR test, the center point reamins and the trust region shrinks
                    delta_T = gamma_1 * delta_T
                elif (eta_0 <= rho) & (rho < eta_1):
                    # The center point moves to the new solution and the trust region remains
                    new_solution = candidate_solution
                    new_x = candidate_x
                    # Update incumbent best solution
                    if (problem.minmax * new_solution.objectives_mean
                    > problem.minmax * best_solution.objectives_mean):
                        best_solution = new_solution
                        recommended_solns.append(new_solution)
                        intermediate_budgets.append(expended_budget)
                else:
                    # The center point moves to the new solution and the trust region enlarges
                    delta_T = gamma_2*delta_T
                    new_solution = candidate_solution
                    new_x = candidate_x
                    # Update incumbent best solution
                    if (problem.minmax * new_solution.objectives_mean
                    > problem.minmax * best_solution.objectives_mean):
                        best_solution = new_solution
                        recommended_solns.append(new_solution)
                        intermediate_budgets.append(expended_budget)
                r = int(np.ceil(1.01 * r))
            # Stage II 
            # When trust region size is very small, use the quadratic design
            else:
                n_onbound = np.sum(BdsCheck != 0)
                if n_onbound <= 1:
                    NumOfEval = problem.dim **2
                else:
                    NumOfEval = problem.dim **2 + problem.dim - math.factorial(n_onbound) / (math.factorial(2), math.factorial(n_onbound - 2))
                # step1 Build the quadratic model
                grad, Hessian = self.finite_diff(new_x, new_solution, delta_T, BdsCheck, 2, problem, r)
                expended_budget += NumOfEval * r
                # step2 Solve the subproblem
                # Cauchy reduction
                if np.matmul(np.matmul(grad, Hessian), grad) <= 0:
                    tau = 1
                else:
                    tau = min(1, norm(grad) ** 3 / (delta_T * np.matmul(np.matmul(grad, Hessian), grad)))

                grad = np.reshape(grad, (1, problem.dim))[0]
                candidate_x = new_x - tau * delta_T * grad / norm(grad)

                for i in range(problem.dim):
                    if candidate_x[i] < problem.lower_bounds[i]:
                        candidate_x[i] = problem.lower_bounds[i] + 0.01
                    elif candidate_x[i] > problem.upper_bounds[i]:
                        candidate_x[i] = problem.upper_bounds[i] - 0.01

                candidate_solution = self.create_new_solution(tuple(candidate_x), problem)
                # step 3: compute the ratio
                # Use r simulated observations to estimate g(x_start\)
                problem.simulate(candidate_solution, r)
                expended_budget += r
                # Find the old objective value and the new objective value
                g_old = -1 * problem.minmax[0] * new_solution.objectives_mean
                g_new = -1 * problem.minmax[0] * candidate_solution.objectives_mean
                # construct the polynomial 
                r_old = g_old
                r_new = g_old + np.matmul(np.subtract(candidate_x, new_x), grad) + (1/2)* np.matmul(np.matmul(np.subtract(candidate_x, new_x), Hessian), np.subtract(candidate_x, new_x))
                rho = (g_old - g_new)/(r_old - r_new)
                # step4 Update the trust region size and determine to accept or reject the solution
                if (rho < eta_0) | ((g_old - g_new) <= 0) | ((r_old - r_new ) <= 0):
                    # Inner Loop
                    rr_old = r_old
                    g_b_old = rr_old
                    sub_counter = 1
                    result_solution = new_solution
                    result_x = new_x

                    while np.sum(result_x != new_x) == 0:
                        if expended_budget > problem.factors["budget"]: 
                            break
                        # step1 Build the quadratic model
                        G, H = self.finite_diff(new_x, new_solution, delta_T, BdsCheck, 2, problem, (sub_counter + 1) * r)
                        expended_budget += NumOfEval * (sub_counter + 1) * r
                        # step2 determine the new inner solution based on the accumulated design matrix X
                        if np.matmul(np.matmul(G, H), G) <= 0:
                            tau = 1
                        else:
                            tau = min(1, norm(G) ** 3 / (delta_T * np.matmul(np.matmul(G, H), G)))

                        G = np.reshape(G, (1, problem.dim))[0]
                        try_x = new_x - tau * delta_T * G / norm(G)

                        for i in range(problem.dim):
                            if try_x[i] < problem.lower_bounds[i]:
                                try_x[i] = problem.lower_bounds[i] + 0.01
                            elif try_x[i] > problem.upper_bounds[i]:
                                try_x[i] = problem.upper_bounds[i] - 0.01
                        try_solution = self.create_new_solution(tuple(try_x), problem)
                        # step 3
                        problem.simulate(try_solution, int(r + np.ceil(sub_counter**1.01)))
                        expended_budget += int(r + np.ceil(sub_counter**1.01))
                        g_b_new = -1 * problem.minmax[0] * try_solution.objectives_mean
                        dummy_solution = new_solution
                        problem.simulate(dummy_solution, int(np.ceil(sub_counter**1.01) - np.ceil((sub_counter - 1)**1.01)))
                        expended_budget += int(np.ceil(sub_counter**1.01) - np.ceil((sub_counter - 1)**1.01))
                        dummy = -1 * problem.minmax[0] * dummy_solution.objectives_mean
                        # update g_old
                        g_b_old = (g_b_old * (r + np.ceil((sub_counter - 1)**1.01)) + (np.ceil(sub_counter**1.01) - np.ceil((sub_counter - 1)**1.01)) * dummy) / (r + np.ceil(sub_counter**1.01))
                        rr_new = g_b_old + np.matmul(np.subtract(try_x, new_x), G) + (1/2)* np.matmul(np.matmul(np.subtract(try_x, new_x), H), np.subtract(try_x, new_x))
                        rr_old = g_b_old
                        rrho = (g_b_old - g_b_new) / (rr_old - rr_new)
                        if (rrho < eta_0) | ((g_b_old - g_b_new) <= 0) | ((rr_old - rr_new) <= 0):
                            delta_T = gamma_1 * delta_T
                            result_solution = new_solution
                            result_x = new_x

                        elif (eta_0 <= rrho) & (rrho < eta_1):
                            result_solution = try_solution
                            result_x = try_x #accept the solution and remains the size of trust region
                            rr_old = g_b_new
                        else:
                            delta_T = gamma_2 * delta_T
                            result_solution = try_solution 
                            result_x = try_x #accept the solution and expand the size of trust reigon
                            rr_old = g_b_new
                        sub_counter = sub_counter + 1
                    new_solution = result_solution
                    new_x = result_x
                    # Update incumbent best solution
                    if (problem.minmax * new_solution.objectives_mean
                    > problem.minmax * best_solution.objectives_mean):
                        best_solution = new_solution
                        recommended_solns.append(new_solution)
                        intermediate_budgets.append(expended_budget)
                elif (eta_0 <= rho) & (rho < eta_1):
                    # The center point moves to the new solution and the trust region remains
                    new_solution = candidate_solution
                    new_x = candidate_x
                    # Update incumbent best solution
                    if (problem.minmax * new_solution.objectives_mean
                    > problem.minmax * best_solution.objectives_mean):
                        best_solution = new_solution
                        recommended_solns.append(new_solution)
                        intermediate_budgets.append(expended_budget)
                else:
                    # The center point moves to the new solution and the trust region enlarges
                    delta_T = gamma_2*delta_T
                    new_solution = candidate_solution
                    new_x = candidate_x
                    # Update incumbent best solution
                    if (problem.minmax * new_solution.objectives_mean
                    > problem.minmax * best_solution.objectives_mean):
                        best_solution = new_solution
                        recommended_solns.append(new_solution)
                        intermediate_budgets.append(expended_budget)
                r = int(np.ceil(1.01 * r))

        return recommended_solns, intermediate_budgets
    
    # Finite difference for calculating gradients and BFGS for calculating Hessian Matrix
    def finite_diff(self, new_x, new_solution, delta_T, BdsCheck, stage, problem, r):
        # Store values for each dimension
        FnPlusMinus = np.zeros((problem.dim, 3)) 
        grad = np.zeros(problem.dim)
        Hessian = np.zeros((problem.dim, problem.dim))

        for i in range(problem.dim):
            # initialization
            x1 = list(new_x)
            x2 = list(new_x)
            steph1 = delta_T # forward stepsize
            steph2 = delta_T # backward stepsize
            
            # check variable bounds
            if x1[i] + steph1 > problem.upper_bounds[i]:
                steph1 = np.abs(problem.upper_bounds[i] - x1[i])
            if x2[i] - steph2 < problem.lower_bounds[i]:
                steph2 = np.abs(x2[i] - problem.lower_bounds[i])
            
            # decide stepsize
            if BdsCheck[i] == 0:   #central diff
                FnPlusMinus[i, 2] = min(steph1, steph2)
                x1[i] = x1[i] + FnPlusMinus[i, 2]
                x2[i] = x2[i] - FnPlusMinus[i, 2]
            elif BdsCheck[i] == 1:    # forward diff
                FnPlusMinus[i, 2] = steph1
                x1[i] = x1[i] + FnPlusMinus[i, 2]
            else:    # backward diff
                FnPlusMinus[i, 2] = steph2
                x2[i] = x2[i] - FnPlusMinus[i,2]
            x1_solution = self.create_new_solution(tuple(x1), problem)
            if BdsCheck[i] != -1:
                problem.simulate(x1_solution, 1)
                fn1 = -1 * problem.minmax[0] * x1_solution.objectives_mean
                FnPlusMinus[i, 0] = fn1 # first column is f(x+h,y)
            x2_solution = self.create_new_solution(tuple(x2), problem)
            if BdsCheck[i] != 1:
                problem.simulate(x2_solution, 1)
                fn2 = -1 * problem.minmax[0] * x2_solution.objectives_mean
                FnPlusMinus[i, 1] = fn2 # second column is f(x-h,y)
            
            # Calculate gradient
            fn = -1 * problem.minmax[0] * new_solution.objectives_mean
            if BdsCheck[i] == 0:
                grad[i] = (fn1 - fn2) / (2 * FnPlusMinus[i,2])
            elif BdsCheck[i] == 1:
                grad[i] = (fn1 - fn) / FnPlusMinus[i, 2]
            elif BdsCheck[i] == -1:
                grad[i] = (fn - fn2) / FnPlusMinus[i, 2]
        
        if stage == 2:
            # diagonal in Hessian
            for i in range(problem.dim):
                if BdsCheck[i] == 0:
                    Hessian[i, i] = (FnPlusMinus[i, 0] - 2 * fn + FnPlusMinus[i, 1])/(FnPlusMinus[i, 2]**2)
                elif BdsCheck[i] == 1:
                    x3 = list(new_x)
                    x3[i] = x3[i] + FnPlusMinus[i, 2] / 2
                    x3_solution = self.create_new_solution(tuple(x3), problem)
                    # check budget
                    problem.simulate(x3_solution, r)
                    fn3 = -1 * problem.minmax[0] * x3_solution.objectives_mean
                    Hessian[i, i] = 4 * (FnPlusMinus[i, 1] - 2 * fn3 + fn) / (FnPlusMinus[i, 2]**2)
                elif BdsCheck[i] == -1:
                    x4 = list(new_x)
                    x4[i] = x4[i] - FnPlusMinus[i, 2] / 2
                    x4_solution = self.create_new_solution(tuple(x4), problem)
                    # check budget
                    problem.simulate(x4_solution, r)
                    fn4 = -1 * problem.minmax[0] * x4_solution.objectives_mean
                    Hessian[i, i] = 4 * (fn - 2 * fn4 + FnPlusMinus[i, 1])/(FnPlusMinus[i, 2]**2)
                
                # upper triangle in Hessian
                for j in range(i + 1, problem.dim):
                    if BdsCheck[i]**2 + BdsCheck[j]**2 == 0: # neither x nor y on boundary
                        # f(x+h,y+k)
                        x5 = list(new_x)
                        x5[i] = x5[i] + FnPlusMinus[i, 2]
                        x5[j] = x5[j] + FnPlusMinus[j, 2]
                        x5_solution = self.create_new_solution(tuple(x5), problem)
                        # check budget
                        problem.simulate(x5_solution, r)
                        fn5 = -1 * problem.minmax[0] * x5_solution.objectives_mean
                        # f(x-h,y-k)
                        x6 = list(new_x)
                        x6[i] = x6[i] - FnPlusMinus[i, 2]
                        x6[j] = x6[j] - FnPlusMinus[j, 2]
                        x6_solution = self.create_new_solution(tuple(x5), problem)
                        # check budget
                        problem.simulate(x6_solution , r)
                        fn6 = -1 * problem.minmax[0] * x6_solution .objectives_mean
                        # compute second order gradient
                        Hessian[i, j] = (fn5 - FnPlusMinus[i, 0] - FnPlusMinus[j, 0] + 2 * fn - FnPlusMinus[i, 1] - FnPlusMinus[j, 1] + fn6) / (2 * FnPlusMinus[i, 2] * FnPlusMinus[j ,2])
                        Hessian[j, i] = Hessian[i, j]
                    elif BdsCheck[j] == 0: # x on boundary, y not
                        # f(x+/-h,y+k)
                        x5 = list(new_x)
                        x5[i] = x5[i] + BdsCheck[i] * FnPlusMinus[i, 2]
                        x5[j] = x5[j] + FnPlusMinus[j, 2]
                        x5_solution = self.create_new_solution(tuple(x5), problem)
                        # check budget
                        problem.simulate(x5_solution, r)
                        fn5 = -1 * problem.minmax[0] * x5_solution.objectives_mean
                        # f(x+/-h,y-k)
                        x6 = list(new_x)
                        x6[i] = x6[i] + BdsCheck[i] * FnPlusMinus[i, 2]
                        x6[j] = x6[j] - FnPlusMinus[j, 2]
                        x6_solution = self.create_new_solution(tuple(x6), problem)
                        # check budget
                        problem.simulate(x6_solution, r)
                        fn6 = -1 * problem.minmax[0] * x6_solution.objectives_mean
                        # compute second order gradient
                        Hessian[i, j] = (fn5 - FnPlusMinus[j, 0] - fn6 + FnPlusMinus[j, 1]) / (2 * FnPlusMinus[i, 2] * FnPlusMinus[j, 2] * BdsCheck[i])
                        Hessian[j, i] = Hessian[i, j]
                    elif BdsCheck[i] == 0: # y on boundary, x not
                        # f(x+h,y+/-k)
                        x5 = list(new_x)
                        x5[i] = x5[i] + FnPlusMinus[i, 2]
                        x5[j] = x5[j] + BdsCheck[j] * FnPlusMinus[j, 2]
                        x5_solution = self.create_new_solution(tuple(x5), problem)                        
                        # check budget
                        problem.simulate(x5_solution, r)
                        fn5 = -1 * problem.minmax[0] * x5_solution.objectives_mean
                        # f(x-h,y+/-k)
                        x6 = list(new_x)
                        x6[i] = x6[i] + FnPlusMinus[i, 2]
                        x6[j] = x6[j] + BdsCheck[j] * FnPlusMinus[j, 2]
                        x6_solution = self.create_new_solution(tuple(x6), problem) 
                        # check budget
                        problem.simulate(x6_solution, r)
                        fn6 = -1 * problem.minmax[0] * x6_solution.objectives_mean
                        # compute second order gradient
                        Hessian[i, j] = (fn5 - FnPlusMinus[i, 0] - fn6 + FnPlusMinus[i, 1])/(2*FnPlusMinus[i, 2]*FnPlusMinus[j, 2]*BdsCheck[j])
                        Hessian[j, i] = Hessian[i, j]
                    elif BdsCheck[i] == 1:
                        if BdsCheck[j] == 1:
                            # f(x+h,y+k)
                            x5 = list(new_x)
                            x5[i] = x5[i] + FnPlusMinus[i, 2]
                            x5[j] = x5[j] + FnPlusMinus[j, 2]
                            x5_solution = self.create_new_solution(tuple(x5), problem) 
                            #check budget
                            problem.simulate(x5_solution, r)
                            fn5 = -1 * problem.minmax[0] * x5_solution.objectives_mean
                            # compute second order gradient
                            Hessian[i, j] = (fn5 - FnPlusMinus[i, 0] - FnPlusMinus[j, 0] + fn)/(FnPlusMinus[i, 2]*FnPlusMinus[j, 2])
                            Hessian[j, i] = Hessian[i, j]
                        else:
                            # f(x+h,y-k)
                            x5 = list(new_x)
                            x5[i] = x5[i] + FnPlusMinus[i, 2]
                            x5[j] = x5[j] - FnPlusMinus[j, 2]
                            x5_solution = self.create_new_solution(tuple(x5), problem) 
                            #check budget
                            problem.simulate(x5_solution, r)
                            fn5 = -1 * problem.minmax[0] * x5_solution.objectives_mean
                            # compute second order gradient
                            Hessian[i, j] = (FnPlusMinus[i, 0] - fn5 - fn + FnPlusMinus[j, 1])/(FnPlusMinus[i, 2]*FnPlusMinus[j, 2])
                            Hessian[j, i] = Hessian[i, j]
                    elif BdsCheck[i] == -1:
                        if BdsCheck[j] == 1:
                            # f(x-h,y+k)
                            x5 = list(new_x)
                            x5[i] = x5[i] - FnPlusMinus[i, 2]
                            x5[j] = x5[j] + FnPlusMinus[j, 2]
                            x5_solution = self.create_new_solution(tuple(x5), problem) 
                            #check budget
                            problem.simulate(x5_solution, r)
                            fn5 = -1 * problem.minmax[0] * x5_solution.objectives_mean
                            # compute second order gradient
                            Hessian[i, j] = (FnPlusMinus[j, 0] - fn - fn5 + FnPlusMinus[i, 1])/(FnPlusMinus[i, 2]*FnPlusMinus[j, 2])
                            Hessian[j, i] = Hessian[i, j]
                        else:
                            # f(x-h,y-k)
                            x5 = list(new_x)
                            x5[i] = x5[i] - FnPlusMinus[i, 2]
                            x5[j] = x5[j] - FnPlusMinus[j, 2]
                            x5_solution = self.create_new_solution(tuple(x5), problem) 
                            #check budget
                            problem.simulate(x5_solution, r)
                            fn5 = -1 * problem.minmax[0] * x5_solution.objectives_mean
                            # compute second order gradient
                            Hessian[i, j] = (fn - FnPlusMinus[j, 1] - FnPlusMinus[i, 1] + fn5)/(FnPlusMinus[i, 2]*FnPlusMinus[j, 2])
                            Hessian[j, i] = Hessian[i, j]

        return grad, Hessian
    