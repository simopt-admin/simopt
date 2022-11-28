
"""
Summary
-------
ASTRODF
Based on the sample average approximation, the solver makes the surrogate model within the trust region at each iteration k.
The sample sizes are determined adaptively.
Solve the subproblem and decide whether the algorithm take the candidate solution as next ieration center point or not.
Cannot handle stochastic constraints.


[DONE] then use the new version of astrodf (with no c_lambda and eps_lambda)
then do 100-3 and 100-4 and pick the best and potentially simple_solve=F
choose gamma = 2 instead of 1.5
change san_experiment demo file with that of David's branch??
"""
from base import Solver
from numpy.linalg import inv
from numpy.linalg import norm
import numpy as np
from math import log, ceil
import warnings
from scipy.optimize import NonlinearConstraint
from scipy.optimize import minimize
warnings.filterwarnings("ignore")

class ASTRODF(Solver):
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
    def __init__(self, name="ASTRODF", fixed_factors={}):
        self.name = name
        self.objective_type = "single"
        self.constraint_type = "box"
        self.variable_type = "continuous"
        self.gradient_needed = False
        self.specifications = {
            "crn_across_solns": {
                "description": "CRN across solutions?",
                "datatype": bool,
                "default": True
            },
            "delta_max": {
                "description": "maximum value of the trust-region radius",
                "datatype": float,
                "default": 200.0
            },
            "eta_1": {
                "description": "threshhold for any success at all",
                "datatype": float,
                "default": 0.1
            },
            "eta_2": {
                "description": "threshhold for good success",
                "datatype": float,
                "default": 0.5
            },
            "gamma_01": {
                "description": "initial trust-region radius parameter tuning coefficient 1",
                "datatype": float,
                "default": 0.08
            },
            "gamma_02": {
                "description": "initial trust-region radius parameter tuning coefficient 2",
                "datatype": float,
                "default": 0.5
            },
            "gamma_1": {
                "description": "very successful step trust-region radius increase",
                "datatype": float,
                "default": 1.5
            },
            "gamma_2": {
                "description": "unsuccessful step trust-region radius decrease",
                "datatype": float,
                "default": 0.75
            },
            "w": {
                "description": "trust-region radius rate of shrinkage in contracation loop",
                "datatype": float,
                "default": 0.85
            },
            "mu": {
                "description": "trust-region radius ratio upper bound in contraction loop",
                "datatype": int,
                "default": 1000
            },
            "beta": {
                "description": "trust-region radius ratio lower bound in contraction loop",
                "datatype": int,
                "default": 10
            },
            "c_lambda": {
                "description": "minimum sample size coefficient",
                "datatype": float,
                "default": 0.1
            },
            "epsilon_lambda": {
                "description": "minimum sample size exponent",
                "datatype": float,
                "default": 0.00001 ## less means faster increase at the beginning
            },
            "simple_solve": {
                "description": "subproblem solver with Cauchy point or the built-in solver? True: Cauchy point, False: built-in solver",
                "datatype": bool,
                "default": True
            },
            "select_criticality": {
                "description": "True: skip contraction loop if not near critical region, False: always run contraction loop",
                "datatype": bool,
                "default": True
            },
            "criticality_threshold": {
                "description": "threshold on gradient norm indicating near-critical region",
                "datatype": float,
                "default": 0.1
            }
        }
        self.check_factor_list = {
            "crn_across_solns": self.check_crn_across_solns,
            "delta_max": self.check_delta_max,
            "eta_1": self.check_eta_1,
            "eta_2": self.check_eta_2,
            "gamma_01": self.check_gamma_01,
            "gamma_02": self.check_gamma_02,
            "gamma_1": self.check_gamma_1,
            "gamma_2": self.check_gamma_2,
            "w": self.check_w,
            "beta": self.check_beta,
            "mu": self.check_mu,
            "c_lambda": self.check_c_lambda,
            "epsilon_lambda": self.check_epsilon_lambda,
            "criticality_threshold": self.check_criticality_threshold
        }
        super().__init__(fixed_factors)

    def check_delta_max(self):
        return self.factors["delta_max"] > 0

    def check_eta_1(self):
        return self.factors["eta_1"] > 0

    def check_eta_2(self):
        return self.factors["eta_2"] > self.factors["eta_1"]

    def check_gamma_01(self):
        return (self.factors["gamma_01"] > 0 and self.factors["gamma_01"] < 1 )

    def check_gamma_02(self):
        return (self.factors["gamma_02"] > self.factors["gamma_01"] and self.factors["gamma_02"] < 1 )

    def check_gamma_1(self):
        return self.factors["gamma_1"] > 1

    def check_gamma_2(self):
        return (self.factors["gamma_2"] < 1 and self.factors["gamma_2"] > 0)

    def check_w(self):
        return (self.factors["w"] < 1 and self.factors["w"] > 0)

    def check_beta(self):
        return (self.factors["beta"] < self.factors["mu"] and self.factors["beta"] > 0)

    def check_mu(self):
        return self.factors["mu"] > 0

    def check_c_lambda(self):
        return self.factors["c_lambda"] > 0

    def check_epsilon_lambda(self):
        return self.factors["epsilon_lambda"] > 0

    def check_criticality_threshold(self):
        return self.factors["criticality_threshold"] > 0

    def standard_basis(self, size, index):
        arr = np.zeros(size)
        arr[index] = 1.0
        return arr

    def local_model_evaluate(self, x_k, q):
        X = [1]
        X = np.append(X, np.array(x_k))
        X = np.append(X, np.array(x_k) ** 2)
        return np.matmul(X, q)

    def stoppingtime(self, k, sig2, delta, kappa, dim):
        c_lambda = self.factors["c_lambda"]
        epsilon_lambda = self.factors["epsilon_lambda"]

        lambda_k = (max(4, dim) + c_lambda) * max(log(k+ c_lambda, 10) ** (1 + epsilon_lambda),1)
        # compute sample size
        N_k = ceil(max(lambda_k, lambda_k * sig2 / ((kappa ** 2) * delta ** 4)))
        ## for later: could we normalize f's before computing sig2?
        return N_k

    def model_construction(self, x_k, delta, k, problem, expended_budget, kappa, new_solution):
        interpolation_solns = []
        w = self.factors["w"]
        mu = self.factors["mu"]
        beta = self.factors["beta"]
        select_criticality = self.factors["select_criticality"]
        criticality_threshold = self.factors["criticality_threshold"]
        j = 0
        d = problem.dim
        budget = problem.factors["budget"]
        while True:
            fval = []
            j = j + 1
            delta_k = delta * w ** (j - 1)

            # construct the interpolation set
            Y = self.interpolation_points(x_k, delta_k, problem)
            for i in range(2 * d + 1):
                # For X_0, we don't need to simulate the system
                if (k == 1) and (i==0):
                    fval.append(-1 * problem.minmax[0] * new_solution.objectives_mean)
                    interpolation_solns.append(new_solution)
                # Otherwise, we need to simulate the system
                else:
                    new_solution = self.create_new_solution(tuple(Y[i][0]), problem)
                    # check if there is existing result
                    problem.simulate(new_solution, 1)
                    expended_budget += 1
                    sample_size = 1

                    # Adaptive sampling
                    while True:
                        problem.simulate(new_solution, 1)
                        expended_budget += 1
                        sample_size += 1
                        sig2 = new_solution.objectives_var
                        if sample_size >= self.stoppingtime(k, sig2, delta_k, kappa, problem.dim) or expended_budget >= budget:
                            break
                    fval.append(-1 * problem.minmax[0] * new_solution.objectives_mean)
                    interpolation_solns.append(new_solution)

            Z = self.interpolation_points(np.array(x_k) - np.array(x_k), delta_k, problem)

            # construct the model and get the model coefficients
            q, grad, Hessian = self.coefficient(Z, fval, problem)

            if not select_criticality:
                # check the condition and break
                if norm(grad) > criticality_threshold:
                    break

            if delta_k <= mu * norm(grad):
                break

        delta_k = min(max(beta * norm(grad), delta_k), delta)

        return fval, Y, q, grad, Hessian, delta_k, expended_budget, interpolation_solns

    def coefficient(self, Y, fval, problem):
        M = []
        d = problem.dim
        for i in range(0, 2 * d + 1):
            M.append(1)
            M[i] = np.append(M[i], np.array(Y[i]))
            M[i] = np.append(M[i], np.array(Y[i]) ** 2)

        q = np.matmul(np.linalg.pinv(M), fval) # pinv returns the inverse of your matrix when it is available and the pseudo inverse when it isn't.
        grad = q[1:d + 1]
        grad = np.reshape(grad, d)
        Hessian = q[d + 1:2 * d + 1]
        Hessian = np.reshape(Hessian, d)
        return q, grad, Hessian

    def interpolation_points(self, x_k, delta, problem):
        Y = [[x_k]]
        d = problem.dim
        epsilon = 0.01
        for i in range(0, d):
            plus = Y[0] + delta * self.standard_basis(d, i)
            minus = Y[0] - delta * self.standard_basis(d, i)

            if sum(x_k) != 0:
                # block constraints
                if minus[0][i] <= problem.lower_bounds[i]:
                    minus[0][i] = problem.lower_bounds[i] + epsilon
                    # Y[0][i] = (minus[0][i]+plus[0][i])/2
                if plus[0][i] >= problem.upper_bounds[i]:
                    plus[0][i] = problem.upper_bounds[i] - epsilon
                    # Y[0][i] = (minus[0][i]+plus[0][i])/2

            Y.append(plus)
            Y.append(minus)
        return Y

    def parameter_tuning(self, delta, delta_max, problem): # use the delta_max determined in the solve(...) function
        recommended_solns = []
        intermediate_budgets = []
        expended_budget = 0
        k = 0  # iteration number

        # default values
        eta_1 = self.factors["eta_1"]
        eta_2 = self.factors["eta_2"]
        gamma_1 = self.factors["gamma_1"]
        gamma_2 = self.factors["gamma_2"]
        simple_solve = self.factors["simple_solve"]
        budget = problem.factors["budget"]


        # Start with the initial solution
        new_x = problem.factors["initial_solution"]
        new_solution = self.create_new_solution(tuple(new_x), problem)
        recommended_solns.append(new_solution)
        intermediate_budgets.append(expended_budget)
        delta_k = delta
        kappa = 1

        while expended_budget < budget * 0.01:

            # calculate kappa
            k += 1
            if k == 1:
                problem.simulate(new_solution, 1)
                expended_budget += 1
                sample_size = 1
                while True:
                    problem.simulate(new_solution, 1)
                    expended_budget += 1
                    sample_size += 1
                    fn = new_solution.objectives_mean
                    sig2 = new_solution.objectives_var
                    if sample_size >= self.stoppingtime(k, sig2, delta, fn/(delta**2), problem.dim) or expended_budget >= budget * 0.01:
                        kappa = fn/(delta**2)
                        break

            fval, Y, q, grad, Hessian, delta_k, expended_budget, interpolation_solns = self.model_construction(new_x, delta_k, k, problem, expended_budget, kappa, new_solution)
            if simple_solve == True:
                # Cauchy reduction
                if np.dot(np.multiply(grad, Hessian), grad) <= 0:
                    tau = 1
                else:
                    tau = min(1, norm(grad) ** 3 / (delta * np.dot(np.multiply(grad, Hessian), grad)))
                grad = np.reshape(grad, (1, problem.dim))[0]
                candidate_x = new_x - tau * delta * grad / norm(grad)
                candidate_solution = self.create_new_solution(tuple(candidate_x), problem)
            else:
                # Search engine - solve subproblem
                def subproblem(s):
                    return fval[0] + np.dot(s, grad) + np.dot(np.multiply(s, Hessian), s)

                con_f = lambda s: norm(s)
                nlc = NonlinearConstraint(con_f, 0, delta_k)
                solve_subproblem = minimize(subproblem, np.zeros(problem.dim), method='trust-constr', constraints=nlc)
                candidate_x = new_x + solve_subproblem.x
                candidate_solution = self.create_new_solution(tuple(candidate_x), problem)

            for i in range(problem.dim):
                if candidate_x[i] <= problem.lower_bounds[i]:
                    candidate_x[i] = problem.lower_bounds[i] + 0.01
                elif candidate_x[i] >= problem.upper_bounds[i]:
                    candidate_x[i] = problem.upper_bounds[i] - 0.01

            # adaptive sampling needed
            problem.simulate(candidate_solution, 1)
            expended_budget += 1
            sample_size = 1

            # Adaptive sampling
            while True:
                problem.simulate(candidate_solution, 1)
                expended_budget += 1
                sample_size += 1
                sig2 = candidate_solution.objectives_var
                if sample_size >= self.stoppingtime(k, sig2, delta_k, kappa, problem.dim) or expended_budget >= budget * 0.01:
                    break

            # calculate success ratio
            fval_tilde = -1 * problem.minmax[0] * candidate_solution.objectives_mean

            # replace the candidate x if the interpolation set has lower objective function value
            if min(fval) < fval_tilde:
                minpos = fval.index(min(fval))
                fval_tilde = min(fval)
                candidate_x = Y[minpos][0]
                candidate_solution = interpolation_solns[minpos]

            if (self.local_model_evaluate(np.zeros(problem.dim), q) - self.local_model_evaluate
                    (np.array(candidate_x) - np.array(new_x), q)) == 0:
                rho = 0
            else:
                rho = (fval[0] - fval_tilde) / \
                            (self.local_model_evaluate(np.zeros(problem.dim), q) - self.local_model_evaluate(
                        candidate_x - new_x, q));

            if rho >= eta_2:  # very successful
                new_x = candidate_x
                final_ob = candidate_solution.objectives_mean
                delta_k = min(gamma_1 * delta_k, delta_max)
                recommended_solns.append(candidate_solution)
                intermediate_budgets.append(expended_budget)
            elif rho >= eta_1:  # successful
                new_x = candidate_x
                final_ob = candidate_solution.objectives_mean
                delta_k = min(delta_k, delta_max)
                recommended_solns.append(candidate_solution)
                intermediate_budgets.append(expended_budget)
            else:
                delta_k = min(gamma_2 * delta_k, delta_max)
                final_ob = fval[0]

        return final_ob, k, delta_k, recommended_solns, intermediate_budgets, expended_budget, new_x, kappa

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
        budget = problem.factors["budget"]
        delta_max = min( self.factors["delta_max"] , problem.upper_bounds[0] - problem.lower_bounds[0])
        gamma_01 = self.factors["gamma_01"]
        gamma_02 = self.factors["gamma_02"]
        delta_start = delta_max * gamma_01
        delta_candidate = [gamma_02 * delta_start, delta_start, delta_start / gamma_02]

        # default values
        eta_1 = self.factors["eta_1"]
        eta_2 = self.factors["eta_2"]
        gamma_1 = self.factors["gamma_1"]
        gamma_2 = self.factors["gamma_2"]
        simple_solve = self.factors["simple_solve"]
        
        
        k = 0  # iteration number

        # Start with the initial solution
        new_x = problem.factors["initial_solution"]
        new_solution = self.create_new_solution(tuple(new_x), problem)
        recommended_solns.append(new_solution)
        intermediate_budgets.append(expended_budget)

        # Parameter tuning run
        tp_final_ob_pt, k, delta, recommended_solns, intermediate_budgets, expended_budget, new_x, kappa = self.parameter_tuning(
            delta_candidate[0], delta_max, problem)
        for i in range(1, 3):
            final_ob_pt, k_pt, delta_pt, recommended_solns_pt, intermediate_budgets_pt, expended_budget_pt, new_x_pt, kappa_pt = self.parameter_tuning(
                delta_candidate[i], delta_max, problem)
            expended_budget += expended_budget_pt
            if -1 * problem.minmax[0] * final_ob_pt < -1 * problem.minmax[0] * tp_final_ob_pt:
                k = k_pt
                delta = delta_pt
                recommended_solns = recommended_solns_pt
                intermediate_budgets = intermediate_budgets_pt
                new_x = new_x_pt
                kappa = kappa_pt

        intermediate_budgets = (
                intermediate_budgets + 2 * np.ones(len(intermediate_budgets)) * budget * 0.01).tolist()
        intermediate_budgets[0] = 0
        delta_k = delta

        while expended_budget < budget:
            k += 1
            fval, Y, q, grad, Hessian, delta_k, expended_budget, interpolation_solns = self.model_construction(
                new_x, delta_k, k, problem, expended_budget, kappa, new_solution)

            if simple_solve == True:
                # Cauchy reduction
                if np.dot(np.multiply(grad, Hessian), grad) <= 0:
                    tau = 1
                else:
                    tau = min(1, norm(grad) ** 3 / (delta * np.dot(np.multiply(grad, Hessian), grad)))
                grad = np.reshape(grad, (1, problem.dim))[0]
                candidate_x = new_x - tau * delta * grad / norm(grad)
            else:
                # Search engine - solve subproblem
                def subproblem(s):
                    return fval[0] + np.dot(s, grad) + np.dot(np.multiply(s, Hessian), s)

                con_f = lambda s: norm(s)
                nlc = NonlinearConstraint(con_f, 0, delta_k)
                solve_subproblem = minimize(subproblem, np.zeros(problem.dim), method='trust-constr', constraints=nlc)
                candidate_x = new_x + solve_subproblem.x

            for i in range(problem.dim):
                if candidate_x[i] <= problem.lower_bounds[i]:
                    candidate_x[i] = problem.lower_bounds[i] + 0.01
                elif candidate_x[i] >= problem.upper_bounds[i]:
                    candidate_x[i] = problem.upper_bounds[i] - 0.01

            candidate_solution = self.create_new_solution(tuple(candidate_x), problem)

            # adaptive sampling needed
            problem.simulate(candidate_solution, 1)
            expended_budget += 1
            sample_size = 1

            # Adaptive sampling
            while True:
                problem.simulate(candidate_solution, 1)
                expended_budget += 1
                sample_size += 1
                sig2 = candidate_solution.objectives_var
                if sample_size >= self.stoppingtime(k, sig2, delta_k, kappa, problem.dim) or expended_budget >= budget:
                    break

            # calculate success ratio
            fval_tilde = -1 * problem.minmax[0] * candidate_solution.objectives_mean

            # replace the candidate x if the interpolation set has lower objective function value
            if min(fval) < fval_tilde:
                minpos = fval.index(min(fval))
                fval_tilde = min(fval)
                candidate_x = Y[minpos][0]
                candidate_solution = interpolation_solns[minpos]

            if (self.local_model_evaluate(np.zeros(problem.dim), q) - self.local_model_evaluate(
                    np.array(candidate_x) - np.array(new_x), q)) == 0:
                rho = 0
            else:
                rho = (fval[0] - fval_tilde) / (
                        self.local_model_evaluate(np.zeros(problem.dim), q) - self.local_model_evaluate(
                    candidate_x - new_x, q));

            if rho >= eta_2:  # very successful
                new_x = candidate_x
                delta_k = min(gamma_1 * delta_k, delta_max)
                recommended_solns.append(candidate_solution)
                intermediate_budgets.append(expended_budget)
            elif rho >= eta_1:  # successful
                new_x = candidate_x
                delta_k = min(delta_k, delta_max)
                recommended_solns.append(candidate_solution)
                intermediate_budgets.append(expended_budget)
            else:
                delta_k = min(gamma_2 * delta_k, delta_max)

        return recommended_solns, intermediate_budgets