"""
Summary
-------
The ASTRO-DF solver progressively builds local models (quadratic with diagonal Hessian) using interpolation on a set of points on the coordinate bases of the best (incumbent) solution. Solving the local models within a trust region (closed ball around the incumbent solution) at each iteration suggests a candidate solution for the next iteration. If the candidate solution is worse than the best interpolation point, it is replaced with the latter (a.k.a. direct search). The solver then decides whether to accept the candidate solution and expand the trust-region or reject it and shrink the trust-region based on a success ratio test. The sample size at each visited point is determined adaptively and based on closeness to optimality.
A detailed description of the solver can be found `here <https://simopt.readthedocs.io/en/latest/astrodf.html>`_.
"""
from numpy.linalg import pinv
from numpy.linalg import norm
import numpy as np
from math import log, ceil
import warnings
from scipy.optimize import NonlinearConstraint
from scipy.optimize import minimize
warnings.filterwarnings("ignore")

from ..base import Solver


class ASTRODF(Solver):
    """The ASTRO-DF solver.

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
    def __init__(self, name="ASTRODF", fixed_factors=None):
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
            "delta_max": {
                "description": "maximum value of the trust-region radius",
                "datatype": float,
                "default": 200.0
            },
            "eta_1": {
                "description": "threshhold for a successful iteration",
                "datatype": float,
                "default": 0.1
            },
            "eta_2": {
                "description": "threshhold for a very successful iteration",
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
            "lambda_min": {
                "description": "minimum sample size value",
                "datatype": int,
                "default": 4
            },
            "simple_solve": {
                "description": "subproblem solver with Cauchy point or the built-in solver? True - Cauchy point, False - built-in solver",
                "datatype": bool,
                "default": True
            },
            "criticality_select": {
                "description": "True - skip contraction loop if not near critical region, False - always run contraction loop",
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
            "gamma_1": self.check_gamma_1,
            "gamma_2": self.check_gamma_2,
            "w": self.check_w,
            "beta": self.check_beta,
            "mu": self.check_mu,
            "lambda_min": self.check_lambda_min,
            "criticality_threshold": self.check_criticality_threshold
        }
        super().__init__(fixed_factors)

    def check_delta_max(self):
        return self.factors["delta_max"] > 0

    def check_eta_1(self):
        return self.factors["eta_1"] > 0

    def check_eta_2(self):
        return self.factors["eta_2"] > self.factors["eta_1"]

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

    def check_lambda_min(self):
        return self.factors["lambda_min"] > 2

    def check_criticality_threshold(self):
        return self.factors["criticality_threshold"] > 0

    def get_standard_basis(self, size, index):
        arr = np.zeros(size)
        arr[index] = 1.0
        return arr

    def evaluate_model(self, x_k, q):
        X = [1]
        X = np.append(X, np.array(x_k))
        X = np.append(X, np.array(x_k) ** 2)
        return np.matmul(X, q)

    def get_stopping_time(self, k, sig2, delta, kappa, dim):
        if kappa == 0:
            kappa = 1

        lambda_min = self.factors["lambda_min"]
        lambda_k = max(lambda_min, .5 * dim) * max(log(k + 0.1, 10) ** (1.01), 1)
        # compute sample size
        N_k = ceil(max(lambda_k, lambda_k * sig2 / ((kappa ** 2) * delta ** 4)))
        return N_k

    def construct_model(self, x_k, delta, k, problem, expended_budget, kappa, new_solution):
        interpolation_solns = []
        w = self.factors["w"]
        mu = self.factors["mu"]
        beta = self.factors["beta"]
        lambda_min = self.factors["lambda_min"]
        criticality_select = self.factors["criticality_select"]
        criticality_threshold = self.factors["criticality_threshold"]
        j = 0
        d = problem.dim
        budget = problem.factors["budget"]
        while True:
            fval = []
            j = j + 1
            delta_k = delta * w ** (j - 1)

            # construct the interpolation set
            Y = self.get_interpolation_points(x_k, delta_k, problem)
            for i in range(2 * d + 1):
                # for X_0, we don't need to simulate the new solution
                if (k == 1) and (i == 0):
                    fval.append(-1 * problem.minmax[0] * new_solution.objectives_mean)
                    interpolation_solns.append(new_solution)
                # otherwise, we need to simulate the new solution
                else:
                    new_solution = self.create_new_solution(tuple(Y[i][0]), problem)
                    # pilot run # ??check if there is existing result
                    pilot_run = int(max(lambda_min, .5 * problem.dim) - 1)
                    problem.simulate(new_solution, pilot_run)
                    expended_budget += pilot_run
                    sample_size = pilot_run

                    # adaptive sampling
                    while True:
                        problem.simulate(new_solution, 1)
                        expended_budget += 1
                        sample_size += 1
                        sig2 = new_solution.objectives_var
                        if sample_size >= self.get_stopping_time(k, sig2, delta_k, kappa, problem.dim) or expended_budget >= budget:
                            break
                    fval.append(-1 * problem.minmax[0] * new_solution.objectives_mean)
                    interpolation_solns.append(new_solution)

            Z = self.get_interpolation_points(np.array(x_k) - np.array(x_k), delta_k, problem)

            # construct the model and obtain the model coefficients
            q, grad, Hessian = self.get_model_coefficients(Z, fval, problem)

            if not criticality_select:
                # check the condition and break
                if norm(grad) > criticality_threshold:
                    break

            if delta_k <= mu * norm(grad):
                break

        delta_k = min(max(beta * norm(grad), delta_k), delta)

        return fval, Y, q, grad, Hessian, delta_k, expended_budget, interpolation_solns

    def get_model_coefficients(self, Y, fval, problem):
        M = []
        d = problem.dim
        for i in range(0, 2 * d + 1):
            M.append(1)
            M[i] = np.append(M[i], np.array(Y[i]))
            M[i] = np.append(M[i], np.array(Y[i]) ** 2)

        q = np.matmul(pinv(M), fval)  # pinv returns the inverse of your matrix when it is available and the pseudo inverse when it isn't.
        grad = q[1:d + 1]
        grad = np.reshape(grad, d)
        Hessian = q[d + 1:2 * d + 1]
        Hessian = np.reshape(Hessian, d)
        return q, grad, Hessian

    def get_interpolation_points(self, x_k, delta, problem):
        Y = [[x_k]]
        d = problem.dim
        epsilon = 0.01
        for i in range(0, d):
            plus = Y[0] + delta * self.get_standard_basis(d, i)
            minus = Y[0] - delta * self.get_standard_basis(d, i)

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

    def tune_parameters(self, delta, delta_max, problem):  # use the delta_max determined in the solve(...) function
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
        lambda_min = self.factors["lambda_min"]

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
                # pilot run
                pilot_run = int(max(lambda_min, .5 * problem.dim) - 1)
                problem.simulate(new_solution, pilot_run)
                expended_budget += pilot_run
                sample_size = pilot_run
                while True:
                    problem.simulate(new_solution, 1)
                    expended_budget += 1
                    sample_size += 1
                    fn = new_solution.objectives_mean
                    sig2 = new_solution.objectives_var
                    if sample_size >= self.get_stopping_time(k, sig2, delta, fn / (delta ** 2), problem.dim) or expended_budget >= budget * 0.01:
                        kappa = fn / (delta ** 2)
                        break

            fval, Y, q, grad, Hessian, delta_k, expended_budget, interpolation_solns = self.construct_model(new_x, delta_k, k, problem, expended_budget, kappa, new_solution)
            if simple_solve:
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

            # pilot run
            pilot_run = int(max(lambda_min, .5 * problem.dim) - 1)
            problem.simulate(candidate_solution, pilot_run)
            expended_budget += pilot_run
            sample_size = pilot_run

            # adaptive sampling
            while True:
                problem.simulate(candidate_solution, 1)
                expended_budget += 1
                sample_size += 1
                sig2 = candidate_solution.objectives_var
                if sample_size >= self.get_stopping_time(k, sig2, delta_k, kappa, problem.dim) or expended_budget >= budget * 0.01:
                    break

            # calculate success ratio
            fval_tilde = -1 * problem.minmax[0] * candidate_solution.objectives_mean

            # replace the candidate x if the interpolation set has lower objective function value
            if min(fval) < fval_tilde:
                minpos = fval.index(min(fval))
                fval_tilde = min(fval)
                candidate_x = Y[minpos][0]
                candidate_solution = interpolation_solns[minpos]

            if (self.evaluate_model(np.zeros(problem.dim), q) - self.evaluate_model
                    (np.array(candidate_x) - np.array(new_x), q)) == 0:
                rho = 0
            else:
                rho = (fval[0] - fval_tilde) / \
                      (self.evaluate_model(np.zeros(problem.dim), q) - self.evaluate_model(candidate_x - new_x, q))

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
        delta_max = min(self.factors["delta_max"], problem.upper_bounds[0] - problem.lower_bounds[0])
        gamma_01 = 0.08  # self.factors["gamma_01"]
        gamma_02 = 0.5  # self.factors["gamma_02"]
        delta_start = delta_max * gamma_01
        delta_candidate = [gamma_02 * delta_start, delta_start, delta_start / gamma_02]

        # default values
        eta_1 = self.factors["eta_1"]
        eta_2 = self.factors["eta_2"]
        gamma_1 = self.factors["gamma_1"]
        gamma_2 = self.factors["gamma_2"]
        simple_solve = self.factors["simple_solve"]
        lambda_min = self.factors["lambda_min"]

        k = 0  # iteration number

        # Start with the initial solution
        new_x = problem.factors["initial_solution"]
        new_solution = self.create_new_solution(tuple(new_x), problem)
        recommended_solns.append(new_solution)
        intermediate_budgets.append(expended_budget)

        # Parameter tuning run
        tp_final_ob_pt, k, delta, recommended_solns, intermediate_budgets, expended_budget, new_x, kappa = self.tune_parameters(
            delta_candidate[0], delta_max, problem)
        for i in range(1, 3):
            final_ob_pt, k_pt, delta_pt, recommended_solns_pt, intermediate_budgets_pt, expended_budget_pt, new_x_pt, kappa_pt = self.tune_parameters(
                delta_candidate[i], delta_max, problem)
            expended_budget += expended_budget_pt
            if -1 * problem.minmax[0] * final_ob_pt < -1 * problem.minmax[0] * tp_final_ob_pt:
                k = k_pt
                delta = delta_pt
                recommended_solns = recommended_solns_pt
                intermediate_budgets = intermediate_budgets_pt
                new_x = new_x_pt
                kappa = kappa_pt

        intermediate_budgets = (intermediate_budgets + 2 * np.ones(len(intermediate_budgets)) * budget * 0.01).tolist()
        intermediate_budgets[0] = 0
        delta_k = delta

        while expended_budget < budget:
            k += 1
            fval, Y, q, grad, Hessian, delta_k, expended_budget, interpolation_solns = self.construct_model(
                new_x, delta_k, k, problem, expended_budget, kappa, new_solution)

            if simple_solve:
                # Cauchy reduction
                if np.dot(np.multiply(grad, Hessian), grad) <= 0:
                    tau = 1
                else:
                    tau = min(1, norm(grad) ** 3 / (delta * np.dot(np.multiply(grad, Hessian), grad)))
                grad = np.reshape(grad, (1, problem.dim))[0]
                candidate_x = new_x - tau * delta * grad / norm(grad)
            else:
                # subproblem
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

            # pilot run
            pilot_run = int(max(lambda_min, .5 * problem.dim) - 1)
            problem.simulate(candidate_solution, pilot_run)
            expended_budget += pilot_run
            sample_size = pilot_run

            # adaptive sampling
            while True:
                problem.simulate(candidate_solution, 1)
                expended_budget += 1
                sample_size += 1
                sig2 = candidate_solution.objectives_var
                if sample_size >= self.get_stopping_time(k, sig2, delta_k, kappa, problem.dim) or expended_budget >= budget:
                    break

            # calculate success ratio
            fval_tilde = -1 * problem.minmax[0] * candidate_solution.objectives_mean

            # replace the candidate x if the interpolation set has lower objective function value
            if min(fval) < fval_tilde:
                minpos = fval.index(min(fval))
                fval_tilde = min(fval)
                candidate_x = Y[minpos][0]
                candidate_solution = interpolation_solns[minpos]

            if (self.evaluate_model(np.zeros(problem.dim), q) - self.evaluate_model(
                    np.array(candidate_x) - np.array(new_x), q)) == 0:
                rho = 0
            else:
                rho = (fval[0] - fval_tilde) / (self.evaluate_model(np.zeros(problem.dim), q) - self.evaluate_model(candidate_x - new_x, q))

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
