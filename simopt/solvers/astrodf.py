"""
Summary
-------
ASTRODF
Based on the sample average approximation, the solver makes the surrogate model within the trust region at each iteration k.
The sample sizes are determined adaptively.
Solve the subproblem and decide whether the algorithm take the candidate solution as next ieration center point or not.
Cannot handle stochastic constraints.
"""
from base import Solver
from numpy.linalg import inv
from numpy.linalg import norm
import numpy as np
import math
import warnings
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
        self.constraint_type = "deterministic"
        self.variable_type = "continuous"
        self.gradient_needed = False
        self.specifications = {
            "crn_across_solns": {
                "description": "Use CRN across solutions?",
                "datatype": bool,
                "default": True
            },
            "delta_max": {
                "description": "maximum value of the radius",
                "datatype": float,
                "default": 200
            },
            "tau": {
                "description": "shirink ratio for delta_candidate",
                "datatype": float,
                "default": 0.5
            },
            "eta_1": {
                "description": "threshhold for decent success",
                "datatype": float,
                "default": 0.1
            },
            "eta_2": {
                "description": "threshhold for good success",
                "datatype": float,
                "default": 0.5
            },
            "gamma_1": {
                "description": "very successful step radius increase",
                "datatype": float,
                "default": 1.25
            },
            "gamma_2": {
                "description": "unsuccessful step radius decrease",
                "datatype": float,
                "default": 0.8
            },
            "w": {
                "description": "decreasing rate for delta in contracation loop",
                "datatype": float,
                "default": 0.9
            },
            "mu": {
                "description": "the constant to make upper bound for delta in contraction loop",
                "datatype": float,
                "default": 100
            },
            "beta": {
                "description": "the constant to make the delta in main loop not too small",
                "datatype": float,
                "default": 50
            },
            "c_lambda": {
                "description": "hyperparameter to determine sample size",
                "datatype": float,
                "default": 0
            },
            "epsilon_lambda": {
                "description": "hyperparameter to determine sample size",
                "datatype": float,
                "default": 0.5
            },
            "kappa": {
                "description": "hyperparameter to determine sample size",
                "datatype": float,
                "default": 100
            }
        }
        self.check_factor_list = {
            "crn_across_solns": self.check_crn_across_solns,
            "sample_size": self.check_sample_size
        }
        super().__init__(fixed_factors)

    def check_sample_size(self):
        return self.factors["sample_size"] > 0
    '''
    def check_solver_factors(self):
        pass
    '''
    def standard_basis(self, size, index):
        arr = np.zeros(size)
        arr[index] = 1.0
        return arr

    def local_model_evaluate(self, x_k, q):
        X = [1]
        X = np.append(X, np.array(x_k))
        X = np.append(X, np.array(x_k) ** 2)
        return np.matmul(X, q)

    def samplesize(self, k, sig, delta):
        c_lambda = self.factors["c_lambda"]
        epsilon_lambda = self.factors["epsilon_lambda"]
        kappa = self.factors["kappa"]
        lambda_k = (10 + c_lambda) * math.log(k, 10) ** (1 + epsilon_lambda)
        # lambda_k = 10*math.log(k,10)**1.5

        # S_k = math.floor(max(3,lambda_k,(lambda_k*sig)/((kappa^2)*delta**(2*(1+1/alpha_k)))))
        S_k = math.floor(max(lambda_k, (lambda_k * sig) / ((kappa ^ 2) * delta ** 4)))
        return S_k

    def model_construction(self, x_k, delta, k, problem, expended_budget):
        w = self.factors["w"]
        mu = self.factors["mu"]
        beta = self.factors["beta"]
        j = 0
        d = problem.dim
        while True:
            fval = []
            j = j + 1
            delta_k = delta * w ** (j - 1)

            # make the interpolation set
            Y = self.interpolation_points(x_k, delta_k, problem)
            for i in range(2 * d + 1):
                new_solution = self.create_new_solution(Y[i][0], problem)

                # need to check there is existing result
                problem.simulate(new_solution, 1)
                expended_budget += 1
                sample_size = 1

                # Adaptive sampling
                while True:
                    problem.simulate(new_solution, 1)
                    expended_budget += 1
                    sample_size += 1
                    sig = new_solution.objectives_var
                    if sample_size >= self.samplesize(k, sig, delta_k):
                        break
                fval.append(-1 * problem.minmax[0] * new_solution.objectives_mean)

            Z = self.interpolation_points(np.array(x_k) - np.array(x_k), delta_k, problem)

            # make the model and get the model parameters
            q, grad, Hessian = self.coefficient(Z, fval, problem)

            # check the condition and break
            if norm(grad) > 0.1:
                break

            if delta_k <= mu * norm(grad):
                break

        delta_k = min(max(beta * norm(grad), delta_k), delta)
        return fval, Y, q, grad, Hessian, delta_k, expended_budget

    def coefficient(self, Y, fval, problem):
        M = []
        d = problem.dim
        for i in range(0, 2 * d + 1):
            M.append(1)
            M[i] = np.append(M[i], np.array(Y[i]))
            M[i] = np.append(M[i], np.array(Y[i]) ** 2)

        q = np.matmul(inv(M), fval)
        Hessian = np.diag(q[d + 1:2 * d + 1])
        return q, q[1:d + 1], Hessian

    def interpolation_points(self, x_k, delta, problem):
        Y = [[x_k]]
        d = problem.dim
        epsilon = 0.01
        for i in range(0, d):
            plus = Y[0] + delta * self.standard_basis(d, i)
            minus = Y[0] - delta * self.standard_basis(d, i)

            if sum(x_k) != 0:
                # block constraints
                if minus[0][i] < problem.lower_bounds[i]:
                    minus[0][i] = problem.lower_bounds[i] + epsilon
                    # Y[0][i] = (minus[0][i]+plus[0][i])/2
                if plus[0][i] > problem.upper_bounds[i]:
                    plus[0][i] = problem.upper_bounds[i] - epsilon
                    # Y[0][i] = (minus[0][i]+plus[0][i])/2

            Y.append(plus)
            Y.append(minus)
        return Y

    def parameter_tuning(self, delta, problem):
        recommended_solns = []
        intermediate_budgets = []
        expended_budget = 0

        # default values
        delta_max = self.factors["delta_max"]
        eta_1 = self.factors["eta_1"]
        eta_2 = self.factors["eta_2"]
        gamma_1 = self.factors["gamma_1"]
        gamma_2 = self.factors["gamma_2"]

        k = 0  # iteration number

        # Start with the initial solution
        new_x = problem.factors["initial_solution"]
        new_solution = self.create_new_solution(new_x, problem)
        recommended_solns.append(new_solution)
        intermediate_budgets.append(expended_budget)

        while expended_budget < problem.factors["budget"] * 0.01:
            k += 1
            fval, Y, q, grad, Hessian, delta_k, expended_budget = self.model_construction(new_x, delta, k, problem, expended_budget)

            # Cauchy reduction
            if np.matmul(np.matmul(grad, Hessian), grad) <= 0:
                tau = 1
            else:
                tau = min(1, norm(grad) ** 3 / (delta * np.matmul(np.matmul(grad, Hessian), grad)))

            grad = np.reshape(grad, (1, problem.dim))[0]
            candidate_x = new_x - tau * delta * grad / norm(grad)
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
                sig = candidate_solution.objectives_var
                if sample_size >= self.samplesize(k, sig, delta_k):
                    break

            # calculate success ratio
            fval_tilde = -1 * problem.minmax[0] * candidate_solution.objectives_mean

            # replace the candidate x if the interpolation set has lower objective function value
            if min(fval) < fval_tilde:
                minpos = fval.index(min(fval))
                fval_tilde = min(fval)
                candidate_x = Y[minpos][0]

            if (self.local_model_evaluate(np.zeros(problem.dim), q) - self.local_model_evaluate(
                    np.array(candidate_x) - np.array(new_x), q)) == 0:
                rho = 0
            else:
                rho = (fval[0] - fval_tilde) / (
                            self.local_model_evaluate(np.zeros(problem.dim), q) - self.local_model_evaluate(
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

        return final_ob, k, delta_k, recommended_solns, intermediate_budgets, expended_budget, new_x

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
        delta_max = self.factors["delta_max"]
        tau = self.factors["tau"]
        delta_candidate = [tau * delta_max, delta_max, delta_max / tau]
        #print(delta_candidate)

        # default values
        eta_1 = self.factors["eta_1"]
        eta_2 = self.factors["eta_2"]
        gamma_1 = self.factors["gamma_1"]
        gamma_2 = self.factors["gamma_2"]
        k = 0  # iteration number

        # Start with the initial solution
        new_x = problem.factors["initial_solution"]
        new_solution = self.create_new_solution(new_x, problem)
        recommended_solns.append(new_solution)
        intermediate_budgets.append(expended_budget)

        # Parameter tuning run
        tp_final_ob_pt, k, delta, recommended_solns, intermediate_budgets, expended_budget, new_x = self.parameter_tuning(
            delta_candidate[0], problem)
        for i in range(1, 3):
            final_ob_pt, k_pt, delta_pt, recommended_solns_pt, intermediate_budgets_pt, expended_budget_pt, new_x_pt = self.parameter_tuning(
                delta_candidate[i], problem)
            expended_budget += expended_budget_pt
            if -1 * problem.minmax[0] * final_ob_pt < -1 * problem.minmax[0] * tp_final_ob_pt:
                k = k_pt
                delta = delta_pt
                recommended_solns = recommended_solns_pt
                intermediate_budgets = intermediate_budgets_pt
                new_x = new_x_pt

        intermediate_budgets = (
                    intermediate_budgets + 2 * np.ones(len(intermediate_budgets)) * problem.factors["budget"] * 0.01).tolist()
        intermediate_budgets[0] = 0

        while expended_budget < problem.factors["budget"]:
            k += 1
            fval, Y, q, grad, Hessian, delta_k, expended_budget = self.model_construction(new_x, delta, k, problem,
                                                                                          expended_budget)

            # Cauchy reduction
            if np.matmul(np.matmul(grad, Hessian), grad) <= 0:
                tau = 1
            else:
                tau = min(1, norm(grad) ** 3 / (delta * np.matmul(np.matmul(grad, Hessian), grad)))

            grad = np.reshape(grad, (1, problem.dim))[0]
            candidate_x = new_x - tau * delta * grad / norm(grad)

            for i in range(problem.dim):
                if candidate_x[i] < problem.lower_bounds[i]:
                    candidate_x[i] = problem.lower_bounds[i] + 0.01
                elif candidate_x[i] > problem.upper_bounds[i]:
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
                sig = candidate_solution.objectives_var
                if sample_size >= self.samplesize(k, sig, delta_k):
                    break

            # calculate success ratio
            fval_tilde = -1 * problem.minmax[0] * candidate_solution.objectives_mean

            # replace the candidate x if the interpolation set has lower objective function value
            if min(fval) < fval_tilde:
                minpos = fval.index(min(fval))
                fval_tilde = min(fval)
                candidate_x = Y[minpos][0]

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
