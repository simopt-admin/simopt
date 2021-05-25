"""
Summary
-------
ASTRODF
Cannot handle stochastic constraints.
"""
from base import Solver, Solution
from numpy.linalg import inv
from numpy.linalg import norm
import numpy as np
import math

class ASTRODF(Solver):
    """
    A solver that randomly samples solutions from the feasible region.
    Take a fixed number of replications at each solution.

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
    fixed_factors : dict
        fixed_factors of the solver

    See also
    --------
    base.Solver
    """
    def __init__(self, fixed_factors={}):
        self.name = "ASTRODF"
        self.objective_type = "single"
        self.constraint_type = "deterministic"
        self.variable_type = "continuous"
        self.gradient_needed = False
        self.specifications = {
            "sample_size": {
                "description": "Sample size per solution",
                "datatype": int,
                "default": 10
            }
        }
        self.check_factor_list = {
            "sample_size": self.check_sample_size,
        }
        super().__init__(fixed_factors)

    def check_sample_size(self):
        return self.factors["sample_size"] > 0

    def check_solver_factors(self):
        pass
    
    def standard_basis(self, size, index):
        arr = np.zeros(size)
        arr[index] = 1.0
        return arr
    
    def local_model_evaluate(self,x_k,q):
        X = [1] 
        X = np.append(X, np.array(x_k))
        X = np.append(X, np.array(x_k)**2)
        return np.matmul(X,q)
    
    def samplesize(self,k,sig,delta):
        alpha_k = 1
        lambda_k = 10*math.log(k,10)**1.5
        kappa = 10**2
        S_k = math.floor(max(5,lambda_k,(lambda_k*sig)/((kappa^2)*delta**(2*(1+1/alpha_k)))))
        #S_k = math.floor(max(lambda_k,(lambda_k*sig)/((kappa^2)*delta**(2*(1+1/alpha_k)))))
        return S_k
    
    def model_construction(self,x_k,delta,k,lin_quad,problem,expended_budget, crn_across_solns):        
        w = 0.9 
        mu = 100
        beta = 50
        j = 0
        d = problem.dim
        while True:
            fval = []
            j = j+1
            delta_k = delta*w**(j-1)
            
            # make the interpolation set
            Y = self.interpolation_points(x_k,delta,lin_quad,problem)
            
            for i in range(2*d+1):              
                new_solution = self.create_new_solution(Y[i][0], problem, crn_across_solns)
                
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
                    if sample_size >= self.samplesize(k,sig,delta_k):
                        break
                fval.append(-1*problem.minmax[0]*new_solution.objectives_mean)
                #print(-1*problem.minmax[0]*new_solution.objectives_mean)
                #print(new_solution.objectives_mean)
                #fval.append(new_solution.objectives_mean)
            
            
            Z = self.interpolation_points(np.array(x_k)-np.array(x_k),delta,lin_quad,problem)
            # make the model and get the model parameters
            q,grad,Hessian = self.coefficient(Z,fval,lin_quad,problem)
                       
            # check the condition and break
            if delta_k <= mu*norm(grad):
                break
        
        delta_k = min(max(beta*norm(grad), delta_k),delta)
        return fval,Y,q,grad,Hessian,delta_k,expended_budget
    
    def coefficient(self,Y,fval,lin_quad,problem):
        M = []
        d = problem.dim
        for i in range(0,2*d+1):
            M.append(1) 
            M[i] = np.append(M[i], np.array(Y[i]))
            M[i] = np.append(M[i], np.array(Y[i])**2)   
        
        q = np.matmul(inv(M),fval)
        Hessian = np.diag(q[d+1:2*d+1])
        return q, q[1:d+1], Hessian
            
    def interpolation_points(self,x_k,delta,lin_quad,problem):
        Y = [[x_k]]
        d = problem.dim
        if lin_quad == 2:
            for i in range(0,d):
                plus = Y[0] + delta * self.standard_basis(d,i)    
                minus = Y[0] - delta * self.standard_basis(d,i)  
                Y.append(plus)
                Y.append(minus)
        return Y
    
    def solve(self, problem, crn_across_solns):
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
        delta_max = 50
        delta = delta_max
        
        # default values
        eta_1 = 0.10            #threshhold for decent success
        eta_2 = 0.50            #threshhold for good success
        gamma_1 = 1.25          #successful step radius increase
        gamma_2 = 1/gamma_1     #unsuccessful step radius decrease
        lin_quad = 2            #quadratic or linear
        k = 0                   #iteration number

        # Start with the initial solution
        new_x = problem.initial_solution
        new_solution = self.create_new_solution(new_x, problem, crn_across_solns)
        recommended_solns.append(new_solution)
        intermediate_budgets.append(expended_budget)
        
        while expended_budget < problem.budget:
            k += 1
            #print(k)
            fval,Y,q,grad,Hessian,delta_k,expended_budget = self.model_construction(new_x,delta,k,lin_quad,problem,expended_budget, crn_across_solns)

            # Cauchy reduction
            if np.matmul(np.matmul(grad,Hessian),grad) <= 0:
                tau = 1
            else:
                tau = min(1, norm(grad)**3/(delta*np.matmul(np.matmul(grad,Hessian),grad)))

            grad = np.reshape(grad, (1, problem.dim))[0]            
            candidate_x = new_x - tau*delta*grad/norm(grad)     
            candidate_solution = self.create_new_solution(tuple(candidate_x), problem, crn_across_solns)
            
            #adaptive sampling needed        
            problem.simulate(candidate_solution, 1)
            expended_budget += 1
            sample_size = 1
            
            # Adaptive sampling
            while True:
                problem.simulate(candidate_solution, 1)
                expended_budget += 1
                sample_size += 1
                sig = candidate_solution.objectives_var
                if sample_size >= self.samplesize(k,sig,delta_k):
                    break
            
            # calculate success ratio            
            fval_tilde = -1*problem.minmax[0]*candidate_solution.objectives_mean
            
            # replace the candidate x if the interpolation set has lower objective function value
            if min(fval) < fval_tilde:
                minpos = fval.index(min(fval)) 
                fval_tilde = min(fval)
                candidate_x = Y[minpos][0]
                
            rho = (fval[0] - fval_tilde)/(self.local_model_evaluate(np.zeros(problem.dim),q) - self.local_model_evaluate(candidate_x-new_x,q));
 
            if rho >= eta_2: #very successful
                new_x = candidate_x
                #print(new_x)
                #print(grad)
                delta_k = min(gamma_1*delta_k, delta_max)
                recommended_solns.append(candidate_solution)
                intermediate_budgets.append(expended_budget)
            elif rho >= eta_1: #successful
                new_x = candidate_x
                #print(new_x)
                #print(grad)
                delta_k = min(delta_k, delta_max)
                recommended_solns.append(candidate_solution)
                intermediate_budgets.append(expended_budget)
            else:
                delta_k = min(gamma_2*delta_k, delta_max)
                
        return recommended_solns, intermediate_budgets
