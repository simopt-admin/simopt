""" 
Summary

Simultaneous perturbation stochastic approximation (SPSA) is an algorithmic method for optimizing systems with multiple unknown parameters.

"""

from random import random
from base import Solver
from base import random
import numpy as np

class SPSA(solver):
    """
    Simultaneous perturbation stochastic approximation (SPSA) is an algorithmic method for optimizing systems with multiple unknown parameters.
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
    def __init__(self, name="SPSA", fixed_factors={}):
            self.name = name
            self.objective_type = "single"                  ### Need to update these
            self.constraint_type = "stochastic"
            self.variable_type = "mixed"
            self.gradient_needed = False
            self.specifications = {
                "sensititvity": {
                    "description": "shrinking scale for VarBds",
                    "datatype": float,
                    "default": 10^(-7)
                },
                "alpha": {
                    "description": "Non negative coefficient",
                    "datatype": float,
                    "default": 0.602
                },
                "gamma": {
                    "description": "Non negative coefficient",
                    "datatype": float,
                    "default": 0.101
                },
                "step": {
                    "description": "The initial desired magnitude of change in the theta elements",
                    "datatype": float,
                    "default": 0.1
                },
                "gavg": {
                    "description": "How many averaged SP gradients will be used per iteration?",
                    "datatype": float,
                    "default": 1
                },
                "r": {
                    "description": "Number of replications takes at each solution",   #increased r means more accurate gradient
                    "datatype": int,
                    "default": 30
                },
                "NL": {
                    "description": "How many loss function evaluations do you want to use in this gain calculation?",
                    "datatype": int,
                    "default": 2
                }
            }
            self.check_factor_list = {
                "sensititvity": self.check_sensititvity,
                "alpha": self.check_alpha,
                "gamma": self.check_gamma,
                "step": self.check_step,
                "gavg": self.check_gavg,
                "r": self.check_r,
                "NL": self.check_NL
            }
            super().__init__(fixed_factors)

    def check_sensititvity(self):
        return self.factors["sensititvity"] > 0

    def check_alpha(self):
        return self.factors["alpha"] > 0

    def gamma(self):
        return self.factors["check_gamma"] > 0

    def step(self):
        return self.factors["check_step"] > 0

    def check_gavg(self):
        return self.factors["gavg"] > 0

    def check_r(self):
        return self.factors["r"] > 0

    def check_NL(self):
        return self.factors["NL"] > 0

    def gen_simul_pert_vec(self, inital_solution):
        """Generates a new simulatanious pertubation vector to be applied, uses input of inital solution to get vector length"""
        
        SP_vect = []
        
        for i in inital_solution:
            delta = random.choices([-1,1], [.5,.5])
            SP_vect.append(delta)
        
        return SP_vect

    def solve(self, problem):
        """
        Run a single macroreplication of a solver on a problem.
        Arguments
        ---------
        problem : Problem object
            simulation-optimization problem to solve
        
>>      Should I have anything else in this section?

        Returns
        -------
        Reccomended_solutions

        Intermediate_budgets 
        
        #loss function is embedded in the solution object
        
        Anncalls : Array
            An array (size = 'NumSoln' X 1) of budget expended
        A : Array
             An array (size = 'NumSoln' X 'dim') of solutions returned by solver
        AFnMean : Array
            An array (size = 'NumSoln' X 1) of estimates of expected objective function value
        AFnVar : Array 
            An array of variances corresponding to the objective function at A 
            Equals NaN if solution is infeasible
        AFnGrad : Array
            An array of gradient estimates at A; not reported
        AFnGradCov : Array 
            An array of gradient covariance matrices at A; not reported
        AConstraint : Vector
            A vector of constraint function estimators; not applicable
        AConstraintCov : Arrary
            An array of covariance matrices corresponding to the constraint function at A; not applicable
        AConstraintGrad : Array
            An array of constraint gradient estimators at A; not applicable
        AConstraintGradCov: Array
            An array of covariance matrices of constraint gradient estimators at A; not applicable
        """
        recommended_solns = []
        intermediate_budgets = []
        expended_budget = 0
        ghat = [0,0]
        c = max(fthetaVar/self.factors["gavg"]^0.5,.0001)   #check line 113 in matlab code, need to determine fthetavar
        A = 1 + int(problem.factors["budget"]/self.factors["r"])
        k = 0
        # Designate random number generator for random sampling.
        find_next_soln_rng = self.rng_list[2] #the rng list is automatically integrated into the .random class
        # Sequentially generate random solutions and simulate them.
        while expended_budget < problem.factors["budget"]:
            if expended_budget == 0:
                # Start at initial solution and record as best.
                new_x = problem.factors["initial_solution"] #CRNs = Common Random Numbers, not what is used here, need a specic RN Stream #3, Look at line 97 on RandSearch
                new_solution = new_x #function of a solver super class under base.py
                best_solution = new_solution                            #Can use random.choices([-1,1], .5,.5)
                recommended_solns.append(new_solution)
                intermediate_budgets.append(expended_budget)
            else:
                #generate Simulanious Pertubation Vector
                k += 1
                delta_k = self.gen_simul_pert_vec(new_solution) #outputs either +tv or -tv integers aka +1 or -1 for every variable 
                #Loss function evaluation
                ck = c/(k^self.factors(["gamma"]))
                ak = A/(k + )
                #loss func 1
                L1 = np.dot(new_solution,np.add(delta_k,np.multiply(ck,delta_k)))
                #loss func 2
                L2 = np.dot(new_solution,np.subtract(delta_k,np.multiply(ck,delta_k)))
                #Gradient Approximation
                ghat = np.dot(np.divide(L1-L2, 2 * ck), np.transform(delta_k))
                #Updating previous solution estimate
                new_solution = new_solution - ghat
                recommended_solns.append(new_solution)
                #iteration or termnination
                
                
                
            # Also check for feasibility w.r.t. stochastic constraints.
            if (problem.minmax * new_solution.objectives_mean
                    > problem.minmax * best_solution.objectives_mean and
                    all(new_solution.stoch_constraints_mean[idx] >= 0 for idx in range(problem.n_stochastic_constraints))):
                # If better, record incumbent solution as best.
                best_solution = new_solution
                recommended_solns.append(new_solution)
                intermediate_budgets.append(expended_budget)
        return recommended_solns, intermediate_budgets
