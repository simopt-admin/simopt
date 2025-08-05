#https://github.com/bodono/apgpy
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
#from apgwrapper import NumpyWrapper
#from functools import partial

import warnings
warnings.filterwarnings("ignore")

from ..base import Solver

class CSA_LP(Solver):
    """
    modified Cooperative Stochastic Approximation method by Lan, G. and Zhou Z.
    by improving multiple constraints at the same time
    """
    def __init__(self, name="CSA_LP", fixed_factors={"max_iters": 300}):
        self.name = name
        self.objective_type = "single"
        self.constraint_type = "stochastic"
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
            "h": {
                "description": "difference in finite difference gradient",
                "datatype": float,
                "default": 0.1
            },
            "step_f": {
                "description": "step size function",
                "datatype": "function",
                "default": self.default_step_f
            },
            "ratio": {
                "description": "decay ratio in line search",
                "datatype": float,
                "default": 0.8
            },
            "tolerance":{
                "description": "tolerence function",
                "datatype": "function",
                "default": 0.01
            },
            "max_iters": {
                "description": "maximum iterations",
                "datatype": int,
                "default": 300
            }
        }

        self.check_factor_list = {
            "crn_across_solns": self.check_crn_across_solns,
            "r": self.check_r,
            "max_iters": self.check_max_iters
        }
        super().__init__(fixed_factors)
        
    def default_step_f(self,k):
        """
        take in the current iteration k
        """
        return 1/(k+1)
    
    def check_r(self):
        
        return self.factors['r'] > 0
    
    def check_max_iters(self):
        
        return self.factors['max_iters'] > 0
    
    def is_feasible(self, x, Ci,di,Ce,de,lower,upper):
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
        res = True
        
        if(lower is not None):
            res = res & np.all(x >= lower)
        if(upper is not None):
            res = res & np.all(x <= upper)
        
        if (Ci is not None) and (di is not None):
            res = res & np.all(Ci @ x <= di)
        if (Ce is not None) and (de is not None):
            res = res & (np.allclose(np.dot(Ce, x), de))
        return res 
    
        
    def get_simulated_values(self,problem,x,value = 'both'):
        """
        getting either sample path or gradient. The return "value"
        can be specified to "val"|"gradient"|"both"
        """
        r = self.factors["r"]
        sol =  self.create_new_solution(tuple(x), problem)
        problem.simulate(sol, r)
        budget = 0
        
        #getting the function evaluation
        if((value == "both") or (value == "val")):
            budget += r
            Fval = -1 * problem.minmax[0] * sol.objectives_mean
        
        if((value == "both") or (value == "gradient")):
            if problem.gradient_available:
                # Use IPA gradient if available.
                gradient = -1 * problem.minmax[0] * sol.objectives_gradients_mean[0]
            else:
                gradient, budget_spent = self.get_FD_grad(x, problem, self.factors["h"], self.factors["r"])
                budget += budget_spent
                
        if(value == "val"):
            return Fval, budget
        elif(value == "gradient"):
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

        if(d == 1):
            #xsol = self.create_new_solution(tuple(x), problem)
            x1 = x + h/2
            x2 = x - h/2
            
            x1 = self.create_new_solution(tuple(x1), problem)
            problem.simulate(x1, r)
            f1 = -1 * problem.minmax[0] * x1.objectives_mean

            x2 = self.create_new_solution(tuple(x2), problem)
            problem.simulate(x2, r)
            f2 = -1 * problem.minmax[0] * x2.objectives_mean
            grad = (f1-f2)/h
        else:
            I = np.eye(d)
            grad = 0
            
            for i in range(d):
                x1 = x + h*I[:,i]/2
                x2 = x - h*I[:,i]/2
                
                x1 = self.create_new_solution(tuple(x1), problem)
                problem.simulate_up_to([x1], r)
                f1 = -1 * problem.minmax[0] * x1.objectives_mean

                x2 = self.create_new_solution(tuple(x2), problem)
                problem.simulate_up_to([x2], r)
                f2 = -1 * problem.minmax[0] * x2.objectives_mean

                grad += ((f1-f2)/h)*I[:,i]
              
        return grad, (2*d*r) 
    
    def get_violated_constraints_grads(self,constraints_results,grads):
        '''
        get all violated constraints gradients
        '''
        n_cons, n = grads.shape
        violated_cons_grads = []
        
        for i in range(n_cons):
            if(constraints_results[i] > self.factors['tolerance']):
                violated_cons_grads.append(grads[i])
    
        return np.array(violated_cons_grads)
    
    def get_constraints_dir(self,grads):
        '''
        compute search direction by LP to improve 
        multiple constraints at the same time
        
        grads: collects vectors of gradients from multiple 
        violated constraint(s), expect 2d array
        
        solve for d s.t. 
        
        max theta s.t. gi^Td >= theta
        '''
        #normalization ?
        #print("grads: ", grads)
        
        n_violated_cons, n = grads.shape
        
        if(n_violated_cons == 1):
            d = grads[0]
        else:
            #normalization
            grads = grads/np.linalg.norm(grads,axis = 1).reshape(n_violated_cons,1)

            direction = cp.Variable(n)
            theta = cp.Variable()

            objective = cp.Maximize(theta)
            constraints = [theta >= 0, theta <= 1]

            for i in range(n_violated_cons):
                constraints += [grads[i]@direction >= theta]

            prob = cp.Problem(objective, constraints)
            prob.solve()

            d = direction.value
            #d = d/np.linalg.norm(d)
        
        return d
    
    
    def prox_fn(self,a,cur_x,Ci,di,Ce,de,lower,upper):
        '''
        prox function for CSA
        'a' is an input, typically use 'step*grad'
        solve the minimization problem (z)
        
        aTz + 0.5*||z - cur_x||^2
        
        by default, use the euclidean distance
        '''
        #print("a: ", a)
        #print("cur x: ", cur_x)
        n = len(cur_x)
        z = cp.Variable(n)

        objective = cp.Minimize(a@z + 0.5*(cp.norm(cur_x-z)**2))
        constraints = []
        
        if((lower is not None) and (lower > -np.inf).all()):
            constraints += [z >= lower]
        if((upper is not None) and (upper < np.inf).all()):
            constraints += [z <= upper]
        
        if (Ci is not None) and (di is not None):
            constraints += [Ci@z <= di]
        if (Ce is not None) and (de is not None):
            constraints += [Ce@z == de]
            
        prob = cp.Problem(objective, constraints)
        prob.solve()
        
        return z.value
        

    def solve(self, problem):
        max_iters = self.factors['max_iters']
        r = self.factors["r"]
        #max_gamma = self.factors["max_gamma"]
            
        #t = 1 #first max step size
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
        
        k = 0 #iteration
        
        # Start with the initial solution.
        new_solution = self.create_new_solution(problem.factors["initial_solution"], problem)
        
        # Use r simulated observations to estimate the objective value.
        problem.simulate(new_solution, r)
        expended_budget += r
        best_solution = new_solution
        recommended_solns.append(new_solution)
        intermediate_budgets.append(expended_budget)
        
        #numiter = 0
        numviolated = 0
        last_is_feasible = 1
        infeasible_step = 1

        while expended_budget < problem.factors["budget"]:
            cur_x = new_solution.x

            #check if the constraints are violated
            if(problem.n_stochastic_constraints > 0):
                #constraint_results = problem.stoch_constraint(cur_x) #multiple dim of constraints in the form E[Y] <= 0
                constraint_results = new_solution.stoch_constraints_mean
                #print(constraint_results)
                is_violated = max(constraint_results) > self.factors['tolerance'] #0.01
                #print("constraint violation: ", max(constraint_results))
            else:
                #problems with no stoch constraints
                is_violated = 0
            
            #if the constraints are violated, then improve the feasibility
            if(is_violated):
                #print("violated!")
                #find the gradient of the constraints
                #violated_index = np.argmax(constraint_results)
                grads = np.array(new_solution.stoch_constraints_gradients_mean)
                violated_grads = self.get_violated_constraints_grads(constraint_results,grads)
                #print("num violated cons: ", len(violated_grads))
                #print("violated grads: ", violated_grads)
                #direction for improving multiple constraints, but call it 'grad' for convenience
                grad = self.get_constraints_dir(violated_grads) 

                numviolated += 1
            else:
                #print("cons pass!")
                #if constraints are not violated, then conpute gradients
                #computeing the gradients
                if problem.gradient_available:
                    # Use IPA gradient if available.
                    grad = -1 * problem.minmax[0] * new_solution.objectives_gradients_mean[0]
                else:
                    # Use finite difference to estimate gradient if IPA gradient is not available.
                    #grad, budget_spent = self.finite_diff(new_solution, problem, r, stepsize = alpha)
                    grad, budget_spent = self.get_FD_grad(cur_x, problem, self.factors["h"], self.factors["r"])
                    expended_budget += budget_spent
                    
            if(is_violated):
                #if this iteration is infeasible again, increase step size
                if(last_is_feasible == 0):
                    t = infeasible_step/self.factors['ratio']
                    #infeasible_step = t
                else:
                    t = infeasible_step*self.factors['ratio']
                    #infeasible_step = t
                infeasible_step = t
                last_is_feasible = 0
            else:
                t = self.factors["step_f"](k)
                    
            #step-size 
            #t = self.factors["step_f"](k)
            #print("step: ", t)
            #print("grad: ", grad)
            #new_x = cur_x + t * direction
            new_x = self.prox_fn(t*grad,cur_x,Ci,di,Ce,de,lower,upper)
            
            candidate_solution = self.create_new_solution(tuple(new_x), problem)
            # Use r simulated observations to estimate the objective value.
            problem.simulate(candidate_solution, r)
            expended_budget += r
            
            new_solution = candidate_solution
            
            # Append new solution.
            if (problem.minmax[0] * new_solution.objectives_mean > problem.minmax[0] * best_solution.objectives_mean):
                best_solution = new_solution
                #recommended_solns.append(candidate_solution)
                recommended_solns.append(new_solution)
                intermediate_budgets.append(expended_budget)

            k += 1
            #print("----------------------")
        #print("==========================")
        return recommended_solns, intermediate_budgets    
    