#https://github.com/bodono/apgpy
import numpy as np
import cvxpy as cp
#from apgwrapper import NumpyWrapper
#from functools import partial

import warnings
warnings.filterwarnings("ignore")

from ..base import Solver

class BoomProxGD(Solver):
    """
    
    """
    def __init__(self, name="Boom-PGD", fixed_factors={}):
        self.name = name
        self.objective_type = "single"
        self.constraint_type = "deterministic"
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
            "max_iters": {
                "description": "maximum iterations",
                "datatype": int,
                "default": 300
            },
            "theta": {
                "description": "constant in the line search condition",
                "datatype": int,
                "default": 0.2
            },
            "line_search_max_iters": {
                "description": "maximum iterations for line search",
                "datatype": int,
                "default": 20
            },
            "ratio": {
                "description": "decay ratio in line search",
                "datatype": float,
                "default": 0.8
            },
            "max_gamma":{
                "description": "max distance to the next iteration",
                "datatype": float,
                "default": 1
            },
            "backtrack":{
                "description": "an indicator whether we do the backtrack",
                "datatype": bool,
                "default": 0
            },
            "proj_thres":{
                "description": "proportion of the max iters to stop if have too many projections",
                "datatype": float,
                "default": 0.1
            }  
        }
        
        self.check_factor_list = {
            "crn_across_solns": self.check_crn_across_solns,
            "r": self.check_r,
            "max_iters": self.check_max_iters,
            "proj_thres":self.check_proj_thres
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
    
    def check_proj_thres(self):
        
        return self.factors["proj_thres"] > 0 and self.factors["proj_thres"] < 1
    
    def proj(self,z,Ci,di,Ce,de,lower,upper):
        '''
        project a point z onto the constraint
        Ax <= b depending on the constraint type
        '''
        n = len(z)
        u = cp.Variable(n)

        objective = cp.Minimize(cp.square(cp.norm(u-z)))
        constraints = []
        
        if(lower is not None):
            constraints += [u >= lower]
        if(upper is not None):
            constraints += [u <= upper]
        
        if (Ci is not None) and (di is not None):
            constraints += [Ci@u <= di]
        if (Ce is not None) and (de is not None):
            constraints += [Ce@u == de]

        #constraints = [A@u <= b]
        prob = cp.Problem(objective, constraints)
        prob.solve()#solver=cp.ECOS

        return u.value
    
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
    
    def LineSearch(self,cur_sol,grad,d,max_step,problem,expended_budget):
        """
        carry out line search on the function F where we 
        min F(x + alpha*d) s.t. alpha >=0 

        cur_sol: starting point
        d: direction
        grad: gradient at the point cur_sol
        max_step: literally max step
        ratio: decay ratio if fails
        """
        
        r = self.factors["r"]
        ratio = self.factors["ratio"]
        max_iter = self.factors["line_search_max_iters"]
        
        cur_iter = 0
        step_size = max_step
        added_budget = 0
        
        cur_x = cur_sol.x
        curF = -1 * problem.minmax[0] * cur_sol.objectives_mean
        
        while True:
            if(expended_budget + added_budget > problem.factors["budget"] or cur_iter >= max_iter):
                break
                
            new_x = cur_x + step_size*d
            new_sol =  self.create_new_solution(tuple(new_x), problem)
            problem.simulate(new_sol, r)
            added_budget += r
            
            newF = -1 * problem.minmax[0] * new_sol.objectives_mean
            
            if(newF < curF + self.factors['theta'] * step_size * np.dot(grad, d)):
            #if(newF < curF):
                break
            step_size = step_size*ratio
            cur_iter += 1
        #print("newF: ", newF)
        #print("linear F: ", curF + self.factors['theta'] * step_size * np.dot(grad, d))
        #print("inner iter: ", cur_iter)   
        return step_size, added_budget
    
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

    def solve(self, problem):
        
        #print("Starting PGD")
        
        max_iters = self.factors['max_iters']
        proj_thres = self.factors['proj_thres']
        r = self.factors["r"]
        dim = problem.dim
        
        Ci = problem.Ci
        di = problem.di
        Ce = problem.Ce
        de = problem.de
        
        lower = np.array(problem.lower_bounds)
        upper = np.array(problem.upper_bounds)
        
        #print("Ci: ", Ci)
        #print("di: ", di)
        
        recommended_solns = []
        intermediate_budgets = []
        expended_budget = 0
        
        #store consecutive projections
        consec_proj = 0
        k = 0
        
        # Start with the initial solution.
        new_solution = self.create_new_solution(problem.factors["initial_solution"], problem)
        #cur_x = new_solution.x
        
        # Use r simulated observations to estimate the objective value.
        problem.simulate(new_solution, r)
        expended_budget += r
        best_solution = new_solution
        recommended_solns.append(new_solution)
        intermediate_budgets.append(expended_budget)

        while expended_budget < problem.factors["budget"] and consec_proj < proj_thres*max_iters:
            cur_x = new_solution.x

            proj_trace = int(proj_thres*max_iters)
            
            #computeing the gradients
            if problem.gradient_available:
                # Use IPA gradient if available.
                grad = -1 * problem.minmax[0] * new_solution.objectives_gradients_mean[0]
            else:
                # Use finite difference to estimate gradient if IPA gradient is not available.
                #grad, budget_spent = self.finite_diff(new_solution, problem, r, stepsize = alpha)
                grad, budget_spent = self.get_FD_grad(cur_x, problem, self.factors["h"], self.factors["r"])
                expended_budget += budget_spent
                # A while loop to prevent zero gradient.
                #while np.all((grad == 0)):
                #    if expended_budget > problem.factors["budget"]:
                #        break
                    #grad, budget_spent  = self.finite_diff(new_solution, problem, r)
                    #grad, budget_spent = self.get_FD_grad(self, x, problem, h, r)
                #    grad, budget_spent = self.get_FD_grad(cur_x, problem, self.factors["h"], self.factors["r"])
                #    expended_budget += budget_spent
                    # Update r after each iteration.
                    #r = int(self.factors["lambda"] * r)
            direction = -grad/np.linalg.norm(grad)
            #step sizes
            if(self.factors["backtrack"]):
                t, added_budget = self.LineSearch(new_solution,grad,direction,self.factors["max_gamma"],problem,expended_budget)
                expended_budget += added_budget
            else:
                #t = min(self.factors["step_f"](k),self.factors["max_gamma"])
                t = self.factors["step_f"](k)#*direction#np.linalg.norm(grad)

            #print("grad: ", grad)
            #print("step: ", t)
            #t = self.factors['step_f'](k) 
            #new_x = cur_x - t * grad
            new_x = cur_x + t * direction
            
            #print("new_x before proj: ", new_x)

            #if the new iterate is feasible, then no need to project
            if(not self.is_feasible(new_x, Ci,di,Ce,de,lower,upper)):
                new_x = self.proj(new_x,Ci,di,Ce,de,lower,upper)
                consec_proj += 1
            else:
                if(consec_proj > 0):
                    consec_proj -= 1
            #print("new_x after proj: ", new_x)
            
            candidate_solution = self.create_new_solution(tuple(new_x), problem)
            # Use r simulated observations to estimate the objective value.
            problem.simulate(candidate_solution, r)
            expended_budget += r
            
            new_solution = candidate_solution
            recommended_solns.append(candidate_solution)
            intermediate_budgets.append(expended_budget)
            
            k += 1
        #print("------------------------------------")   
        return recommended_solns, intermediate_budgets    
    