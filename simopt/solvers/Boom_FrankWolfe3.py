import numpy as np
import cvxpy as cp
import gurobipy
import matplotlib.pyplot as plt
#import cdd


import warnings
warnings.filterwarnings("ignore")

from ..base import Solver
#env = gurobipy.Env()
#env.setParam('FeasibilityTol', 1e-9)
#env.setParam('MIPGap',0)


class BoomFrankWolfe3(Solver):
    """
    """
    
    def __init__(self, name="FW-SS", fixed_factors={}):
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
            "theta": {
                "description": "constant in the line search condition",
                "datatype": int,
                "default": 0.1
            },
            "max_iters": {
                "description": "maximum iterations",
                "datatype": int,
                "default": 300
            },  
            "gamma": {
                "description": "constant for shrinking the step size",
                "datatype": int,
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
            "alpha": {
                "description": "initial step size",
                "datatype": int,
                "default": 1
            },
            "epsilon_f": {
                "description": "additive constant in the Armijo condition",
                "datatype": int,
                "default": 1e-3  # In the paper, this value is estimated for every epoch but a value > 0 is justified in practice.
            },
            "LSmethod":{
                "description": "methods for line search algorithm",
                "datatype":str,
                "default":'stepsearch'
            },
            "line_search_max_iters": {
                "description": "maximum iterations for line search",
                "datatype": int,
                "default": 40
            },
            "ratio": {
                "description": "decay ratio in line search",
                "datatype": float,
                "default": 0.8
            },
            "curve_const": {
                "description": "constant in curvature wolfe conditions, usually greater than theta",
                "datatype": float,
                "default": 0.3
            },
            "zoom_init_ratio": {
                "description": "ratio of the max step size in Zoom lien search",
                "datatype": float,
                "default": 0.2
            },
            "zoom_inc_ratio": {
                "description": "increment ratio in Zoom lien search",
                "datatype": float,
                "default": 1.5
            },
            "atom_vectors":{
                "description": "atom vectors for away/pairwise frank-wolfe",
                "datatype": "matrix",
                "default": None
            },
            "max_gamma":{
                "description": "max distance to the next iteration",
                "datatype": float,
                "default": 1
            },
            "backtrack":{
                "description": "an indicator whether we do the backtrack",
                "datatype": bool,
                "default": 1
            },
            "algorithm":{
                "description": "type of FW algorithm",
                "datatype": str,
                "default": "away"
                #away, pairwise
            }
            
        }
        self.check_factor_list = {
            "crn_across_solns": self.check_crn_across_solns,
            "r": self.check_r,
            "max_iters": self.check_alpha_max,
        }
        super().__init__(fixed_factors)
        
    def check_r(self):
        return self.factors["r"] > 0
    
    def check_alpha_max(self):
        
        return self.factors["alha_max"] > 0
    
    def check_max_iters(self):
        
        return self.factors['max_iters'] > 0
        
    def default_step_f(self,k):
        """
        take in the current iteration k
        """
        
        return 1/(k+1)

    def is_feasible(self, x, Ci,di,Ce,de,lower, upper, tol = 1e-8):
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
            res = res & np.all(Ci @ x <= di + tol)
        if (Ce is not None) and (de is not None):
            res = res & (np.allclose(np.dot(Ce, x), de))
        return res
    
    def get_max_gamma_ratio_test(self, cur_x, d, Ce, Ci, de, di, lower, upper):
        '''
        perform a ratio test to find the max step size
        '''
        #step = cp.Variable()
        #objective = cp.Maximize(step)
        #constraints = [step >= 0]
        #ratio test: (bi - ai^Tx)/(ai^Td)
        ratio_val = []
        denom = []
        dim = len(cur_x)
        
        if(lower is not None):
            #constraints += [(cur_x + step*d) >= lower]
            #vals += [(lower[0] - cur_x[i])/(d[i]) for i in range(dim)]
            ratio_val += list((cur_x - lower)/-d)
            denom += list(-d)
        if(upper is not None):
            #constraints += [(cur_x + step*d) <= upper]
            #vals += [(upper[0] - cur_x[i])/(d[i]) for i in range(dim)]
            ratio_val += list((upper - cur_x)/d)
            denom += list(d)
        if((Ci is not None) and (di is not None)):
            #constraints += [Ci@(cur_x + step*d) <= di]
            ratio_val += list((di - Ci@cur_x)/(Ci@d))
            #print("type: ", type(Ci@d))
            if(type(Ci@d) != float and type(Ci@d) != np.float64 and type(Ci@d) != np.float32):
                denom += list(Ci@d)
            else:
                denom += [Ci@d]
        #print(denom)
        #print("ratio: ", ratio_val)
        #print("denom: ", denom)
        ratio_val = np.array(ratio_val)
        denom = np.array(denom)
        #print("denom: ", denom)
        #print("ratio_val: ", ratio_val)
        
        return min(ratio_val[denom > 1e-6])
        #prob = cp.Problem(objective, constraints)
        #prob.solve()
    
    def get_dir(self,g,Ce, Ci, de, di,lower, upper):
        '''
        solve for the direction in each iteration
        given a gradient vector g, find min_s{sg}
        s.t. problem is feasible
        '''
    
        n = len(g)
        s = cp.Variable(n)

        objective = cp.Minimize(s@g)
        constraints = []
        
        if(lower is not None):
            constraints += [s >= lower]
        if(upper is not None):
            constraints += [s <= upper]
        if((Ci is not None) and (di is not None)):
            constraints += [Ci@s <= di]
        if((Ce is not None) and (de is not None)):
            constraints += [Ce@s == de]

        prob = cp.Problem(objective, constraints)
        #prob.solve(solver=cp.GUROBI,env=env)#solver=cp.ECOS
        prob.solve(solver=cp.SCIPY)
        
        return s.value
    
    def get_dir_unbd(self,g,Ce, Ci, de, di,lower, upper):
        '''
        solve for the direction in each iteration
        given a gradient vector g, find min_s{sg}
        s.t. problem is feasible
        '''
    
        n = len(g)
        s = cp.Variable(n)

        objective = cp.Minimize(s@g)
        constraints = []
        
        if(lower is not None):
            constraints += [s >= lower]
        if(upper is not None):
            constraints += [s <= upper]
        if((Ci is not None) and (di is not None)):
            constraints += [Ci@s <= di]
        if((Ce is not None) and (de is not None)):
            constraints += [Ce@s == de]

        prob = cp.Problem(objective, constraints)
        #prob.solve(solver=cp.GUROBI,env=env)#solver=cp.ECOS
        prob.solve(solver=cp.GUROBI,InfUnbdInfo= 1)
        
        if('unbounded' in prob.status):
            result = np.array([prob.solver_stats.extra_stats.getVars()[j].unbdray for j in range(n)])
            is_bounded = False
        else:
            result = s.value
            is_bounded = True
        
        return result, is_bounded
    
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
            x2 = self.create_new_solution(tuple(x2), problem)
            problem.simulate_up_to([x1], r)
            problem.simulate_up_to([x2], r)
            f1 = -1 * problem.minmax[0] * x1.objectives_mean
            f2 = -1 * problem.minmax[0] * x2.objectives_mean
            grad = (f1-f2)/h
        else:
            I = np.eye(d)
            grad = 0
            
            for i in range(d):
                x1 = x + h*I[:,i]/2
                x2 = x - h*I[:,i]/2
                
                x1 = self.create_new_solution(tuple(x1), problem)
                x2 = self.create_new_solution(tuple(x2), problem)
                problem.simulate_up_to([x1], r)
                problem.simulate_up_to([x2], r)
                
                f1 = -1 * problem.minmax[0] * x1.objectives_mean
                f2 = -1 * problem.minmax[0] * x2.objectives_mean
                
                grad += ((f1-f2)/h)*I[:,i]
              
        return grad, (2*d*r)   
    
    #def min_quadratic(div0,f0,):
    #    """
    #    find the (arg)minimum of the quadratic function from
    #    the given info q'(0), q(0), q(alpha) in the interval
    #    [a,b] where a < b
    #    """
    def get_gradient(self,problem,x,sol):
        """
        getting the gradient of the function at point x where
        sol is the solution data structure
        """
        budget = 0
        #get the gradient of the new solution grad f(x + step*d) for curvature condition
        if problem.gradient_available:
            # Use IPA gradient if available.
            gradient = -1 * problem.minmax[0] * sol.objectives_gradients_mean[0]
        else:
            gradient, budget_spent = self.get_FD_grad(x, problem, self.factors["h"], self.factors["r"])
            gradient = -1 * problem.minmax[0] * gradient
            budget += budget_spent
            
        return gradient, budget
    
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
    
    def full_min_quadratic(self, div_a,Fa,Fb,a,b,problem):
        '''
        return the minimum point which is the 
        next step size usig the quadratic 
        interpolation  with the information q(a),
        q(b), q'(a) and q'(b) where a < b
        '''
        #print("div: ",div_a)
        #print("Fa,Fb: ", (Fa,Fb))
        #print("(a,b): ", (a,b))
        #numerator = (a**2 - b**2)*div_a - 2*a*(Fa - Fb)
        #denominator = 2*((a-b)*div_a - (Fa - Fb))
        #result = numerator/denominator
        A = div_a/(a - b) - (Fa - Fb)/((a-b)**2)
        B = div_a - 2*A*a
        result = -B/(2*A)
        
        if(-problem.minmax[0] == np.sign(A)):
            #if A and problem have the same sign, i.e. min and A > 0
            if(result < a):
                return a
            elif(result > b):
                return b
            else:
                return result
        else:
            if(problem.minmax[0] > 0):
                #maximization but A > 0
                return [a,b][np.argmax([Fa,Fb])]
              
            else:
                #minization but A < 0
                return [a,b][np.argmin([Fa,Fb])]
            
    def quadratic_interpolate(self,x1,x2,div_x1,div_x2,Fx1,Fx2,problem):
        '''
        interpolate the quadratic polynomial using given points
        and return the lowest (arg)point
        '''

        if(x2 > x1):
            #we use div_x1,x1,x2
            #return min_quadratic(div_x1,Fx1,Fx2,x2)
            return self.full_min_quadratic(div_x1,Fx1,Fx2,x1,x2,problem)
        else:
            #we use div_x2,x2,x1
            #return min_quadratic(div_x2,Fx2,Fx1,x1)
            return self.full_min_quadratic(div_x2,Fx2,Fx1,x2,x1,problem)
    
    def backtrackLineSearch(self,cur_sol,grad,d,max_step,problem,expended_budget):
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
        #print("cur_x: ", cur_x)
        curF = -1 * problem.minmax[0] * cur_sol.objectives_mean
        #print("FW-BT Line Search...")
        #xrange = np.arange(0,max_step,0.01)
        #nn = len(xrange)
        #fval = np.zeros(nn)
        #for i in range(nn):
        #    temp_x = cur_sol.x + xrange[i]*d
        #    temp_sol = self.create_new_solution(tuple(temp_x), problem)
        #    problem.simulate(temp_sol, r)
        #    fval[i] = temp_sol.objectives_mean
        #plt.scatter(xrange,fval)
        #plt.show()
        while True:
            #if(expended_budget + added_budget > problem.factors["budget"] or cur_iter >= max_iter):
            if(cur_iter >= max_iter):
                break;
                  
            new_x = cur_x + step_size*d
            new_sol =  self.create_new_solution(tuple(new_x), problem)
            problem.simulate(new_sol, r)
            added_budget += r
          
            newF = -1 * problem.minmax[0] * new_sol.objectives_mean
            #print("newF: ",newF)
            
            if(newF < curF + self.factors['theta'] * step_size * np.dot(grad, d)):
                break;
                
            step_size = step_size*ratio
            cur_iter += 1
            #print("---------------")
        #print("step from backtrack: ",step_size)
        
        return step_size, added_budget
    
    def stepLineSearch(self,cur_sol,grad,d,max_step,problem,expended_budget):
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
        theta = self.factors['theta']
        epsilon_f = self.factors['epsilon_f']
        alpha_max = self.factors['alpha_max']
        gamma = self.factors['gamma']
        alpha = self.factors['alpha']
        alpha_0 = max_step
        
        step_size = max_step
        added_budget = 0
        
        x = cur_sol.x
        fx = -1 * problem.minmax[0] * cur_sol.objectives_mean
        # step_size = self.factors['alpha_0']
        new_solution = cur_sol
        x_new = x + step_size * d
        # Create a solution object for x_new.
        x_new_solution = self.create_new_solution(tuple(x_new), problem)
        # Use r simulated observations to estimate the objective value.
        problem.simulate(x_new_solution, r)
        added_budget += r
            
        # Check the modified Armijo condition for sufficient decrease.
        if (-1 * problem.minmax[0] * x_new_solution.objectives_mean) <= (
                -1 * problem.minmax[0] * cur_sol.objectives_mean + alpha * theta * np.dot(grad, d) + 2 * epsilon_f):
            # Successful step
            new_solution = x_new_solution
            # Enlarge step size.
            # alpha = min(alpha_max, alpha / gamma)
            step_size = min(alpha_0, step_size/gamma)
        else:
            # Unsuccessful step - reduce step size.
            # alpha = gamma * alpha
            step_size = gamma * step_size
            
        # self.factors['alpha'] = alpha
        
        return step_size, added_budget
    
    def interpolateLineSearch(self,cur_sol,grad,d,max_step,problem,expended_budget):
        """
        carry out interpolation line search on the function F where we 
        min F(x + alpha*d) s.t. alpha >=0  where phi(a) = F(x + ad)
        """
        #print("Interpolation LS")
        r = self.factors["r"]
        ratio = self.factors["ratio"]
        max_iter = self.factors["line_search_max_iters"]
        sign = -problem.minmax[0]
        
        cur_iter = 0
        step_size = max_step
        added_budget = 0
        
        cur_x = cur_sol.x
        curF = -1 * problem.minmax[0] * cur_sol.objectives_mean
        #print("max_step: ", max_step)
        if(max_step == 0):
            return max_step, added_budget
        
        #xrange = np.arange(0,max_step,0.01)
        #nn = len(xrange)
        #fval = np.zeros(nn)
        #for i in range(nn):
        #    temp_x = cur_x + xrange[i]*d
        #    temp_sol = self.create_new_solution(tuple(temp_x), problem)
        #    problem.simulate(temp_sol, r)
        #    fval[i] = temp_sol.objectives_mean
        #plt.scatter(xrange,fval)
        #line = -1*problem.minmax[0]*curF + self.factors["theta"]*xrange*(-1*problem.minmax[0]* grad.dot(d))
        #plt.scatter(xrange,line,color='red')
        #plt.show()
        
        while True:
        #while(not suff dec and cur iter)
        #while((newF >= curF + self.factors['theta'] * step_size * np.dot(grad, d)) and (cur_iter < max_iter)):
            #if(expended_budget + added_budget > problem.factors["budget"] or cur_iter >= max_iter):
            if(cur_iter >= max_iter):
                break;
            #print("cur step size: ", step_size)    
            new_x = cur_x + step_size*d
            #print("LS new x: ",new_x)
            new_sol =  self.create_new_solution(tuple(new_x), problem)
            problem.simulate(new_sol, r)
            added_budget += r
            #print("dir size: ", np.linalg.norm(d))
            newF = -1 * problem.minmax[0] * new_sol.objectives_mean
            
            #sufficient decrease
            if(newF < curF + self.factors['theta'] * step_size * np.dot(grad, d)):
                #print("sufficient by interpolation")
                break;
            
            #quadratic interpolation using phi(0), phi'(0), phi(step)
            #new_step_size = ((-grad.dot(d))*(step_size**2))/(2*(newF-curF-step_size*(grad.dot(d))))
            new_step_size = self.full_min_quadratic(sign*grad.dot(d),sign*curF,sign*newF,0,step_size,problem)
            #print("grad . d: ", grad.dot(d))
            #print("opt new step: ", new_step_size)
            if(abs(new_step_size) >= 1e-4):
                #if we can make some progress
                #new_step_size = ((-grad.dot(d))*(step_size**2))/(2*(newF-curF-step_size*(grad.dot(d))))
                step_size = min(new_step_size,max_step)
            else:
                #if we did not make progress, use the informaiton {step*ratio}
                temp_x = cur_x + (step_size*ratio)*d
                temp_sol =  self.create_new_solution(tuple(temp_x), problem)
                problem.simulate(temp_sol, r)
                added_budget += r
                newF = -1 * problem.minmax[0] * temp_sol.objectives_mean
                #print("another newF: ", newF)
                #new_step_size = ((-grad.dot(d))*((step_size*ratio)**2))/(2*(newF-curF-(step_size*ratio)*(grad.dot(d))))
                new_step_size = self.full_min_quadratic(sign*grad.dot(d),sign*curF,sign*newF,0,step_size*ratio,problem)
                #check if it's in the interval
                if(new_step_size <= 1e-15): #outside interval (too small)
                    step_size = 0
                    break;
                elif(new_step_size > step_size*ratio): #outside interval (too big)
                    step_size = step_size*ratio
                else:
                    step_size = new_step_size
                
            #print("new step: ", step_size)
            cur_iter += 1
            #print("iteration: ", cur_iter)
            #print("=============")  
        #print("Inter step: ",step_size)
        #print("-------end of LS--------")
        return step_size, added_budget
    
    def zoomLineSearch(self,cur_sol,grad,d,max_step,problem,expended_budget):
        """
        carry out interpolation line search on the function F where we 
        min F(x + alpha*d) s.t. alpha >=0  where phi(a) = F(x + ad)
        
        NOTE: in this method, we increase the step size
        """
        if(max_step == 0):
            return 0,0
        r = self.factors["r"]
        ratio = self.factors["ratio"]
        max_iter = self.factors["line_search_max_iters"]
        sign = -problem.minmax[0]
        
        cur_iter = 0
        #step_size = max_step
        cur_step_size = max_step*self.factors["zoom_init_ratio"]
        last_step_size = 0
        last_grad = grad
        added_budget = 0

        #xrange = np.arange(0,max_step,0.01)
        #nn = len(xrange)
        #fval = np.zeros(nn)
        #for i in range(nn):
        #    temp_x = cur_sol.x + xrange[i]*d
        #    temp_sol = self.create_new_solution(tuple(temp_x), problem)
        #    problem.simulate(temp_sol, r)
            #fval[i] = -1 * problem.minmax[0] * temp_sol.objectives_mean
        #    fval[i] = temp_sol.objectives_mean
        #plt.scatter(xrange,fval)
        #plt.show()

        cur_x = cur_sol.x
        curF = -1 * problem.minmax[0] * cur_sol.objectives_mean
        lastF = curF
        nextF, next_grad, budget_spent = self.get_simulated_values(problem,cur_x + cur_step_size*d,value = 'both')
        added_budget += budget_spent
        
        returned_steps = [0]
        returned_vals = [curF]
        #line = -curF + self.factors["theta"]*xrange*(-1*problem.minmax[0]* grad.dot(d))
        #plt.scatter(xrange,line,color='red')
        #plt.show()

        while True:
        #while(not suff dec and cur iter)
            #if(expended_budget + added_budget > problem.factors["budget"] or cur_iter >= max_iter):
            if(cur_iter >= max_iter):
                break;
              
            #print("cur_grad: ", grad.dot(d))
            #print("next_grad: ", next_grad.dot(d))
            #sufficient decrease doesn't satisfy, zoom into an interval
            if((nextF >= curF + self.factors['theta'] * cur_step_size * np.dot(grad, d))):
                #zoom into the interval {last_step,cur_step}
                #step_lo, step_hi, Flo, Fhi, div_lo, div_hi
                #print("zooming, NO SF")
                return self.zoomSearch(last_step_size,cur_step_size,lastF,nextF,
                                       last_grad.dot(d),next_grad.dot(d),problem,
                                       cur_x,curF,grad,d,added_budget,cur_iter)
                
            #last_grad = next_grad
            #get the gradient of the new solution grad f(x + step*d) for curvature condition
            #next_grad, B = self.get_gradient(problem,next_x,new_sol)
            #added_budget += B
 
            #check curvature, if satisfies then return 
            if((abs(next_grad.dot(d)) <= self.factors['curve_const']*abs(grad.dot(d)))):
                #print("Satisfied - upper")
                step_size = cur_step_size
                break;
            if((next_grad.dot(d)) >= 0):
                #zoom
                #print("zooming, sign")
                return self.zoomSearch(cur_step_size,last_step_size,nextF,lastF,
                                       next_grad.dot(d),last_grad.dot(d),problem,
                                       cur_x,curF,grad,d,added_budget,cur_iter)
                
            returned_steps.append(cur_step_size)
            returned_vals.append(nextF)
            #print("new step: ", cur_step_size)
            #print("sign*Fval: ",nextF)
            
            last_step_size = cur_step_size
            cur_step_size = min(max_step,cur_step_size*self.factors["zoom_inc_ratio"])

            if(cur_step_size >= max_step):
                break;
            #    return max_step, added_budget

            lastF = nextF
            last_grad = next_grad
            nextF, next_grad, budget_spent = self.get_simulated_values(problem,cur_x + cur_step_size*d,value = 'both')
            added_budget += budget_spent

            cur_iter += 1
            #print("new step: ", cur_step_size)
            #print("---------------")  
        if(cur_iter == self.factors["line_search_max_iters"] or (cur_step_size >= max_step)):
            #if use all iterations, let's return the step which optimizes the sufficient decrease
            return returned_steps[np.argmin(returned_vals)] ,added_budget
            #return max_step*self.factors["zoom_init_ratio"], added_budget
        else:
            return cur_step_size, added_budget
    
    def zoomSearch(self,step_lo, step_hi, Flo, Fhi, div_lo, div_hi, problem,cur_x,curF, grad,d,added_budget,cur_iter):
        """
        carry out the zoom search into the interval {}
        *two of these are not ordered*
        """
        max_iter = self.factors["line_search_max_iters"]
        r = self.factors["r"]
        sign = -1*problem.minmax[0]
        
        while(True):
            if(cur_iter >= max_iter):
                break;
                
            m1 = min([step_lo,step_hi])
            m2 = max([step_lo,step_hi])
            #print("zooming:: (",str(m1) + "," + str(m2) + ")") 
            #print("zooming:: (",str(step_lo) + "," + str(step_hi) + ")")
            #use the actual value without the sign
            new_step = self.quadratic_interpolate(step_lo,step_hi,sign*div_lo,sign*div_hi,sign*Flo,sign*Fhi,problem)
            if(step_lo < step_hi):
                left_dif = sign*div_lo;right_dif = sign*div_hi
                left_val = sign*Flo;right_val = sign*Fhi
            else:
                left_dif = sign*div_hi;right_dif = sign*div_lo
                left_val = sign*Fhi;right_val = sign*Flo

            #print("left div: ", left_dif)
            #print("right div: ", right_dif)
            #print("left val: ", left_val)
            #print("right val: ", right_val)
            
            #print("new step: ", new_step)

            #xrange = np.arange(0,1,0.02)
            #xrange = np.arange(m1,m2,(m2-m1)/20)
            #nn = len(xrange)
            #fval = np.zeros(nn)
            #for i in range(nn):
            #    temp_x = cur_x + xrange[i]*d
            #    temp_sol = self.create_new_solution(tuple(temp_x), problem)
            #    problem.simulate(temp_sol, r)
                #fval[i] = -1 * problem.minmax[0] * temp_sol.objectives_mean
            #    fval[i] = temp_sol.objectives_mean
            #plt.scatter(xrange,fval)
            #plt.show()

            if(abs(new_step - step_lo) < 1e-4 or abs(new_step - step_hi) < 1e-4):
                return new_step, added_budget

            #new_grad = grad_f(cur_x + new_step*d).dot(d)
            #newF = F(cur_x + new_step*d)
            newF, new_grad, budget_spent = self.get_simulated_values(problem,cur_x + new_step*d,value = 'both')
            added_budget += budget_spent
            
            #is_suff_decrese(nextF, curF, theta, cur_grad, cur_step_size, d)
            #if(not is_suff_decrese(newF, curF, theta, grad.dot(d), new_step)):
            if((newF >= curF + self.factors['theta'] * new_step * np.dot(grad, d))):
                step_hi = new_step
                #Fhi = F(cur_x + step_hi*d)
                Fhi, div_hi, budget_spent = self.get_simulated_values(problem,cur_x + step_hi*d,value = 'both')
                div_hi = div_hi.dot(d)
            else:
                #if(is_strong_curvature(new_grad, grad.dot(d), rho)):
                #if(is_curvature(new_grad, grad.dot(d), rho)):
                if((abs(new_grad.dot(d)) <= self.factors['curve_const']*abs(grad.dot(d)))):
                    return new_step, added_budget 
                if((new_grad.dot(d))*(step_hi - step_lo) >= 0):
                    step_hi = step_lo
                    
                    Fhi, div_hi, budget_spent = self.get_simulated_values(problem,cur_x + step_hi*d,value = 'both')
                    div_hi = div_hi.dot(d)
                    added_budget += budget_spent

                step_lo = new_step
                #Flo = F(cur_x + step_lo*d)
                #Fhi = F(cur_x + step_hi*d)
              
                Flo, div_lo, budget_spent = self.get_simulated_values(problem,cur_x + step_lo*d,value = 'both')
                div_lo = div_lo.dot(d)
                added_budget += budget_spent
                #Fhi, budget_spent = self.get_simulated_values(problem,cur_x + step_hi*d,value = 'val')
                #added_budget += budget_spent
                
            cur_iter += 1
        
        return new_step, added_budget   
    
    def find_feasible_initial(self, problem, Ce, Ci, de, di,lower, upper, tol = 1e-8):
        '''
        Find an initial feasible solution (if not user-provided)
        by solving phase one simplex.

        Arguments
        ---------
        problem : Problem object
            simulation-optimization problem to solve
        C: ndarray
            constraint coefficient matrix
        d: ndarray
            constraint coefficient vector

        Returns
        -------
        x0 : ndarray
            an initial feasible solution
        tol: float
            Floating point comparison tolerance
        '''
        # Define decision variables.
        x = cp.Variable(problem.dim)

        # Define constraints.
        constraints = []

        if(lower is not None):
            constraints += [x >= lower]
        if(upper is not None):
            constraints += [x <= upper]
        if (Ce is not None) and (de is not None):
            constraints += [Ce @ x == de]
        if (Ci is not None) and (di is not None):
            constraints += [Ci @ x <= di]

        # Define objective function.
        obj = cp.Minimize(0)
        
        # Create problem.
        model = cp.Problem(obj, constraints)

        # Solve problem.
        model.solve()

        # Check for optimality.
        if model.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] :
            raise ValueError("Could not find feasible x0")
        x0 = x.value
        if not self.is_feasible(x0, problem, tol):
            raise ValueError("Could not find feasible x0")

        return x0

    def get_atom_vectors(self,Ci,di):
        """
        get vertices of a polytope defined by the 
        constraints Ci <= di
        """
        a,b = Ci.shape
        mat = np.concatenate((di.reshape(a,1),-Ci),axis = 1)
        mat = cdd.Matrix(mat,linear=False,number_type='float')
        
        P = cdd.Polyhedron(mat)
        poly = cdd.Polyhedron(mat)
        ext = poly.get_generators()

        return np.array(ext)[:,1:] 
    
    def get_random_vertex(self,Ci,di,lower,upper):
        
        num_var = Ci.shape[1]
        x = cp.Variable(num_var)
        #objective = cp.Minimize(cp.norm(Ci@x - di,1))
        objective = cp.Maximize(cp.sum(x))
        constraints = [Ci@x <= di]
        if(lower is not None):
            constraints += [x >= lower]
        if(upper is not None):
            constraints += [x <= upper]
            
        problem = cp.Problem(objective, constraints)
        #problem.solve(solver=cp.GUROBI,env=env)
        problem.solve(solver=cp.SCIPY)
        return x.value
    
    def get_alpha_vec(self,x0,atom_vectors):
        """
        get the coefficients of convex combination of the x0
        """
        
        m,n = atom_vectors.shape
        x = cp.Variable(m)

        objective = cp.Minimize(cp.norm(atom_vectors.T @ x - x0) + cp.norm(x,1))
        constraints = [x >= 0,
                       x <= 1]

        prob = cp.Problem(objective, constraints)
        prob.solve()
        
        return x.value
    
    def solve(self, problem):    
        
        max_iters = self.factors['max_iters']
        ls_type = self.factors['LSmethod']
        #self.factors['problem'] = problem
        #print(ls_type)
        if(ls_type == 'backtracking'):
            self.factors["LSfn"] = self.backtrackLineSearch
        elif(ls_type == 'interpolation'):
            self.factors["LSfn"] = self.interpolateLineSearch
        elif(ls_type == 'zoom'):
            self.factors["LSfn"] = self.zoomLineSearch
        else:
            self.factors["LSfn"] = self.stepLineSearch
            
        
        #print("Solved by Frank Wolfe - " + self.factors["algorithm"])
        
        if(self.factors["algorithm"] == "normal"):
            return self.normal_FrankWolfe(problem)
        elif(self.factors["algorithm"] == "away"):
            return self.away_FrankWolfe(problem)
        elif(self.factors["algorithm"] == "normal_unbd"):
            return self.normal_FrankWolfe_unbd(problem)
        else:
            return self.pairwise_FrankWolfe(problem)
            
    def normal_FrankWolfe(self, problem):
        #print("Starting Frank Wolfe")
        recommended_solns = []
        intermediate_budgets = []
        expended_budget = 0
        dim = problem.dim
        
        Ci = problem.Ci
        di = problem.di
        Ce = problem.Ce
        de = problem.de
        
        lower = np.array(problem.lower_bounds)
        upper = np.array(problem.upper_bounds)
        
        r = self.factors["r"]
        ratio = self.factors["ratio"]
        LSmax_iter = self.factors["line_search_max_iters"]
        
        # Start with the initial solution.
        new_solution = self.create_new_solution(problem.factors["initial_solution"], problem)
        new_x = new_solution.x
        
        if(not self.is_feasible(new_x, Ci,di,Ce,de,lower,upper)):
            new_x = self.find_feasible_initial(problem, Ce, Ci, de, di,lower,upper)
            new_solution = self.create_new_solution(tuple(new_x), problem)

        # Use r simulated observations to estimate the objective value.
        problem.simulate(new_solution, r)
        expended_budget += r
        best_solution = new_solution
        recommended_solns.append(new_solution)
        intermediate_budgets.append(expended_budget)
        
        k = 0
        
        while expended_budget < problem.factors["budget"]:
            cur_x = new_solution.x
            
            #getting the gradient
            if problem.gradient_available:
                # Use IPA gradient if available.
                grad = -1 * problem.minmax[0] * new_solution.objectives_gradients_mean[0]
            else:
                # Use finite difference to estimate gradient if IPA gradient is not available.
                #grad, budget_spent = self.finite_diff(new_solution, problem, r, stepsize = alpha)
                grad, budget_spent = self.get_FD_grad(cur_x, problem, self.factors["h"], self.factors["r"])
                expended_budget += budget_spent
  
            v = self.get_dir(grad,Ce, Ci, de, di,lower,upper) 
            #direction = (v-cur_x)/np.linalg.norm(v-cur_x)
            direction = (v-cur_x)
            #print("grad: ", grad)
            #print("dir: ", v)
            #max_gamma = self.get_max_gamma_ratio_test(cur_x, direction, Ce, Ci, de, di, lower, upper)
            #max_gamma = max_gamma*self.factors["max_gamma"]
            max_gamma = 1
            
            if(self.factors["backtrack"]):
                #gamma, added_budget = self.LineSearch(new_solution,grad,direction,self.factors["max_gamma"],problem,expended_budget)
                #gamma, added_budget = self.LineSearch(new_solution,grad,direction,max_gamma,problem,expended_budget)
                gamma, added_budget = self.factors['LSfn'](new_solution,grad,direction,max_gamma,problem,expended_budget)
                expended_budget += added_budget
            else:
                #gamma = min(self.factors["step_f"](k),self.factors["max_gamma"])
                gamma = min(self.factors["step_f"](k),max_gamma)
            
            #k = self.get_max_gamma_ratio_test(cur_x, direction, Ce, Ci, de, di, lower, upper)
            #new_x = (1 - gamma)*np.array(cur_x) + gamma*v
            new_x = np.array(cur_x) + gamma*direction
            #print("new x: ",new_x)
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
            #print("obj: ",candidate_solution.objectives_mean)
        #print("-----------------------------")   
        return recommended_solns, intermediate_budgets 
    
    def normal_FrankWolfe_unbd(self, problem):
        #print("Starting Frank Wolfe")
        recommended_solns = []
        intermediate_budgets = []
        expended_budget = 0
        dim = problem.dim
        
        Ci = problem.Ci
        di = problem.di
        Ce = problem.Ce
        de = problem.de
        
        lower = np.array(problem.lower_bounds)
        upper = np.array(problem.upper_bounds)
        
        r = self.factors["r"]
        ratio = self.factors["ratio"]
        LSmax_iter = self.factors["line_search_max_iters"]
        
        # Start with the initial solution.
        new_solution = self.create_new_solution(problem.factors["initial_solution"], problem)
        new_x = new_solution.x
        
        if(not self.is_feasible(new_x, Ci,di,Ce,de,lower,upper)):
            new_x = self.find_feasible_initial(problem, Ce, Ci, de, di,lower,upper)
            new_solution = self.create_new_solution(tuple(new_x), problem)

        # Use r simulated observations to estimate the objective value.
        problem.simulate(new_solution, r)
        expended_budget += r
        best_solution = new_solution
        recommended_solns.append(new_solution)
        intermediate_budgets.append(expended_budget)
        
        k = 0
        
        while expended_budget < problem.factors["budget"]:
            cur_x = new_solution.x
            
            #getting the gradient
            if problem.gradient_available:
                # Use IPA gradient if available.
                grad = -1 * problem.minmax[0] * new_solution.objectives_gradients_mean[0]
            else:
                # Use finite difference to estimate gradient if IPA gradient is not available.
                #grad, budget_spent = self.finite_diff(new_solution, problem, r, stepsize = alpha)
                grad, budget_spent = self.get_FD_grad(cur_x, problem, self.factors["h"], self.factors["r"])
                expended_budget += budget_spent
  
            v, is_bounded = self.get_dir_unbd(grad,Ce, Ci, de, di,lower,upper) 
            max_gamma = 1
            
            if(is_bounded):#go to a vertex
                direction = v - cur_x
            else:#go to the open space
                direction = v
            #print("dir: ", direction)
            
            if(self.factors["backtrack"]):
                #gamma, added_budget = self.LineSearch(new_solution,grad,direction,self.factors["max_gamma"],problem,expended_budget)
                gamma, added_budget = self.LineSearch(new_solution,grad,direction,max_gamma,problem,expended_budget)
                expended_budget += added_budget
            else:
                #gamma = min(self.factors["step_f"](k),self.factors["max_gamma"])
                gamma = min(self.factors["step_f"](k),max_gamma)
            
            #print("gamma: ", gamma)
            new_x = np.array(cur_x) + gamma*direction
            #print("new x: ",new_x)
            candidate_solution = self.create_new_solution(tuple(new_x), problem)
            # Use r simulated observations to estimate the objective value.
            problem.simulate(candidate_solution, r)
            expended_budget += r
            
            new_solution = candidate_solution
            recommended_solns.append(candidate_solution)
            intermediate_budgets.append(expended_budget)
            
            k += 1
            #print("obj: ",candidate_solution.objectives_mean)
        #print("-----------------------------")   
        return recommended_solns, intermediate_budgets 
    
    def away_FrankWolfe(self, problem):
        
        recommended_solns = []
        intermediate_budgets = []
        expended_budget = 0
        last_step = []
        last_gamma = []
        
        Ci = problem.Ci
        di = problem.di
        Ce = problem.Ce
        de = problem.de
        
        lower = np.array(problem.lower_bounds)
        upper = np.array(problem.upper_bounds)
        
        scale_factor = self.factors["ratio"]
        LSmax_iter = self.factors["line_search_max_iters"]
        r = self.factors["r"]       
        
        # Start with the initial solution.
        new_solution = self.create_new_solution(problem.factors["initial_solution"], problem)
        new_x = new_solution.x
        
        if(not self.is_feasible(new_x, Ci,di,Ce,de,lower,upper)):
            #new_x = self.find_feasible_initial(problem, Ce, Ci, de, di,lower,upper)
            new_x = self.get_random_vertex(Ci,di,lower,upper)
            new_solution = self.create_new_solution(tuple(new_x), problem)
        
        #new_x = self.get_random_vertex(Ci,di,lower,upper)
        new_solution = self.create_new_solution(tuple(new_x), problem)
        atom_vectors = np.array([new_x])
        problem.simulate(new_solution, r)
        
        #initializing active set and all alpha coefficients, contains only one vector here
        #active_vectors = {0:[]}
        active_vectors = [np.array(new_x)]
        alphas = {tuple(new_x):1}
        #store the "active" infinite search direction
        active_dirs = []
        betas = {}
        
        # Use r simulated observations to estimate the objective value.
        problem.simulate(new_solution, r)
        expended_budget += r
        best_solution = new_solution
        recommended_solns.append(new_solution)
        intermediate_budgets.append(expended_budget)
        
        k = 0
        
        while expended_budget < problem.factors["budget"]:
            cur_x = new_solution.x
            
            if problem.gradient_available:
                # Use IPA gradient if available.
                grad = -1 * problem.minmax[0] * new_solution.objectives_gradients_mean[0]
            else:
                # Use finite difference to estimate gradient if IPA gradient is not available.
                #grad, budget_spent = self.finite_diff(new_solution, problem, r, stepsize = alpha)
                grad, budget_spent = self.get_FD_grad(cur_x, problem, self.factors["h"], self.factors["r"])
                expended_budget += budget_spent
                   
            #print("grad: ", grad)
            #print("active set: ")
            #the list dot product values [grad_f.a for a in atom]
            #s = self.get_dir(grad,Ce, Ci, de, di,lower, upper)
            s, is_bounded = self.get_dir_unbd(grad,Ce, Ci, de, di,lower, upper)
            
            #list of dot product of [grad_f.v for v in active set]
            #gv = np.array([grad.dot(a) for a in active_vectors[k]])
            if(len(active_vectors) > 0):
                gv = np.array([grad.dot(a) for a in active_vectors])
                #v = active_vectors[k][np.argmax(gv)]
                v = active_vectors[np.argmax(gv)]
                d_away = cur_x - v
            else:
                d_away = np.zeros(problem.dim)
                v = None
                
            #compute the directions of normal Frank-Wolfe
            if(is_bounded):
                d_FW = s - cur_x
            else:#go to the open space
                d_FW = s
                s = d_FW
            
            #there is no way to move further since we finished early
            if((d_FW == 0).all() and (d_away == 0).all()):
                direction = d_FW #by default since it has no effect anyway
                gamma = 0
            
            elif((-grad.dot(d_FW) >= -grad.dot(d_away)) or (d_away == 0).all() or (not is_bounded)):
                #FW step
                #print("foward")
                #ind.append('FW')
                direction = d_FW
                #print("dir: ", direction)
                #max_gamma = 1
                #max_gamma = self.get_max_gamma_ratio_test(cur_x, direction, Ce, Ci, de, di, lower, upper)
                #max_gamma = max_gamma*self.factors["max_gamma"]
                #print("gamma: ", gamma)
                #if(self.factors["backtrack"]):
                    #gamma, added_budget = self.LineSearch(new_solution,grad,direction,max_gamma,problem,expended_budget)
                #    gamma, added_budget = self.factors['LSfn'](new_solution,grad,direction,max_gamma,problem,expended_budget)
                #    expended_budget += added_budget
                #else:
                #    gamma = min(self.factors["step_f"](k),max_gamma)
                
                if(is_bounded):
                    #print("bounded")
                    max_gamma = 1
                    if(self.factors["backtrack"]):
                        #gamma, added_budget = self.LineSearch(new_solution,grad,direction,max_gamma,problem,expended_budget)
                        gamma, added_budget = self.factors['LSfn'](new_solution,grad,direction,max_gamma,problem,expended_budget)
                        expended_budget += added_budget
                    else:
                        gamma = min(self.factors["step_f"](k),max_gamma)
                    #update the active set
                    if(gamma < 1):
                        add = 0
                        #check whether we have added this vertex s before
                        for vec in active_vectors:
                            if((vec != s).any()):
                                add = 1 #if could not find it, we must add it
                            else:
                                add = 0
                                break;
                        if(add): #adding the new vertex s
                            active_vectors.append(s)
                            alphas[tuple(s)] = 0
                    else:
                        #go the vertex s
                        active_vectors = [s]
                        alphas = {tuple(s):0}

                    #print("active set change in forward: ", active_vectors)
                    #for atom in active_vectors[k]:
                    for atom in active_vectors:
                        if((atom == s).all()):
                            alphas[tuple(atom)] = (1-gamma)*alphas[tuple(atom)] + gamma
                        else:
                            alphas[tuple(atom)] = (1-gamma)*alphas[tuple(atom)]
                            
                    for dirs in active_dirs:
                        betas[tuple(dirs)] = (1-gamma)*betas[tuple(dirs)]
                    last_step.append("bounded")
                else:
                    #print("unbounded")
                    #if we have consecutive extreme search
                    if(k > 0 and last_step[-1] == 'unbounded'):
                        gamma = last_gamma[-1]/self.factors["ratio"]
                    else:
                        max_gamma = 1
                        #gamma = 1
                        #print("max step: ",max_gamma)
                        if(self.factors["backtrack"]):
                            #gamma, added_budget = self.LineSearch(new_solution,grad,direction,max_gamma,problem,expended_budget)
                            gamma, added_budget = self.factors['LSfn'](new_solution,grad,direction,max_gamma,problem,expended_budget)
                            expended_budget += added_budget
                        else:
                            gamma = min(self.factors["step_f"](k),max_gamma)
                    #searching in the infinite search direction
                    #add a new inf direction if not found yet
                    if(len(active_dirs) == 0):
                        #It's the first time we add the search direction
                        active_dirs.append(s)
                        betas[tuple(s)] = gamma
                    else: #we added some extreme direction before
                        diffs = np.array([sum(abs(vec - s)) for vec in active_dirs])
                        if((diffs > 1e-6).all()):#s is a new inf direction
                            active_dirs.append(s)
                            betas[tuple(s)] = gamma
                        else:
                            betas[tuple(s)] += gamma
                    last_step.append("unbounded")
                    
            else:
                #away step
                #print("away")
                #ind.append('away')
                direction = d_away #xt - v
                #print("dir: ", direction)
                #direction = d_away/np.linalg.norm(d_away)
                #gamma = gamma_f(k)
                #max_gamma = alphas[tuple(v)]/(1 - alphas[tuple(v)])
                gamma_star = self.get_max_gamma_ratio_test(cur_x, direction, Ce, Ci, de, di, lower, upper)
                #print("gamma_star: ", gamma_star)
                #gamma_star = gamma_star*self.factors["max_gamma"]
                #print("the alpha in ratio: ",alphas[tuple(v)])
                #direction = direction*gamma_star #d' = gamma_star*d
                #max_dist = 1
                #max_dist = min(1,alphas[tuple(v)]/(gamma_star*(1 - alphas[tuple(v)])))
                max_dist = min(gamma_star,alphas[tuple(v)]/((1 - alphas[tuple(v)])))
                #max_gamma = alphas[v]/(1 - alphas[v])
                #print("max_dist: ", max_dist)
                #active_vectors[k+1] = active_vectors[k]

                if(self.factors["backtrack"]):
                    #gamma, added_budget = self.LineSearch(new_solution,grad,direction,max_dist,problem,expended_budget)
                    gamma, added_budget = self.factors['LSfn'](new_solution,grad,direction,max_dist,problem,expended_budget)
                    expended_budget += added_budget
                else:
                    gamma = min(self.factors["step_f"](k),self.factors["max_gamma"])

                #if gamma_max, then update St \ {vt}
                if(gamma == 1 or gamma <= scale_factor**LSmax_iter):
                    #print("dropping: ", v)
                    #active_vectors[k+1] = []
                    #for vec in active_vectors[k]:
                    new_active = []
                    for vec in active_vectors:
                        if((vec != v).any()): 
                        #if((np.linalg.norm(vec - v)) > 1e-4):
                            #active_vectors[k+1].append(vec)
                            new_active.append(vec)
                    active_vectors = new_active
                    removed_atom = alphas.pop(tuple(v))
                    
                for atom in active_vectors:
                    if((atom == v).all()):
                        #alphas[tuple(atom)] = (1+gamma)*alphas[tuple(atom)] - gamma
                        alphas[tuple(atom)] = (1+gamma*gamma_star)*alphas[tuple(atom)] - gamma*gamma_star
                    else:
                        alphas[tuple(atom)] = (1+gamma*gamma_star)*alphas[tuple(atom)]
                last_step.append("away")
            #print("alphas: ", alphas)
            #print("Displaying Alphas:")
            #for key,val in alphas.items():
            #    print(key)
            #    print(val)
            #    print('**************')
            
            
            #print("max_gamma: ", max_gamma)
            #print("gamma: ", gamma)
            #print("dir: ", tuple(direction))
            last_gamma.append(gamma)
            new_x = cur_x + gamma*direction
            #print("new_x: ",tuple(new_x))
            
            candidate_solution = self.create_new_solution(tuple(new_x), problem)
            # Use r simulated observations to estimate the objective value.
            problem.simulate(candidate_solution, r)
            #print("obj: ",candidate_solution.objectives_mean)
            expended_budget += r
            
            new_solution = candidate_solution
            
            # Append new solution.
            if (problem.minmax[0] * new_solution.objectives_mean > problem.minmax[0] * best_solution.objectives_mean):
                best_solution = new_solution
                #recommended_solns.append(candidate_solution)
                recommended_solns.append(new_solution)
                intermediate_budgets.append(expended_budget)

                #print("obj: ",new_solution.objectives_mean)
            
            k += 1
            #print("--------------")
        return recommended_solns, intermediate_budgets 
    
    def pairwise_FrankWolfe(self, problem):
        
        recommended_solns = []
        intermediate_budgets = []
        expended_budget = 0
        
        Ci = problem.Ci
        di = problem.di
        Ce = problem.Ce
        de = problem.de

        r = self.factors["r"]
        
        lower = np.array(problem.lower_bounds)
        upper = np.array(problem.upper_bounds)
        
        # Start with the initial solution.
        #new_solution = self.create_new_solution(problem.factors["initial_solution"], problem)
        #new_x = new_solution.x
        
        #if(not self.is_feasible(new_x, problem)):
        #    new_x = self.find_feasible_initial(problem, Ce, Ci, de, di)
        #    new_solution = self.create_new_solution(tuple(new_x), problem)
        
        # Start with the initial solution.
        #new_solution = self.create_new_solution(problem.factors["initial_solution"], problem)
        #new_x = new_solution.x
        
        #if(not self.is_feasible(new_x, Ci,di,Ce,de,lower,upper)):
            #new_x = self.find_feasible_initial(problem, Ce, Ci, de, di,lower,upper)
        #    new_x = self.get_random_vertex(Ci,di,lower,upper)
        #    new_solution = self.create_new_solution(tuple(new_x), problem)
        
        new_x = self.get_random_vertex(Ci,di,lower,upper)
        new_solution = self.create_new_solution(tuple(new_x), problem)
        #initiailizing a dictionary of atom vectors and their coefficients
        #atom_vectors = self.factors["atom_vectors"]
        #if(self.factors["atom_vectors"] is None):
        #    atom_vectors = self.get_atom_vectors(Ci,di)
        #    num_atoms = atom_vectors.shape[0]
        #    alpha_vec = np.zeros(num_atoms)
        #    alpha_vec[0] = 1
            
        #    new_x = atom_vectors[0]
        #    new_solution = self.create_new_solution(tuple(new_x), problem)
        #else:
        #    atom_vectors = self.factors["atom_vectors"]
        #    num_atoms = atom_vectors.shape[0]
        #    alpha_vec = self.get_alpha_vec(new_x,atom_vectors)
        
        #initiailizing a dictionary of atom vectors and their coefficients
        #atom_vectors = self.factors["atom_vectors"]
        #atom_vectors = self.get_atom_vectors(Ci,di)
        #num_atoms = atom_vectors.shape[0]
        #active_vectors = {0:[]}
        #alphas = {tuple(v):0 for v in atom_vectors}
        
        atom_vectors = np.array([new_x])
        active_vectors = [np.array(new_x)]
        #alphas = {tuple(v):0 for v in atom_vectors}
        alphas = {tuple(new_x):1}
        
        #new_x = atom_vectors[0]
        #new_solution = self.create_new_solution(tuple(new_x), problem)
        
        #for i in range(num_atoms): 
        #    alphas[tuple(atom_vectors[i])] = alpha_vec[i]
        #    if(alpha_vec[i] > 0):
        #        active_vectors[0].append(atom_vectors[i])
        
        # Use r simulated observations to estimate the objective value.
        problem.simulate(new_solution, r)
        expended_budget += r
        best_solution = new_solution
        recommended_solns.append(new_solution)
        intermediate_budgets.append(expended_budget)
        
        k = 0
        
        while expended_budget < problem.factors["budget"]:
            cur_x = new_solution.x
            
            #print("cur_x: ", cur_x)
            
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
                #    grad, budget_spent  = self.finite_diff(new_solution, problem, r)
                #    expended_budget += budget_spent
                    # Update r after each iteration.
                #    r = int(self.factors["lambda"] * r)

            s = self.get_dir(grad,Ce, Ci, de, di,lower, upper)
            
            #compute the directions
            if(len(active_vectors) > 0):
                gv = np.array([grad.dot(a) for a in active_vectors])
                #v = active_vectors[k][np.argmax(gv)]
                v = active_vectors[np.argmax(gv)]
                d_pw = s-v
            else:
                d_pw = np.zeros(problem.dim)
            #direction = s - v
            d_FW = s - cur_x
            
            #print("s-v: ", s-v)
            #print("foward direction: ", s)
            #print("pairwise direction: ", s-v)
            #print("grad :", grad)
            #print("current point: ",cur_x)
            #print("dir: ", direction)
            #print("v: ", v)
            #max_gamma = min(alphas[tuple(v)]*np.linalg.norm(s-v),self.factors["max_gamma"])
            #max_gamma = self.get_max_gamma_ratio_test(cur_x, direction, Ce, Ci, de, di, lower, upper)
            #away vector v = 0
            if((-grad.dot(d_FW) >= -grad.dot(d_pw)) or (d_pw == 0).all()):
                #print('Forward')
                direction = d_FW
                #print("direcition: ", direction)
                max_gamma = 1
                
                if(self.factors["backtrack"]):
                    #gamma = LineSearch(F=F,x=cur_x,d=d_away,max_step=max_gamma/2)
                    gamma, added_budget = self.LineSearch(new_solution,grad,direction,max_gamma,problem,expended_budget)
                    expended_budget += added_budget
                else:
                    #gamma = self.factors["step_f"](k)
                    gamma = min(self.factors["step_f"](k),max_gamma)
                    
                #update the active set
                if(gamma < 1):
                    for vec in active_vectors:
                        if((s != vec).any()):
                            add = 1
                        else:
                            add = 0
                            break;
                    if(add):
                        #active_vectors[k+1].append(s)
                        active_vectors.append(s)
                        alphas[tuple(s)] = 0
                else:
                    active_vectors = [s]
                    alphas = {tuple(s):0}

                #updating weights/coefficients
                for atom in active_vectors:
                    if((atom == s).all()):
                        alphas[tuple(atom)] = (1-gamma)*alphas[tuple(atom)] + gamma
                    else:
                        alphas[tuple(atom)] = (1-gamma)*alphas[tuple(atom)]
                
            else:
                #print("pairwise")
                direction = d_pw
                #print("direcition: ", direction)
                max_gamma = alphas[tuple(v)]
                #print("max_step: ", max_gamma)
                if(self.factors["backtrack"]):
                    #gamma = LineSearch(F=F,x=cur_x,d=d_away,max_step=max_gamma/2)
                    gamma, added_budget = self.LineSearch(new_solution,grad,direction,max_gamma,problem,expended_budget)
                    expended_budget += added_budget
                else:
                    #gamma = self.factors["step_f"](k)
                    gamma = min(self.factors["step_f"](k),max_gamma)
                #print("active set in pairwise: ", active_vectors)
                #found a new vertex not in the past vertices
                for vec in active_vectors:
                    #different/a new vertex
                    if((s != vec).any()):
                    #if(sum(abs(s-vec)/(problem.dim*(vec+1e-10))) > 1e-2):
                        add = 1
                    else:
                        add = 0
                        break;
                if(add):
                    active_vectors.append(s)
                    alphas[tuple(s)] = 0
                #print("active set in pairwise: ", active_vectors)
                alphas[tuple(s)] = alphas[tuple(s)] + gamma
                alphas[tuple(v)] = alphas[tuple(v)] - gamma
                
            #print("Displaying Alphas:")
            #for key,val in alphas.items():
            #    print(key)
            #    print(val)
            #    print('**************')
            
        
            #print("max_gamma: ", max_gamma)
            #print("gamma: ", gamma)
            new_x = cur_x + gamma*direction
            #print("next x: ",new_x)
            candidate_solution = self.create_new_solution(tuple(new_x), problem)
            # Use r simulated observations to estimate the objective value.
            problem.simulate(candidate_solution, r)
            expended_budget += r
           
            
            new_solution = candidate_solution
            recommended_solns.append(candidate_solution)
            intermediate_budgets.append(expended_budget)
            
            k += 1
            #print("obj: ",candidate_solution.objectives_mean)
            #print("------------------")
            #print("------------------")
            
        return recommended_solns, intermediate_budgets 
        