import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
#import cdd


import warnings
warnings.filterwarnings("ignore")

from ..base import Solver
#env = gurobipy.Env()
#env.setParam('FeasibilityTol', 1e-9)
#env.setParam('MIPGap',0)


class boostFrankWolfe(Solver):
    """
    """
    
    def __init__(self, name="boost-FW", fixed_factors={}):
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
                "default":"backtracking"
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
            "delta": {
                "description": "improvement in boost FW",
                "datatype": float,
                "default": 1e-3
            },
            "max_oracle_iter": {
                "description": "maximum number of iterations for boosting oracle",
                "datatype": int,
                "default": 10
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
        
        return min(ratio_val[denom > 1e-10])
        #return min(ratio_val[denom > 0])
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
        #print("max step: ", max_step)
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
            #print(new_x)
            new_sol =  self.create_new_solution(tuple(new_x), problem)
            problem.simulate(new_sol, r)
            added_budget += r
          
            newF = -1 * problem.minmax[0] * new_sol.objectives_mean
            #print("newF: ",newF)
            #print("newX: ", new_x)
            
            if(newF < curF + self.factors['theta'] * step_size * np.dot(grad, d)):
                break;
                
            step_size = step_size*ratio
            cur_iter += 1
            #print("---------------")
        #print("step from backtrack: ",step_size)
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
        
#         xrange = np.arange(0,max_step,max_step/50)
#         nn = len(xrange)
#         fval = np.zeros(nn)
#         for i in range(nn):
#             temp_x = cur_x + xrange[i]*d
#             temp_sol = self.create_new_solution(tuple(temp_x), problem)
#             problem.simulate(temp_sol, r)
#             fval[i] = temp_sol.objectives_mean
#         plt.scatter(xrange,fval)
#         line = -1*problem.minmax[0]*curF + self.factors["theta"]*xrange*(-1*problem.minmax[0]* grad.dot(d))
#         plt.scatter(xrange,line,color='red')
#         plt.show()
        
        if(max_step == 0):
            return max_step, added_budget
        
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
                #print('changing the endpoint')
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
                #break;
                return max_step, added_budget

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

    def combine_constraint(self,Ci,di,Ce,de,lower, upper, problem):
        '''
        combine all constraints together
        '''
        
        # Remove redundant upper/lower bounds.
        ub_inf_idx = np.where(~np.isinf(upper))[0]
        lb_inf_idx = np.where(~np.isinf(lower))[0]
        
        # Form a constraint coefficient matrix where all the equality constraints are put on top and
        # all the bound constraints in the bottom and a constraint coefficient vector.  
        if (Ce is not None) and (de is not None) and (Ci is not None) and (di is not None):
            A = np.vstack((Ce,  Ci))
            b = np.vstack((de.T, di.T))
        elif (Ce is not None) and (de is not None):
            A = Ce
            b = de.T
            A = np.vstack((A, -Ce))
            b = np.vstack((b, -de.T))
        elif (Ci is not None) and (di is not None):
            A = Ci
            b = di.T
        else:
            A = np.empty([1, problem.dim])
            b = np.empty([1, 1])
        
        if len(ub_inf_idx) > 0:
            A = np.vstack((A, np.identity(upper.shape[0])))
            #b = np.vstack((b, upper[np.newaxis].T))
            
            if(len(b.shape) == len(upper[np.newaxis].shape)):
                b = np.vstack((b, upper[np.newaxis].T))
            else:
                b = np.vstack((b[np.newaxis].T, upper[np.newaxis].T))
        if len(lb_inf_idx) > 0:
            A = np.vstack((A, -np.identity(lower.shape[0])))
            
            if(len(b.shape) == len(lower[np.newaxis].shape)):
                b = np.vstack((b, -lower[np.newaxis].T))
            else:
                b = np.vstack((b[np.newaxis].T, -lower[np.newaxis].T))
                
        #if initialize A,b with empty array, remove those values
        if((Ce is None) and (de is None) and (Ci is None) and (di is None)):
            A = A[1:,]
            b = b[1:,]
            
        return A, b
        
    
    def finite_diff(self, new_solution, problem, r, A, b, stepsize = 1e-4, tol = 1e-7):
        '''
        Finite difference for approximating objective gradient at new_solution.

        Arguments
        ---------
        new_solution : Solution object
            a solution to the problem
        problem : Problem object
            simulation-optimization problem to solve
        r : int 
            number of replications taken at each solution
        C : ndarray
            constraint matrix
        d : ndarray
            constraint vector
        stepsize: float
            step size for finite differences

        Returns
        -------
        grad : ndarray
            the estimated objective gradient at new_solution
        budget_spent : int
            budget spent in finite difference
        '''

        BdsCheck = np.zeros(problem.dim)
        fn = -1 * problem.minmax[0] * new_solution.objectives_mean
        new_x = new_solution.x
        grad = np.zeros(problem.dim)
        h = np.zeros(problem.dim)
        budget_spent = 0

        for i in range(problem.dim):
            # Initialization.
            x1 = list(new_x)
            x2 = list(new_x)
            # Forward stepsize.
            steph1 = stepsize
            # Backward stepsize.
            steph2 = stepsize

            dir1 = np.zeros(problem.dim)
            dir1[i] = 1
            dir2 = np.zeros(problem.dim)
            dir2[i] = -1 

            ra = b.flatten() - A @ new_x
            ra_d = A @ dir1
            # Initialize maximum step size.
            temp_steph1 = np.inf
            # Perform ratio test.
            for j in range(len(ra)):
                if ra_d[j] - tol > 0:
                    s = ra[j]/ra_d[j]
                    if s < temp_steph1:
                        temp_steph1 = s
            steph1 = min(temp_steph1, steph1)

            if np.isclose(steph1, 0 , atol= tol):
                # Address numerical stability of step size.
                steph1 = 0

            ra_d = A @ dir2
            # Initialize maximum step size.
            temp_steph2 = np.inf
            # Perform ratio test.
            for j in range(len(ra)):
                if ra_d[j] - tol > 0:
                    s = ra[j]/ra_d[j]
                    if s < temp_steph2:
                        temp_steph2 = s
            steph2 = min(temp_steph2, steph2)

            if np.isclose(steph2, 0 , atol= tol):
                # Address numerical stability of step size.
                steph2 = 0
        
            # Determine whether to use central diff, backward diff, or forward diff.
            if (steph1 != 0) & (steph2 != 0):
                BdsCheck[i] = 0
            elif (steph1 == 0) & (steph2 != 0):
                BdsCheck[i] = -1
            elif (steph2 == 0) & (steph1 != 0):
                BdsCheck[i] = 1
            else:
                # Set gradient to 0 if unable to move.
                grad[i] = 0
                continue
            
            # Decide stepsize
            # Central diff.
            if BdsCheck[i] == 0:
                h[i] = min(steph1, steph2)
                x1[i] = x1[i] + h[i]
                x2[i] = x2[i] - h[i]
            # Forward diff.
            elif BdsCheck[i] == 1:
                h[i] = steph1
                x1[i] = x1[i] + h[i]
            # Backward diff.
            else:
                h[i] = steph2
                x2[i] = x2[i] - h[i]

            # Evaluate solutions
            x1_solution = self.create_new_solution(tuple(x1), problem)
            if BdsCheck[i] != -1:
                # x+h
                problem.simulate_up_to([x1_solution], r)
                budget_spent += r -x1_solution.n_reps
                fn1 = -1 * problem.minmax[0] * x1_solution.objectives_mean
            x2_solution = self.create_new_solution(tuple(x2), problem)
            if BdsCheck[i] != 1:
                # x-h
                problem.simulate_up_to([x2_solution], r)
                budget_spent += r -x2_solution.n_reps
                fn2 = -1 * problem.minmax[0] * x2_solution.objectives_mean

            # Calculate gradient.
            if BdsCheck[i] == 0:
                grad[i] = (fn1 - fn2) / (2 * h[i])
            elif BdsCheck[i] == 1:
                grad[i] = (fn1 - fn) / h[i]
            elif BdsCheck[i] == -1:
                grad[i] = (fn - fn2) / h[i]

            #print("r inside: ", r)
            #print("n rep: ", x1_solution.n_reps)
        #print("check: ", BdsCheck)
        #print("grad: ", grad)
        #print("budget spent: ", budget_spent)
        return grad, budget_spent

    
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
    
    def get_weigts(grad, components):
        '''
        return weights of all components so that the
        cosine similarity is the smallest (closest to
        the negative gradient)
        '''
        num_components, dim = components.shape

        weights = cp.Variable(num_components)
        direction = cp.Variable(dim) #the combination of all components
        norm = np.linalg.norm(grad)

        #objective = cp.Minimize(grad@direction/norm)
        objective = cp.Minimize(cp.sum_squares(direction + grad))
        constraints = [sum(weights) <= 10,
                       weights >= 0,
                      direction == (components.T)@weights]

        prob = cp.Problem(objective, constraints)
        prob.solve()

        return weights.value
    
    def LPoracle(self, c,A,b):
        '''
        solving min LP under constraint Ax <= b
        '''
        n = len(c)
        u = cp.Variable(n)
        
        num_con = b.shape[0]
        b = b.reshape(num_con,)

        objective = cp.Maximize(u@c)
        constraints = [A@u <= b]

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.GUROBI,InfUnbdInfo= 1)#solver=cp.ECOS
        #prob.solve(solver=cp.SCIPY)
        #check if the soln is unbounded
        if('unbounded' in prob.status):
            result = np.array([prob.solver_stats.extra_stats.getVars()[j].unbdray for j in range(n)])
            is_bounded = False
        else:
            result = u.value
            is_bounded = True

        return result, is_bounded
    
    def align(self, u,v):
        '''
        return the cosine of angle align between the vector u,v
        (u is constant wrt v in general: assume that u is not zero)
        '''
        if((v == 0).all() or (u == 0).all()):
            return -1
        else:
            return (u.dot(v))/(np.linalg.norm(u)*np.linalg.norm(v))
    
    def get_weights(self, grad, components):
        '''
        return weights of all components so that the
        cosine similarity is the smallest (closest to
        the negative gradient)
        '''
        num_components, dim = components.shape
        
        #print("comp: ", components)
        #print("grad: ", grad)

        weights = cp.Variable(num_components)
        direction = cp.Variable(dim) #the combination of all components
        norm = np.linalg.norm(grad)

        objective = cp.Minimize(grad@direction/norm)
        #objective = cp.Minimize(cp.sum_squares(direction + grad))
        constraints = [sum(weights) <= 10,
                       weights >= 0,
                      direction == (components.T)@weights]
        
        for i in range(num_components):
            cur_comp = components[i]
            cur_comp = cur_comp/np.linalg.norm(cur_comp)
            constraints += [-grad@direction/norm >= cur_comp@(-grad/norm)]

        prob = cp.Problem(objective, constraints)
        prob.solve()

        return weights.value
    
    
    def boostConic(self, grad, cur_x, A,b):
        '''
        Similar to the previous boostedDir, 
        this function uses conic representation
        of components
        '''
        num_iter = 0
        dim = len(cur_x)
        d = np.zeros(dim) #final search direction result vector
        delta = self.factors['delta']
        
        #store all components we found, initialize the matrix for storage
        #size is (#comp, dim)
        components = np.array([]).reshape(0,dim)
        
        for i in range(self.factors["max_oracle_iter"]):
            #solving an LP
            residual = -grad - d
            v, is_bounded = self.LPoracle(residual,A,b)
            num_iter += 1

            if(is_bounded):
                #print("bounded")
                u1 = v - cur_x
                #print("v: ", v)
            else:
                #print("unbd")
                #normalize the recession direction
                v = v/np.linalg.norm(v)
                #print("v: ", v)
                u1 = v
                
            if((d == 0).all()): #when d is initialized, first iteration of boosting
                u = u1
            else:
                #comparing the new direction d with the previous d
                u2 = -d/np.linalg.norm(d)
                res1 = u1.dot(residual)
                res2 = u2.dot(residual)
                if(res1 > res2):
                    #print("add new dir")
                    #the new direction has better improvement
                    u = u1
                else:
                    #print("prev dir")
                    #previous d is better
                    #u = u2
                    break;
                    
            if((u == 0).all()):
                return u, 1
            proj_weight = ((-grad).dot(u))/(np.linalg.norm(u))**2
            u = (u/np.linalg.norm(u))*proj_weight
            components = np.concatenate((components,u.reshape(1,dim)),axis = 0) 
            #print("comp: ", components)
            #if(num_iter > 1):
            weights = self.get_weights(grad/np.linalg.norm(grad), components)
            #else:
            #    weights = np.array([1])
            temp_d = (components.T)@weights #the sum of weighted components
            
            
            
            #print("prev align: ",self.align(-grad,d))
            #print("new align: ",self.align(-grad,temp_d))
            
            #check if temp d is strictly better than the previous d
            if(self.align(-grad,temp_d) - self.align(-grad,d) >= delta):
                d = temp_d     
            else:
                break;
                
        #print("weights: ", weights)   
        return d, num_iter
    
    
    def boostDir(self, grad, cur_x, A,b):
        '''
        determine a direction d which is 
        close to negative gradient -grad
        where d is the conic rep of the extreme dir
        and V - x_k
        '''
        num_iter = 0
        #cur_x = cur_sol.x
        
        #print("boosted Dir")
        
        dim = len(cur_x)
        d = np.zeros(dim) #final search direction result vector
        Wsum = 0 #sum of all weights (initialized)
        #A = factors['A'];b = factors['b']
        delta = self.factors['delta']

        for i in range(self.factors["max_oracle_iter"]):
            #solving an LP
            residual = -grad - d
            v, is_bounded = self.LPoracle(residual,A,b)
            num_iter += 1

            if(is_bounded):
                print("bounded")
                u1 = v - cur_x
                print("v: ", v)
            else:
                print("unbd")
                #normalize the recession direction
                v = v/np.linalg.norm(v)
                print("v: ", v)
                u1 = v
            
            if((d == 0).all()): #when d is initialized, first iteration of boosting
                u = u1
            else:
                #comparing the new direction d with the previous d
                u2 = -d/np.linalg.norm(d)
                res1 = u1.dot(residual);res2 = u2.dot(residual)
                if(res1 > res2):
                    #the new direction has better improvement
                    u = u1
                else:
                    #previous d is better
                    u = u2
                    
            #finding the weight of this new 'u' (projection step)
            cur_weight = (residual.dot(u))/(np.linalg.norm(u))**2
            temp_d = d + cur_weight*u 
            #print("u: ",u)
            #print("temp d: ", temp_d)
            #print("old d: ", d)
            #print("*************")
            #print("new align: ", self.align(-grad,temp_d))
            #print("prev align: ", self.align(-grad,d))

            #check if temp d is strictly better than the previous d by some 'delta'
            if(self.align(-grad,temp_d) - self.align(-grad,d) >= delta):
                if((d == 0).all()): #the very first iteration of boosting 
                    Wsum += cur_weight
                else:
                    if(res1 > res2):
                        #print("add dir")
                        #print("add u: ", u1)
                        #the new direction has better improvement
                        Wsum += cur_weight
                    else:
                        #print("use prev d")
                        #previous d is better
                        Wsum = Wsum*(1 - cur_weight/np.linalg.norm(d))

                d = temp_d     
            else:
                #if we cannot improve the alignment by 'delta', then break
                break;

        if(Wsum == 0):
            #the loop break in ONE iteration: no combination
            d = u1
        else:
            d = d/Wsum

        #print("normalizing: ", Wsum)
        return d, num_iter
    
    def update_vertices(self,v,vertices_collection):
        '''
        update seen vertices for away-step direction
        '''
        #compute the difference to check if v is one of 
        #the vertices
        #print("v: ", v)
        #print("collection: ", vertices_collection)
        diff_norm = np.linalg.norm(vertices_collection - v,axis = 1)
        
        #if v is a new vertex, add to the collection
        if(min(diff_norm) < 1e-15):
            #v is already in the collection
            return vertices_collection
        else:
            return np.vstack((vertices_collection,v))
    
    def boostDirAway(self, grad, cur_x, A,b,vertices_collection):
        '''
        determine a direction d which is 
        close to negative gradient -grad
        where d is the conic rep of the extreme dir
        and V - x_k
        '''
        num_iter = 0
        #cur_x = cur_sol.x
        
        #print("boosted Dir")
        
        dim = len(cur_x)
        d = np.zeros(dim) #final search direction result vector
        Wsum = 0 #sum of all weights (initialized)
        #A = factors['A'];b = factors['b']
        delta = self.factors['delta']
        is_away = 0
        
        #initialize the vertices collection for the next iteration
        #this matrix stores vertices found in this iteration

        for i in range(self.factors["max_oracle_iter"]):
            #solving an LP
            residual = -grad - d
            v, is_bounded = self.LPoracle(residual,A,b) #max resTu s.t. A@u <= b
            num_iter += 1

            if(is_bounded):
                #print("bounded")
                u1 = v - cur_x
                #print("v: ", v)
            else:
                #print("unbd")
                #normalize the recession direction
                v = v/np.linalg.norm(v)
                #print("v: ", v)
                u1 = v
            
            if((d == 0).all()): #when d is initialized, first iteration of boosting
                u = u1
            else:
                #comparing the new direction d with the previous d
                u2 = -d/np.linalg.norm(d)
                res1 = u1.dot(residual);res2 = u2.dot(residual)
                if(res1 > res2):
                    #the new direction has better improvement
                    #print("new dir")
                    u = u1
                else:
                    #previous d is better
                    #print("prev d")
                    u = u2
                #check if this u is better than away direction
                if(vertices_collection.shape[0] > 0):
                    v_away = vertices_collection[np.argmin(vertices_collection.dot(residual))] #collect away vertices
                    scaled_xv = np.linalg.norm(u)*(cur_x - v_away)/np.linalg.norm(cur_x - v_away)
                    if((u.dot(residual)) <= ((scaled_xv).dot(residual))):
                        #print("res: ", tuple(residual))
                        #print("u: ", tuple(u))
                        is_away = 1
                        u = cur_x - v_away
                        
                        #drop the vertex if we take away step
                        row = np.where(v_away == vertices_collection)[0][0]
                        vertices_collection = np.delete(vertices_collection,row,axis=0)
                        
                    
            #finding the weight of this new 'u' (projection step)
            cur_weight = (residual.dot(u))/(np.linalg.norm(u))**2
            temp_d = d + cur_weight*u 
            
            #print("prev align: ", self.align(-grad,d))
            #print("new align: ", self.align(-grad,temp_d))
            #check if temp d is strictly better than the previous d by some 'delta'
            if(self.align(-grad,temp_d) - self.align(-grad,d) >= delta):
                if((d == 0).all()): #the very first iteration of boosting 
                    Wsum += cur_weight
                else:
                    if(is_away):
                        #print("take away step")
                        Wsum += cur_weight
                        is_away = 0 #reset indicator
                        #print("v_away: ", v_away)
                    else:
                        if(res1 > res2):
                            #print("add dir")
                            #print("add u: ", u1)
                            #the new direction has better improvement
                            Wsum += cur_weight
                            #add new resource if havn't seen for the vertex
                            if(is_bounded):
                                #print("add collection: ", v)
                                if(vertices_collection.shape[0] == 0):
                                    vertices_collection = np.vstack((vertices_collection,v))
                                else:
                                    vertices_collection = self.update_vertices(v,vertices_collection)

                        else:
                            #print("use prev d")
                            #previous d is better
                            Wsum = Wsum*(1 - cur_weight/np.linalg.norm(d))

                d = temp_d     
            else:
                #if we cannot improve the alignment by 'delta', then break
                break;

        if(Wsum == 0):
            #the loop break in ONE iteration: no combination
            d = u1
        else:
            d = d/Wsum

        #print("normalizing: ", Wsum)
        return d, num_iter, vertices_collection
    
    def solve(self, problem):    
        
        max_iters = self.factors['max_iters']
        ls_type = self.factors['LSmethod']
        #self.factors['problem'] = problem
        #print(ls_type)
        if(ls_type == 'backtracking'):
            self.factors["LSfn"] = self.backtrackLineSearch
        elif(ls_type == 'interpolation'):
            self.factors["LSfn"] = self.interpolateLineSearch
        else:
            self.factors["LSfn"] = self.zoomLineSearch
            
        #print("Solved by Frank Wolfe - " + self.factors["algorithm"])
        
        return self.boostFW(problem)
    
    def boostFW(self, problem):
        
        recommended_solns = []
        intermediate_budgets = []
        expended_budget = 0
        last_step = []
        last_gamma = []
        
        print("Boosting - ", self.factors["boostmethod"])
        
        #extracting all linear constraints
        Ci = problem.Ci
        di = problem.di
        Ce = problem.Ce
        de = problem.de
        
        lower = np.array(problem.lower_bounds)
        upper = np.array(problem.upper_bounds)
        
        #combine all linear constraints
        A, b = self.combine_constraint(Ci,di,Ce,de,lower, upper,problem)
        
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
        
        vertices_collection = np.array([]).reshape(0,problem.dim)
        rays_collection = np.array([]).reshape(0,problem.dim)
        
        # Use r simulated observations to estimate the objective value.
        problem.simulate(new_solution, r)
        expended_budget += r
        best_solution = new_solution
        recommended_solns.append(new_solution)
        intermediate_budgets.append(expended_budget)
        
        k = 0
        K = []
        
        while expended_budget < problem.factors["budget"]:
            cur_x = new_solution.x
            #print("cur x: ", tuple(cur_x))
            if problem.gradient_available:
                # Use IPA gradient if available.
                grad = -1 * problem.minmax[0] * new_solution.objectives_gradients_mean[0]
            else:
                # Use finite difference to estimate gradient if IPA gradient is not available.
                #grad, budget_spent = self.finite_diff(new_solution, problem, r, stepsize = alpha)
                #grad, budget_spent = self.get_FD_grad(cur_x, problem, self.factors["h"], self.factors["r"])
                #expended_budget += budget_spent
                grad, budget_spent = self.finite_diff(new_solution, problem, r, A, b, stepsize=self.factors["h"])
                expended_budget += budget_spent
                while np.all((grad == 0)):
                    #print("recompute gradient")
                    #print("current budget: ", expended_budget)
                    #print("r: ", r)
                    if expended_budget > problem.factors["budget"]:
                        break;
                    grad, budget_spent  = self.finite_diff(new_solution, problem, r, A, b)
                    #if budget = 0, this means we cannot move any further and grad = 0
                    if(budget_spent == 0):
                        budget_spent = r + problem.factors["budget"] - expended_budget
                    
                    expended_budget += budget_spent
                    # Update r after each iteration.
                    #r = int(self.factors["lambda"] * r)
                    r = int(2 * r)
                
            #sub-problem: finding the direction close to the negative grad
            #direction, sub_iter = self.boostDir(grad, cur_x, A, b)
            #the away-step version must take the stored components**********
            if(self.factors["boostmethod"] == "away"):
                direction, sub_iter, vertices_collection = self.boostDirAway(grad, cur_x, A, b, vertices_collection)
            else:
                direction, sub_iter = self.boostDir(grad, cur_x, A, b)
            #if(self.factors["boostmethod"] == "conic"):
            #    direction, sub_iter = self.boostConic(grad, cur_x, A, b)
            #else:
            #    direction, sub_iter = self.boostDir(grad, cur_x, A, b)
            #print("grad: ", tuple(grad))
            #print("dir: ",tuple(direction))
            print("error angle: ", 180*(np.arccos((-grad.dot(direction))/(np.linalg.norm(grad)*np.linalg.norm(direction)))/np.pi))
            print("num sub iter: ", sub_iter)
            K.append(sub_iter)
            
            #add more constraints every iteration, hoping that it could close the fesible set
            if(self.factors["boostmethod"] == "cutting"): 
                A = np.vstack((A,grad))
                b = np.vstack((b,grad.dot(cur_x)))
                #print("add new constraint")
                #print("A shape: ", A.shape)
                if((Ci is None) and (di is None)):
                    #print("initial")
                    Ci = grad
                    di = np.array([grad.dot(cur_x)])
                else:
                    #print("adding")
                    Ci = np.vstack((Ci,grad))
                    di = np.hstack((di,grad.dot(cur_x)))

            #max_gamma = 1
            #gamma_star = self.get_max_gamma_ratio_test(cur_x, direction, A, b)
            #gamma_star = self.get_max_gamma_ratio_test(cur_x, direction, Ce, Ci, de, di, lower, upper)
            if((max(abs(direction)) < 1e-10)):
                #print("direction too small")
                gamma_star = 0 #if unfortunately, we have zero direction, then just take step = 0
            else:
                #print("ratio test")
                gamma_star = self.get_max_gamma_ratio_test(cur_x, direction, Ce, Ci, de, di, lower, upper)
             
            if(gamma_star == np.inf):
                max_gamma = 1
                #print("extreme dir")
                if(self.factors["boostmethod"] == "neggrad"):
                    print("neg Grad dir")
                    direction = -grad/np.linalg.norm(grad)
                    
                if(self.factors["boostmethod"] == "infcutting"):
                    #modify constraints A, b by adding constriant 
                    #gT(x - cur_x) <= 0 or gTx <= gTcur_x
                    #print("add new constraint")
                    #print("modify constraints")
                    #print("A: ", A)
                    #print("grad: ", grad)
                    #print("A shape: ", A.shape)
                    A = np.vstack((A,grad))
                    b = np.vstack((b,grad.dot(cur_x)))
                    
                    if((Ci is None) and (di is None)):
                        #print("initial")
                        Ci = grad
                        di = np.array([grad.dot(cur_x)])
                    else:
                        #print("adding")
                        Ci = np.vstack((Ci,grad))
                        di = np.hstack((di,grad.dot(cur_x)))
                    
            else:
                #print("bounded dir")
                max_gamma = gamma_star

            print("max step: ", gamma_star)
            #line search to find step size 
            #gamma = LSfn(F,cur_x,grad,direction,max_gamma,factors)
            gamma, added_budget = self.factors['LSfn'](new_solution,grad,direction,max_gamma,problem,expended_budget)
            expended_budget += added_budget
            
            #print("cur x: ", cur_x)
            print("step: ", gamma)
            
            new_x = cur_x + gamma*direction
            #print("new_x: ",tuple(new_x))

            #updating current iteration
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
                print("obj: ",new_solution.objectives_mean)
                
            print("obj: ",new_solution.objectives_mean)

            k += 1
            print("--------------")
        print("=========================")
        return recommended_solns, intermediate_budgets 
            
            
    
    
    
   