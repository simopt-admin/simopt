#https://github.com/bodono/apgpy
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
#from apgwrapper import NumpyWrapper
#from functools import partial

import warnings
warnings.filterwarnings("ignore")

from ..base import Solver

class BoomProxGD(Solver):
    """
    
    """
    def __init__(self, name="Boom-PGD", fixed_factors={"max_iters": 300, "backtrack": 1, "curve_const": 0.3, "LSmethod": 'zoom', "algorithm": 'away'}):
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
            "LSmethod": {
                "description": "method",
                "datatype": str,
                "default": 'backtracking'
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
            "max_step_size": {
                "description": "maximum possible step size",
                "datatype": float,
                "default": 10
            },
            "max_iters": {
                "description": "maximum iterations",
                "datatype": int,
                "default": 300
            },
            "theta": {
                "description": "constant in the line search condition",
                "datatype": int,
                "default": 0.1
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
            "max_gamma":{
                "description": "max distance possible",
                "datatype": float,
                "default": 10
            },
            "backtrack":{
                "description": "an indicator whether we do the backtrack",
                "datatype": bool,
                "default": 1
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
        
        if((lower is not None) and (lower > -np.inf).all()):
            constraints += [u >= lower]
        if((upper is not None) and (upper < np.inf).all()):
            constraints += [u <= upper]
        
        if (Ci is not None) and (di is not None):
            constraints += [Ci@u <= di]
        if (Ce is not None) and (de is not None):
            constraints += [Ce@u == de]

        #constraints = [A@u <= b]
        prob = cp.Problem(objective, constraints)
        prob.solve(abstol=1e-6)#solver=cp.ECOS

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
            denom += list([Ci@d])
        
        #print("ratio: ", ratio)
        ratio_val = np.array(ratio_val)
        denom = np.array(denom)
        #print("denom: ", denom)
        #print("ratio_val: ", ratio_val)
        
        #if(len(ratio_val[denom > 1e-6]) == 0):
        
        return min(ratio_val[denom > 1e-6])
    
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
        
        #if(result < a):
        #    return a
        #elif(result > b):
        #    return b
        #else:
        #    return result

        #return numerator/denominator
        A = div_a/(a - b) - (Fa - Fb)/((a-b)**2)
        B = div_a - 2*A*a
        result = -B/(2*A)
        
        if(-problem.minmax[0] == np.sign(A)):
            #if A and problem have the same sign, i.e. min and A > 0 or max and A < 0
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
        #print("backtrack LS")
        #print("max step: ", max_step)
        r = self.factors["r"]
        ratio = self.factors["ratio"]
        max_iter = self.factors["line_search_max_iters"]
        
        cur_iter = 0
        step_size = max_step
        added_budget = 0
        
        cur_x = cur_sol.x
        #print("cur_x: ", cur_x)
        curF = -1 * problem.minmax[0] * cur_sol.objectives_mean
        #print("Line Search...")
        while True:
            #if(expended_budget + added_budget > problem.factors["budget"] or cur_iter >= max_iter):
            if(cur_iter >= max_iter):
                break;
                
            new_x = cur_x + step_size*d
            #print("next x: ",new_x)
            new_sol =  self.create_new_solution(tuple(new_x), problem)
            problem.simulate(new_sol, r)
            added_budget += r
            
            newF = -1 * problem.minmax[0] * new_sol.objectives_mean
            
            #newF, budget_spent = self.get_simulated_values(problem,cur_x + step_size*d,value = 'val')
            #added_budget += budget_spent
            #new_grad = -1 * problem.minmax[0] * new_sol.objectives_gradients_mean[0]
            #print("newF: ",newF)
            
            if(newF < curF + self.factors['theta'] * step_size * np.dot(grad, d)):
                break;
                
            step_size = step_size*ratio
            cur_iter += 1
            #print("---------------") 
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
            
            newF = -1 * problem.minmax[0] * new_sol.objectives_mean
            
            #xrange = np.arange(0,1,0.01)
            #nn = len(xrange)
            #fval = np.zeros(nn)
            #for i in range(nn):
            #    temp_x = cur_x + xrange[i]*d
            #    temp_sol = self.create_new_solution(tuple(temp_x), problem)
            #    problem.simulate(temp_sol, r)
            #    fval[i] = -1 * problem.minmax[0] * temp_sol.objectives_mean
            #    fval[i] = temp_sol.objectives_mean
            #plt.scatter(xrange,fval)
            #line = -1*problem.minmax[0]*curF + self.factors["theta"]*xrange*(-1*problem.minmax[0]* grad.dot(d))
            #plt.scatter(xrange,line,color='red')
            #plt.show()
            
            #sufficient decrease
            if(newF < curF + self.factors['theta'] * step_size * np.dot(grad, d)):
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
            #elif(new_step_size == 0):
            #    step_size = 0
            #    break;
            else:
                #if we did not make progress, use the informaiton {step/2}
                temp_x = cur_x + (step_size*ratio)*d
                temp_sol =  self.create_new_solution(tuple(temp_x), problem)
                problem.simulate(temp_sol, r)
                added_budget += r
                newF = -1 * problem.minmax[0] * temp_sol.objectives_mean
                #print("another newF: ", newF)
                #new_step_size = ((-grad.dot(d))*((step_size/2)**2))/(2*(newF-curF-(step_size/2)*(grad.dot(d))))
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
            #print("=============")  
        #print("determined step: ", step_size)
        return step_size, added_budget
    
    def zoomLineSearch(self,cur_sol,grad,d,max_step,problem,expended_budget):
        """
        carry out interpolation line search on the function F where we 
        min F(x + alpha*d) s.t. alpha >=0  where phi(a) = F(x + ad)
        
        NOTE: in this method, we increase the step size
        """
        #print("ZOOM LS")
        #print("max step: ",max_step)
        if(max_step == 0):
            return 0,0
        r = self.factors["r"]
        ratio = self.factors["ratio"]
        max_iter = self.factors["line_search_max_iters"]
        
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

        #line = -curF + self.factors["theta"]*xrange*(-1*problem.minmax[0]* grad.dot(d))
        #plt.scatter(xrange,line,color='red')
        #plt.show()
        returned_steps = []
        returned_vals = []

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
            
            if(cur_step_size >= max_step):
                break;
            
            last_step_size = cur_step_size
            cur_step_size = min(max_step,cur_step_size*self.factors["zoom_inc_ratio"])

            lastF = nextF
            last_grad = next_grad
            nextF, next_grad, budget_spent = self.get_simulated_values(problem,cur_x + cur_step_size*d,value = 'both')
            added_budget += budget_spent

            cur_iter += 1
            #print("iter: ",cur_iter)
            #print("added budget: ",added_budget)
            #print("---------------") 
            #print("max step: ",max_step)
        
        if((cur_iter == max_iter) or (cur_step_size >= max_step)):
            #return max_step*self.factors["zoom_init_ratio"], added_budget
            #print("return from iteration or max step")
            return returned_steps[np.argmin(returned_vals)] ,added_budget
            #return max_step, added_budget
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
                
            #m1 = min([step_lo,step_hi])
            #m2 = max([step_lo,step_hi])
            #print("zooming:: (",str(m1) + "," + str(m2) + ")") 
            #use the actual value without the sign
            new_step = self.quadratic_interpolate(step_lo,step_hi,sign*div_lo,sign*div_hi,sign*Flo,sign*Fhi,problem)
            #print("left div: ", left_dif)
            #print("right div: ", right_dif)
            #print("left val: ", left_val)
            #print("right val: ", right_val)
            
            #print("new step: ", new_step)
            
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
        max_gamma = self.factors["max_gamma"]
        
        ls_type = self.factors['LSmethod']
        #self.factors['problem'] = problem
        #print(ls_type)
        if(ls_type == 'backtracking'):
            self.factors["LSfn"] = self.backtrackLineSearch
        elif(ls_type == 'interpolation'):
            self.factors["LSfn"] = self.interpolateLineSearch
        else:
            self.factors["LSfn"] = self.zoomLineSearch
            
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
        action = "normal" #storing whether we do the projection in each step
        last_action = "normal"
        
        #store consecutive projections
        consec_proj = 0
        k = 0
        max_step = 1 #initial max step
        last_normal_maxstep = 1
        last_proj_maxstep = 1
        
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
                    
            #print("max_step: ",max_step)
            direction = -grad/np.linalg.norm(grad)
            temp_x = cur_x + max_step * direction
            #print("cur x: ",cur_x)
            #print("temp x: ",temp_x)
            
            #if the new iterate is feasible, then no need to project
            if(not self.is_feasible(temp_x, Ci,di,Ce,de,lower,upper)):
                action = "project"
                proj_x = self.proj(temp_x,Ci,di,Ce,de,lower,upper)
                #print("proj x: ",proj_x)
                direction = proj_x - cur_x #change direction to the projected point
                max_step = 1
                
                #if(last_action == "project"):
                    #consecutive projection: should increase max proj step
                 #   max_step = min(self.factors["max_step_size"],last_proj_maxstep/self.factors["ratio"])
                #else:
                    #last step is normal
                    #max step is to go to the boundary
               #     max_step = 1
                consec_proj += 1
            else:
                action = "normal"
                #decrease consecutive projection if we don't have the projection
                if(consec_proj > 0):
                    #consec_proj -= 1
                    consec_proj = 0
                    
            #max_step_feas = self.get_max_gamma_ratio_test(cur_x, direction, Ce, Ci, de, di, lower, upper)
            #print("max step: ", max_step)
            #step sizes
            if(self.factors["backtrack"]):
                #t, added_budget = self.LineSearch(new_solution,grad,direction,self.factors["max_gamma"],problem,expended_budget)
                #t, added_budget = self.LineSearch(new_solution,grad,direction,t,problem,expended_budget)
                t, added_budget = self.factors['LSfn'](new_solution,grad,direction,max_step,problem,expended_budget)
                expended_budget += added_budget
            else:
                #t = min(self.factors["step_f"](k),self.factors["max_gamma"])
                t = self.factors["step_f"](k)#*direction#np.linalg.norm(grad)

            if(action == "normal"):
                #store the last max step size of the normal iteratopnm
                last_normal_maxstep = max_step
                if(t == max_step):
                    #if we reach max step, then next iteration should move further
                    #max_step = min(max_gamma,max_step/self.factors["ratio"])
                    max_step = min(self.factors["max_step_size"],max_step/self.factors["ratio"])
                #t = min(t,max_step_feas)
                else:
                    max_step = max(1,max_step*self.factors["ratio"])
            else:
                #we have the projection, next max step is the max step from 
                #the iteration before projection
                #store the max step in the projection iteration
                #last_proj_maxstep = max_step 
                if(t == max_step):
                    #print("full projection")
                    #max_step = min(self.factors["max_step_size"],last_proj_maxstep/self.factors["ratio"])
                    last_proj_maxstep = min(self.factors["max_step_size"],last_proj_maxstep/self.factors["ratio"])
                    max_step = last_proj_maxstep
                else:
                    #use this for the next iteration, assume to be normal
                    last_proj_maxstep = max(1,last_proj_maxstep*self.factors["ratio"])
                    max_step = last_normal_maxstep 
            #print("act: ", action)
            #if(t == max_step):
            #    max_step = min(max_gamma,max_step/self.factors["ratio"])
            #print("grad: ", grad)
            #print("max step: ", max_step)
            #print("step: ", t)
 
            #new_x = cur_x - t * grad
            new_x = cur_x + t * direction
            last_action = action
            #update the max step for the next iteration
            #t = min(self.factors["max_step_size"],t/self.factors["ratio"])
            #print("new_x before proj: ", new_x)

            #print("new_x after proj: ", new_x)
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
            
            #print("current budget: ",expended_budget)
            #print("========================")
            
            k += 1
        #print("obj: ",candidate_solution.objectives_mean)
        #print("------------------------------------")   
        return recommended_solns, intermediate_budgets    
    