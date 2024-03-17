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


class FrankWolfeSS(Solver):
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
            #print("newX: ", new_x)
            
            if(newF < curF + self.factors['theta'] * step_size * np.dot(grad, d)):
                break;
                
            step_size = step_size*ratio
            cur_iter += 1
            #print("---------------")
        #print("step from backtrack: ",step_size)
        return step_size, added_budget
    
    def stepLineSearch(self,cur_sol,grad,d,max_step,last_step_size,problem,expended_budget):
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
        #alpha_max = self.factors['alpha_max']
        ratio = self.factors['ratio']
        #alpha = self.factors['alpha']
        #alpha_0 = max_step
        
        step_size = last_step_size
        added_budget = 0
        
        x = cur_sol.x
        #fx = -1 * problem.minmax[0] * cur_sol.objectives_mean
        # step_size = self.factors['alpha_0']
        new_solution = cur_sol
        x_temp = x + step_size * d
        # Create a solution object for x_new.
        x_temp_solution = self.create_new_solution(tuple(x_temp), problem)
        # Use r simulated observations to estimate the objective value.
        problem.simulate(x_temp_solution, r)
        added_budget += r
            
        # Check the modified Armijo condition for sufficient decrease.
        if (-1 * problem.minmax[0] * x_temp_solution.objectives_mean) < (
                -1 * problem.minmax[0] * cur_sol.objectives_mean + step_size * theta * np.dot(grad, d) + 2 * epsilon_f):
            # Successful step
            #new_solution = x_temp_solution
            # Enlarge step size.
            # alpha = min(alpha_max, alpha / gamma)
            step_size = min(max_step, step_size/ratio)
        else:
            # Unsuccessful step - reduce step size.
            # alpha = gamma * alpha
            step_size = ratio * step_size
            
        # self.factors['alpha'] = alpha
        
        return step_size, added_budget

    
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
        
        recommended_solns = []
        intermediate_budgets = []
        expended_budget = 0
        last_step = []
        
        Ci = problem.Ci
        di = problem.di
        Ce = problem.Ce
        de = problem.de
        
        lower = np.array(problem.lower_bounds)
        upper = np.array(problem.upper_bounds)
        
        ratio = self.factors["ratio"]
        LSmax_iter = self.factors["line_search_max_iters"]
        theta = self.factors["theta"]
        epsilon_f = self.factors['epsilon_f']
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
        
        #last_gamma = self.factors['alpha_0']
        gamma = self.factors['alpha_0']
        
        k = 0
        
        while expended_budget < problem.factors["budget"]:
            cur_x = new_solution.x
            
            #determine gradients
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
            #determine directions
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

                if(is_bounded):
                    #print("bounded")
                    max_gamma = 1
                    gamma = min(gamma, max_gamma)

                    new_x = cur_x + gamma*direction
                    candidate_solution = self.create_new_solution(tuple(new_x), problem)
                    # Use r simulated observations to estimate the objective value.
                    problem.simulate(candidate_solution, r)
                    #print("obj: ",candidate_solution.objectives_mean)
                    expended_budget += r
                    
                    if (-1 * problem.minmax[0] * candidate_solution.objectives_mean) < (
                -1 * problem.minmax[0] * new_solution.objectives_mean + gamma * theta * np.dot(grad, direction) + 2 * epsilon_f):
                        # Successful step
                        new_solution = candidate_solution
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

                        #for atom in active_vectors[k]:
                        for atom in active_vectors:
                            if((atom == s).all()):
                                alphas[tuple(atom)] = (1-gamma)*alphas[tuple(atom)] + gamma
                            else:
                                alphas[tuple(atom)] = (1-gamma)*alphas[tuple(atom)]

                        for dirs in active_dirs:
                            betas[tuple(dirs)] = (1-gamma)*betas[tuple(dirs)]

                        # Enlarge step size.
                        gamma = min(max_gamma, gamma/ratio)
                    else:
                        # Unsuccessful step - reduce step size.
                        # alpha = gamma * alpha
                        gamma = ratio * gamma
                  
                    last_step.append("bounded")
                else:
                    #print("unbounded")
                    #searching in the infinite search direction
                    #if we have consecutive extreme search
                    if(k == 0 or last_step[-1] != 'unbounded'):
                        #gamma = gamma/self.factors["ratio"]
                        max_gamma = 1
                        gamma = min(gamma, max_gamma)
                    else:
                        max_gamma = self.factors["alpha_max"]

                    new_x = cur_x + gamma*direction
                    candidate_solution = self.create_new_solution(tuple(new_x), problem)
                    # Use r simulated observations to estimate the objective value.
                    problem.simulate(candidate_solution, r)
                    #print("obj: ",candidate_solution.objectives_mean)
                    expended_budget += r
                    
                    if (-1 * problem.minmax[0] * candidate_solution.objectives_mean) < (
                -1 * problem.minmax[0] * new_solution.objectives_mean + gamma * theta * np.dot(grad, direction) + 2 * epsilon_f):
                        # Successful step
                        new_solution = candidate_solution
                        
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

                        # Enlarge step size.
                        gamma = min(max_gamma, gamma/ratio)
                        
                    else:
                        # Unsuccessful step - reduce step size.
                        # alpha = gamma * alpha
                        gamma = ratio * gamma

                    last_step.append("unbounded")
                          
            else:
                #away step
                #print("away")
                #ind.append('away')
                direction = d_away #xt - v
                #print("dir: ", direction)
                #max_gamma = alphas[tuple(v)]/(1 - alphas[tuple(v)])
                gamma_star = self.get_max_gamma_ratio_test(cur_x, direction, Ce, Ci, de, di, lower, upper)
                #max_dist = min(1,alphas[tuple(v)]/(gamma_star*(1 - alphas[tuple(v)])))
                max_dist = min(gamma_star,alphas[tuple(v)]/((1 - alphas[tuple(v)])))
                #print("max_dist: ", max_dist)
                
                #max_gamma = 1
                gamma = min(gamma, max_dist)

                new_x = cur_x + gamma*direction
                candidate_solution = self.create_new_solution(tuple(new_x), problem)
                # Use r simulated observations to estimate the objective value.
                problem.simulate(candidate_solution, r)
                #print("obj: ",candidate_solution.objectives_mean)
                expended_budget += r
                
                if (-1 * problem.minmax[0] * candidate_solution.objectives_mean) < (
                -1 * problem.minmax[0] * new_solution.objectives_mean + gamma * theta * np.dot(grad, direction) + 2 * epsilon_f):
                    # Successful step
                    new_solution = candidate_solution
                    #update the active set
                    
                    #if gamma_max, then update St \ {vt}
                    if(gamma == 1 or gamma <= ratio**LSmax_iter):
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
                    # Enlarge step size.
                    gamma = min(max_dist, gamma/ratio)
                
                else:
                    # Unsuccessful step - reduce step size.
                    # alpha = gamma * alpha
                    gamma = ratio * gamma

                last_step.append("away")
            
            
            #print("max_gamma: ", max_gamma)
            #print("gamma: ", gamma)
            #print("dir: ", tuple(direction))
            #last_gamma.append(gamma)
            #last_gamma = gamma
            #print("last gamma: ", last_gamma)
            #new_x = cur_x + gamma*direction
            #print("new_x: ",tuple(new_x))
            
            #print("obj: ",candidate_solution.objectives_mean)
            #new_solution = candidate_solution

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
    