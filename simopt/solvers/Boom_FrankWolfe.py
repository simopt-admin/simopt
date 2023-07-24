import numpy as np
import cvxpy as cp
import cdd


import warnings
warnings.filterwarnings("ignore")

from ..base import Solver


class BoomFrankWolfe(Solver):
    """
    """
    
    def __init__(self, name="Boom-FW", fixed_factors={}):
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
                "default": 0.2
            },
            "max_iters": {
                "description": "maximum iterations",
                "datatype": int,
                "default": 300
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
            "curve_const": {
                "description": "constant in curvature wolfe conditions, usually greater than theta",
                "datatype": float,
                "default": 0.3
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
                "default": 0
            },
            "algorithm":{
                "description": "type of FW algorithm",
                "datatype": str,
                "default": "normal"
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
        step = cp.Variable()
        objective = cp.Maximize(step)
        constraints = [step >= 0]
        
        if(lower is not None):
            constraints += [(cur_x + step*d) >= lower]
        if(upper is not None):
            constraints += [(cur_x + step*d) <= upper]
        if((Ci is not None) and (di is not None)):
            constraints += [Ci@(cur_x + step*d) <= di]
        if((Ce is not None) and (de is not None)):
            constraints += [Ce@(cur_x + step*d) == de]
           
            
        prob = cp.Problem(objective, constraints)
        prob.solve()
        
        if(step.value is None):
            return 0
        else:
            return step.value
    
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
        prob.solve()#solver=cp.ECOS

        return s.value
    
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
        #print("LineSearch...")
        #print("cur_x: ", cur_x)
        curF = -1 * problem.minmax[0] * cur_sol.objectives_mean
        
        while True:
            if(expended_budget + added_budget > problem.factors["budget"] or cur_iter >= max_iter):
                break;
                
            new_x = cur_x + step_size*d
            #print("step: ", step_size)
            #print("new_x: ", new_x)
            new_sol =  self.create_new_solution(tuple(new_x), problem)
            problem.simulate(new_sol, r)
            added_budget += r
            
            #get the gradient of the new solution grad f(x + step*d) for 
            #curvature condition
            #if problem.gradient_available:
                # Use IPA gradient if available.
            #    next_grad = -1 * problem.minmax[0] * new_sol.objectives_gradients_mean[0]
            #else:
            #    next_grad, budget_spent = self.get_FD_grad(new_x, problem, self.factors["h"], self.factors["r"])
            #    added_budget += budget_spent
            
            #problem.simulate(new_sol, r)
            #added_budget += r
            
            newF = -1 * problem.minmax[0] * new_sol.objectives_mean
            
            #armijo = (newF < curF + self.factors['theta'] * step_size * np.dot(grad, d))
            #curvature = np.dot(next_grad,d) > self.factors["curve_const"] * np.dot(grad, d)
            
            if(newF < curF + self.factors['theta'] * step_size * np.dot(grad, d)):
            #if(armijo and curvature):
                break
            step_size = step_size*ratio
            cur_iter += 1
        #print("newF: ", newF)
        #print("linear F: ", curF + self.factors['theta'] * step_size * np.dot(grad, d))
        #print("inner iter: ", cur_iter)   
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
        
        #print("Solved by Frank Wolfe - " + self.factors["algorithm"])
        
        if(self.factors["algorithm"] == "normal"):
            return self.normal_FrankWolfe(problem)
        elif(self.factors["algorithm"] == "away"):
            return self.away_FrankWolfe(problem)
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
                # A while loop to prevent zero gradient.
                #while np.all((grad == 0)):
                #    if expended_budget > problem.factors["budget"]:
                #        break
                #    grad, budget_spent  = self.finite_diff(new_solution, problem, r)
                #    expended_budget += budget_spent
                    # Update r after each iteration.
                #    r = int(self.factors["lambda"] * r)
  
            v = self.get_dir(grad,Ce, Ci, de, di,lower,upper) 
            direction = (v-cur_x)/np.linalg.norm(v-cur_x)
            #direction = (v-cur_x)
            #print("dir: ", v)
            max_gamma = self.get_max_gamma_ratio_test(cur_x, direction, Ce, Ci, de, di, lower, upper)
            max_gamma = max_gamma*self.factors["max_gamma"]
            #max_gamma = 1
            
            if(self.factors["backtrack"]):
                #gamma, added_budget = self.LineSearch(new_solution,grad,direction,self.factors["max_gamma"],problem,expended_budget)
                gamma, added_budget = self.LineSearch(new_solution,grad,direction,max_gamma,problem,expended_budget)
                expended_budget += added_budget
            else:
                #gamma = min(self.factors["step_f"](k),self.factors["max_gamma"])
                gamma = min(self.factors["step_f"](k),max_gamma)
            
            #k = self.get_max_gamma_ratio_test(cur_x, direction, Ce, Ci, de, di, lower, upper)
            #print("current max gamma: ", k)
            #print("gamma: ", gamma)
            #print("direction: ",direction)
            #print("grad*direction: ", np.linalg.norm(grad.dot(direction)))
            #new_x = (1 - gamma)*np.array(cur_x) + gamma*v
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
        #print("-----------------------------")   
        return recommended_solns, intermediate_budgets 
    
    def away_FrankWolfe(self, problem):
        
        recommended_solns = []
        intermediate_budgets = []
        expended_budget = 0
        
        Ci = problem.Ci
        di = problem.di
        Ce = problem.Ce
        de = problem.de
        
        lower = np.array(problem.lower_bounds)
        upper = np.array(problem.upper_bounds)
        
        #ratio = self.factors["ratio"]
        #LSmax_iter = self.factors["line_search_max_iters"]
        r = self.factors["r"]
        
        # Start with the initial solution.
        new_solution = self.create_new_solution(problem.factors["initial_solution"], problem)
        new_x = new_solution.x
        
        if(not self.is_feasible(new_x, Ci,di,Ce,de,lower,upper)):
            new_x = self.find_feasible_initial(problem, Ce, Ci, de, di,lower,upper)
            new_solution = self.create_new_solution(tuple(new_x), problem)
        
        #initiailizing a dictionary of atom vectors and their coefficients
        #atom_vectors = self.factors["atom_vectors"]
        if(self.factors["atom_vectors"] is None):
            atom_vectors = self.get_atom_vectors(Ci,di)
            num_atoms = atom_vectors.shape[0]
            alpha_vec = np.zeros(num_atoms)
            alpha_vec[0] = 1
            
            #initialize the solution to just be a vertex
            #print("start at a vertex")
            new_x = atom_vectors[0]
            new_solution = self.create_new_solution(tuple(new_x), problem)
            
            #initial solution is a centroid
            #print("start at a centroid")
            #new_x = sum(atom_vectors)/num_atoms
            #new_solution = self.create_new_solution(tuple(new_x), problem)
        else:
            atom_vectors = self.factors["atom_vectors"]
            num_atoms = atom_vectors.shape[0]
            alpha_vec = self.get_alpha_vec(new_x,atom_vectors)
        
        
        #atom_vectors = self.get_atom_vectors(Ci,di)
        #num_atoms = atom_vectors.shape[0]
        active_vectors = {0:[]}
        alphas = {tuple(v):0 for v in atom_vectors}
        
        #new_x = atom_vectors[0]
        #new_solution = self.create_new_solution(tuple(new_x), problem)
        #print("x0: ", new_x)
      
        #m,n = atom_vectors.shape
        #determine the weights on the atom vectors(polytope's vertices)
        #if(m != n):
        #    alpha_vec = np.linalg.solve(atom_vectors.dot(atom_vectors.T),atom_vectors.dot(new_x))
        #else:
        #    alpha_vec = np.linalg.solve(atom_vectors.T,new_x)
        
        #alpha_vec = self.get_alpha_vec(new_x,atom_vectors)
        #alpha_vec = np.zeros(m)
        #alpha_vec[0] = 1
        
        for i in range(num_atoms): 
            alphas[tuple(atom_vectors[i])] = alpha_vec[i]
            if(alpha_vec[i] > 0):
                active_vectors[0].append(atom_vectors[i])
        
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
                # A while loop to prevent zero gradient.
                #while np.all((grad == 0)):
                #    if expended_budget > problem.factors["budget"]:
                #        break
                #    grad, budget_spent  = self.finite_diff(new_solution, problem, r)
                #    expended_budget += budget_spent
                    # Update r after each iteration.
                #    r = int(self.factors["lambda"] * r)
                   
            #print("grad: ", grad)
            #the list dot product values [grad_f.a for a in atom]
            gs = np.array([grad.dot(a) for a in atom_vectors])
            s = atom_vectors[np.argmin(gs)]
            #list of dot product of [grad_f.v for v in active set]
            gv = np.array([grad.dot(a) for a in active_vectors[k]])
            v = active_vectors[k][np.argmax(gv)]
            #compute the directions
            d_FW = s - cur_x
            d_away = cur_x - v
            
            #d_FW = d_FW/np.linalg.norm(d_FW)
            #d_FW = d_away/np.linalg.norm(d_away)
            
            #print("s: ",s)
            #print("v: ",v)
            
            if((-grad.dot(d_FW) >= -grad.dot(d_away)) or (d_away == 0).all()):
                #FW step
                #print("foward")
                #ind.append('FW')
                direction = d_FW
                #direction = d_FW/np.linalg.norm(d_FW)
                max_gamma = self.factors["max_gamma"]
                #max_gamma = self.get_max_gamma_ratio_test(cur_x, direction, Ce, Ci, de, di, lower, upper)
                #max_gamma = max_gamma*self.factors["max_gamma"]

                if(self.factors["backtrack"]):
                    gamma, added_budget = self.LineSearch(new_solution,grad,direction,max_gamma,problem,expended_budget)
                    #gamma = self.LineSearch(F=F,x=cur_x,d=d_FW,max_step=max_gamma/2)
                    #gamma, added_budget = self.LineSearch(cur_x,grad,direction,max_gamma,ratio,LSmax_iter,problem,r)
                    expended_budget += added_budget
                else:
                    gamma = min(self.factors["step_f"](k),max_gamma)
                
                #print("alpha: ", alphas)
                #update the active set
                if(gamma < 1):
                    active_vectors[k+1] = active_vectors[k] 
                    for vec in active_vectors[k]:
                        if((vec != s).all()):
                            add = 1
                        else:
                            add = 0
                            break;
                    if(add):
                        active_vectors[k+1].append(s)
                else:
                    active_vectors[k+1] = [s]


                for atom in active_vectors[k]:
                    if((atom == s).all()):
                        alphas[tuple(atom)] = (1-gamma)*alphas[tuple(atom)] + gamma
                    else:
                        alphas[tuple(atom)] = (1-gamma)*alphas[tuple(atom)]
            
            else:
                #away step
                #print("away")
                #ind.append('away')
                direction = d_away #xt - v
                #direction = d_away/np.linalg.norm(d_away)
                #gamma = gamma_f(k)
                #max_gamma = alphas[tuple(v)]/(1 - alphas[tuple(v)])
                gamma_star = self.get_max_gamma_ratio_test(cur_x, direction, Ce, Ci, de, di, lower, upper)
                gamma_star = gamma_star*self.factors["max_gamma"]
                direction = direction*gamma_star #d' = gamma_star*d
                max_dist = min(1,alphas[tuple(v)]/(gamma_star*(1 - alphas[tuple(v)])))
                #max_dist = 1
                #max_gamma = alphas[v]/(1 - alphas[v])
                #print("max_dist: ", max_dist)
                active_vectors[k+1] = active_vectors[k]

                if(self.factors["backtrack"]):
                    gamma, added_budget = self.LineSearch(new_solution,grad,direction,max_dist,problem,expended_budget)
                    expended_budget += added_budget
                else:
                    gamma = min(self.factors["step_f"](k),self.factors["max_gamma"])

                #if gamma_max, then update St \ {vt}

                if(gamma == 1):
                    active_vectors[k+1] = []
                    for vec in active_vectors[k]:
                        if(not (vec != v).all()):
                            active_vectors[k+1].append(vec)

                for atom in active_vectors[k]:
                    if((atom == v).all()):
                        #alphas[tuple(atom)] = (1+gamma)*alphas[tuple(atom)] - gamma
                        alphas[tuple(atom)] = (1+gamma*gamma_star)*alphas[tuple(atom)] - gamma*gamma_star
                    else:
                        alphas[tuple(atom)] = (1+gamma*gamma_star)*alphas[tuple(atom)]
            #print("alphas: ", alphas)
            #print("Displaying Alphas:")
            #for key,val in alphas.items():
            #    print(key)
            #    print(val)
            #    print('**************')
            
            #print("max_gamma: ", max_gamma)
            #print("gamma: ", gamma)
            new_x = cur_x + gamma*direction
            #print("dir: ", direction)
            #print("new_x: ", new_x)
            candidate_solution = self.create_new_solution(tuple(new_x), problem)
            # Use r simulated observations to estimate the objective value.
            problem.simulate(candidate_solution, r)
            expended_budget += r
            print("obj: ",candidate_solution.objectives_mean)
           
            
            new_solution = candidate_solution
            recommended_solns.append(candidate_solution)
            intermediate_budgets.append(expended_budget)
            
            k += 1
            print("--------------")
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
        new_solution = self.create_new_solution(problem.factors["initial_solution"], problem)
        new_x = new_solution.x
        
        if(not self.is_feasible(new_x, Ci,di,Ce,de,lower,upper)):
            new_x = self.find_feasible_initial(problem, Ce, Ci, de, di,lower,upper)
            new_solution = self.create_new_solution(tuple(new_x), problem)
        
        #initiailizing a dictionary of atom vectors and their coefficients
        #atom_vectors = self.factors["atom_vectors"]
        if(self.factors["atom_vectors"] is None):
            atom_vectors = self.get_atom_vectors(Ci,di)
            num_atoms = atom_vectors.shape[0]
            alpha_vec = np.zeros(num_atoms)
            alpha_vec[0] = 1
            
            new_x = atom_vectors[0]
            new_solution = self.create_new_solution(tuple(new_x), problem)
        else:
            atom_vectors = self.factors["atom_vectors"]
            num_atoms = atom_vectors.shape[0]
            alpha_vec = self.get_alpha_vec(new_x,atom_vectors)
        
        #initiailizing a dictionary of atom vectors and their coefficients
        #atom_vectors = self.factors["atom_vectors"]
        #atom_vectors = self.get_atom_vectors(Ci,di)
        #num_atoms = atom_vectors.shape[0]
        active_vectors = {0:[]}
        alphas = {tuple(v):0 for v in atom_vectors}
        
        #new_x = atom_vectors[0]
        #new_solution = self.create_new_solution(tuple(new_x), problem)
        
        #m,n = atom_vectors.shape
        #determine the weights on the atom vectors(polytope's vertices)
        #if(m != n):
        #    alpha_vec = np.linalg.solve(atom_vectors.dot(atom_vectors.T),atom_vectors.dot(new_x))
        #else:
        #    alpha_vec = np.linalg.solve(atom_vectors.T,new_x)

        #alpha_vec = get_alpha_vec(self,new_x,atom_vectors)
        #alpha_vec = np.zeros(num_atoms)
        #alpha_vec[0] = 1
        
        for i in range(num_atoms): 
            alphas[tuple(atom_vectors[i])] = alpha_vec[i]
            if(alpha_vec[i] > 0):
                active_vectors[0].append(atom_vectors[i])
        
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
                   
            
            #the list dot product values [grad_f.a for a in atom]
            gs = np.array([grad.dot(a) for a in atom_vectors])
            s = atom_vectors[np.argmin(gs)]
            #list of dot product of [grad_f.v for v in active set]
            gv = np.array([grad.dot(a) for a in active_vectors[k]])
            v = active_vectors[k][np.argmax(gv)]
            #compute the directions
            direction = s - v
            
            if((direction == 0).all()):
                direction = np.zeros(problem.dim)
            else:
                direction = direction/np.linalg.norm(direction)
                
            
            #print("s-v: ", s-v)
            
            #print("dir: ", direction)
            #print("v: ", v)
            #max_gamma = min(alphas[tuple(v)]*np.linalg.norm(s-v),self.factors["max_gamma"])
            max_gamma = self.get_max_gamma_ratio_test(cur_x, direction, Ce, Ci, de, di, lower, upper)
            max_gamma = max_gamma*self.factors["max_gamma"]
            
            #print("max_step: ", max_gamma)
            if(self.factors["backtrack"]):
                #gamma = LineSearch(F=F,x=cur_x,d=d_away,max_step=max_gamma/2)
                gamma, added_budget = self.LineSearch(new_solution,grad,direction,max_gamma,problem,expended_budget)
                expended_budget += added_budget
            else:
                #gamma = self.factors["step_f"](k)
                gamma = min(self.factors["step_f"](k),max_gamma)

            #updating alphas
            alphas[tuple(s)] = alphas[tuple(s)] + gamma
            alphas[tuple(v)] = alphas[tuple(v)] - gamma
            
            active_vectors[k+1] = []
        
            for atom in atom_vectors:
                if(alphas[tuple(atom)] > 0):
                    active_vectors[k+1].append(atom)
            
            new_x = cur_x + gamma*direction
            candidate_solution = self.create_new_solution(tuple(new_x), problem)
            # Use r simulated observations to estimate the objective value.
            problem.simulate(candidate_solution, r)
            expended_budget += r
           
            
            new_solution = candidate_solution
            recommended_solns.append(candidate_solution)
            intermediate_budgets.append(expended_budget)
            
            k += 1
            
        return recommended_solns, intermediate_budgets 
        