"""
Summary
-------
ACTIVESET: An active set algorithm for problems with linear constraints i.e., Ce@x = de, Ci@x <= di.
A detailed description of the solver can be found `here <https://simopt.readthedocs.io/en/latest/active_set.html>`_.
"""
import numpy as np
import cvxpy as cp
import warnings
warnings.filterwarnings("ignore")

from ..base import Solver


class ACTIVESET(Solver):
    """
    The Active Set solver.

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
    def __init__(self, name="ACTIVESET", fixed_factors={}):
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
            "alpha": {
                "description": "tolerance for sufficient decrease condition.",
                "datatype": float,
                "default": 0.2
            },
            "beta": {
                "description": "step size reduction factor in line search.",
                "datatype": float,
                "default": 0.9
            },
            "alpha_max": {
                "description": "maximum step size.",
                "datatype": float,
                "default": 10.0
            },
            "lambda": {
                "description": "magnifying factor for r inside the finite difference function",
                "datatype": int,
                "default": 2
            },
            "tol": {
                "description": "floating point tolerance for checking tightness of constraints",
                "datatype": float,
                "default": 1e-7
            },
            "tol2": {
                "description": "floating point tolerance for checking closeness of dot product to zero",
                "datatype": float,
                "default": 1e-7
            },
            "finite_diff_step": {
                "description": "step size for finite difference",
                "datatype": float,
                "default": 1e-5
            }
        }
        self.check_factor_list = {
            "crn_across_solns": self.check_crn_across_solns,
            "r": self.check_r,
            "alpha": self.check_alpha,
            "beta": self.check_beta,
            "alpha_max": self.check_alpha_max,
            "lambda": self.check_lambda,
            "tol": self.check_tol,
            "tol2": self.check_tol2,
            "finite_diff_step": self.check_finite_diff_step
        }
        super().__init__(fixed_factors)

    def check_r(self):
        return self.factors["r"] > 0

    def check_alpha(self):
        return self.factors["alpha"] > 0

    def check_beta(self):
        return self.factors["beta"] > 0 & self.factors["beta"] < 1

    def check_alpha_max(self):
        return self.factors["alpha_max"] > 0
    
    def check_lambda(self):
        return self.factors["lambda"] > 0

    def check_tol(self):
        return self.factors["tol"] > 0
    
    def check_tol2(self):
        return self.factors["tol2"] > 0
    
    def check_finite_diff_step(self):
        return self.factors["finite_diff_step"] > 0

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

        # Default values.
        r = self.factors["r"]
        alpha = self.factors["alpha"]
        beta = self.factors["beta"]
        tol = self.factors["tol"]
        tol2 = self.factors["tol2"]
        max_step = self.factors["alpha_max"] # Maximum step size

        # Upper bound and lower bound.
        lower_bound = np.array(problem.lower_bounds)
        upper_bound = np.array(problem.upper_bounds)

        # Input inequality and equlaity constraint matrix and vector.
        # Cix <= di
        # Cex = de
        Ci = problem.Ci
        di = problem.di
        Ce = problem.Ce
        de = problem.de

        # Remove redundant upper/lower bounds.
        ub_inf_idx = np.where(~np.isinf(upper_bound))[0]
        lb_inf_idx = np.where(~np.isinf(lower_bound))[0]

        # Form a constraint coefficient matrix where all the equality constraints are put on top and
        # all the bound constraints in the bottom and a constraint coefficient vector.  
        if (Ce is not None) and (de is not None) and (Ci is not None) and (di is not None):
            C = np.vstack((Ce,  Ci))
            d = np.vstack((de.T, di.T))
        elif (Ce is not None) and (de is not None):
            C = Ce
            d = de.T
        elif (Ci is not None) and (di is not None):
            C = Ci
            d = di.T
        else:
          C = np.empty([1, problem.dim])
          d = np.empty([1, 1])
        
        if len(ub_inf_idx) > 0:
            C = np.vstack((C, np.identity(upper_bound.shape[0])))
            d = np.vstack((d, upper_bound[np.newaxis].T))
        if len(lb_inf_idx) > 0:
            C = np.vstack((C, -np.identity(lower_bound.shape[0])))
            d = np.vstack((d, -lower_bound[np.newaxis].T))

        # Number of equality constraints.
        if (Ce is not None) and (de is not None):
            neq = len(de)
        else:
            neq = 0

        # Checker for whether the problem is unconstrained.
        unconstr_flag = (Ce is None) & (Ci is None) & (di is None) & (de is None) & (all(np.isinf(lower_bound))) & (all(np.isinf(upper_bound)))

        # Start with the initial solution.
        new_solution = self.create_new_solution(problem.factors["initial_solution"], problem)
        new_x = new_solution.x

        # If the initial solution is not feasible, generate one using phase one simplex.
        if (not unconstr_flag) & (not self._feasible(new_x, problem, tol)):
            new_x = self.find_feasible_initial(problem, Ce, Ci, de, di, tol)
            new_solution = self.create_new_solution(tuple(new_x), problem)
        
        # Use r simulated observations to estimate the objective value.
        problem.simulate(new_solution, r)
        expended_budget += r
        best_solution = new_solution
        recommended_solns.append(new_solution)
        intermediate_budgets.append(expended_budget)

        # Active constraint index vector.
        acidx = []
        if not unconstr_flag:
            # Initialize the active set to be the set of indices of the tight constraints.
            cx = np.dot(C, new_x)
            for j in range(cx.shape[0]):
                if j < neq or np.isclose(cx[j], d[j], rtol=0, atol= tol):
                    acidx.append(j)

        while expended_budget < problem.factors["budget"]:
            new_x = new_solution.x
            # Check variable bounds.
            forward = np.isclose(new_x, lower_bound, atol = tol).astype(int)
            backward = np.isclose(new_x, upper_bound, atol = tol).astype(int)
            # BdsCheck: 1 stands for forward, -1 stands for backward, 0 means central diff.
            BdsCheck = np.subtract(forward, backward)

            if problem.gradient_available:
                # Use IPA gradient if available.
                grad = -1 * problem.minmax[0] * new_solution.objectives_gradients_mean[0]
            else:
                # Use finite difference to estimate gradient if IPA gradient is not available.
                grad = self.finite_diff(new_solution, BdsCheck, problem, r, self.factors["finite_diff_step"])
                expended_budget += (2 * problem.dim - np.sum(BdsCheck != 0)) * r
                # A while loop to prevent zero gradient
                while np.all((grad == 0)):
                    if expended_budget > problem.factors["budget"]:
                        break
                    grad = self.finite_diff(new_solution, BdsCheck, problem, alpha, r)
                    expended_budget += (2 * problem.dim - np.sum(BdsCheck != 0)) * r
                    # Update r after each iteration.
                    r = int(self.factors["lambda"] * r)

            # If the active set is empty, search on negative gradient.
            if len(acidx) == 0:
                dir = -grad
            else:
                # Find the search direction and Lagrange multipliers of the direction-finding problem.
                dir, lmbd, = self.compute_search_direction(acidx, grad, problem, C)
            # If the optimal search direction is 0
            if (np.isclose(np.dot(grad, dir), 0, rtol=0, atol=tol2)):
                # Terminate if Lagrange multipliers of the inequality constraints in the active set are all nonnegative.
                if unconstr_flag or np.all(lmbd[neq:] >= 0):
                    break
                # Otherwise, drop the inequality constraint in the active set with the most negative Lagrange multiplier.
                else:
                    q = acidx[neq + np.argmin(lmbd[neq:][lmbd[neq:] < 0])]
                    acidx.remove(q)
            else:
                if not unconstr_flag:
                    idx = list(set(np.arange(C.shape[0])) - set(acidx)) # Constraints that are not in the active set.
                # If all constraints are feasible.
                if unconstr_flag or np.all(C[idx,:] @ dir <= 0):
                    # Line search to determine a step_size.
                    new_solution, step_size, expended_budget = self.line_search(problem, expended_budget, r, grad, new_solution, max_step, dir, alpha, beta)
                    # Update maximum step size for the next iteration.
                    max_step = step_size

                # Ratio test to determine the maximum step size possible
                else:
                    # Get all indices not in the active set such that Ai^Td>0
                    r_idx = list(set(idx).intersection(set((C @ dir > 0).nonzero()[0])))
                    # Compute the ratio test
                    ra = d[r_idx,:].flatten() - C[r_idx, :] @ new_x
                    ra_d = C[r_idx, :] @ dir
                    # Initialize maximum step size.
                    s_star = np.inf
                    # Initialize blocking constraint index.
                    q = -1
                    # Perform ratio test.
                    for i in range(len(ra)):
                        if ra_d[i] - tol > 0:
                            s = ra[i]/ra_d[i]
                            if s < s_star:
                                s_star = s
                                q = r_idx[i]
                    # If there is no blocking constraint (i.e., s_star >= 1)
                    if s_star >= 1:
                        # Line search to determine a step_size.
                        new_solution, step_size, expended_budget = self.line_search(problem, expended_budget, r, grad, new_solution, s_star, dir, alpha, beta)
                    # If there is a blocking constraint (i.e., s_star < 1)
                    else:
                        # Add blocking constraint to the active set.
                        if q not in acidx:
                            acidx.append(q)
                        # No need to do line search if s_star is 0.
                        if s_star > 0:
                            # Line search to determine a step_size.
                            new_solution, step_size, expended_budget = self.line_search(problem, expended_budget, r, grad, new_solution, s_star, dir, alpha, beta)

            # Append new solution.
            if (problem.minmax[0] * new_solution.objectives_mean > problem.minmax[0] * best_solution.objectives_mean):
                best_solution = new_solution
                recommended_solns.append(new_solution)
                intermediate_budgets.append(expended_budget)

        return recommended_solns, intermediate_budgets


    def compute_search_direction(self, acidx, grad, problem, C):
        '''
        Compute a search direction by solving a direction-finding quadratic subproblem at solution x.

        Arguments
        ---------
        acidx: list
            list of indices of active constraints
        grad : ndarray
            the estimated objective gradient at new_solution
        problem : Problem object
            simulation-optimization problem to solve
        C : ndarray
            constraint matrix

        Returns
        -------
        d : ndarray
            search direction
        lmbd : ndarray
            Lagrange multipliers for this LP
        '''
        # Define variables.
        d = cp.Variable(problem.dim)

        # Define constraints.
        constraints = [C[acidx, :] @ d == 0]
        
        # Define objective.
        obj = cp.Minimize(grad @ d + 0.5 * cp.quad_form(d, np.identity(problem.dim)))
        prob = cp.Problem(obj, constraints)
        prob.solve()
        # Get Lagrange multipliers
        lmbd = prob.constraints[0].dual_value

        dir = np.array(d.value)
        dir[np.abs(dir) < self.factors["tol"]] = 0

        return dir, lmbd


    def finite_diff(self, new_solution, BdsCheck, problem, r, stepsize = 1e-5):
        '''
        Finite difference for approximating objective gradient at new_solution.

        Arguments
        ---------
        new_solution : Solution object
            a solution to the problem
        BdsCheck : ndarray
            an array that checks for lower/upper bounds at each dimension
        problem : Problem object
            simulation-optimization problem to solve
        r : int 
            number of replications taken at each solution
        stepsize: float
            step size for finite differences

        Returns
        -------
        grad : ndarray
            the estimated objective gradient at new_solution
        '''
        lower_bound = problem.lower_bounds
        upper_bound = problem.upper_bounds
        fn = -1 * problem.minmax[0] * new_solution.objectives_mean
        new_x = new_solution.x
        # Store values for each dimension.
        FnPlusMinus = np.zeros((problem.dim, 3))
        grad = np.zeros(problem.dim)

        for i in range(problem.dim):
            # Initialization.
            x1 = list(new_x)
            x2 = list(new_x)
            # Forward stepsize.
            steph1 = stepsize
            # Backward stepsize.
            steph2 = stepsize

            # Check variable bounds.
            if x1[i] + steph1 > upper_bound[i]:
                steph1 = np.abs(upper_bound[i] - x1[i])
            if x2[i] - steph2 < lower_bound[i]:
                steph2 = np.abs(x2[i] - lower_bound[i])
            
            # Decide stepsize.
            # Central diff.
            if BdsCheck[i] == 0:
                FnPlusMinus[i, 2] = min(steph1, steph2)
                x1[i] = x1[i] + FnPlusMinus[i, 2]
                x2[i] = x2[i] - FnPlusMinus[i, 2]
            # Forward diff.
            elif BdsCheck[i] == 1:
                FnPlusMinus[i, 2] = steph1
                x1[i] = x1[i] + FnPlusMinus[i, 2]
            # Backward diff.
            else:
                FnPlusMinus[i, 2] = steph2
                x2[i] = x2[i] - FnPlusMinus[i, 2]
            x1_solution = self.create_new_solution(tuple(x1), problem)
            if BdsCheck[i] != -1:
                problem.simulate_up_to([x1_solution], r)
                fn1 = -1 * problem.minmax[0] * x1_solution.objectives_mean
                # First column is f(x+h,y).
                FnPlusMinus[i, 0] = fn1
            x2_solution = self.create_new_solution(tuple(x2), problem)
            if BdsCheck[i] != 1:
                problem.simulate_up_to([x2_solution], r)
                fn2 = -1 * problem.minmax[0] * x2_solution.objectives_mean
                # Second column is f(x-h,y).
                FnPlusMinus[i, 1] = fn2
            # Calculate gradient.
            if BdsCheck[i] == 0:
                grad[i] = (fn1 - fn2) / (2 * FnPlusMinus[i, 2])
            elif BdsCheck[i] == 1:
                grad[i] = (fn1 - fn) / FnPlusMinus[i, 2]
            elif BdsCheck[i] == -1:
                grad[i] = (fn - fn2) / FnPlusMinus[i, 2]

        return grad
    
    def line_search(self, problem, expended_budget, r, grad, cur_sol, alpha_0, d, alpha, beta):
        """
        A backtracking line-search along [x, x + rd] assuming all solution on the line are feasible. 

        Arguments
        ---------
        problem : Problem object
            simulation-optimization problem to solve
        expended_budget: int
            current expended budget
        r : int
            number of replications taken at each solution
        grad : ndarray
            objective gradient of cur_sol
        cur_sol : Solution object
            current solution
        alpha_0 : float
            maximum step size allowed
        d : ndarray
            search direction
        alpha: float
            tolerance for sufficient decrease condition
        beta: float
            step size reduction factor

        Returns
        -------
        x_new_solution : Solution
            a solution obtained by line search
        step_size : float
            computed step size
        expended_budget : int
            updated expended budget
        """
        x = cur_sol.x
        fx = -1 * problem.minmax[0] * cur_sol.objectives_mean
        step_size = alpha_0
        count = 0
        x_new_solution = self.create_new_solution(tuple(x), problem)
        while True:
            if expended_budget > problem.factors["budget"]:
                break
            x_new = x + step_size * d
            # Create a solution object for x_new.
            x_new_solution = self.create_new_solution(tuple(x_new), problem)
            # Use r simulated observations to estimate the objective value.
            problem.simulate(x_new_solution, r)
            expended_budget += r
            # Check the sufficient decrease condition.
            f_new = -1 * problem.minmax[0] * x_new_solution.objectives_mean
            if f_new < fx + alpha * step_size * np.dot(grad, d):
                break
            step_size *= beta
            count += 1
        return x_new_solution, step_size, expended_budget

    def find_feasible_initial(self, problem, Ae, Ai, be, bi, tol):
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
        upper_bound = np.array(problem.upper_bounds)
        lower_bound = np.array(problem.lower_bounds)

        # Define decision variables.
        x = cp.Variable(problem.dim)

        # Define constraints.
        constraints = []

        if (Ae is not None) and (be is not None):
            constraints.append(Ae @ x == be.ravel())
        if (Ai is not None) and (bi is not None):
            constraints.append(Ai @ x <= bi.ravel())

        # Removing redundant bound constraints.
        ub_inf_idx = np.where(~np.isinf(upper_bound))[0]
        if len(ub_inf_idx) > 0:
            for i in ub_inf_idx:
                constraints.append(x[i] <= upper_bound[i])
        lb_inf_idx = np.where(~np.isinf(lower_bound))
        if len(lb_inf_idx) > 0:
            for i in lb_inf_idx:
                constraints.append(x[i] >= lower_bound[i])

        # Define objective function.
        obj = cp.Minimize(0)
        
        # Create problem.
        model = cp.Problem(obj, constraints)

        # Solve problem.
        model.solve(solver = cp.SCIPY)

        # Check for optimality.
        if model.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] :
            raise ValueError("Could not find feasible x0")
        x0 = x.value
        if not self._feasible(x0, problem, tol):
            raise ValueError("Could not find feasible x0")

        return x0

    def _feasible(self, x, problem, tol):
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
        lb = np.asarray(problem.lower_bounds)
        ub = np.asarray(problem.upper_bounds)
        res = True
        if (problem.Ci is not None) and (problem.di is not None):
            res = res & np.all(problem.Ci @ x <= problem.di + tol)
        if (problem.Ce is not None) and (problem.de is not None):
            res = res & (np.allclose(np.dot(problem.Ce, x), problem.de, rtol=0, atol=tol))
        return res & (np.all(x >= lb)) & (np.all(x <= ub))