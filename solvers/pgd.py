"""
Summary
-------
PGD
projected gradient descent for problems with linear constraints Ce@x <= de, Ci@x <= di
"""

from base import Solver
import numpy as np
import cvxpy as cp
import warnings
warnings.filterwarnings("ignore")


class PGD(Solver):
    """
    Description.

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
    def __init__(self, name="PGD", fixed_factors={}):
        self.name = name
        self.objective_type = "single"
        self.constraint_type = "box"
        self.variable_type = "deterministic"
        self.gradient_needed = False
        self.specifications = {
            "crn_across_solns": {
                "description": "Use CRN across solutions?",
                "datatype": bool,
                "default": True
            },
            "r": {
                "description": "number of replications taken at each solution",
                "datatype": int,
                "default": 10
            },
            "alpha": {
                "description": "Tolerance for sufficient decrease condition.",
                "datatype": float,
                "default": 0.2
            },
            "beta": {
                "description": "Step size reduction factor in line search.",
                "datatype": float,
                "default": 0.9
            },
            "alpha_0": {
                "description": "Maximum step size.",
                "datatype": float,
                "default": 10.0
            },
            "lambda": {
                "description": "magnifying factor for n_r inside the finite difference function",
                "datatype": int,
                "default": 2
            },
        }
        self.check_factor_list = {
            "crn_across_solns": self.check_crn_across_solns,
            "r": self.check_r,
            "alpha": self.check_alpha,
            "beta": self.check_beta,
            "alpha_0": self.check_alpha_0,
            "lambda": self.check_lambda
        }
        super().__init__(fixed_factors)

    def check_r(self):
        return self.factors["r"] > 0

    def check_alpha(self):
        return self.factors["alpha"] > 0

    def check_beta(self):
        return self.factors["beta"] > 0 & self.factors["beta"] < 1

    def check_alpha_0(self):
        return self.factors["alpha_0"] > 0
    
    def check_lambda(self):
        return self.factors["lambda"] > 0

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
        alpha_0 = self.factors["alpha_0"]

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

        # Start with the initial solution.
        new_solution = self.create_new_solution(problem.factors["initial_solution"], problem)
        new_x = new_solution.x
        problem.simulate(new_solution, r)
        expended_budget += r
        best_solution = new_solution
        recommended_solns.append(new_solution)
        intermediate_budgets.append(expended_budget)

        # If the initial solution is not feasible, generate one using phase one simplex.
        # if not self._feasible(new_x, problem):
        if True:
            new_x = self.find_feasible_initial(problem, Ce, Ci, de, di)
            new_solution = self.create_new_solution(tuple(new_x), problem)

        while expended_budget < problem.factors["budget"]:
            new_x = new_solution.x
            # Check variable bounds.
            forward = [int(new_x[i] == lower_bound[i]) for i in range(problem.dim)]
            backward = [int(new_x[i] == upper_bound[i]) for i in range(problem.dim)]
            # BdsCheck: 1 stands for forward, -1 stands for backward, 0 means central diff.
            BdsCheck = np.subtract(forward, backward)

            if problem.gradient_available:
                # Use IPA gradient if available.
                grad = -1 * problem.minmax[0] * new_solution.objectives_gradients_mean[0]
            else:
                # Use finite difference to estimate gradient if IPA gradient is not available.
                grad = self.finite_diff(new_solution, BdsCheck, problem, r)
                expended_budget += (2 * problem.dim - np.sum(BdsCheck != 0)) * r
                # A while loop to prevent zero gradient.
                while np.all((grad == 0)):
                    if expended_budget > problem.factors["budget"]:
                        break
                    grad = self.finite_diff(new_solution, BdsCheck, problem, r)
                    expended_budget += (2 * problem.dim - np.sum(BdsCheck != 0)) * r
                    # Update r after each iteration.
                    r = int(self.factors["lambda"] * r)

            # Line search to determine a step_size.
            step_size, expended_budget = self.line_search(problem, expended_budget, r, grad, new_solution, alpha_0, -grad, alpha, beta)
            # Get a temp solution.
            temp_x = new_x - step_size * grad

            if self._feasible(temp_x, problem):
                new_solution = self.create_new_solution(tuple(temp_x), problem)
            else:
                # If not feasible, project temp_x back to the feasible set.
                proj_x = self.project_grad(problem, temp_x, Ce, Ci, de, di)
                new_solution = self.create_new_solution(tuple(proj_x), problem)

            # Use r simulated observations to estimate the objective value.
            problem.simulate(new_solution, r)
            expended_budget += r

            # Append new solution.
            if (problem.minmax[0] * new_solution.objectives_mean > problem.minmax[0] * best_solution.objectives_mean):
                best_solution = new_solution
                recommended_solns.append(new_solution)
                intermediate_budgets.append(expended_budget)

        return recommended_solns, intermediate_budgets


    # TODO: check the input stepsizes
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
        """
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


    def _feasible(self, x, problem):
        """
        Check whether a solution vector x is feasible to the problem.
        
        Arguments
        ---------
        x : ndarray
            a solution vector
        problem : Problem object
            simulation-optimization problem to solve
        """
        return (np.dot(problem.Ci, x) <= problem.di) & \
            (np.allclose(np.dot(problem.Ce, x), problem.de, rtol=0, atol=1e-05)) & \
            (np.all(x >= problem.lower_bounds)) & (np.all(x <= problem.upper_bounds))
    
    def project_grad(self, problem, x, Ae, Ai, be, bi):
        """
        Project the vector x onto the hyperplane H: Ax = b by solving a quadratic projection problem:

        min d^Td
        s.t. Ae(x + d) = be
             Ai(x + d) <= bi
             (x + d) >= lb
             (x + d) <= ub
        
        Arguments
        ---------
        problem : Problem object
            simulation-optimization problem to solve
        x : ndarray
            vector to be projected
        Ae: ndarray
            equality constraint coefficient matrix
        be: ndarray
            equality constraint coefficient vector
        Ai: ndarray
            inequality constraint coefficient matrix
        bi: ndarray
            inequality constraint coefficient vector 
        Returns
        -------
        x_new : ndarray
            the projected vector
        """
        # Define variables.
        d = cp.Variable(problem.dim)

        # Define objective.
        obj = cp.Minimize(cp.quad_form(d, np.identity(problem.dim)))

        # Define constraints.
        constraints = [Ae @ (x + d) == be,
                        Ai @ (x + d) <= bi,
                        (x + d) >= problem.lower_bounds,
                        (x + d) <= problem.upper_bounds]

        # Form and solve problem.
        prob = cp.Problem(obj, constraints)
        prob.solve()

        # Get the projected vector.
        x_new = x + d.value

        return x_new

    def line_search(self, problem, expended_budget, r, grad, cur_sol, alpha_0, d, alpha, beta):
        """
        A backtracking line-search along [x, x + rd] assuming all solution on the line are feasible. #TODO : change step size

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
        step_size : float
            computed step size
        expended_budget : int
            updated expended budget
        """
        x = cur_sol.x
        fx = -1 * problem.minmax[0] * cur_sol.objectives_mean
        step_size = alpha_0
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
        return step_size, expended_budget

    # TODO: test this function and use cvxpy instead.
    def find_feasible_initial(self, problem, Ae, Ai, be, bi):
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
        '''
        # Define decision variables.
        x = cp.Variable(problem.dim)

        # Define objective function.
        obj = cp.Minimize(0)

        # Define constraints.
        constraints = [Ae @ x == be,
                       Ai @ x <= bi,
                        x >= problem.lower_bounds,
                        x <= problem.upper_bounds]
        
        # Create problem.
        model = cp.Problem(obj, constraints)

        # Solve problem.
        model.solve()
        print(model.constraints)

        # Check for optimality.
        if model.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] :
            raise ValueError("Simplex phase 1: could not find feasible x0")
        x0 = x.value
        print('Initial feasible pt', x0)
        if not self._feasible(x.value, problem):
            raise ValueError("Simplex phase 1: could not find feasible x0")

        return x0
