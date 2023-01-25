"""
Summary
-------
PGD
projected gradient descent for problems with linear constraints
"""

from base import Solver
from numpy.linalg import norm
import numpy as np
from scipy.optimize import linprog
import warnings
warnings.filterwarnings("ignore")


class PGD2(Solver):
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
                "default": 30
            },
            "beta_1": {
                "description": "Exponential decay of the rate for the first moment estimates.",
                "datatype": int,
                "default": 0.9
            },
            "beta_2": {
                "description": "Exponential decay rate for the second-moment estimates.",
                "datatype": int,
                "default": 0.999
            },
            "theta": {
                "description": "Constant in the Armijo condition.",
                "datatype": int,
                "default": 0.2
            },
            "gamma": {
                "description": "Constant for shrinking the step size.",
                "datatype": int,
                "default": 0.8
            },
            "alpha_0": {
                "description": "Initial step size.",
                "datatype": int,
                "default": 0.1
            },
            "s": {
                "description": "Initial step size after projection.",
                "datatype": int,
                "default": 0.1
            },
            "epsilon": {
                "description": "A small value to prevent zero-division.",
                "datatype": int,
                "default": 10**(-8)
            },
            "epsilon_f": {
                "description": "Additive constant in the Armijo condition.",
                "datatype": int,
                "default": 1  # In the paper, this value is estimated for every epoch but a value > 0 is justified in practice.
            },
            "sensitivity": {
                "description": "Shrinking scale for variable bounds.",
                "datatype": float,
                "default": 10**(-7)
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
            "beta_1": self.check_beta_1,
            "beta_2": self.check_beta_2,
            "theta": self.check_theta,
            "gamma": self.check_gamma,
            "alpha_0": self.check_alpha_0,
            "s": self.check_s,
            "epsilon": self.check_epsilon,
            "epsilon_f": self.check_epsilon_f,
            "sensitivity": self.check_sensitivity,
            "lambda": self.check_lambda
        }
        super().__init__(fixed_factors)

    def check_r(self):
        return self.factors["r"] > 0

    def check_beta_1(self):
        return self.factors["beta_1"] > 0 & self.factors["beta_1"] < 1

    def check_beta_2(self):
        return self.factors["beta_2"] > 0 & self.factors["beta_2"] < 1

    def check_theta(self):
        return self.factors["theta"] > 0 & self.factors["theta"] < 1

    def check_gamma(self):
        return self.factors["gamma"] > 0 & self.factors["gamma"] < 1

    def check_alpha_0(self):
        return self.factors["alpha_0"] > 0

    def check_s(self):
        return self.factors["s"] > 0 & self.factors["s"] < 1
    
    def check_epsilon(self):
        return self.factors["epsilon"] > 0

    def check_epsilon_f(self):
        return self.factors["epsilon_f"] > 0

    def check_sensitivity(self):
        return self.factors["sensitivity"] > 0
    
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
        theta = self.factors["theta"]
        gamma = self.factors["gamma"]
        alpha_0 = self.factors["alpha_0"]
        epsilon_f = self.factors["epsilon_f"]
        s = self.factors["s"]

        # Shrink the bounds to prevent floating errors.
        # lower_bound = np.array(problem.lower_bounds) + np.array((self.factors['sensitivity'],) * problem.dim)
        # upper_bound = np.array(problem.upper_bounds) - np.array((self.factors['sensitivity'],) * problem.dim)
        lower_bound = np.array(problem.lower_bounds)
        upper_bound = np.array(problem.upper_bounds)

        #TODO: check coefficient matrix for full rank

        # Cix <= di
        # Cex = de
        Ci = problem.Ci
        di = problem.di
        Ce = problem.Ce
        de = problem.de

        atol = 1e-05

        neq = len(de) # number of equality constraints

        # Form a constraint coefficient matrix where all the equality constraints are put on top and
        # all the bound constraints in the bottom.

        C = np.vstack((Ce, Ci, np.identity(upper_bound.shape[0]), -np.identity(lower_bound.shape[0])))
        # Form a constraint coefficient vector
        d = np.vstack((de, di, upper_bound[np.newaxis].T, -lower_bound[np.newaxis].T))

        # Active constraint index vector
        acidx = np.ndarray((0, 1))

        # Initialize stepsize.
        alpha = alpha_0

        # Start with the initial solution.
        new_solution = self.create_new_solution(problem.factors["initial_solution"], problem)
        new_x = new_solution.x
        # If the initial solution is not feasible, generate one using phase one simplex.
        if not self._feasible(np.array(problem.factors["initial_solution"]), problem):
            new_x = self.find_feasible_initial(problem, C, d)
            new_solution = self.create_new_solution(tuple(new_x), problem)

        # Initialize active set
        # equality constraints always in active set.
        # save the indeces into the full matrix.
        # add any inequality constraints to the
        # active working set that are equal to the
        # constraint value, i.e.
        # x>=0 is active if x=0.
        # TODO: debug add_active_constraint
        cx = np.dot(C, new_x)
        # print('cx', cx)
        # print("C", C)
        # print('d', d)
        for j in range(cx.shape[0]):
            if j < neq or np.isclose(cx[j], d[j], rtol=0, atol=atol):
                # print('j', j)
                acidx = self.add_active_constraint(j, acidx)
        # print('acidx - initial', acidx)
        problem.simulate(new_solution, r)
        expended_budget += r
        best_solution = new_solution
        recommended_solns.append(new_solution)
        intermediate_budgets.append(expended_budget)


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
                print('IPA grad', grad)
            else:
                # Use finite difference to estimate gradient if IPA gradient is not available.
                grad = self.finite_diff(new_solution, BdsCheck, problem, alpha, r)
                expended_budget += (2 * problem.dim - np.sum(BdsCheck != 0)) * r
                # A while loop to prevent zero gradient
                while np.all((grad == 0)):
                    if expended_budget > problem.factors["budget"]:
                        break
                    grad = self.finite_diff(new_solution, BdsCheck, problem, alpha, r)
                    expended_budget += (2 * problem.dim - np.sum(BdsCheck != 0)) * r
                    # Update r after each iteration.
                    r = int(self.factors["lambda"] * r)
   
            # Get a temp solution.
            temp_x = new_x - grad
            print('temp_x', temp_x)
            # Update the active set
            acidx = np.ndarray((0, 1))
            cx = np.dot(C, temp_x)
            for j in range(cx.shape[0]):
                # TODO : isclose or not isclose
                if j < neq or np.isclose(cx[j], d[j], rtol=0, atol=atol):
                    print('C', C[j])
                    print('d[j]', d[j])
                    acidx = self.add_active_constraint(j, acidx)
            # Project it onto the active set
            mask = np.isin(np.arange(C.shape[0]), acidx, assume_unique=True)
            # print('acidx', acidx)
            # print('temp_x', temp_x)
            # print('mask', mask)
            proj_x = self.project_grad(temp_x, C[mask], d[mask])

            # Get new direction
            dir = proj_x - new_x

            print('proj_x', proj_x)
            print('temp_x', temp_x)
            print('dir', dir)

            # Ratio test to determine the maximum step size possible
            ineq_idx = []
            cx = np.dot(C, new_solution.x)
            for j in range(cx.shape[0]):
                if not np.isclose(cx[j], d[j], rtol=0, atol=atol):
                    ineq_idx.append(j) # Find all constraints with strict inequality
            mask = np.array(ineq_idx)
            # TODO:
            # g - Gx_k 
            RHS = d[mask] - np.matmul(C[mask], temp_x.T)[:,np.newaxis]
            # Gd_k
            LHS = np.matmul(C[mask], dir[np.newaxis].T)
            print("Cmask shape", C[mask].shape)
            # print('dir shape', (dir[np.newaxis].T).shape)
            print("dmask shape", d[mask].shape)
            print("RHS", RHS)
            print("LHS", LHS)
            # Ratio Test
            s = 1
            min_ratio = np.inf
            for i in range(len(mask)):
                try:
                    ratio = LHS[i] / RHS[i]
                except:
                    ratio = LHS[i] / RHS[i]
                print('LHS i', LHS[i])
                print("RHS i", RHS[i])
                print('ratio', ratio)
                if (ratio > 0) & (ratio < min_ratio):
                    min_ratio = ratio
            if min_ratio != np.inf:
                s = min_ratio
        
            # Calculate the candidate solution.
            print('max step size: ', s)
            print('dir', dir)
            print('new_x', new_x)
            candidate_x = new_x + s * dir
            candidate_solution = self.create_new_solution(tuple(candidate_x), problem)

            # Use r simulated observations to estimate the objective value.
            problem.simulate(candidate_solution, r)
            expended_budget += r

            # Use Armijo condition for line search
            while not (-1 * problem.minmax[0] * candidate_solution.objectives_mean) <= (-1 * problem.minmax[0] * new_solution.objectives_mean - alpha * theta * norm(dir)**2 + 2 * epsilon_f):
                if expended_budget > problem.factors["budget"]:
                    break
                # Unsuccessful step.
                s = gamma * s
                # Calculate the candidate solution.
                candidate_x = new_x + s * dir
                candidate_solution = self.create_new_solution(tuple(candidate_x), problem)
                # Use r simulated observations to estimate the objective value.
                problem.simulate(candidate_solution, r)
                expended_budget += r
            print('final step size', s)
            # Successful step.
            new_solution = candidate_solution

            # Updating active set.
            acidx = np.ndarray((0, 1))
            cx = np.dot(C, new_solution.x)
            for j in range(cx.shape[0]):
                if j < neq or np.isclose(cx[j], d[j], rtol=0, atol=atol):
                    acidx = self.add_active_constraint(j, acidx)

            print('new x', new_solution.x)
            print("x sum:", np.sum(new_solution.x))
            # Append new solution.
            if (problem.minmax[0] * new_solution.objectives_mean > problem.minmax[0] * best_solution.objectives_mean):
                print('objective', problem.minmax[0] * new_solution.objectives_mean)
                best_solution = new_solution
                recommended_solns.append(new_solution)
                intermediate_budgets.append(expended_budget)

        return recommended_solns, intermediate_budgets

    # Finite difference for approximating gradients.
    def finite_diff(self, new_solution, BdsCheck, problem, stepsize, r):
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
        return (np.dot(problem.Ci, x) <= problem.di) & \
            (np.allclose(np.dot(problem.Ce, x), problem.de, rtol=0, atol=1e-05)) & \
            (np.all(x >= problem.lower_bounds)) & (np.all(x <= problem.upper_bounds))


    # Find the initial feasible solution
    # (if not user-provided) by solving the
    # equality-constrained linear program
    def find_feasible_initial(self, problem, C, d):
        lower_bound = np.array(problem.lower_bounds)
        upper_bound = np.array(problem.upper_bounds)
        neq = len(problem.Ci)
        nbds = len(lower_bound) + len(upper_bound)
        # trivial solution if only bounds constraints
        if nbds == C.shape[0]:
            if len(lower_bound) > 0:
                x0 = lower_bound
            else:
                x0 = upper_bound
            if not self._feasible(x0, problem):
                raise ValueError('ActiveSet: cannot find feasible x0')
            return x0
        # remove bounds constraints
        if nbds:
            Ce = self.C[:-nbds]
            de = self.d[:-nbds]
        else:
            Ce = C
            de = d
        # add slack variables to inequality constraints
        s = np.ones(shape=(Ce.shape[0],))
        Ce = np.hstack((Ce, np.diag(s)[:, neq:]))
        # initialize objective constraint
        co = np.zeros(shape=(Ce.shape[1],))
        # add artificial variables to equality constraints
        s = np.ones(shape=(Ce.shape[0],))
        Ce = np.hstack((Ce, np.diag(s)[:, :neq]))
        co = np.hstack((co, np.ones(shape=(neq,))))
        # add artificial variables to inequality constraints
        # with a negative target
        for i in range(neq, Ce.shape[0]):
            if de[i] < 0:
                col = np.zeros(shape=(Ce.shape[0], 1))
                col[i] = 1
                Ce = np.hstack((Ce, col))
                co = np.hstack((co, [1]))
        # convert bounds array to list
        cl = lower_bound.tolist()
        cu = upper_bound.tolist()
        # create a list of bounds 2-tuples
        bds = []
        for i in range(Ce.shape[1]):
            if i < problem.dim:
                if len(cl) and len(cu):
                    bd = (cl[i], cu[i])
                elif len(cl):
                    bd = (cl[i], np.Inf)
                elif len(cu):
                    bd = (0, cu[i])
                else:
                    bd = (0, np.Inf)
            else:
                bd = (0, np.Inf)
            bds.append(bd)
        # solve the linear programs
        res = linprog(co, A_eq=Ce, b_eq=de, bounds=bds, method='interior-point')
        if not res.success:
            raise ValueError('ActiveSet: could not find feasible x0')
        x0 = res.x[:C.shape[1]].reshape(C.shape[1], 1)
        if not self._feasible(x0, problem):
            raise ValueError('ActiveSet: could not find feasible x0')
        return x0

    # add constraint to the active set
    def add_active_constraint(self, cidx, acidx):
        return np.vstack((acidx, np.array([cidx])))
    
    # # remove constraint from the active set
    # def remove_active_constraint(self, cidx, acidx):
    #     # remove the constraint index
    #     acidx = np.delete(acidx, (cidx), axis=0)
    # TODO: check correctness of the function
    def project_grad(self, x, A, b):
        """
        Project the vector x onto the hyperplane H: Ax = b.
        """
        print('A', A)
        print('b', b)
        print('input x', x)
        x = x[np.newaxis].T
        lamb = np.linalg.solve(A@A.T, A@x-b)
        proj_x = x - A.T@lamb
        print('proj_x', proj_x.reshape((-1,)))
        return proj_x.reshape((-1,))