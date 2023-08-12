"""
Summary
-------
Nelder-Mead: An algorithm that maintains a simplex of points that moves around the feasible
region according to certain geometric operations: reflection, expansion,
contraction, and shrinking.
A detailed description of the solver can be found 
`here <https://simopt.readthedocs.io/en/latest/neldmd.html>`_.
"""
import numpy as np
import cvxpy as cp
import warnings
warnings.filterwarnings("ignore")

from ..base import Solution, Solver

class NelderMead(Solver):
    """The Nelder-Mead algorithm, which maintains a simplex of points that moves around the feasible
    region according to certain geometric operations: reflection, expansion,
    contraction, and shrinking.

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
    rng_list : list of mrg32k3a.mrg32k3a.MRG32k3a objects
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
    def __init__(self, name="NELDMD", fixed_factors=None):
        if fixed_factors is None:
            fixed_factors = {}
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
                "description": "reflection coefficient > 0",
                "datatype": float,
                "default": 1.
            },
            "gammap": {
                "description": "expansion coefficient > 1",
                "datatype": float,
                "default": 2.
            },
            "betap": {
                "description": "contraction coefficient > 0, < 1",
                "datatype": float,
                "default": 0.5
            },
            "delta": {
                "description": "shrink factor > 0, < 1",
                "datatype": float,
                "default": 0.5
            },
            "sensitivity": {
                "description": "shrinking scale for bounds",
                "datatype": float,
                "default": 10**(-7)
            },
            "initial_spread": {
                "description": "1 - fraction of the distance towards the centroid for shrinking the initial points",
                "datatype": float,
                "default": 1/2
            },
            "tol": {
                "description": "floating point tolerance for checking tightness of constraints",
                "datatype": float,
                "default": 1e-7
            },
        }
        self.check_factor_list = {
            "crn_across_solns": self.check_crn_across_solns,
            "r": self.check_r,
            "alpha": self.check_alpha,
            "gammap": self.check_gammap,
            "betap": self.check_betap,
            "delta": self.check_delta,
            "sensitivity": self.check_sensitivity,
            "initial_spread": self.check_initial_spread,
            "tol": self.check_tol
        }
        super().__init__(fixed_factors)

    def check_r(self):
        return self.factors["r"] > 0

    def check_alpha(self):
        return self.factors["alpha"] > 0

    def check_gammap(self):
        return self.factors["gammap"] > 1

    def check_betap(self):
        return (self.factors["betap"] > 0) & (self.factors["betap"] < 1)

    def check_delta(self):
        return (self.factors["delta"] > 0) & (self.factors["delta"] < 1)

    def check_sensitivity(self):
        return self.factors["sensitivity"] > 0

    def check_initial_spread(self):
        return self.factors["initial_spread"] > 0
    
    def check_tol(self):
        return self.factors["tol"] > 0

    def solve(self, problem):
        """
        Run a single macroreplication of a solver on a problem.

        Arguments
        ---------
        problem : Problem object
            simulation-optimization problem to solve

        Returns
        -------
        recommended_solns : list of Solution objects
            list of solutions recommended throughout the budget
        intermediate_budgets : list of ints
            list of intermediate budgets when recommended solutions changes
        """
        # Designate random number generator for random sampling.
        get_rand_soln_rng = self.rng_list[1]
        n_pts = problem.dim + 1
        # Check for sufficiently large budget.
        if problem.factors["budget"] < self.factors["r"] * n_pts:
            print('Budget is too small for a good quality run of Nelder-Mead.')
            return
        
        # Initialize lists to track budget and best solutions.
        intermediate_budgets = []
        recommended_solns = []

        # Track overall budget spent.
        budget_spent = 0
        r = self.factors["r"]  # For increasing replications.
        tol = self.factors["tol"]

        
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

        # Checker for whether the problem is unconstrained.
        unconstr_flag = (Ce is None) & (Ci is None) & (di is None) & (de is None) & (all(np.isinf(lower_bound))) & (all(np.isinf(upper_bound)))

        # Initial dim + 1 points.
        sol = []
        new_solution = self.create_new_solution(problem.factors["initial_solution"], problem)
        new_x = new_solution.x

        if (not unconstr_flag) & (not self._feasible(new_x, problem, tol)):
            new_x = self.find_feasible_initial(problem, Ce, Ci, de, di, tol)
            new_solution = self.create_new_solution(tuple(new_x), problem)

        sol.append(new_solution)

        if (problem.name == "CASCADE-1") or (problem.name == "CASCADETIME-1") or (problem.name == "CASCADETIME-2") :
            xs = problem.get_multiple_random_solution(get_rand_soln_rng, n_pts - 1)
            for rand_x in xs:
                sol.append(self.create_new_solution(rand_x, problem))
        else:
            for i in range(1, n_pts):
                rand_x = problem.get_random_solution(get_rand_soln_rng)
                sol.append(self.create_new_solution(rand_x, problem))

        # Record initial solution data.
        intermediate_budgets.append(0)
        recommended_solns.append(sol[0])
        
        # Restrict starting shape by shrinking nodes other than the initial solution towards the centroid
        p_cent = tuple(np.mean(tuple([s.x for s in sol]), axis=0))
        small_sols = [sol[0]]
        for i in range(1, len(sol)):
            small_sol = np.array(sol[i].x) + (np.array(p_cent) - np.array(sol[i].x)) * self.factors["initial_spread"]
            small_sols.append(self.create_new_solution(tuple(small_sol), problem))
        
        sol = small_sols

        # Start Solving.
        # Evaluate solutions in initial structure.
        for solution in sol:
            problem.simulate(solution, self.factors["r"])
            budget_spent += self.factors["r"]

        # Sort solutions by obj function estimate.
        sort_sol = self.sort_and_end_update(problem, sol)

        # Maximization problem is converted to minimization by using minmax.
        while budget_spent <= problem.factors["budget"]:
            # Reflect worst and update sort_sol.
            p_high = sort_sol[-1]  # Current worst point.
            p_cent = tuple(np.mean(tuple([s.x for s in sort_sol[0:-1]]), axis=0))  # Centroid for other pts.
            orig_pt = p_high  # Save the original point.
            p_refl = tuple(map(lambda i, j: i - j, tuple((1 + self.factors["alpha"]) * i for i in p_cent),
                               tuple(self.factors["alpha"] * i for i in p_high.x)))  # Reflection.
            p_refl_copy = p_refl
            if not self._feasible(p_refl, problem, tol):
                p_refl = self.check_const(p_refl, orig_pt.x,  C, d, tol)

            # # Shrink towards best if out of bounds.
            # if not np.allclose(p_refl, p_refl_copy):
            #     print('here')
            #     while not np.allclose(p_refl, p_refl_copy):
            #         p_low = sort_sol[0]
            #         for i in range(1, len(sort_sol)):
            #             p_new2 = p_low
            #             p_new = tuple(map(lambda i, j: i + j, tuple(self.factors["delta"] * i for i in sort_sol[i].x),
            #                               tuple((1 - self.factors["delta"]) * i for i in p_low.x)))
            #             if not self._feasible(p_new, problem, tol):
            #                 p_new = self.check_const(p_new, p_new2.x, C, d, tol)
            #             p_new = Solution(tuple(p_new), problem)
            #             p_new.attach_rngs(rng_list=self.solution_progenitor_rngs, copy=True)
            #             problem.simulate(p_new, r)
            #             budget_spent += r

            #             # Update sort_sol.
            #             sort_sol[i] = p_new  # p_new replaces pi.

            #         # Sort & end updating.
            #         sort_sol = self.sort_and_end_update(problem, sort_sol)

            #         p_high = sort_sol[-1]  # Current worst point.
            #         p_cent = tuple(np.mean(tuple([s.x for s in sort_sol[0:-1]]), axis=0))  # Centroid for other pts.
            #         orig_pt = p_high  # Save the original point.
            #         p_refl = tuple(map(lambda i, j: i - j, tuple((1 + self.factors["alpha"]) * i for i in p_cent),
            #                            tuple(self.factors["alpha"] * i for i in p_high.x)))  # Reflection.
            #         p_refl_copy = np.array(p_refl)
            #         if not self._feasible(p_refl, problem, tol):
            #             p_refl = self.check_const(p_refl, orig_pt.x,  C, d, tol)

            # Evaluate reflected point.
            p_refl = Solution(tuple(p_refl), problem)
            p_refl.attach_rngs(rng_list=self.solution_progenitor_rngs, copy=True)
            problem.simulate(p_refl, r)
            budget_spent += r
            refl_fn_val = tuple([-1 * i for i in problem.minmax]) * p_refl.objectives_mean

            # Track best, worst, and second worst points.
            p_low = sort_sol[0]  # Current best pt.
            fn_low = tuple([-1 * i for i in problem.minmax]) * sort_sol[0].objectives_mean
            fn_sec = tuple([-1 * i for i in problem.minmax]) * sort_sol[-2].objectives_mean  # Current 2nd worst obj fn.
            fn_high = tuple([-1 * i for i in problem.minmax]) * sort_sol[-1].objectives_mean  # Worst obj fn from unreflected structure.

            # Check if accept reflection.
            if fn_low <= refl_fn_val and refl_fn_val <= fn_sec:
                sort_sol[-1] = p_refl  # The new point replaces the previous worst.

                # Sort & end updating.
                sort_sol = self.sort_and_end_update(problem, sort_sol)

                # Best solution remains the same, so no reporting.

            # Check if accept expansion (of reflection in the same direction).
            elif refl_fn_val < fn_low:
                p_exp2 = p_refl
                p_exp = tuple(map(lambda i, j: i + j, tuple(self.factors["gammap"] * i for i in p_refl.x),
                                  tuple((1 - self.factors["gammap"]) * i for i in p_cent)))
                if not self._feasible(p_exp, problem, tol):
                    p_exp = self.check_const(p_exp, p_exp2.x, C, d, tol)

                # Evaluate expansion point.
                p_exp = Solution(tuple(p_exp), problem)
                p_exp.attach_rngs(rng_list=self.solution_progenitor_rngs, copy=True)
                problem.simulate(p_exp, r)
                budget_spent += r
                exp_fn_val = tuple([-1 * i for i in problem.minmax]) * p_exp.objectives_mean

                # Check if expansion point is an improvement relative to simplex.
                if exp_fn_val < fn_low:
                    sort_sol[-1] = p_exp  # p_exp replaces p_high.

                    # Sort & end updating.
                    sort_sol = self.sort_and_end_update(problem, sort_sol)
                    # print('pexp', p_exp.x)
                    # print('pexp', p_exp.objectives_mean)
                    # Record data from expansion point (new best).
                    if budget_spent <= problem.factors["budget"]:
                        # print('here')
                        intermediate_budgets.append(budget_spent)
                        recommended_solns.append(p_exp)
                else:
                    sort_sol[-1] = p_refl  # p_refl replaces p_high.

                    # Sort & end updating.
                    sort_sol = self.sort_and_end_update(problem, sort_sol)

                    # Record data from expansion point (new best).
                    if budget_spent <= problem.factors["budget"]:
                        intermediate_budgets.append(budget_spent)
                        recommended_solns.append(p_refl)

            # Check if accept contraction or shrink.
            elif refl_fn_val > fn_sec:
                if refl_fn_val <= fn_high:
                    p_high = p_refl  # p_refl replaces p_high.
                    fn_high = refl_fn_val  # Replace fn_high.

                # Attempt contraction or shrinking.
                p_cont2 = p_high
                p_cont = tuple(map(lambda i, j: i + j, tuple(self.factors["betap"] * i for i in p_high.x),
                                   tuple((1 - self.factors["betap"]) * i for i in p_cent)))
                if not self._feasible(p_cont, problem, tol):
                    p_cont = self.check_const(p_cont, p_cont2.x, C, d, tol)

                # Evaluate contraction point.
                p_cont = Solution(tuple(p_cont), problem)
                p_cont.attach_rngs(rng_list=self.solution_progenitor_rngs, copy=True)
                problem.simulate(p_cont, r)
                budget_spent += r
                cont_fn_val = tuple([-1 * i for i in problem.minmax]) * p_cont.objectives_mean

                # Accept contraction.
                if cont_fn_val <= fn_high:
                    sort_sol[-1] = p_cont  # p_cont replaces p_high.

                    # Sort & end updating.
                    sort_sol = self.sort_and_end_update(problem, sort_sol)

                    # Check if contraction point is new best.
                    if cont_fn_val < fn_low:
                        # Record data from contraction point (new best).
                        if budget_spent <= problem.factors["budget"]:
                            intermediate_budgets.append(budget_spent)
                            recommended_solns.append(p_cont)
                else:  # Contraction fails -> simplex shrinks by delta with p_low fixed.
                    sort_sol[-1] = p_high  # Replaced by p_refl.

                    # Check for new best.
                    new_best = 0

                    for i in range(1, len(sort_sol)):
                        p_new2 = p_low
                        p_new = tuple(map(lambda i, j: i + j, tuple(self.factors["delta"] * i for i in sort_sol[i].x),
                                          tuple((1 - self.factors["delta"]) * i for i in p_low.x)))
                        if not self._feasible(p_new, problem, tol):
                            p_new = self.check_const(p_new, p_new2.x, C, d, tol)
                        p_new = Solution(tuple(p_new), problem)
                        p_new.attach_rngs(rng_list=self.solution_progenitor_rngs, copy=True)
                        problem.simulate(p_new, r)
                        budget_spent += r
                        new_fn_val = tuple([-1 * i for i in problem.minmax]) * p_new.objectives_mean

                        # Check for new best.
                        if new_fn_val <= fn_low:
                            new_best = 1

                        # Update sort_sol.
                        sort_sol[i] = p_new  # p_new replaces pi.

                    # Sort & end updating.
                    sort_sol = self.sort_and_end_update(problem, sort_sol)

                    # Record data if there is a new best solution in the contraction.
                    if new_best == 1 and budget_spent <= problem.factors["budget"]:
                        intermediate_budgets.append(budget_spent)
                        recommended_solns.append(sort_sol[0])
        return recommended_solns, intermediate_budgets

    # HELPER FUNCTIONS

    def sort_and_end_update(self, problem, sol):
        sort_sol = sorted(sol, key=lambda s: tuple([-1 * i for i in problem.minmax]) * s.objectives_mean)
        return sort_sol
    
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

    def check_const(self, candidate_x, cur_x, C, d, tol):
        # handle the box and linear constraints using ratio test
        dir = np.array(candidate_x) - np.array(cur_x)

        # Get all indices not in the active set such that Ai^Td>0
        r_idx = list((C @ dir > 0).nonzero()[0])

        # Compute the ratio test
        ra = d[r_idx,:].flatten() - C[r_idx, :] @ cur_x
        ra_d = C[r_idx, :] @ dir
        # Initialize maximum step size.
        s_star = np.inf
        # Perform ratio test.
        for i in range(len(ra)):
            if ra_d[i] - tol > 0:
                s = ra[i]/ra_d[i]
                if s < s_star:
                    s_star = s

        new_x = cur_x + min(1, s_star) * dir

        return new_x