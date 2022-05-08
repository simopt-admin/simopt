"""
Summary
-------
Nelder-Mead
The algorithm maintains a simplex of points that moves around the feasible
region according to certain geometric operations: reflection, expansion,
scontraction, and shrinking.
"""
from base import Solution, Solver
import numpy as np
import warnings
warnings.filterwarnings("ignore")


class NelderMead(Solver):
    """
    REFERENCE
    Russell R. Barton, John S. Ivey, Jr., (1996)
    Nelder-Mead Simplex Modifications for Simulation
    Optimization. Management Science 42(7):954-973.

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
    def __init__(self, name="NELDMD", fixed_factors={}):
        self.name = name
        self.objective_type = "single"
        self.constraint_type = "box"
        self.variable_type = "continuous"
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
                "description": "fraction of the distance between bounds used to select initial points",
                "datatype": float,
                "default": 1 / 10
            }
        }
        self.check_factor_list = {
            "crn_across_solns": self.check_crn_across_solns,
            "r": self.check_r,
            "alpha": self.check_alpha,
            "gammap": self.check_gammap,
            "betap": self.check_betap,
            "delta": self.check_delta,
            "sensitivity": self.check_sensitivity,
            "initial_spread": self.check_initial_spread
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
        # Shrink variable bounds to avoid floating errors.
        if problem.lower_bounds is not None and problem.lower_bounds != (-np.inf,) * problem.dim:
            self.lower_bounds = tuple(map(lambda i: i + self.factors["sensitivity"], problem.lower_bounds))
        else:
            self.lower_bounds = None
        if problem.upper_bounds is not None and problem.upper_bounds != (np.inf,) * problem.dim:
            self.upper_bounds = tuple(map(lambda i: i - self.factors["sensitivity"], problem.upper_bounds))
        else:
            self.upper_bounds = None
        # Initial dim + 1 points.
        sol = []
        sol.append(self.create_new_solution(problem.factors["initial_solution"], problem))
        if self.lower_bounds is None or self.upper_bounds is None:
            for _ in range(1, n_pts):
                rand_x = problem.get_random_solution(get_rand_soln_rng)
                sol.append(self.create_new_solution(rand_x, problem))
        else:  # Restrict starting shape/location.
            for i in range(problem.dim):
                distance = (self.upper_bounds[i] - self.lower_bounds[i]) * self.factors["initial_spread"]
                new_pt = list(problem.factors["initial_solution"])
                new_pt[i] += distance
                # Try opposite direction if out of bounds.
                if new_pt[i] > self.upper_bounds[i] or new_pt[i] < self.lower_bounds[i]:
                    new_pt[i] -= 2 * distance
                # Set to bound if neither direction works.
                if new_pt[i] > self.upper_bounds[i] or new_pt[i] < self.lower_bounds[i]:
                    if problem.minmax[i] == -1:
                        new_pt[i] = self.lower_bounds[i]
                    else:
                        new_pt[i] = self.upper_bounds[i]
                sol.append(self.create_new_solution(new_pt, problem))

        # Initialize lists to track budget and best solutions.
        intermediate_budgets = []
        recommended_solns = []
        # Track overall budget spent.
        budget_spent = 0
        r = self.factors["r"]  # For increasing replications.

        # Start Solving.
        # Evaluate solutions in initial structure.
        for solution in sol:
            problem.simulate(solution, self.factors["r"])
            budget_spent += self.factors["r"]
        # Record initial solution data.
        intermediate_budgets.append(0)
        recommended_solns.append(sol[0])
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
            p_refl = self.check_const(p_refl, orig_pt.x)

            # Shrink towards best if out of bounds.
            if p_refl != p_refl_copy:
                while p_refl != p_refl_copy:
                    p_low = sort_sol[0]
                    for i in range(1, len(sort_sol)):
                        p_new2 = p_low
                        p_new = tuple(map(lambda i, j: i + j, tuple(self.factors["delta"] * i for i in sort_sol[i].x),
                                          tuple((1 - self.factors["delta"]) * i for i in p_low.x)))
                        p_new = self.check_const(p_new, p_new2.x)
                        p_new = Solution(p_new, problem)
                        p_new.attach_rngs(rng_list=self.solution_progenitor_rngs, copy=True)
                        problem.simulate(p_new, r)
                        budget_spent += r

                        # Update sort_sol.
                        sort_sol[i] = p_new  # p_new replaces pi.

                    # Sort & end updating.
                    sort_sol = self.sort_and_end_update(problem, sort_sol)

                    p_high = sort_sol[-1]  # Current worst point.
                    p_cent = tuple(np.mean(tuple([s.x for s in sort_sol[0:-1]]), axis=0))  # Centroid for other pts.
                    orig_pt = p_high  # Save the original point.
                    p_refl = tuple(map(lambda i, j: i - j, tuple((1 + self.factors["alpha"]) * i for i in p_cent),
                                       tuple(self.factors["alpha"] * i for i in p_high.x)))  # Reflection.
                    p_refl_copy = p_refl
                    p_refl = self.check_const(p_refl, orig_pt.x)

            # Evaluate reflected point.
            p_refl = Solution(p_refl, problem)
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
                p_exp = self.check_const(p_exp, p_exp2.x)

                # Evaluate expansion point.
                p_exp = Solution(p_exp, problem)
                p_exp.attach_rngs(rng_list=self.solution_progenitor_rngs, copy=True)
                problem.simulate(p_exp, r)
                budget_spent += r
                exp_fn_val = tuple([-1 * i for i in problem.minmax]) * p_exp.objectives_mean

                # Check if expansion point is an improvement relative to simplex.
                if exp_fn_val < fn_low:
                    sort_sol[-1] = p_exp  # p_exp replaces p_high.

                    # Sort & end updating.
                    sort_sol = self.sort_and_end_update(problem, sort_sol)

                    # Record data from expansion point (new best).
                    if budget_spent <= problem.factors["budget"]:
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
                p_cont = self.check_const(p_cont, p_cont2.x)

                # Evaluate contraction point.
                p_cont = Solution(p_cont, problem)
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
                        p_new = self.check_const(p_new, p_new2.x)
                        p_new = Solution(p_new, problem)
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

    # Check & modify (if needed) the new point based on bounds.
    def check_const(self, pt, pt2):
        col = len(pt2)
        step = tuple(map(lambda i, j: i - j, pt, pt2))
        tmax = np.ones(col)
        for i in range(col):
            if step[i] > 0 and self.upper_bounds is not None:  # Move pt to ub.
                tmax[i] = (self.upper_bounds[i] - pt2[i]) / step[i]
            elif step[i] < 0 and self.lower_bounds is not None:  # Move pt to lb.
                tmax[i] = (self.lower_bounds[i] - pt2[i]) / step[i]
        t = min(1, min(tmax))
        modified = list(map(lambda i, j: i + t * j, pt2, step))
        # Remove rounding error.
        for i in range(col):
            if abs(modified[i]) < self.factors["sensitivity"]:
                modified[i] = 0
        return tuple(modified)
