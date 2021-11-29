"""
Summary
-------
Nelder-Mead
The algorithm maintains a simplex of points that moves around the feasible 
region according to certain geometric operations: reflection, expansion, 
scontraction, and shrinking.
"""
from base import Solver
import numpy as np
import math
import warnings
warnings.filterwarnings("ignore")

class NELDMD(Solver):
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
        self.constraint_type = "deterministic"
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
                "datatype": float,
                "default": 30.
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
                "description": "shrinking scale for VarBds (bounds)",
                "datatype": float,
                "default": 10**(-7)
            }
        }
        self.check_factor_list = {
            "crn_across_solns": self.check_crn_across_solns,
            "r": self.check_r,
            "alpha": self.check_alpha,
            "gammap": self.check_gammap,
            "betap": self.check_betap,
            "delta": self.check_delta,
            "sensitivity": self.check_sensitivity
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
        n_pts = problem.dim + 1
        # Check for sufficiently large budget
        if problem.factors["budget"] <  self.factors["r"]*n_pts:
            print('Budget is too small for a good quality run of Nelder-Mead.')
            return
        # Determine max number of solutions that can be sampled within budget
        max_num_sol = int(np.floor(problem.factors["budget"]/self.factors["r"]))
        # Shrink variable bounds to avoid floating errors
        lower_bounds = problem.lower_bounds + self.factors["sensitivity"]
        upper_bounds = problem.upper_bounds - self.factors["sensitivity"]
        # Initial dim + 1 random points
        sol = np.zeros(n_pts)
        new_x = problem.factors["initial_solution"]
        sol[0] = new_x
        for i in range(1, n_pts):
            sol[i] = self.create_new_solution(new_x, problem)  # use self.get_random_solution() instead?

        # Initialize larger than necessary (extra point for end of budget)
        n_calls = np.zeros(max_num_sol)
        A = np.zeros(((max_num_sol+1),problem.dim))
        fn_mean = np.zeros(max_num_sol)
        fn_var = np.zeros(max_num_sol)
        # Using CRN: for each solution, start at substream 1
        problemseed = 1
        # Track overall budget spent
        budget_spent = 0

        # Start Solving
        # Evaluate solutions in initial structure
        fn_val = np.zeros(n_pts)
        fn_var_val = np.zeros(n_pts)
        for i in range(n_pts):
            problem.simulate(sol[i], self.factors["r"])
            budget_spent += self.factors["r"]
            fn_val[i] = -1*problem.minmax*sol[i].objectives_mean
            fn_var_val[i] = sol[i].objectives_var
        # Record initial solution data
        n_calls[0] = 0
        A[0] = sol[0]
        fn_mean[0] = -1*problem.minmax*fn_val[0]
        fn_var[0] = fn_var_val[0]
        # Sort solutions by obj function estimate
        fn_val_idx = np.argsort(fn_val)
        sort_fn_val = fn_val[fn_val_idx]
        sort_fn_var_val = fn_var_val[fn_val_idx]
        sort_sol = sol[fn_val_idx]
        # Record only when recommended solution changes
        record_idx = 1

        # Reflect worst and update sort_sol
        # Maximization problem is converted to minimization by -z
        while budget_spent <= problem.factors["budget"]:
            # Reflect worse point
            p_high = sort_sol[-1]  # current worst point
            p_cent = np.mean(sort_sol[1:-1])  # centroid for other pts
            orig_pt = p_high  # save the original point
            p_refl = (1 + self.factors["alpha"])*p_cent - self.factors["alpha"]*p_high  # reflection
            p_refl = checkCons() ### TODO: write helper function
            
            # Evaluate reflected point
            problem.simulate(p_refl, self.factors["r"])
            budget_spent += self.factors["r"]
            refl_fn_val = -1*problem.minmax*p_refl.objectives_mean
            refl_fn_var_val = p_refl.objectives_var
            
            # Track best, worst, and second worst points
            p_low = sort_sol[0]  # current best pt
            fn_low = sort_fn_val[0]
            fn_sec = sort_fn_val[-2]  # current 2nd worst z
            fn_high = sort_fn_val[-1]  # worst z from unreflected structure

            # Check if accept reflection
            if fn_low <= refl_fn_val and refl_fn_val <= fn_sec:
                sort_sol[-1] = p_refl  # the new point replaces the previous worst
                sort_fn_val[-1] = refl_fn_val
                sort_fn_var_val[-1] = refl_fn_var_val

                # Sort & end updating
                fn_val_idx = np.argsort(sort_fn_val)
                sort_fn_val = sort_fn_val[fn_val_idx]
                sort_fn_var_val = sort_fn_var_val[fn_val_idx]
                sort_sol = sort_sol[fn_val_idx]

                # Best solution remains the same, so no reporting

            # Check if accept expansion (of reflection in the same direction)
            elif refl_fn_val < fn_low:
                p_exp2 = p_refl
                p_exp = self.factors["gammap"]*p_refl + (1-self.factors["gammap"])*p_cent
                p_exp = checkCons()  ### TODO: helper function

                # Evaluate expansion point
                problem.simulate(p_exp, self.factors["r"])
                budget_spent += self.factors["r"]
                exp_fn_val = -1*problem.minmax*p_exp.objectives_mean
                exp_fn_var_val = p_refl.objectives_var

                # Check if expansion point is an improvement relative to simplex
                if exp_fn_val < fn_low:
                    sort_sol[-1] = p_exp  # p_exp replaces p_high
                    sort_fn_val[-1] = exp_fn_val
                    sort_fn_var_val[-1] = exp_fn_var_val

                    # Sort & end updating
                    fn_val_idx = np.argsort(sort_fn_val)
                    sort_fn_val = sort_fn_val[fn_val_idx]
                    sort_fn_var_val = sort_fn_var_val[fn_val_idx]
                    sort_sol = sort_sol[fn_val_idx] 

                    # Record data from expansion point (new best)
                    if budget_spent <= problem.factors["budget"]:
                        n_calls[record_idx] = budget_spent
                        A[record_idx] = p_exp
                        fn_mean[record_idx] = -1*problem.minmax*exp_fn_val  # flip sign back
                        fn_var[record_idx] = exp_fn_var_val
                        record_idx += 1
                else:
                    sort_sol[-1] = p_refl  # p_refl replaces p_high
                    sort_fn_val[-1] = refl_fn_val
                    sort_fn_var_val[-1] = refl_fn_var_val

                    # Sort & end updating
                    fn_val_idx = np.argsort(sort_fn_val)
                    sort_fn_val = sort_fn_val[fn_val_idx]
                    sort_fn_var_val = sort_fn_var_val[fn_val_idx]
                    sort_sol = sort_sol[fn_val_idx] 

                    # Record data from expansion point (new best)
                    if budget_spent <= problem.factors["budget"]:
                        n_calls[record_idx] = budget_spent
                        A[record_idx] = p_refl
                        fn_mean[record_idx] = -1*problem.minmax*refl_fn_val  # flip sign back
                        fn_var[record_idx] = refl_fn_var_val
                        record_idx += 1
            
            # Check if accept contraction or shrink
            elif refl_fn_val > fn_sec:  # line 238

                        
    

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# what is Varbds in matlab?
# How do you set rng? self.rng_list[0]?
# What's ssolsM
problem.factors["budget"]
problem.minmax
problem.dim
self.factors["r"]


n_pts ~ numExtPts in matlab
max_num_sol ~ MaxNumSoln