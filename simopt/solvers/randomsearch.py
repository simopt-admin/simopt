"""
Summary
-------
Randomly sample solutions from the feasible region. 
"""
from base import Solver, Solution
import numpy as np

class RandomSearch(Solver):
    """
    A solver that randomly samples solutions to evaluate from the feasible region.
    Take a fixed number of replications at each solution.

    Attributes
    ----------
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
        changeable factors of the solver
    rng_list : list of rng.MRG32k3a objects
        list of random-number generators used for the solver's internal purposes    

    Arguments
    ---------
    fixed_factors : dict
        fixed_factors of the solver

    See also
    --------
    base.Solver
    """
    def __init__(self, fixed_factors={}):
        self.objective_type = "single"
        self.constraint_type = "deterministic"
        self.variable_type = "mixed"
        self.gradient_needed = False
        self.factors = fixed_factors
        self.specifications = {
            "sample_size": {
                "description": "Sample size per solution",
                "datatype": int,
                "default": 10
            }
        }
        self.check_factor_list = {
            "sample_size": self.check_sample_size,
        }
        # NEED TO WRITE THIS INTO SOLVER SUPER CLASS __INIT__
        # set factors of the simulation oracle
        super().__init__(fixed_factors)

    # NEED TO WRITE CHECK FACTORS FRAMEWORK IN SOLVER SUPER CLASS
    # Check for valid factors
    def check_sample_size(self):
        return self.factors["sample_size"] > 0

    def check_simulatable_factors(self):
        pass
    
    def solver(self, problem):
        """
        Run a single macroreplciation of a solver on a problem.

        Arguments
        ---------
        problem : Problem object
            simulation-optimization problem to solve
        crn_across_solns : bool 
            indicates if CRN will be used when simulating different solutions
        
        Returns
        -------
        recommended_solns : list of Solution objects
            list of solutions recommended throughout the budget
        intermediate_budgets : list of ints
            list of intermediate budgets at which recommended solution changes
        """
        # AT WRAPPER LEVEL, CHECK COMPATIBILITY OF SOLVER AND PROBLEM
        # initialize
        recommended_solns = []
        intermediate_budgets = []
        # designate random number generator
        find_next_soln_rng = self.rng_list[0]
        
        expended_budget = 0
        while expended_budget < problem.budget:
            new_x = problem.get_random_solution() # Needs to use find_next_soln_rng
            new_solution = Solution(new_x, problem)
            problem.simulate(solution=new_solution, m=self.factors["sample_size"])
            expended_budget += self.factors["sample_size"]
            # account for first iteration
            # track best incumbent solution
            if problem.minmax*new_solution.objectives_mean > problem.minmax*best_solution.objectives_mean:
                recommended_solns.append(new_solution)
                intermediate_budgets.append(expended_budget)
        # return recommended solutions and intermediate budgets
        return recommended_solns, intermediate_budgets