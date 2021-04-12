"""
Summary
-------
Randomly sample solutions from the feasible region.
"""
from base import Solver, Solution


class RandomSearch(Solver):
    """
    A solver that randomly samples solutions from the feasible region.
    Take a fixed number of replications at each solution.

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
    fixed_factors : dict
        fixed_factors of the solver

    See also
    --------
    base.Solver
    """
    def __init__(self, fixed_factors={}):
        self.name = "RNDSRCH"
        self.objective_type = "single"
        self.constraint_type = "deterministic"
        self.variable_type = "mixed"
        self.gradient_needed = False
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
        super().__init__(fixed_factors)

    def check_sample_size(self):
        return self.factors["sample_size"] > 0

    def check_solver_factors(self):
        pass

    def solve(self, problem, crn_across_solns):
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
        # Designate random number generator for random sampling.
        find_next_soln_rng = self.rng_list[0]
        # Sequentially generate random solutions and simulate them.
        while expended_budget < problem.budget:
            if expended_budget == 0:
                # Start at initial solution and record as best.
                new_x = problem.initial_solution
                new_solution = Solution(new_x, problem)
                best_solution = new_solution
                recommended_solns.append(new_solution)
                intermediate_budgets.append(expended_budget)
            else:
                # Identify new solution to simulate.
                new_x = problem.get_random_solution(find_next_soln_rng)
                new_solution = Solution(new_x, problem)
            # Manipulate RNGs, simulate new solution, and update budget.
            self.prepare_sim_new_soln(problem, crn_across_solns)
            problem.simulate(new_solution, self.factors["sample_size"])
            expended_budget += self.factors["sample_size"]
            # Check for improvement relative to incumbent best solution.
            if (problem.minmax * new_solution.objectives_mean
                    > problem.minmax * best_solution.objectives_mean):
                # If better, record incumbent solution as best.
                best_solution = new_solution
                recommended_solns.append(new_solution)
                intermediate_budgets.append(expended_budget)
        # At final budget, record best solution.
        recommended_solns.append(best_solution)
        intermediate_budgets.append(problem.budget)
        return recommended_solns, intermediate_budgets
