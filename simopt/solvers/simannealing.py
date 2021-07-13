"""
Summary
-------
Simulated Annealing in Noisy Environments (SANE).
"""
import numpy as np
import scipy.stats as ss

from base import Solver, Solution


class SANE(Solver):
    """
    Simulated Annealing in Noisy Environments (SANE)
    "Simulated Annealing in the Presence of Noise"
    Jurgen Branke, Stephan Meisel and Christian Schmidt
    Journal of Heuristics (2008) 14: 627--654.

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
    def __init__(self, name="SANE", fixed_factors={}):
        self.name = name
        self.objective_type = "single"
        self.constraint_type = "deterministic"
        self.variable_type = "mixed"
        self.gradient_needed = False
        self.specifications = {
            "crn_across_solns": {
                "description": "Use CRN across solutions?",
                "datatype": bool,
                "default": True
            },
            "sampling_variance": {
                "description": "Variance of difference in objective values",
                "datatype": float,
                "default": 100.0
            },
            "init_temp": {
                "description": "Initial temperature",
                "datatype": float,
                "default": 10.0
            },
            "cooling_coeff": {
                "description": "Coefficient for geometric cooling temperature schedule",
                "datatype": float,
                "default": 0.95**(1/100)
            }
        }
        self.check_factor_list = {
            "crn_across_solns": self.check_crn_across_solns,
            "sampling_variance": self.check_sampling_variance,
            "init_temp": self.check_init_temp,
            "cooling_coeff": self.check_cooling_coeff
        }
        super().__init__(fixed_factors)

    def check_sampling_variance(self):
        return self.factors["sample_variance"] > 0

    def check_init_temp(self):
        return self.factors["init_temp"] > 0

    def check_cooling_coeff(self):
        return 0 < self.factors["cooling_coeff"] < 1

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
        recommended_solns = []
        intermediate_budgets = []
        expended_budget = 0
        temperature = self.factors["init_temp"]
        # self.rng_list[0] is unused.
        # Designate random number generator for random sampling.
        find_next_soln_rng = self.rng_list[1]
        # Designate random number generator for switching to new solutions.
        switch_soln_rng = self.rng_list[2]
        # Sequentially generate a random neighboring solution, assess its
        # quality, and switch based on estimated differences and current 
        # temperature.
        # TO DO: Double-check how RNGs are to be used to simulate solutions.
        while expended_budget < problem.factors["budget"]:
            if expended_budget == 0:
                # Start at initial solution and record as best.
                current_x = problem.factors["initial_solution"]
                current_solution = self.create_new_solution(current_x, problem)
                recommended_solns.append(current_solution)
                intermediate_budgets.append(expended_budget)
            if temperature >= 1./np.sqrt(8.0/(np.pi*self.factors["sampling_variance"])):
                #print("First Case")
                # Simulate one replication of current solution.
                # Fresh sampling, so create new solution objects.
                current_solution = self.create_new_solution(current_x, problem)
                problem.simulate(current_solution, m=1)
                expended_budget += 1
                # Simulate one replication at new neighboring solution
                # Fresh sampling, so create new solution objects.
                new_x = problem.get_random_solution(find_next_soln_rng)
                new_solution = self.create_new_solution(new_x, problem)
                problem.simulate(new_solution, m=1)
                expended_budget += 1
                # Follow Ceperley and Dewing acceptance condition.
                # See Equation (15) on pg. 638 of Branke et al. (2008).
                delta_hat = problem.minmax * (current_solution.objectives_mean - new_solution.objectives_mean)
                if delta_hat <= -0.5*self.factors["sampling_variance"]/temperature:
                    prob_switch = 1
                else:
                    prob_switch = np.exp(-1*(delta_hat/temperature + 0.5*self.factors["sampling_variance"]/temperature**2))
                # Switch to new solution with probability prob_switch
                coin_flip = switch_soln_rng.random()
                if coin_flip < prob_switch:
                    #print("Switched")
                    recommended_solns.append(new_solution)
                    intermediate_budgets.append(expended_budget)
                    current_x = new_x
            else:
                #print("Second Case")
                #print(expended_budget)
                # Create a fresh solution object for current solution
                current_solution = self.create_new_solution(current_x, problem)
                # Identify new neighboring solution to simulate.
                # TO DO: generalize to neighborhood of current solution.
                new_x = problem.get_random_solution(find_next_soln_rng)
                new_solution = self.create_new_solution(new_x, problem)
                # Do sequential sampling until error probability matches Glauber probability
                prob_error = 1
                prob_glauber = 0
                sample_size = 0
                while prob_error > prob_glauber:
                    problem.simulate(current_solution, m=1)
                    expended_budget += 1
                    problem.simulate(new_solution, m=1)
                    expended_budget += 1
                    sample_size += 1
                    # Estimate difference in objective value.
                    delta_hat = problem.minmax * (current_solution.objectives_mean - new_solution.objectives_mean)
                    prob_error = ss.norm.cdf(-np.abs(delta_hat)*np.sqrt(sample_size)/np.sqrt(self.factors["sampling_variance"]))
                    prob_glauber = 1.0/(1.0 + np.exp(np.abs(delta_hat)/temperature))
                #print(expended_budget)
                # Accept new solution.
                recommended_solns.append(new_solution)
                intermediate_budgets.append(expended_budget)
                current_x = new_x
            # Update temperature according to cooling schedule.
            temperature = self.factors["init_temp"]*self.factors["cooling_coeff"]**expended_budget
        return recommended_solns, intermediate_budgets
