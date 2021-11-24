"""
Summary
-------
Simulate multiple periods of arrival and seating at a restaurant.
"""
import numpy as np

import math

import scipy.stats as scs

from base import Model, Problem


class TableAllocation(Model):
    """
    An model that simulates a table capacity allocation problem at a restaurant 
    with a homogenous Poisson arrvial process and exponential service times.
    Returns expected maximal operating profit.

    Attributes
    ----------
    name : str
        name of model
    n_rngs : int
        number of random-number generators used to run a simulation replication
    n_responses : int
        number of responses (performance measures)
    factors : dict
        changeable factors of the simulation model
    specifications : dict
        details of each factor (for GUI, data validation, and defaults)
    check_factor_list : dict
        switch case for checking factor simulatability

    Parameters
    ----------
    fixed_factors : dict
        fixed_factors of the simulation model

        ``n_min``
            Number of minutes to simulate (`int`)
        ``capacity``
            Maximum total capacity (`int`)
        ``table_cap``
            Capacity of each type of table (`int`)
        ``lambda``
            Average number of arrivals per hour (`flt`)
        ``service_time_means``
            Mean service time in minutes (`flt`)
        ``rev_per_table``
            Per table revenue earned (`flt`)
        ``num_tables``
            Number of tables of each capacity (`int`)

    See also
    --------
    base.Model
    """
        ``n_min``
        ``capacity``
        ``table_size``
        ``lambda``
        ``arrival_rate``
        ``service_time_means``
        ``rev_per_table``
        ``num_tables``


    def __init__(self, fixed_factors={}):
        self.name = "TABLEALLOCATION"
        self.n_rngs = 1
        self.n_responses = 1
        self.factors = fixed_factors
        self.specifications = {

            "n_min": {
                "description": "Number of minutes to simulate.",
                "datatype": int,
                "default": 180
            },
            "capacity": {
                "description": "Maximum total capacity.",
                "datatype": int,
                "default": 80
            },
            "table_cap": {
                "description": "Capacity of each type of table.",
                "datatype": float,
                "default": [2,4,6,8]
            },
            "lambda": {
                "description": "Average number of arrivals per hour.",
                "datatype": float,
                "default": [3, 6, 3, 3, 2, 4/3, 6/5, 1]
            },
            "service_time_means": {
                "description": "Mean service time in minutes.",
                "datatype": float,
                "default": [20, 25, 30, 35, 40, 45, 50, 60]
            },
            "rev_per_table": {
                "description": "Per table revenue earned.",
                "datatype": int,
                "default": [15, 30, 45, 60, 75, 90, 105, 120]
            },
            "num_tables": {
                "description": "Number of tables of each capacity.",
                "datatype": float,
                "default": 5.00
            }

        }

        self.check_factor_list = {
            "n_days": self.check_n_days,
            "initial_inv": self.check_initial_inv,
            "cost_reg": self.check_cost_reg,
            "cost_exp": self.check_cost_exp,
            "lead_reg": self.check_lead_reg,
            "lead_exp": self.check_lead_exp,
            "holding_cost": self.check_holding_cost,
            "penalty_cost": self.check_penalty_cost,
            "distribution": self.check_distribution,
            "st_dev": self.check_st_dev,
            "mu": self.check_mu,
            "order_level_reg": self.check_order_level_reg,
            "order_level_exp": self.check_order_level_exp,
            
        }
        # Set factors of the simulation model
        super().__init__(fixed_factors)

    # Check for simulatable factors
    def check_n_days(self):
        return self.factors["n_days"] >= 1
    
    def check_initial_inv(self):
        return self.factors["initial_inv"] >= 0

    def check_cost_reg(self):
        return self.factors["cost_reg"] > 0

    def check_cost_exp(self):
        return self.factors["cost_exp"] > 0

    def check_lead_reg(self):
        return self.factors["lead_reg"] >= 0
    
    def check_lead_exp(self):
        return self.factors["lead_exp"] >= 0

    def check_holding_cost(self):
        return self.factors["holding_cost"] > 0

    def check_penalty_cost(self):
        return self.factors["penalty_cost"] > 0

    def check_distribution(self):
        return self.factors["distribution"] in ('Normal', 'Uniform','Exponential')
    
    def check_st_dev(self):
        return self.factors["st_dev"] > 0

    def check_mu(self):
        return self.factors["mu"] > 0

    def check_order_level_reg(self):
        return self.factors["order_level_reg"] >= 0

    def check_order_level_exp(self):
        return self.factors["order_level_exp"] >= 0

    def check_simulatable_factors(self):
        return (self.factors["lead_exp"] < self.factors["lead_reg"]) & (self.factors["cost_exp"] > self.factors["cost_exp"])

    def replicate(self, rng_list):
        """
        Simulate a single replication for the current model factors.

        Arguments
        ---------
        rng_list : [list]  [rng.mrg32k3a.MRG32k3a]
            rngs for model to use when simulating a replication

        Returns
        -------
        responses : dict
            performance measures of interest

            ``total_revenue``
                The total revenue earned over the simulation period.

        """
        # Calculate total arrival rate
        # Possible group sizes
        # Probabilities to decide which group arrived
        # Track total revenue 
        # Initiate system time, total revenue to 0
        # Track table availability 
        # (i,j) is the time that jth table of size i becomes available
        # Generate arrival times
        while time < self.factors['n_min']:
            # Sample arrival 
            # Update system time
            # Determine group size
            # Find smallest available table
            # Sample service time
            # Update table availability


        # Calculate responses from simulation data.
        responses = {"total_revenue": total_revenue
                     }
        gradients = {response_key: {factor_key: np.nan for factor_key in self.specifications} for response_key in responses}
        return responses, gradients


"""
Summary
-------
Minimize the expected total cost for dual-sourcing inventory system.
"""


class DualSourcingMinCost(Problem):
    """
    Class to make dual-sourcing inventory simulation-optimization problems.

    Attributes
    ----------
    name : str
        name of problem
    dim : int
        number of decision variables
    n_objectives : int
        number of objectives
    n_stochastic_constraints : int
        number of stochastic constraints
    minmax : tuple of int (+/- 1)
        indicator of maximization (+1) or minimization (-1) for each objective
    constraint_type : str
        description of constraints types:
            "unconstrained", "box", "deterministic", "stochastic"
    variable_type : str
        description of variable types:
            "discrete", "continuous", "mixed"
    gradient_available : bool
        indicates if gradient of objective function is available
    optimal_value : float
        optimal objective function value
    optimal_solution : tuple
        optimal solution
    model : base.Model
        associated simulation model that generates replications
    model_default_factors : dict
        default values for overriding model-level default factors
    model_fixed_factors : dict
        combination of overriden model-level factors and defaults
    model_decision_factors : set of str
        set of keys for factors that are decision variables
    rng_list : [list]  [rng.mrg32k3a.MRG32k3a]
        list of RNGs used to generate a random initial solution
        or a random problem instance
    factors : dict
        changeable factors of the problem
            initial_solution : tuple
                default initial solution from which solvers start
            budget : int > 0
                max number of replications (fn evals) for a solver to take
    specifications : dict
        details of each factor (for GUI, data validation, and defaults)

    Arguments
    ---------
    name : str
        user-specified name of problem
    fixed_factors : dict
        dictionary of user-specified problem factors
    model_fixed factors : dict
        subset of user-specified non-decision factors to pass through to the model

    See also
    --------
    base.Problem
    """
    def __init__(self, name="DUALSOURCING-1", fixed_factors={}, model_fixed_factors={}):
        self.name = name
        self.dim = 2
        self.n_objectives = 1
        self.n_stochastic_constraints = 0
        self.minmax = (-1,)
        self.constraint_type = "box"
        self.variable_type = "discrete"
        self.lowerbound = (0)
        self.upperbound = (np.inf)
        self.gradient_available = False
        self.optimal_value = None
        self.optimal_solution = None
        self.model_default_factors = {}
        self.model_decision_factors = {"order_level_exp", "order_level_reg"}
        self.factors = fixed_factors
        self.specifications = {
            "initial_solution": {
                "description": "Initial solution from which solvers start.",
                "datatype": tuple,
                "default": (50,80)
            },
            "budget": {
                "description": "Max # of replications for a solver to take.",
                "datatype": int,
                "default": 1000
            }
        }
        self.check_factor_list = {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget
        }
        super().__init__(fixed_factors, model_fixed_factors)
        # Instantiate model with fixed factors and overwritten defaults.
        self.model = DualSourcing(self.model_fixed_factors)

    def vector_to_factor_dict(self, vector):
        """
        Convert a vector of variables to a dictionary with factor keys

        Arguments
        ---------
        vector : tuple
            vector of values associated with decision variables

        Returns
        -------
        factor_dict : dict
            dictionary with factor keys and associated values
        """
        factor_dict = {
            "order_level_exp": vector[0],
            "order_level_reg": vector[1]
        }
        return factor_dict

    def factor_dict_to_vector(self, factor_dict):
        """
        Convert a dictionary with factor keys to a vector
        of variables.

        Arguments
        ---------
        factor_dict : dict
            dictionary with factor keys and associated values

        Returns
        -------
        vector : tuple
            vector of values associated with decision variables
        """
        vector = (factor_dict["order_level_exp"], factor_dict["order_level_reg"])
        return vector

    def response_dict_to_objectives(self, response_dict):
        """
        Convert a dictionary with response keys to a vector
        of objectives.

        Arguments
        ---------
        response_dict : dict
            dictionary with response keys and associated values

        Returns
        -------
        objectives : tuple
            vector of objectives
        """
        objectives = (response_dict["average_ordering_cost"] + response_dict["average_penalty_cost"] + response_dict["average_holding_cost"],)
        return objectives

    def response_dict_to_stoch_constraints(self, response_dict):
        """
        Convert a dictionary with response keys to a vector
        of left-hand sides of stochastic constraints: E[Y] >= 0

        Arguments
        ---------
        response_dict : dict
            dictionary with response keys and associated values

        Returns
        -------
        stoch_constraints : tuple
            vector of LHSs of stochastic constraint
        """
        stoch_constraints = None
        return stoch_constraints

    def deterministic_objectives_and_gradients(self, x):
        """
        Compute deterministic components of objectives for a solution `x`.

        Arguments
        ---------
        x : tuple
            vector of decision variables

        Returns
        -------
        det_objectives : tuple
            vector of deterministic components of objectives
        det_objectives_gradients : tuple
            vector of gradients of deterministic components of objectives
        """
        det_objectives = (0,)
        det_objectives_gradients = ((0,),)
        return det_objectives, det_objectives_gradients

    def deterministic_stochastic_constraints_and_gradients(self, x):
        """
        Compute deterministic components of stochastic constraints
        for a solution `x`.

        Arguments
        ---------
        x : tuple
            vector of decision variables

        Returns
        -------
        det_stoch_constraints : tuple
            vector of deterministic components of stochastic constraints
        det_stoch_constraints_gradients : tuple
            vector of gradients of deterministic components of
            stochastic constraints
        """
        det_stoch_constraints = None
        det_stoch_constraints_gradients = None
        return det_stoch_constraints, det_stoch_constraints_gradients

    def check_deterministic_constraints(self, x):
        """
        Check if a solution `x` satisfies the problem's deterministic
        constraints.

        Arguments
        ---------
        x : tuple
            vector of decision variables

        Returns
        -------
        satisfies : bool
            indicates if solution `x` satisfies the deterministic constraints.
        """
        return (x[0] >= 0 and x[1] >= 0)

    def get_random_solution(self, rand_sol_rng):
        """
        Generate a random solution for starting or restarting solvers.

        Arguments
        ---------
        rand_sol_rng : rng.mrg32k3a.MRG32k3a
            random-number generator used to sample a new random solution

        Returns
        -------
        x : tuple
            vector of decision variables
        """
        x = (rand_sol_rng.randint(40, 60), rand_sol_rng.randint(70, 90))
        return x
