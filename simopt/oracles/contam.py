"""
Summary
-------
Simulate contamination rates.
"""
import numpy as np

from base import Oracle, Problem


class Contamination(Oracle):
    """
    An oracle that simulates a contamination problem with a
    beta distribution.
    Returns the probability of violating contamination upper limit 
    in each level of supply chain.

    Attributes
    ----------
    name : string
        name of oracle
    n_rngs : int
        number of random-number generators used to run a simulation replication
    n_responses : int
        number of responses (performance measures)
    factors : dict
        changeable factors of the simulation model
    specifications : dict
        details of each factor (for GUI and data validation)
    check_factor_list : dict
        switch case for checking factor simulatability

    Arguments
    ---------
    fixed_factors : nested dict
        fixed factors of the simulation model

    See also
    --------
    base.Oracle
    """
    def __init__(self, fixed_factors={}):
        self.name = "CONTAM"
        self.n_rngs = 2
        self.n_responses = 1
        self.specifications = {
            "contam_rate_alpha": {
                "description": "Alpha parameter of beta distribution for growth rate of contamination at each stage.",
                "datatype": float,
                "default": 1.0
            },
            "contam_rate_beta": {
                "description": "Beta parameter of beta distribution for growth rate of contamination at each stage.",
                "datatype": float,
                "default": 17/3
            },
            "restore_rate_alpha": {
                "description": "Alpha parameter of beta distribution for rate that contamination decreases by after prevention effort.",
                "datatype": float,
                "default": 1.0
            },
            "restore_rate_beta": {
                "description": "Beta parameter of beta distribution for rate that contamination decreases by after prevention effort.",
                "datatype": float,
                "default": 3/7
            },
            "initial_rate_alpha": {
                "description": "Alpha parameter of beta distribution for initial contamination fraction.",
                "datatype": float,
                "default": 1.0
            },
            "initial_rate_beta": {
                "description": "Beta parameter of beta distribution for initial contamination fraction.",
                "datatype": float,
                "default": 30.0
            },
            "error_prob": {
                "description": "Error probability.",
                "datatype": list,
                "default": [0.05, 0.05, 0.05, 0.05, 0.05]
            },
            "stages": {
                "description": "Stage of food supply chain.",
                "datatype": int,
                "default": 5
            },
            "prev_decision": {
                "description": "Prevention decision.",
                "datatype": list,
                "default": [0, 0, 0, 0, 0]
            }
        }
        self.check_factor_list = {
            "contam_rate_alpha": self.check_contam_rate_alpha,
            "contam_rate_beta": self.check_contam_rate_beta,
            "restore_rate_alpha": self.check_restore_rate_alpha,
            "restore_rate_beta": self.check_restore_rate_beta,
            "initial_rate_alpha": self.check_initial_rate_alpha,
            "initial_rate_beta": self.check_initial_rate_beta,
            "error_prob": self.check_error_prob,
            "stages": self.check_stages,
            "prev_decision": self.check_prev_decision
        }
        # Set factors of the simulation oracle.
        super().__init__(fixed_factors)

    def check_contam_rate_alpha(self):
        return self.factors["contam_rate_alpha"] > 0

    def check_contam_rate_beta(self):
        return self.factors["contam_rate_beta"] > 0

    def check_restore_rate_alpha(self):
        return self.factors["restore_rate_alpha"] > 0

    def check_restore_rate_beta(self):
        return self.factors["restore_rate_beta"] > 0

    def check_initial_rate_alpha(self):
        return self.factors["initial_rate_alpha"] > 0

    def check_initial_rate_beta(self):
        return self.factors["initial_rate_beta"] > 0

    def check_prev_cost(self):
        return all(cost > 0 for cost in self.factors["prev_cost"])

    def check_error_prob(self):
        return all(error > 0 for error in self.factors["error_prob"])

    def check_stages(self):
        return self.factors["stages"] > 0

    def check_prev_decision(self):
        return all(u >= 0 & u <= 1 for u in self.factors["prev_decision"])

    def check_simulatable_factors(self):
        # Check for matching number of stages.
        if len(self.factors["error_prob"]) != self.factors["stages"]:
            return False
        elif len(self.factors["prev_decision"]) != self.factors["stages"]:
            return False
        else:
            return True


    def replicate(self, rng_list):
        """
        Simulate a single replication for the current oracle factors.

        Arguments
        ---------
        rng_list : list of rng.MRG32k3a objects
            rngs for oracle to use when simulating a replication

        Returns
        -------
        responses : dict
            performance measures of interest
            "level" = a list of contamination levels over time
        gradients : dict of dicts
            gradient estimates for each response
        """
        # Designate separate random number generators.
        # Outputs will be coupled when generating demand.
        contam_rng = rng_list[0]
        restore_rng = rng_list[1]
        # Generate rates with beta distribution.
        X = np.zeros(self.factors["stages"])
        X[0] = restore_rng.betavariate(alpha=self.factors["initial_rate_alpha"], beta=self.factors["initial_rate_beta"])
        u = self.factors["prev_decision"]
        for i in range(1, self.factors["stages"]):
            c = contam_rng.betavariate(alpha=self.factors["contam_rate_alpha"], beta=self.factors["contam_rate_beta"])
            r = restore_rng.betavariate(alpha=self.factors["restore_rate_alpha"], beta=self.factors["restore_rate_beta"])
            X[i] = c*(1-u[i])*(1-X[i-1]) + (1-r*u[i])*X[i-1]
        # Compose responses and gradients.
        responses = {'level': X}
        gradients = {response_key: {factor_key: np.nan for factor_key in self.specifications} for response_key in responses}
        return responses, gradients

"""
Summary
-------
Minimize the (deterministic) total cost of prevention efforts.
"""


class ContaminationTotalCost(Problem):
    """
    Base class to implement simulation-optimization problems.

    Attributes
    ----------
    name : string
        name of problem
    dim : int
        number of decision variables
    n_objectives : int
        number of objectives
    n_stochastic_constraints : int
        number of stochastic constraints
    minmax : tuple of int (+/- 1)
        indicator of maximization (+1) or minimization (-1) for each objective
    constraint_type : string
        description of constraints types:
            "unconstrained", "box", "deterministic", "stochastic"
    variable_type : string
        description of variable types:
            "discrete", "continuous", "mixed"
    lower_bounds : tuple
        lower bound for each decision variable
    upper_bounds : tuple
        upper bound for each decision variable
    gradient_available : bool
        indicates if gradient of objective function is available
    optimal_value : float
        optimal objective function value
    optimal_solution : tuple
        optimal solution
    oracle : Oracle object
        associated simulation oracle that generates replications
    oracle_default_factors : dict
        default values for overriding oracle-level default factors
    oracle_fixed_factors : dict
        combination of overriden oracle-level factors and defaults
    rng_list : list of rng.MRG32k3a objects
        list of RNGs used to generate a random initial solution
        or a random problem instance
    factors : dict
        changeable factors of the problem
            initial_solution : list
                default initial solution from which solvers start
            budget : int > 0
                max number of replications (fn evals) for a solver to take
            prev_cost : list
                cost of prevention
            upper_thres : float > 0
                upper limit of amount of contamination
    specifications : dict
        details of each factor (for GUI, data validation, and defaults)

    Arguments
    ---------
    name : str
        user-specified name for problem
    fixed_factors : dict
        dictionary of user-specified problem factors
    oracle_fixed factors : dict
        subset of user-specified non-decision factors to pass through to the oracle

    See also
    --------
    base.Problem
    """
    def __init__(self, name="CONTAM", fixed_factors={}, oracle_fixed_factors={}):
        self.name = name
        self.dim = 5  # stages
        self.n_objectives = 1
        self.n_stochastic_constraints = 5
        self.minmax = (-1,)
        self.constraint_type = "stochastic"
        self.variable_type = "discrete"
        self.lower_bounds = (0, 0, 0, 0 ,0)
        self.upper_bounds = (1, 1, 1, 1, 1)
        self.gradient_available = True
        self.optimal_value = None
        self.optimal_solution = None  # (185, 185, 185)
        self.oracle_default_factors = {}
        self.factors = fixed_factors
        self.specifications = {
            "initial_solution": {
                "description": "Initial solution.",
                "datatype": list,
                "default": [0, 0, 0, 0, 0]
            },
            "prevention_budget": {
                "description": "Max # of replications for a solver to take.",
                "datatype": int,
                "default": 10000
            },
            "prev_cost": {
                "description": "Cost of prevention.",
                "datatype": list,
                "default": [1, 1, 1, 1, 1]
            },
            "upper_thres": {
                "description": "Upper limit of amount of contamination.",
                "datatype": list,
                "default": [0.1, 0.1, 0.1, 0.1, 0.1]
            }            
        }

        self.check_factor_list = {
            "initial_solution": self.check_initial_solution,
            "prevention_budget": self.check_prevention_budget,
            "prev_cost": self.check_prev_cost,
            # "upper_thres": self.check_upper_thres
        }

        super().__init__(fixed_factors, oracle_fixed_factors)
        # Instantiate oracle with fixed factors and over-riden defaults.
        self.oracle = Contamination(self.oracle_fixed_factors)

    def check_initial_solution(self):
        return all(u >= 0 & u <= 1 for u in self.factors["initial_solution"])

    def check_prev_cost(self):
        if len(self.factors["prev_cost"]) != self.oracle.factors["stages"]:
            return False
        elif any([elem < 0 for elem in self.factors["prev_cost"]]):
            return False
        else:
            return True

    def check_prevention_budget(self):
        return self.factors["prevention_budget"] > 0


    def vector_to_factor_dict(self, vector):
        """
        Convert a vector of variables to a dictionary with factor keys

        Arguments
        ---------
        vector : tuple
            vector of values associated with decision variables

        Returns
        -------
        factor_dict : dictionary
            dictionary with factor keys and associated values
        """
        factor_dict = {
            "level": vector[:]
        }
        return factor_dict

    def factor_dict_to_vector(self, factor_dict):
        """
        Convert a dictionary with factor keys to a vector
        of variables.

        Arguments
        ---------
        factor_dict : dictionary
            dictionary with factor keys and associated values

        Returns
        -------
        vector : tuple
            vector of values associated with decision variables
        """
        vector = tuple(factor_dict["level"])
        return vector

    def response_dict_to_objectives(self, response_dict):
        """
        Convert a dictionary with response keys to a vector
        of objectives.

        Arguments
        ---------
        response_dict : dictionary
            dictionary with response keys and associated values

        Returns
        -------
        objectives : tuple
            vector of objectives
        """
        objectives = (0,)
        return objectives

    def response_dict_to_stoch_constraints(self, response_dict):
        """
        Convert a dictionary with response keys to a vector
        of left-hand sides of stochastic constraints: E[Y] >= 0

        Arguments
        ---------
        response_dict : dictionary
            dictionary with response keys and associated values

        Returns
        -------
        stoch_constraints : tuple
            vector of LHSs of stochastic constraint
        """
        stoch_constraints = tuple(response_dict["level"] <= self.factors["upper_thres"])
        return stoch_constraints

    def deterministic_stochastic_constraints_and_gradients(self, x):
        """
        Compute deterministic components of stochastic constraints for a solution `x`.

        Arguments
        ---------
        x : tuple
            vector of decision variables

        Returns
        -------
        det_stoch_constraints : tuple
            vector of deterministic components of stochastic constraints
        det_stoch_constraints_gradients : tuple
            vector of gradients of deterministic components of stochastic constraints
        """
        det_stoch_constraints = tuple(-np.ones(5) + self.oracle.factors["error_prob"]) # >:/
        det_stoch_constraints_gradients = ((0,),)
        return det_stoch_constraints, det_stoch_constraints_gradients

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
        det_objectives = (np.dot(self.factors["prev_cost"], x),)
        det_objectives_gradients = ((self.factors["prev_cost"],),)
        return det_objectives, det_objectives_gradients

    def check_deterministic_constraints(self, x):
        """
        Check if a solution `x` satisfies the problem's deterministic constraints.

        Arguments
        ---------
        x : tuple
            vector of decision variables

        Returns
        -------
        satisfies : bool
            indicates if solution `x` satisfies the deterministic constraints.
        """
        return np.all(x >= 0) & np.all(x <= 1)

    def get_random_solution(self, rand_sol_rng):
        """
        Generate a random solution for starting or restarting solvers.

        Arguments
        ---------
        rand_sol_rng : rng.MRG32k3a object
            random-number generator used to sample a new random solution

        Returns
        -------
        x : tuple
            vector of decision variables
        """
        x = tuple([300*rand_sol_rng.random() for _ in range(self.dim)])
        return x