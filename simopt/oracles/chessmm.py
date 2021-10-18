"""
Summary
-------
Simulate matching of arriving chess players.
"""
import numpy as np
from scipy import special

from base import Oracle, Problem


class ChessMatchmaking(Oracle):
    """
    An oracle that simulates a matchmaking problem with a
    Elo (truncated normal) distribution of players and Poisson arrivals.
    Returns the average difference between matched players.

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
        self.name = "CHESS"
        self.n_rngs = 3
        self.n_responses = 1
        self.specifications = {
            "elo_mean": {
                "description": "Mean of normal distribution for Elo rating.",
                "datatype": float,
                "default": 1200.0
            },
            "elo_sd": {
                "description": "Standard deviation of normal distribution for Elo rating.",
                "datatype": float,
                "default": 1200/(np.sqrt(2)*special.erfcinv(1/50))
            },
            "poisson_rate": {
                "description": "Rate of Poisson process for player arrivals.",
                "datatype": float,
                "default": 1.0
            },
            "initial_mean": {
                "description": "Mean of normal distribution for multiple starting solutions.",
                "datatype": float,
                "default": 150.0
            },
            "initial_sd": {
                "description": "Standard deviation of normal distribution for multiple starting solutions.",
                "datatype": float,
                "default": 50.0
            },
            "num_players": {
                "description": "Number of players.",
                "datatype": int,
                "default": 10000
            },
            "width": {
                "description": "Maximum allowable difference between Elo ratings.",
                "datatype": float,
                "default": 150.0
            },
            "max_diff_decision": {
                "description": "Max difference between Elo ratings to match decision.",
                "datatype": tuple,
                "default": (0)
            }
        }
        self.check_factor_list = {
            "poisson_rate": self.check_poisson_rate,
            "num_players": self.check_num_players,
            "width": self.check_width,
            "max_diff_decision": self.check_max_diff_decision
        }
        # Set factors of the simulation oracle.
        super().__init__(fixed_factors)

    def check_poisson_rate(self):
        return self.factors["poisson_rate"] > 0

    def check_num_players(self):
        return self.factors["num_players"] > 0

    def check_width(self):
        return self.factors["width"] > 0

    def check_max_diff_decision(self):
        return self.factors["max_diff_decision"] >= 0

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
            "exp_diff" = the average Elo difference between all pairs
            "exp_wait_time" = the average waiting time
        gradients : dict of dicts
            gradient estimates for each response
        """
        # Designate separate random number generators.
        elo_rng = rng_list[0]
        arrival_rng = rng_list[1]
        initial_rng = rng_list[2]

        wait_times = np.zeros(self.factors["num_players"])
        waiting_players = []
        not_matched = []
        total_diff = 0
        elo_diffs = []

        for i in range(self.factors["num_players"]):
          player = elo_rng.normalvariate(self.factors["elo_mean"], self.factors["elo_sd"])
          while np.any(player < 0) or np.any(player > 2400):
              player = elo_rng.normalvariate(self.factors["elo_mean"], self.factors["elo_sd"])
          time = arrival_rng.poissonvariate(self.factors["poisson_rate"])
          old_total = total_diff
          for p in range(len(waiting_players)):
            if abs(player - waiting_players[p]) <= self.factors["width"]:
              total_diff += abs(player - waiting_players[p])
              elo_diffs.append(abs(player - waiting_players[p]))
              del waiting_players[p]
              del not_matched[p]
              break
            else:
              wait_times[p] += time
          if old_total == total_diff:
            waiting_players.append(player)
            not_matched.append(i)
        for i in not_matched:   # unmatched players not included in wait time
            wait_times[i] = 0

        # Compose responses and gradients.
        responses = {
          'exp_diff': np.mean(elo_diffs),
          'exp_wait_time': np.mean(wait_times)
        }
        gradients = {response_key: {factor_key: np.nan for factor_key in self.specifications} for response_key in responses}
        return responses, gradients

"""
Summary
-------
Minimize the average Elo difference between all pairs of matched players.
"""


class ChessAvgDifference(Problem):
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
    def __init__(self, name="CHESS", fixed_factors={}, oracle_fixed_factors={}):
        self.name = name
        self.n_objectives = 1
        self.n_stochastic_constraints = 0
        self.minmax = (-1,)
        self.constraint_type = "stochastic"
        self.variable_type = "continuous"
        self.lower_bounds = (0)
        self.upper_bounds = None
        self.gradient_available = False
        self.optimal_value = None
        self.optimal_solution = None
        self.oracle_default_factors = {}
        self.factors = fixed_factors
        self.specifications = {
            "initial_solution": {
                "description": "Initial solution.",
                "datatype": tuple,
                "default": (150)
            },
            "budget": {
                "description": "Max # of replications for a solver to take.",
                "datatype": int,
                "default": 10000
            },
            "upper_time": {
                "description": "Upper bound on wait time.",
                "datatype": int,
                "default": 5
            }            
        }

        self.check_factor_list = {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget,
            "upper_time": self.check_upper_time,
        }

        super().__init__(fixed_factors, oracle_fixed_factors)
        # Instantiate oracle with fixed factors and over-riden defaults.
        self.oracle = ChessMatchmaking(self.oracle_fixed_factors)
        self.dim = 1
        self.oracle_decision_factors = set("max_diff_decision")

    def check_initial_solution(self):
        if len(self.factors["initial_solution"]) != self.dim:
            return False
        else:
            return True

    def check_budget(self):
        return self.factors["budget"] > 0

    def check_upper_time(self):
        return len(self.factors["upper_time"]) == self.dim

    def check_simulatable_factors(self):
        if len(self.lower_bounds) != self.dim:
            return False
        elif len(self.upper_bounds) != self.dim:
            return False
        else:
            return True


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
            "max_diff_decision": vector[:]
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
        vector = tuple(factor_dict["max_diff_decision"])
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
        objectives = (response_dict["exp_diff"],)
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
        stoch_constraints = (response_dict["exp_wait_time"],)
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
        det_stoch_constraints = (-self.factors["upper_time"],)
        det_stoch_constraints_gradients = (0) # tuple of tuples – of sizes self.dim by self.dim, full of zeros
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
        det_objectives = (0,)
        det_objectives_gradients = None
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
        return np.all(x >= 0)

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
        x = tuple(rand_sol_rng.randint(0, 10000))
        return x