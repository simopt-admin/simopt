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
            }
        }
        self.check_factor_list = {
            "poisson_rate": self.check_poisson_rate,
            "num_players": self.check_num_players,
            "width": self.check_width
        }
        # Set factors of the simulation oracle.
        super().__init__(fixed_factors)

    def check_poisson_rate(self):
        return self.factors["poisson_rate"] > 0

    def check_num_players(self):
        return self.factors["num_players"] > 0

    def check_width(self):
        return self.factors["width"] > 0

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

        wait_times = 10000*np.ones(self.factors["num_players"])
        waiting_players = []
        total_diff = 0

        for i in range(self.factors["num_players"]):
          player = elo_rng.normalvariate(self.factors["elo_mean"], self.factors["elo_sd"])
          while np.any(player < 0) or np.any(player > 2400):
              player = elo_rng.normalvariate(self.factors["elo_mean"], self.factors["elo_sd"])
          time = arrival_rng.poissonvariate(self.factors["poisson_rate"])
          old_total = total_diff
          for p in range(len(waiting_players)):
            if abs(player - p) <= self.factors["width"]:
              del waiting_players[p]
              total_diff += abs(player - p)
              break
            else:
              wait_times[p] += time
          if old_total == total_diff:
            waiting_players.append(player)

        # Compose responses and gradients.
        responses = {
          'exp_diff': total_diff/self.factors["num_players"],
          'exp_wait_time': np.mean(wait_times)
        }
        gradients = {response_key: {factor_key: np.nan for factor_key in self.specifications} for response_key in responses}
        return responses, gradients