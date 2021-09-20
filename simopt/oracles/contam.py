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
        self.n_responses = 3
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
            "prev_cost": {
                "description": "Cost of prevention.",
                "datatype": list,
                "default": [0, 0, 0, 0, 0]
            },
            "error_prob": {
                "description": "Error probability.",
                "datatype": list,
                "default": [0.05, 0.05, 0.05, 0.05, 0.05]
            },
            "upper_thres": {
                "description": "Upper limit of amount of contamination.",
                "datatype": float,
                "default": 0.1
            },
            "init_contam": {
                "description": "Initial contamination fraction.",
                "datatype": float,
                "default": 0
            },
            "stages": {
                "description": "Stage of food supply chain.",
                "datatype": int,
                "default": 5
            }
        }
        self.check_factor_list = {
            "contam_rate_alpha": self.check_contam_rate_alpha,
            "contam_rate_beta": self.check_contam_rate_beta,
            "restore_rate_alpha": self.check_restore_rate_alpha,
            "restore_rate_beta": self.check_restore_rate_beta,
            "prev_cost": self.check_prev_cost,
            "error_prob": self.check_error_prob,
            "upper_thres": self.check_upper_thres,
            "init_contam": self.check_init_contam,
            "stages": self.check_stages
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

    def check_prev_cost(self):
        return all(cost > 0 for cost in self.factors["prev_cost"])

    def check_error_prob(self):
        return all(error > 0 for error in self.factors["error_prob"])
    
    def check_upper_thres(self):
        return self.factors["upper_thres"] > 0
    
    def check_init_contam(self):
        return self.factors["init_contam"] > 0

    def check_stages(self):
        return self.factors["stages"] > 0

    def check_simulatable_factors(self):
        # Check for matching number of stages.
        if len(self.factors["prev_cost"]) != self.factors["stages"]:
            return False
        elif len(self.factors["error_prob"]) != self.factors["stages"]:
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
            TBD
        gradients : dict of dicts
            gradient estimates for each response
        """
        # Designate separate random number generators.
        # Outputs will be coupled when generating demand.
        contam_rng = rng_list[0]
        restore_rng = rng_list[1]
        # Generate rates with beta distribution.
        c = contam_rng.betavariate(alpha=self.factors["contam_rate_alpha"], beta=self.factors["contam_rate_beta"])
        r = restore_rng.betavariate(alpha=self.factors["restore_rate_alpha"], beta=self.factors["restore_rate_beta"])
        # Compose responses and gradients.
        responses = {'testing': 0}
        gradients = {response_key: {factor_key: np.nan for factor_key in self.specifications} for response_key in responses}
        return responses, gradients
