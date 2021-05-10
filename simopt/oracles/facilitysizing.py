"""
Summary
-------
Simulate demand at facilities.
"""
from base import Oracle
import numpy as np


class FacilitySize(Oracle):
    """
    An oracle that simulates a facilitysize problem with a
    multi-variate normal distribution.
    Returns the probability of violating demand in each scenario.

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
        self.name = "FACSIZE"
        self.n_rngs = 1
        self.n_responses = 3
        self.specifications = {
            "mean_vec": {
                "description": "Location parameters of the multivariate normal distribution",
                "datatype": list,
                "default": [100, 100, 100]
            },
            "cov": {
                "description": "Covariance of multivariate normal distribution.",
                "datatype": list,
                "default": [[2000, 1500, 500], [1500, 2000, 750], [500, 750, 2000]]
            },
            "capacity": {
                "description": "Capacity.",
                "datatype": list,
                "default": [150, 300, 400]
            },
            "n_fac": {
                "description": "The number of facilities.",
                "datatype": int,
                "default": 3
            }
        }
        self.check_factor_list = {
            "mean_vec": self.check_mean_vec,
            "cov": self.check_cov,
            "capacity": self.check_capacity,
            "n_fac": self.check_n_fac
        }
        # Set factors of the simulation oracle.
        super().__init__(fixed_factors)

    def check_mean_vec(self):
        return np.all(self.factors["mean_vec"]) > 0

    def check_cov(self):
        try:
            np.linalg.cholesky(np.matrix(self.factors["cov"]))
            return True
        except np.linalg.linalg.LinAlgError as err:
            if 'Matrix is not positive definite' in err.message:
                return False
            else:
                raise
        return

    def check_capacity(self):
        return len(self.factors["capacity"]) == self.factors["n_fac"]

    def check_n_fac(self):
        return self.factors["n_fac"] > 0

    def check_simulatable_factors(self):
        if len(self.factors["capacity"]) != self.factors["n_fac"]:
            return False
        elif len(self.factors["mean_vec"]) != self.factors["n_fac"]:
            return False
        elif len(self.factors["cov"]) != self.factors["n_fac"]:
            return False
        elif len(self.factors["cov"][0]) != self.factors["n_fac"]:
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
            "stockout_flag" = a binary variable
                 0 : all facilities satisfy the demand
                 1 : at least one of the facilities did not satisfy the demand
            "n_fac_stockout" = the number of facilities which cannot satisfy the demand
            "n_cut" = the number of toal demand which cannot be satisfied
        gradients : dict of dicts
            gradient estimates for each response
        """
        # Designate RNG for demands.
        demand_rng = rng_list[0]
        stockout_flag = 0
        n_fac_stockout = 0
        n_cut = 0
        # Generate random demands at facilities from truncated multivariate normal distribution.
        demand = demand_rng.mvnormalvariate(self.factors["mean_vec"], self.factors["cov"], factorized=False)
        while np.any(demand < 0):
            demand = demand_rng.mvnormalvariate(self.factors["mean_vec"], self.factors["cov"], factorized=False)
        # Check for stockouts.
        for i in range(self.factors["n_fac"]):
            if demand[i] > self.factors["capacity"][i]:
                n_fac_stockout = n_fac_stockout + 1
                stockout_flag = 1
                n_cut += demand[i] - self.factors["capacity"][i]
        # Compose responses and gradients.
        responses = {'stockout_flag': stockout_flag,
                     'n_fac_stockout': n_fac_stockout,
                     'n_cut': n_cut}
        gradients = {response_key: {factor_key: np.nan for factor_key in self.specifications} for response_key in responses}
        return responses, gradients
