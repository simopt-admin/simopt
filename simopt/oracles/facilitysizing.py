"""
Summary
-------
Simulate a facilitysize problem.
"""
from base import Oracle
import numpy as np

class facilitysize(Oracle):
    """
    An oracle that simulates a facilitysize problem with a 
    multi-variate normal distribution. 
    Returns the probability of violating demand in each scenario.
    Attributes
    ----------
    n_rngs : int
        number of random-number generators used to run a simulation replication
    rng_list : list of rng.MRG32k3a objects
        list of random-number generators used to run a simulation replication
    n_responses : int
        number of responses (performance measures)
    factors : dict
        changeable factors of the simulation model
    specifications : dict
        details of each factor (for GUI and data validation)

    Arguments
    ---------
    fixed_factors : nested dict
        fixed factors of the simulation model
        
    See also
    --------
    base.Oracle
    """
    def __init__(self, fixed_factors={}):
        self.n_rngs = 1
        self.n_responses = 3
        self.specifications = {
            "mean_vec": {
                "description": "Location parameters of the multivariate normal distribution",
                "datatype": list,
                "default": [10, 230, 221]
            },
            "cov": {
                "description": "Covariance of multivariate normal distribution.",
                "datatype": list,
                "default": [[2276, 1508, 813], [1508, 2206, 1349], [813, 1349, 1865]]
            },
            "capa": {
                "description": "Capacity.",
                "datatype": list,
                "default": [150, 300, 400]
            },
            "number_of_facility": {
                "description": "The number of facilities.",
                "datatype": int,
                "default": 3
            }
        }
        self.check_factor_list = {
            "mean_vec": self.check_mean_vec,
            "capa": self.check_capa,
            "number_of_facility": self.check_number_of_facility            
        }
        # set factors of the simulation oracle
        super().__init__(fixed_factors)
        
    def check_mean_vec(self):
        return np.all(self.factors["mean_vec"]) > 0
    
    def check_capa(self):
        return len(self.factors["capa"]) == self.factors["number_of_facility"]
    
    def check_number_of_facility(self):
        return self.factors["number_of_facility"] > 0
    
    def check_simulatable_factors(self):
        return True
    

    def replicate(self):
        """
        Simulate a single replication for the current oracle factors.

        Returns
        -------
        responses : dict
            performance measures of interest
            "stock_out_flag" = the binary variable
                 0 : all facilities satisfy the demand
                 1 : one of the facilities did not satisfy the demand
            "count" = the number of facilities which cannot satisfy the demand
            "number_of_cut" = the number of toal demand which cannot be satisfied
        gradients : dict of dicts
            gradient estimates for each response
        """

        # designate separate random number generators
        demand_rng = self.rng_list[0]
        stock_out_flag = 0
        count = 0
        number_of_cut = 0
        
        # generate demands
        demand = demand_rng.mvnormalvariate(self.factors["mean_vec"], self.factors["cov"],0)
        while np.any(demand < 0):
            demand = demand_rng.mvnormalvariate(self.factors["mean_vec"], self.factors["cov"],0)

        # check 
        for i in range(self.factors["number_of_facility"]):
            if demand[i] > self.factors["capa"][i]:
                count = count + 1
                stock_out_flag = 1
                number_of_cut += demand[i] - self.factors["capa"][i]
                
        # reponse        
        responses = {'stock_out_flag': stock_out_flag,
                    'count': count,
                    'number_of_cut': number_of_cut} 
        
        # compose gradients
        gradients = {response_key: {factor_key: np.nan for factor_key in self.specifications} for response_key in responses}

        return responses, gradients