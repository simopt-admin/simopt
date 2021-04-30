"""
Summary
-------
Simulate a multi-stage revenue management with inter-temporal dependence problem
"""
from base import Oracle
import numpy as np

class rmitd(Oracle):
    """
    An oracle that simulates a multi-stage revenue management with
    inter-temporal dependence problem. 
    Returns the total revenue.
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
        self.n_rngs = 2
        self.n_responses = 1
        self.specifications = {
            "T": {
                "description": "Period",
                "datatype": int,
                "default": 3
            },
            "price": {
                "description": "Price",
                "datatype": list,
                "default": [100, 300, 400]
            },
            "meanDemand": {
                "description": "Mean demand(Mu_t).",
                "datatype": list,
                "default": [50, 20, 30]
            },
            "cost": {
                "description": "The cost per unit of capacity at t = 0.",
                "datatype": float,
                "default": 80
            },
            "k": {
                "description": "The shape of the gamma distribution",
                "datatype": float,
                "default": 1
            },
            "theta": {
                "description": "The scale of the gamma distribution",
                "datatype": float,
                "default": 1
            },
            "b_r": {
                "description": "Vector of reservations for each period, in order [b r_2 r_3 ... r_T]",
                "datatype": int,
                "default": [100, 50, 30]
            }
        }
        self.check_factor_list = {
            "b_r": self.check_r
        }
        # set factors of the simulation oracle
        super().__init__(fixed_factors)
        
    def check_initial_solution(self):
        return np.all(self.factors["b_r"]) > 0
    
    def check_r(self):
        return self.factors["b_r"][0] >= sum(self.factors["b_r"][1:])
    
    def check_simulatable_factors(self):
        return True
    

    def replicate(self):
        """
        Simulate a single replication for the current oracle factors.

        Returns
        -------
        responses : dict
            performance measures of interest
            "revenue" = total revenue
        gradients : dict of dicts
            gradient estimates for each response
        """

        revenue = 0
        b = self.factors["b_r"][0]
        r = self.factors["b_r"][1:]
        r.append(0)
        
        # designate separate random number generators
        x_rng = self.rng_list[0]
        y_rng = self.rng_list[1]
        
        # generate x and y 
        x = x_rng.gammavariate(self.factors["k"],self.factors["theta"])
        y = [y_rng.expovariate(1) for _ in range(self.factors["T"])]
        
        remainingCapacity = b
        # calculate the revenue
        for i in range(self.factors["T"]):
            D_t = self.factors["meanDemand"][i]*x*y[i]
            sell = min(max(remainingCapacity-r[i],0),D_t)
            remainingCapacity = remainingCapacity - sell
            revenue += sell*self.factors["price"][i]
        
        revenue -= self.factors["cost"]*b
        
        # reponse        
        responses = {'revenue': revenue}
        
        # compose gradients
        gradients = {response_key: {factor_key: np.nan for factor_key in self.specifications} for response_key in responses}

        return responses, gradients
