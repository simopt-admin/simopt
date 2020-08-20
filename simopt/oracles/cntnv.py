"""
Summary
-------
Simulate a Continuous Newsvendor Problem.
"""

from base import Oracle
import numpy as np

class CntNV(Oracle):
    """
    An oracle that simulates an Continuous Newsvendor Problem with a 
    Burr Type XII distribution. Returns the profit in each scenario.

    Attributes
    ----------
    n_rngs : int
        number of random-number generators used to run a simulation replication
    n_responses : int
        number of responses (performance measures)
    factors : dict
        changeable factors of the simulation model
    specifications : dict
        details of each factor (for GUI and data validation)

    Arguments
    ---------
    noise_factors : nested dict
        noise_factors of the simulation model

    See also
    --------
    base.Oracle
    """
    def __init__(self, noise_factors={}):
        self.n_rngs = 1
        self.n_responses = 1 # just profits
        self.factors = noise_factors
        self.specifications = {
            "purchase_price": {
                "description": "Purchasing Cost per unit",
                "datatype": float
            },
            "selling_price": {
                "description": "Selling Price per unit",
                "datatype": float
            },
            "salvage_price": {
                "description": "Salvage cost per unit",
                "datatype": float
            },
            "order_quantity":{
                "description": "Order quantity"
            },
            "alpha": {
                "description": "Burr Type XII cdf parameters", # Shape parameter?
                "datatype": float
            },
            "beta": {
                "description": "Burr Type XII cdf parameters", # Scale parameter?
                "datatype": float
            } 
        } 

        self.check_factor_list = {
        "purchase_price": self.check_purchase_price(),
        "selling_price": self.check_selling_price(),
        "salvage_price": self.check_salvage_price(),
        "order_quantity": self.check_order_quantity(),
        "alpha": self.check_alpha(),
        "beta": self.check_beta()
        }

    # Check for simulatable factors
    def check_purchase_price(self):
        return self.factors["purchase_price"] > 0

    def check_selling_price(self):
        return self.factors["selling_price"] > 0

    def check_salvage_price(self):
        return self.factors["salvage_price"] > 0

    def check_order_quantity(self):
        return self.factors["order_quantity"] > 0

    def check_alpha(self):
        return True

    def check_beta(self):
        return True

    # Do we need simulatable factors for NV?
    # salvage price < cost price < selling price????
    def check_simulatable_factors(self):
        return self.factors["salvage_price"] < self.factors["purchase_price"] < self.factors["selling_price"]

    # Will replicate function run mutiple times?
    # Do we even need replicate for NV since there is an exact solution?
    def replicate(self, decision_factors, rng_list):

         """
        Simulate a single replication at solution described by `decision_factors`.

        Arguments
        ---------
        decision_factors : dict
            decision factors of the simulation model

        Returns
        -------
        responses : dict
            performance measures of interest
            "profit" = profit in this scenario
        """

        # Update factors with user input
        self.factors.update(decision_factors)
        demand_rng = rng_list[0] 

        # Generate Burr Type XII Random number using: ((1-rand(1)).^(-1/beta)-1).^(1/alpha)
        demand = ((1-demand_rng.random())**(-1/self.factors["beta"])-1)**(1/self.factors["alpha"])

        # profit 
        profit = -1*self.factors["purchase_price"]*self.factors["order_quantity"] + min(demand, self.factors["order_quantity"])*self.factors["selling_price"] + max(0, self.factors["order_quantity"]-demand)*self.factors["salvage_price"]
        response = {'profit': profit} 

        return response