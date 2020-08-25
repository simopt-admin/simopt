"""
Summary
-------
Simulate a day's worth of sales for a newsvendor.
"""
from base import Oracle
import numpy as np

class CntNV(Oracle):
    """
    An oracle that simulates a day's worth of sales for a newsvendor 
    with a Burr Type XII demand distribution. Returns the profit, after
    accounting for order costs and salvage.

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
    noise_factors : nested dict
        noise_factors of the simulation model

    See also
    --------
    base.Oracle
    """
    def __init__(self, noise_factors={}):
        self.n_rngs = 1
        self.n_responses = 1
        self.factors = noise_factors
        self.specifications = {
            "purchase_price": {
                "description": "Purchasing Cost per unit",
                "datatype": float
            },
            "sales_price": {
                "description": "Sales Price per unit",
                "datatype": float
            },
            "salvage_price": {
                "description": "Salvage cost per unit",
                "datatype": float
            },
            "order_quantity":{
                "description": "Order quantity",
                "datatype": float # or int
            },
            "Burr_c": {
                "description": "Burr Type XII cdf shape parameter",
                "datatype": float
            },
            "Burr_k": {
                "description": "Burr Type XII cdf shape parameter",
                "datatype": float
            } 
        }
        self.check_factor_list = {
            "purchase_price": self.check_purchase_price,
            "sales_price": self.check_sales_price,
            "salvage_price": self.check_salvage_price,
            "order_quantity": self.check_order_quantity,
            "Burr_c": self.check_Burr_c,
            "Burr_k": self.check_Burr_k
        }

    # Check for simulatable factors
    def check_purchase_price(self):
        return self.factors["purchase_price"] > 0

    def check_sales_price(self):
        return self.factors["sales_price"] > 0

    def check_salvage_price(self):
        return self.factors["salvage_price"] > 0

    def check_order_quantity(self):
        return self.factors["order_quantity"] > 0

    def check_Burr_c(self):
        return self.factors["Burr_c"] > 0

    def check_Burr_k(self):
        return self.factors["Burr_k"] > 0

    def check_simulatable_factors(self):
        return self.factors["salvage_price"] < self.factors["purchase_price"] < self.factors["sales_price"]

    def replicate(self, decision_factors):
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
        # update factors with user input
        self.factors.update(decision_factors)
        # designate random number generator
        demand_rng = self.rng_list[0]
        # generate Burr Type XII random demand
        demand = ((1-demand_rng.random())**(-1/self.factors["Burr_k"])-1)**(1/self.factors["Burr_c"])
        # calculate profit 
        profit = -1*self.factors["purchase_price"]*self.factors["order_quantity"] + min(demand, self.factors["order_quantity"])*self.factors["sales_price"] + max(0, self.factors["order_quantity"]-demand)*self.factors["salvage_price"]
        # calculate gradient of profit w.r.t. order quantity
        if demand > self.factors["order_quantity"]:
            grad_profit_order_quantity = self.factors["sales_price"] - self.factors["purchase_price"]
        elif demand < self.factors["order_quantity"]:
            grad_profit_order_quantity = self.factors["salvage_price"] - self.factors["purchase_price"]
        else:
            grad_profit_order_quantity = np.nan
        # return profit w/ gradient estimate
        responses = {"profit": profit}
        gradients = {
            "profit": {
                "purchase_price": np.nan,
                "sales_price": np.nan,
                "salvage_price": np.nan,
                "order_quantity": grad_profit_order_quantity,
                "Burr_c": np.nan,
                "Burr_k": np.nan
            }
        }
        return responses, gradients