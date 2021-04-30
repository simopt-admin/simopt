"""
Summary
-------
Simulate a day's worth of sales for a newsvendor.
"""
import numpy as np

from base import Oracle


class CntNV(Oracle):
    """
    An oracle that simulates a day's worth of sales for a newsvendor
    with a Burr Type XII demand distribution. Returns the profit, after
    accounting for order costs and salvage.

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
        details of each factor (for GUI, data validation, and defaults)
    check_factor_list : dict
        switch case for checking factor simulatability

    Arguments
    ---------
    fixed_factors : dict
        fixed_factors of the simulation model

    See also
    --------
    base.Oracle
    """
    def __init__(self, fixed_factors={}):
        self.name = "CNTNEWS"
        self.n_rngs = 1
        self.n_responses = 1
        self.factors = fixed_factors
        self.specifications = {
            "purchase_price": {
                "description": "Purchasing Cost per unit",
                "datatype": float,
                "default": 5.0
            },
            "sales_price": {
                "description": "Sales Price per unit",
                "datatype": float,
                "default": 9.0
            },
            "salvage_price": {
                "description": "Salvage cost per unit",
                "datatype": float,
                "default": 1.0
            },
            "order_quantity": {
                "description": "Order quantity",
                "datatype": float,  # or int
                "default": 0.5
            },
            "Burr_c": {
                "description": "Burr Type XII cdf shape parameter",
                "datatype": float,
                "default": 2.0
            },
            "Burr_k": {
                "description": "Burr Type XII cdf shape parameter",
                "datatype": float,
                "default": 20.0
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
        # Set factors of the simulation oracle.
        super().__init__(fixed_factors)

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
        return (self.factors["salvage_price"]
                < self.factors["purchase_price"]
                < self.factors["sales_price"])

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
            "profit" = profit in this scenario
        """
        # Designate random number generator for demand variability.
        demand_rng = rng_list[0]
        # Generate random demand according to Burr Type XII distribution.
        # If U ~ Uniform(0,1) and the Burr Type XII has parameters c and k,
        #   X = ((1-U)**(-1/k - 1))**(1/c) has the desired distribution.
        base = ((1 - demand_rng.random())**(-1 / self.factors["Burr_k"]) - 1)
        exponent = (1 / self.factors["Burr_c"])
        demand = base**exponent
        # Calculate profit.
        order_cost = (self.factors["purchase_price"]
                      * self.factors["order_quantity"])
        sales_revenue = (min(demand, self.factors["order_quantity"])
                         * self.factors["sales_price"])
        salvage_revenue = (max(0, self.factors["order_quantity"] - demand)
                           * self.factors["salvage_price"])
        profit = sales_revenue + salvage_revenue - order_cost
        # Calculate gradient of profit w.r.t. order quantity.
        if demand > self.factors["order_quantity"]:
            grad_profit_order_quantity = (self.factors["sales_price"]
                                          - self.factors["purchase_price"])
        elif demand < self.factors["order_quantity"]:
            grad_profit_order_quantity = (self.factors["salvage_price"]
                                          - self.factors["purchase_price"])
        else:
            grad_profit_order_quantity = np.nan
        # Compose responses and gradients.
        responses = {"profit": profit}
        gradients = {response_key:
                     {factor_key: np.nan for factor_key in self.specifications}
                     for response_key in responses
                     }
        gradients["profit"]["order_quantity"] = grad_profit_order_quantity
        return responses, gradients
