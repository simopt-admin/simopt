"""
Summary
-------
Simulate multiple periods worth of sales for a (s,S) inventory problem
with continuous inventory.
"""
from base import Oracle
import numpy as np


class SSCont(Oracle):
    """
    An oracle that simulates multiple periods' worth of sales for a (s,S)
    inventory problem with continuous inventory, exponentially distributed
    demand, and poisson distributed lead time. Returns the average cost per
    period after accounting for fixed and variable order costs, order rate,
    stockout rate, fraction of demand met with inventory on hand, average
    amount backordered given a stockout occured, and average amount ordered
    given an order occured.

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
        self.name = "SSCONT"
        self.n_rngs = 2
        self.n_responses = 7
        self.factors = fixed_factors
        self.specifications = {
            "demand_mean": {
                "description": "Mean of exponentially distributed demand in each period.",
                "datatype": float,
                "default": 100.0
            },
            "lead_mean": {
                "description": "Mean of Poisson distributed order lead time.",
                "datatype": float,
                "default": 6.0
            },
            "holding_cost": {
                "description": "Holding cost per unit per period.",
                "datatype": float,
                "default": 1.0
            },
            "fixed_cost": {
                "description": "Order fixed cost.",
                "datatype": float,
                "default": 36.0
            },
            "variable_cost": {
                "description": "Order variable cost per unit.",
                "datatype": float,
                "default": 2.0
            },
            "s": {
                "description": "Inventory threshold for placing order.",
                "datatype": float,
                "default": 1000.0
            },
            "S": {
                "description": "Max inventory.",
                "datatype": float,
                "default": 2000.0
            },
            "n_days": {
                "description": "Number of periods to simulate.",
                "datatype": int,
                "default": 100
            },
            "warmup": {
                "description": "Number of periods as warmup before collecting statistics.",
                "datatype": int,
                "default": 20
            }
        }
        self.check_factor_list = {
            "demand_mean": self.check_demand_mean,
            "lead_mean": self.check_lead_mean,
            "holding_cost": self.check_holding_cost,
            "fixed_cost": self.check_fixed_cost,
            "variable_cost": self.check_variable_cost,
            "s": self.check_s,
            "S": self.check_S,
            "n_days": self.check_n_days,
            "warmup": self.check_warmup
        }
        # Set factors of the simulation oracle.
        super().__init__(fixed_factors)

    # Check for simulatable factors
    def check_demand_mean(self):
        return self.factors["demand_mean"] > 0

    def check_lead_mean(self):
        return self.factors["lead_mean"] > 0

    def check_holding_cost(self):
        return self.factors["holding_cost"] > 0

    def check_fixed_cost(self):
        return self.factors["fixed_cost"] > 0

    def check_variable_cost(self):
        return self.factors["variable_cost"] > 0

    def check_s(self):
        return self.factors["s"] > 0

    def check_S(self):
        return self.factors["S"] > 0

    def check_n_days(self):
        return self.factors["n_days"] >= 1

    def check_warmup(self):
        return self.factors["warmup"] >= 0

    def check_simulatable_factors(self):
        return self.factors["s"] < self.factors["S"]

    def replicate(self, rng_list):
        """
        Simulate a single replication for the current oracle factors.
        
        Arguments
        ---------
        rng_list : list of rng.MRG32k3a objects
            rngs for oracle to use when simulating a replication

        Arguments
        ---------
        rng_list : list of rng.MRG32k3a objects
            rngs for oracle to use when simulating a replication

        Returns
        -------
        responses : dict
            performance measures of interest
            "avg_order_costs" = average order costs per period
            "avg_holding_costs" = average holding costs per period
            "on_time_rate" = fraction of demand met with stock on hand in store
            "order_rate" = fraction of periods an order was made
            "stockout_rate" = fraction of periods a stockout occured
            "avg_stockout" = mean amount of product backordered given a stockout occured
            "avg_order" = mean amount of product ordered given an order occured
        """
        # Designate random number generators.
        demand_rng = rng_list[0]
        lead_rng = rng_list[1]
        # Generate exponential random demands.
        demands = [demand_rng.expovariate(1/self.factors["demand_mean"]) for _ in range(self.factors["n_days"] + self.factors["warmup"])]
        # Initialize starting and ending inventories for each period.
        start_inv = np.zeros(self.factors["n_days"] + self.factors["warmup"])
        start_inv[0] = self.factors["s"]  # Start with s units at period 0.
        end_inv = np.zeros(self.factors["n_days"] + self.factors["warmup"])
        # Initialize other quantities to track:
        #   - Amount of product to be received in each period.
        #   - Inventory position each period.
        #   - Amount of product ordered in each period.
        #   - Amount of product outstanding in each period.
        orders_received = np.zeros(self.factors["n_days"] + self.factors["warmup"])
        inv_pos = np.zeros(self.factors["n_days"] + self.factors["warmup"])
        orders_placed = np.zeros(self.factors["n_days"] + self.factors["warmup"])
        orders_outstanding = np.zeros(self.factors["n_days"] + self.factors["warmup"])
        # Run simulation over time horizon.
        for day in range(self.factors["n_days"] + self.factors["warmup"]):
            # Calculate end-of-period inventory on hand and inventory position.
            end_inv[day] = start_inv[day] - demands[day]
            inv_pos[day] = end_inv[day] + orders_outstanding[day]
            # Place orders, keeping track of outstanding orders and when they will be received.
            orders_placed[day] = np.max(((inv_pos[day] < self.factors["s"]) * (self.factors["S"] - inv_pos[day])), 0)
            if orders_placed[day] > 0:
                lead = lead_rng.poissonvariate(self.factors["lead_mean"])
                for future_day in range(day + 1, lead + 1):
                    if future_day <= self.factors["n_days"] + self.factors["warmup"]:
                        orders_outstanding[future_day] = orders_outstanding[future_day] + orders_placed[day]
                if day + lead + 1 < self.factors["n_days"] + self.factors["warmup"]:
                    orders_received[day + lead + 1] = orders_received[day + lead + 1] + orders_placed[day]
            # Calculate starting inventory for next period.
            if day < self.factors["n_days"] + self.factors["warmup"] - 1:
                start_inv[day + 1] = end_inv[day] + orders_received[day + 1]
        # Calculate responses from simulation data.
        order_rate = np.mean(orders_placed[self.factors["warmup"]:] > 0)
        stockout_rate = np.mean(end_inv[self.factors["warmup"]:] < 0)
        avg_order_costs = np.mean(self.factors["fixed_cost"] * (orders_placed[self.factors["warmup"]:] > 0) +
                                  self.factors["variable_cost"] * orders_placed[self.factors["warmup"]:])
        avg_holding_costs = np.mean(self.factors["holding_cost"] * end_inv[self.factors["warmup"]:] * [end_inv[self.factors["warmup"]:] > 0])
        on_time_rate = 1 + np.sum(end_inv[self.factors["warmup"]:]
                                  [np.where(end_inv[self.factors["warmup"]:] < 0)])/np.sum(demands[self.factors["warmup"]:])
        if np.array(np.where(end_inv[self.factors["warmup"]:] < 0)).size == 0:
            avg_stockout = 0
        else:
            avg_stockout = -np.mean(end_inv[self.factors["warmup"]:][np.where(end_inv[self.factors["warmup"]:] < 0)])
        if np.array(np.where(orders_placed[self.factors["warmup"]:] > 0)).size == 0:
            avg_order = 0
        else:
            avg_order = np.mean(orders_placed[self.factors["warmup"]:][np.where(orders_placed[self.factors["warmup"]:] > 0)])
        # Compose responses and gradients.
        responses = {"avg_order_costs": avg_order_costs,
                     "avg_holding_costs": avg_holding_costs,
                     "on_time_rate": on_time_rate,
                     "order_rate": order_rate,
                     "stockout_rate": stockout_rate,
                     "avg_stockout": avg_stockout,
                     "avg_order": avg_order
                     }
        gradients = {response_key: {factor_key: np.nan for factor_key in self.specifications} for response_key in responses}
        return responses, gradients
