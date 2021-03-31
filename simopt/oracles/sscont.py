"""
Summary
-------
Simulate multiple periods worth of sales for a (s,S) inventory problem with continuous inventory.
"""
from base import Oracle
import numpy as np

class SSCont(Oracle):
    """
    An oracle that simulates multiple periods worth of sales for a (s,S) inventory problem with continuous inventory,
    exponential demand distribution, and poisson lead time distribution. 
    Returns the average cost per period after accounting for fixed and variable order costs, order rate, stockout rate, fraction of demand
    met with inventory on hand, average amount backordered given a stockout occured, and average amount ordered given an order occured.

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
        self.n_rngs = 2
        self.n_responses = 6
        self.factors = fixed_factors
        self.specifications = {
            "demand_mean": {
                "description": "Mean of exponentially distributed demand in each period",
                "datatype": float,
                "default": 100.0
            },
            "lead_mean": {
                "description": "Mean of Poisson distributed order lead time",
                "datatype": float,
                "default": 6.0
            },
            "holding_cost": {
                "description": "Holding cost per unit per period",
                "datatype": float,
                "default": 1.0
            },
            "fixed_cost":{
                "description": "Order fixed cost",
                "datatype": float, 
                "default": 36.0
            },
            "variable_cost": {
                "description": "Order variable cost per unit",
                "datatype": float,
                "default": 2.0
            },
            "s": {
                "description": "Inventory threshold for placing order",
                "datatype": float,
                "default": 1000.0
            }, 
            "S": {
                "description": "Max inventory",
                "datatype": float,
                "default": 2000.0
            },
            "n_days": {
                "description": "number of periods to simulate",
                "datatype": int,
                "default": 10000
            },
            "warmup": {
                "description": "Number of periods as warmup before collecting statistics",
                "datatype": int,
                "default": 50
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
        # set factors of the simulation oracle
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

    def replicate(self):
        """
        Simulate a single replication for the current oracle factors.

        Returns
        -------
        responses : dict
            performance measures of interest
            "cost_mean" = average cost per period
            "on_time_rate" = fraction of demand met with stock on hand in store
            "order_rate" = fraction of periods an order was made
            "stockout_rate" = fraction of periods a stockout occured
            "avg_stockout" = mean amount of product backordered given a stockout occured
            "avg_order" = mean amount of product ordered given an order occured
        """
        # designate random number generators
        demand_rng = self.rng_list[0]
        lead_rng = self.rng_list[1]
        # generate exponential random demands and Poisson random lead times
        demands = [demand_rng.expovariate(1/self.factors["demand_mean"]) for _ in range(self.factors["n_days"] + self.factors["warmup"])]
        leads = [lead_rng.poissonvariate(self.factors["lead_mean"]) for _ in range(self.factors["n_days"] + self.factors["warmup"])]
       
        # starting inventory each period
        start_inv = np.zeros(self.factors["n_days"] + self.factors["warmup"])
        start_inv[0] = self.factors["s"] # what to start with?
        # ending inventory each period
        end_inv = np.zeros(self.factors["n_days"] + self.factors["warmup"])
        # amount of product to be received for each period
        orders_received = np.zeros(self.factors["n_days"] + self.factors["warmup"])
        # inventory position each period
        inv_pos = np.zeros(self.factors["n_days"] + self.factors["warmup"])
        # amount of product ordered each period
        orders_placed = np.zeros(self.factors["n_days"] + self.factors["warmup"])
        # amount of product outstanding each period
        orders_outstanding = np.zeros(self.factors["n_days"] + self.factors["warmup"])
        # cost each period
        cost = np.zeros(self.factors["n_days"] + self.factors["warmup"])
        # keeping track of how many orders have been made
        num_ord = 0
    
        # run simulation
        for i in range(self.factors["n_days"] + self.factors["warmup"]):
        
            # calculate end of period inventory on hand and inventory position
            end_inv[i] = start_inv[i] - demands[i]
            inv_pos[i] = end_inv[i] + orders_outstanding[i]
        
            # place orders, keep track of outstanding orders and when they will be received
            orders_placed[i] = np.max(((inv_pos[i] < self.factors["s"])*(self.factors["S"]-inv_pos[i])),0)
            if orders_placed[i]>0:
                for j in range(i+1,leads[num_ord]+1):
                    if j<=self.factors["n_days"] + self.factors["warmup"]:  
                        orders_outstanding[j] = orders_outstanding[j] + orders_placed[i]
                if i + leads[num_ord] + 1 < self.factors["n_days"] + self.factors["warmup"]:
                    orders_received[i+leads[num_ord]+1] = orders_received[i+leads[num_ord]+1] + orders_placed[i]
                num_ord = num_ord + 1
        
            # calculate starting inventory for next period
            if i < self.factors["n_days"] + self.factors["warmup"] - 1:
                start_inv[i+1] = end_inv[i] + orders_received[i+1]
        
        # calculate responses from simulation data
        order_rate = np.mean(orders_placed[self.factors["warmup"]:]>0)
        stockout_rate = np.mean(end_inv[self.factors["warmup"]:]<0)
        cost_mean = np.mean(self.factors["fixed_cost"]*(orders_placed[self.factors["warmup"]:]>0) +
                            self.factors["variable_cost"]*orders_placed[self.factors["warmup"]:] + 
                            self.factors["holding_cost"]*end_inv[self.factors["warmup"]:]*[end_inv[self.factors["warmup"]:]>0])
        on_time_rate = 1 + np.sum(end_inv[self.factors["warmup"]:]
                               [np.where(end_inv[self.factors["warmup"]:]<0)])/np.sum(demands[self.factors["warmup"]:])
        avg_stockout = -np.mean(end_inv[self.factors["warmup"]:][np.where(end_inv[self.factors["warmup"]:]<0)])
        avg_order = np.mean(orders_placed[self.factors["warmup"]:][np.where(orders_placed[self.factors["warmup"]:]>0)]) 
            
        # compose responses
        responses = {"cost_mean": cost_mean, "on_time_rate": on_time_rate, "order_rate": order_rate, "stockout_rate": stockout_rate,
                     "avg_stockout": avg_stockout, "avg_order": avg_order}
        
        # return responses
        return responses