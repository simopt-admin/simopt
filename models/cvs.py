"""
Summary
-------
Simulate multiple periods worth of sales for a (s,S) inventory problem
with continuous inventory for multiple stores with its own demand.

Simulate 
"""
import numpy as np
import itertools
import copy
from base import Model, Problem

class DC:
    """
    A class to represent a distribution center
    imperial: check if DC has an imperial (Boolean)
    totes_list: list of totes
    location: tuple of doubles (lat,long)
    """

    # Create incremental id
    id_iter = itertools.count()

    def __init__(self, dc_loc, dc_totes, dc_stores):
        self.dc_id = 0
        self.dc_loc = dc_loc
        self.dc_totes = dc_totes
        self.dc_stores = dc_stores

    def set_dc_totes(self, dc_totes):
        self.dc_totes = dc_totes 
    
    def get_dc_totes(self):
        return self.get_dc_totes

    def get_dc_loc(self):
        return self.dc_loc

    def set_location(self, lat, lon):
        self._location = (lat, lon)
    
    def get_dc_stores(self):
        return self.dc_stores

class Store:
    """
    A class to represent a store.
    store_id: int store id
    location: tuple of doubles (lat, long)
    totes_list: list of totes
    inventory_level: list of int - inventory of totes at store group by tote color (length of 4)
    demand: int mean demand
    """

    # Create incremental id
    id_iter = itertools.count()
    
    def __init__(self, store_loc, dc, inventory_level, demand):
        self.store_id = 0
        self.store_loc = store_loc
        self.dc = dc
        self.inventory_level = inventory_level
        self.empty_totes = copy.deepcopy(inventory_level)
        self.demand = demand

    def get_store_id(self):
        return self.store_id

    def set_store_id(self, id):
        self.store_id = id

    def get_totes_list(self):
        return self._totes_list

    def set_totes_list(self, ls):
        self._totes_list = ls

    def get_demand(self):
        return self._demand

    def set_demand(self,dem):
        self._demand = dem

class CVS(Model):
    """
    A model that simulates multiple periods' worth of sales for a (s,S)
    inventory problem with continuous inventory, uniformly distributed
    demand, and poisson distributed lead time. Returns the various types of
    average costs per period, order rate, stockout rate, fraction of demand
    met with inventory on hand, average amount backordered given a stockout
    occured, and average amount ordered given an order occured.

    Attributes
    ----------
    name : str
        name of model
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

    Parameters
    ----------
    fixed_factors : dict
        fixed_factors of the simulation model

        ``demand_mean``
            Mean of exponentially distributed demand in each period (`flt`)
        ``lead_mean``
            Mean of Poisson distributed order lead time (`flt`)
        ``backorder_cost``
            Cost per unit of demand not met with in-stock inventory (`flt`)
        ``holding_cost``
            Holding cost per unit per period (`flt`)
        ``fixed_cost``
            Order fixed cost (`flt`)
        ``variable_cost``
            Order variable cost per unit (`flt`)
        ``tote_order_lim``
            Max inventory position (`flt`)
        ``n_days``
            Number of periods to simulate (`int`)
        ``warmup``
            Number of periods as warmup before collecting statistics (`int`)
    See also
    --------
    base.Model
    """
    def __init__(self, fixed_factors={}):
        self.name = "CVS"
        self.n_rngs = 2
        self.n_responses = 7
        self.factors = fixed_factors
        self.specifications = {
            "dc_loc": {
                "description": "Location of the distribution centers.",
                "datatype": tuple,
                "default": ((30.216460, -95.454040),) # DC address: 16597 Interstate 45 South, Conroe, TX 77385
            }, 
            "store_loc": {
                "description": "Location of the stores.",
                "datatype": tuple,
                "default": ((29.49585,-98.554806),) #sample address: 4730 FREDERICKSBURG RD.
            },
            "init_inv_store": {
                "description": "Initial inventory of the stores (differs by demand type of the store).",
                "datatype": dict,
                "default": (100,)
            },
            "demand_store":{
                "description": "The average daily demand at each store.",
                "datatype": tuple,
                "default": (5,)
            },
            "tote_order_lim":{
                "description": "The order-up-to of totes at each store.",
                "datatype": tuple,
                "default": (200,)
            },
            "t_delivery": {
                "description": "Time between deliveries.",
                "datatype": float,
                "default": 7 * 24.0
            },
            "n_days": {
                "description": "Number of periods to simulate.",
                "datatype": int,
                "default": 300
            },
            "warmup": {
                "description": "Number of periods as warmup before collecting statistics.",
                "datatype": int,
                "default": 20
            }
        }
        self.check_factor_list = {
            "demand_store": self.check_demand_store,
            "backorder_cost": self.check_backorder_cost,
            "holding_cost": self.check_holding_cost,
            "fixed_cost": self.check_fixed_cost,
            "variable_cost": self.check_variable_cost,
            "S": self.check_S,
            "n_days": self.check_n_days,
            "warmup": self.check_warmup
        }
        # Set factors of the simulation model.
        super().__init__(fixed_factors)

    # Check for simulatable factors
    def check_demand_store(self):
        return self.factors["demand_store"] > 0

    def check_backorder_cost(self):
        return self.factors["backorder_cost"] > 0

    def check_holding_cost(self):
        return self.factors["holding_cost"] > 0

    def check_fixed_cost(self):
        return self.factors["fixed_cost"] > 0

    def check_variable_cost(self):
        return self.factors["variable_cost"] > 0

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
        Simulate a single replication for the current model factors.

        Arguments
        ---------
        rng_list : [list]  [rng.mrg32k3a.MRG32k3a]
            rngs for model to use when simulating a replication

        Returns
        -------
        responses : dict
            performance measures of interest

            ``avg_backorder_costs``
                average backorder costs per period
            ``avg_order_costs``
                average order costs per period
            ``avg_holding_costs``
                average holding costs per period
            ``on_time_rate``
                fraction of demand met with stock on hand in store
            ``order_rate``
                fraction of periods an order was made
            ``stockout_rate``
                fraction of periods a stockout occured
            ``avg_stockout``
                mean amount of product backordered given a stockout occured
            ``avg_order``
                mean amount of product ordered given an order occured
        """
        # Designate random number generators.
        demand_rng = rng_list[0]
        # lead_rng = rng_list[1]

        num_stores = len(self.factors["demand_store"])
        total_days = self.factors["n_days"] + self.factors["warmup"]
        # Generate uniform random demands.
        demands = []
        for i in range(num_stores):
            demands.extend([demand_rng.uniform(1, 2*self.factors["demand_store"][i]-1) for _ in range(total_days)])
        demands = np.array(demands, ndmin=2)
        # Initialize starting and ending inventories for each period.
        start_inv = np.zeros((num_stores, total_days))
        start_inv[:0] = self.factors["init_inv_store"]
        end_inv = np.zeros((num_stores, total_days))
        # Initialize other quantities to track:
        #   - Amount of product to be received in each period.
        #   - Inventory position each period.
        #   - Amount of product ordered in each period.
        #   - Amount of product outstanding in each period.
        orders_received = np.zeros((num_stores, total_days))
        inv_pos = np.zeros((num_stores, total_days))
        orders_placed = np.zeros((num_stores, total_days//int(self.factors["t_delivery"])))
        orders_outstanding = np.zeros((num_stores, total_days))
        # Run simulation over time horizon.
        for day in range(total_days):
            # Calculate end-of-period inventory on hand and inventory position.
            end_inv[:day] = start_inv[:day] - demands[:day]
            inv_pos[:day] = end_inv[:day] + orders_outstanding[:day]
            # Place orders, keeping track of outstanding orders and when they will be received.
            if day%self.factors["t_delivery"] == 0:
                orders_placed[:day//int(self.factors["t_delivery"])] = np.reshape(self.factors["tote_order_lim"], (num_stores,1)) - inv_pos[:day]
                # Calculate starting inventory for next period.
                if day < total_days - 1:
                    start_inv[:day + 1] = end_inv[:day] + orders_received[:day + 1]
            else:
                if day < total_days - 1:
                    start_inv[:day + 1] = end_inv[:day]
        # Calculate responses from simulation data.
        order_rate = np.mean(orders_placed[self.factors["warmup"]:] > 0)
        stockout_rate = np.mean(end_inv[self.factors["warmup"]:] < 0)
        avg_order_costs = np.mean(self.factors["fixed_cost"] * (orders_placed[self.factors["warmup"]:] > 0) +
                                  self.factors["variable_cost"] * orders_placed[self.factors["warmup"]:])
        avg_holding_costs = np.mean(self.factors["holding_cost"] * end_inv[self.factors["warmup"]:] * [end_inv[self.factors["warmup"]:] > 0])
        on_time_rate = 1 - np.sum(np.min(np.vstack((demands[self.factors["warmup"]:], demands[self.factors["warmup"]:] - start_inv[self.factors["warmup"]:])), axis=0)
                                  * ((demands[self.factors["warmup"]:] - start_inv[self.factors["warmup"]:]) > 0))/np.sum(demands[self.factors["warmup"]:])
        avg_backorder_costs = self.factors["backorder_cost"]*(1 - on_time_rate)*np.sum(demands[self.factors["warmup"]:])/float(self.factors["n_days"])
        if np.array(np.where(end_inv[self.factors["warmup"]:] < 0)).size == 0:
            avg_stockout = 0
        else:
            avg_stockout = -np.mean(end_inv[self.factors["warmup"]:][np.where(end_inv[self.factors["warmup"]:] < 0)])
        if np.array(np.where(orders_placed[self.factors["warmup"]:] > 0)).size == 0:
            avg_order = 0
        else:
            avg_order = np.mean(orders_placed[self.factors["warmup"]:][np.where(orders_placed[self.factors["warmup"]:] > 0)])
        # Compose responses and gradients.
        responses = {"avg_backorder_costs": avg_backorder_costs,
                     "avg_order_costs": avg_order_costs,
                     "avg_holding_costs": avg_holding_costs,
                     "on_time_rate": on_time_rate,
                     "order_rate": order_rate,
                     "stockout_rate": stockout_rate,
                     "avg_stockout": avg_stockout,
                     "avg_order": avg_order
                     }
        gradients = {response_key: {factor_key: np.nan for factor_key in self.specifications} for response_key in responses}
        return responses, gradients


"""
Summary
-------
Minimize the expected total cost for (s, S) inventory system.
"""


class CVSMinCost(Problem):
    """
    Class to make multi-echelon inventory simulation-optimization problems.

    Attributes
    ----------
    name : str
        name of problem
    dim : int
        number of decision variables
    n_objectives : int
        number of objectives
    n_stochastic_constraints : int
        number of stochastic constraints
    minmax : tuple of int (+/- 1)
        indicator of maximization (+1) or minimization (-1) for each objective
    constraint_type : str
        description of constraints types:
            "unconstrained", "box", "deterministic", "stochastic"
    variable_type : str
        description of variable types:
            "discrete", "continuous", "mixed"
    lower_bounds : tuple
        lower bound for each decision variable
    upper_bounds : tuple
        upper bound for each decision variable
    gradient_available : bool
        indicates if gradient of objective function is available
    optimal_value : float
        optimal objective function value
    optimal_solution : tuple
        optimal solution
    model : base.Model
        associated simulation model that generates replications
    model_default_factors : dict
        default values for overriding model-level default factors
    model_fixed_factors : dict
        combination of overriden model-level factors and defaults
    model_decision_factors : set of str
        set of keys for factors that are decision variables
    rng_list : [list]  [rng.mrg32k3a.MRG32k3a]
        list of RNGs used to generate a random initial solution
        or a random problem instance
    factors : dict
        changeable factors of the problem
            initial_solution : tuple
                default initial solution from which solvers start
            budget : int > 0
                max number of replications (fn evals) for a solver to take
    specifications : dict
        details of each factor (for GUI, data validation, and defaults)

    Arguments
    ---------
    name : str
        user-specified name of problem
    fixed_factors : dict
        dictionary of user-specified problem factors
    model_fixed factors : dict
        subset of user-specified non-decision factors to pass through to the model

    See also
    --------
    base.Problem
    """
    def __init__(self, name="CVS-1", fixed_factors={}, model_fixed_factors={}):
        self.name = name
        self.n_objectives = 1
        self.n_stochastic_constraints = 0
        self.minmax = (-1,)
        self.constraint_type = "box"
        self.variable_type = "continuous"

        self.gradient_available = False
        self.optimal_value = None
        self.optimal_solution = None
        self.model_default_factors = {}
        self.model_decision_factors = {"t_delivery", "tote_order_lim"}
        self.factors = fixed_factors
        self.specifications = {
            "initial_solution": {
                "description": "Initial solution from which solvers start.",
                "datatype": tuple,
                "default": (7*24.0, (7*5.0,))
            },
            "budget": {
                "description": "Max # of replications for a solver to take.",
                "datatype": int,
                "default": 1000
            }
        }
        self.check_factor_list = {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget
        }
        super().__init__(fixed_factors, model_fixed_factors)
        # Instantiate model with fixed factors and overwritten defaults.
        self.model = CVS(self.model_fixed_factors)
        self.dim = len(self.model_factors["demand_store"])
        self.lower_bounds = (0, (0,)*self.dim)
        self.upper_bounds = (np.inf, (np.inf,)*self.dim)
        

    def vector_to_factor_dict(self, vector):
        """
        Convert a vector of variables to a dictionary with factor keys

        Arguments
        ---------
        vector : tuple
            vector of values associated with decision variables

        Returns
        -------
        factor_dict : dict
            dictionary with factor keys and associated values
        """
        factor_dict = {
            "t_delivery": vector[0],
            "tote_order_lim": vector[1]
        }
        return factor_dict

    def factor_dict_to_vector(self, factor_dict):
        """
        Convert a dictionary with factor keys to a vector
        of variables.

        Arguments
        ---------
        factor_dict : dict
            dictionary with factor keys and associated values

        Returns
        -------
        vector : tuple
            vector of values associated with decision variables
        """
        vector = (factor_dict["t_delivery"], factor_dict["tote_order_lim"])
        return vector

    def response_dict_to_objectives(self, response_dict):
        """
        Convert a dictionary with response keys to a vector
        of objectives.

        Arguments
        ---------
        response_dict : dict
            dictionary with response keys and associated values

        Returns
        -------
        objectives : tuple
            vector of objectives
        """
        objectives = (response_dict["avg_backorder_costs"] + response_dict["avg_order_costs"] + response_dict["avg_holding_costs"],)
        return objectives

    def response_dict_to_stoch_constraints(self, response_dict):
        """
        Convert a dictionary with response keys to a vector
        of left-hand sides of stochastic constraints: E[Y] >= 0

        Arguments
        ---------
        response_dict : dict
            dictionary with response keys and associated values

        Returns
        -------
        stoch_constraints : tuple
            vector of LHSs of stochastic constraint
        """
        stoch_constraints = None
        return stoch_constraints

    def deterministic_objectives_and_gradients(self, x):
        """
        Compute deterministic components of objectives for a solution `x`.

        Arguments
        ---------
        x : tuple
            vector of decision variables

        Returns
        -------
        det_objectives : tuple
            vector of deterministic components of objectives
        det_objectives_gradients : tuple
            vector of gradients of deterministic components of objectives
        """
        det_objectives = None
        det_objectives_gradients = None
        return det_objectives, det_objectives_gradients

    def deterministic_stochastic_constraints_and_gradients(self, x):
        """
        Compute deterministic components of stochastic constraints
        for a solution `x`.

        Arguments
        ---------
        x : tuple
            vector of decision variables

        Returns
        -------
        det_stoch_constraints : tuple
            vector of deterministic components of stochastic constraints
        det_stoch_constraints_gradients : tuple
            vector of gradients of deterministic components of
            stochastic constraints
        """
        det_stoch_constraints = None
        det_stoch_constraints_gradients = None
        return det_stoch_constraints, det_stoch_constraints_gradients

    def check_deterministic_constraints(self, x):
        """
        Check if a solution `x` satisfies the problem's deterministic
        constraints.

        Arguments
        ---------
        x : tuple
            vector of decision variables

        Returns
        -------
        satisfies : bool
            indicates if solution `x` satisfies the deterministic constraints.
        """
        return (x[0] >= 0 and np.all(np.array(x[1]) >= 0))

    def get_random_solution(self, rand_sol_rng):
        """
        Generate a random solution for starting or restarting solvers.

        Arguments
        ---------
        rand_sol_rng : rng.mrg32k3a.MRG32k3a
            random-number generator used to sample a new random solution

        Returns
        -------
        x : tuple
            vector of decision variables
        """
        x = (rand_sol_rng.uniform(1,20*24.0), tuple(rand_sol_rng.uniform(1,20,self.dim)*self.factors["demand_store"]))
        return x
