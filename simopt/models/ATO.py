"""
Summary
-------
Simulate expected revenue for a hotel.
"""
import numpy as np

from base import Model, Problem


class ATO(Model):
    """
    A model that simulates the assembly operation of a production system with constant Poisson arrival rates.

    Attributes
    ----------
    name : string
        name of model
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
    base.Model
    """
    def __init__(self, fixed_factors={}):
        self.name = "ATO"
        self.n_rngs = 1
        self.n_responses = 1
        self.specifications = {
            "num_products": {
                "description": "Number of different product types:",
                "datatype": int,
                "default": 5
            },
            "lambda": {
                "description": "Constant (Poisson) arrival rates for each product type.",
                "datatype": list,
                "default": [3.6,
                            3,
                            2.4,
                            1.8,
                            1.2]
            },
            "num_items": {
                "description": "Number of different item types.",
                "datatype": int,
                "default": 8
            },
            "item_cost": {
                "description": "Items' profit per item sold and holding cost per unit time per item in inventory.",
                "datatype": list,
                "default": [[1,2]
                            [2,2],
                            [3,2],
                            [4,2],
                            [5,2],
                            [6,2],
                            [7,2],
                            [8,2]]
            },
            "item_cap": {
                "description": "Items' inventory capacity.",
                "datatype": list,
                "default": ([[20]
                            [20],
                            [20],
                            [20],
                            [20],
                            [20],
                            [20],
                            [20]])
            },
            "process_time": {
                "description": "Production time for each item type; normally distributed mean and standard deviation (mu, sigma).",
                "datatype": list,
                "default": [[0.15, 0.0225],
                            [0.40, 0.06],
                            [0.25, 0.0375],
                            [0.15, 0.0225],
                            [0.25, 0.0375],
                            [0.08, 0.012],
                            [0.13, 0.0195],
                            [0.40, 0.06]]
            },
           "product_req": {
                "description": "Required item types/quantity for each product type.",
                "datatype": list,
                "default": [[1, 0, 0, 1, 0, 1, 1, 0],
                            [1, 0, 0, 0, 1, 1, 1, 0],
                            [0, 1, 0, 1, 0, 1, 0, 0],
                            [0, 0, 1, 1, 0, 1, 0, 1],
                            [0, 0, 1, 0, 1, 1, 1, 0]]
            },
            "warm_up_time": {
                "description": "Warm-up period in time units.",
                "datatype": int,
                "default": 20
            },
            "run_time": {
                "description": "Run-length in time units for capturing statistics.",
                "datatype": int,
                "default": 50
            },
        }
        self.check_factor_list = {
            "num_products": self.check_num_products,
            "lambda": self.check_lambda,
            "num_items": self.check_num_items,
            "item_cost": self.check_item_cost,
            "item_cap": self.check_item_cap,
            "process_time": self.check_process_time,
            "product_req": self.check_product_req,
            "warm_up_time": self.warm_up_time,
            "run_time": self.run_time
        }
        # Set factors of the simulation model.
        super().__init__(fixed_factors)

    def check_num_products(self):
        return self.factors["num_products"] > 0

    def check_lambda(self):
        for i in self.factors["lambda"]:
            if i <= 0:
                return False
        return len(self.factors["lambda"]) == self.factors["num_products"]

    def check_num_items(self):
        return self.factors["num_items"] > 0

    def check_item_cost(self):
        for i in self.factors["item_cost"]:
            for j in self.factors["item_cost"][i]:
                if j <= 0:
                    return False
        return len(self.factors["item_cost"]) == self.factors["num_items"]

    def check_item_cap(self):
        for i in self.factors["item_cap"]:
            if i <= 0:
                return False
        return len(self.factors["item_cap"]) == self.factors["num_items"]

    def check_process_time(self):
        for i in self.factors["process_time"]:
            for j in self.factors["process_time"][i]:
                if j <= 0:
                    return False
        return len(self.factors["process_time"]) == self.factors["num_items"]

    def check_product_req(self):
        for i in self.factors["product_req"]:
            if len(i) != self.factors["num_items"]:
                return False
        if len(self.factors["product_req"]) != self.factors["num_products"]:
            return False
        return len(self.factors["product_req"] == self.factors["num_products"])

    def check_warm_up_time(self):
        return self.factors["warm_up_time"] > 0

    def check_run_time(self):
        return self.factors["run_time"] > 0

    def replicate(self, rng_list):
        """
        Simulate a single replication for the current model factors.

        Arguments
        ---------
        rng_list : list of rng.MRG32k3a objects
            rngs for model to use when simulating a replication

        Returns
        -------
        responses : dict
            performance measures of interest
            "revenue" = expected revenue
        gradients : dict of dicts
            gradient estimates for each response
        """
        # Find items needed for product 
        def product_items(product):
            needed_items = list(self.factors["product_req"][(product-1)])
            return needed_items

        # Verify order 
        def verified_items(needed_items, item_invent):
            stock_items = []
            # Check key items are in stock
                # If order can be completed
                # Return list of available items 
            for i in range(6):
                if needed_items[i] == 1:
                    if item_invent[i] >= needed_items[i]:
                        stock_items.append(1)
                    else:
                        return False
                        break
                else:
                    stock_items.append(0)
            for i in range(6, self.factors["num_items"], 1):
                if needed_items[i] == 1:
                    if invent_items[i] >= needed_items[i]:
                        stock_items.append(1)
                    else:
                        stock_items.append(0)
                else:
                    stock_items.append(0)
            
            return stock_items


        # Assemble with key items & available non-key items
            # Update list of quantity of items (type) sold
            # Update inventory 
            # Update holding cost 
            # Update profit

        # Replenish item 
            # Generate production time for every item in list of available items for replenishment
            # Update time lapse/clock



        total_revenue = 0
        total_holding = 0
        item_invent = self.factors["item_cap"]
        # Generate orders/product type arrivals 
        ## Order/Product-type probability of arrivals
        tot = 0 
        prod_probs = []                                                                      # List of product/order arrival probablities 
        for i in range(self.factors["num_products"]): tot += self.factors["lambda"][i]       # Sums up arrival rates
        for j in self.factors["lambda"]: prod_probs.append((self.factors["lambda"][j])/tot)       
        ## Produce for time horizon 
        orders = []                                                                          # List of in stock orders
        orders_time = 0                                                                      # Sum of processing times 
        num_orders = 0                                                                  
        while orders_time < ((self.factors["warm_up_time"] + self.factors["run_time"])):
            
            product = random.choices(np.arange(1, self.factors["num_products"]+1), weights = prod_probs, k = 1)
            verified_items(product_items(product), item_invent)
            orders.append(product[0])
            order_arrival_time = (1/self.factors["lambda"][product[0]-1])                    # Order inter-arrival time
            orders_time += order_arrival_time                                                # Sum of arrival times                                                                                                                                                                 
                
                
                ###############################
                if orders_time <= self.factors["time_horizon"]:                              # Attach if sum is less than time horizon
                    arrival_times_rng.append(orders_time)                                    # Track number of orders
                    num_orders += 1
                    product = random.choices(np.arange(1, self.factors["num_products"]+1), weights = self.factors["product_batch_prob"], k = 1)
                    product_orders_rng.append(product[0])
                else: 
                    break
    



"""
Summary
-------
Maximize the expected profit.
"""


class ATOProfit(Problem):
    """
    Base class to implement simulation-optimization problems.

    Attributes
    ----------
    name : string
        name of problem
    dim : int
        number of decision variables
    n_objectives : int
        number of objectives
    n_stochastic_constraints : int
        number of stochastic constraints
    minmax : tuple of int (+/- 1)
        indicator of maximization (+1) or minimization (-1) for each objective
    constraint_type : string
        description of constraints types:
            "unconstrained", "box", "deterministic", "stochastic"
    variable_type : string
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
    model : Model object
        associated simulation model that generates replications
    model_default_factors : dict
        default values for overriding model-level default factors
    model_fixed_factors : dict
        combination of overriden model-level factors and defaults
    model_decision_factors : set of str
        set of keys for factors that are decision variables
    rng_list : list of rng.MRG32k3a objects
        list of RNGs used to generate a random initial solution
        or a random problem instance
    factors : dict
        changeable factors of the problem
            initial_solution : list
                default initial solution from which solvers start
            budget : int > 0
                max number of replications (fn evals) for a solver to take
    specifications : dict
        details of each factor (for GUI, data validation, and defaults)

    Arguments
    ---------
    name : str
        user-specified name for problem
    fixed_factors : dict
        dictionary of user-specified problem factors
    model_fixed factors : dict
        subset of user-specified non-decision factors to pass through to the model

    See also
    --------
    base.Problem
    """
    def __init__(self, name="HOTEL-1", fixed_factors={}, model_fixed_factors={}):
        self.name = name
        self.n_objectives = 1
        self.n_stochastic_constraints = 1
        self.minmax = (1,)
        self.constraint_type = "box"
        self.variable_type = "discrete"
        self.gradient_available = False
        self.optimal_value = None
        self.optimal_solution = None
        self.model_default_factors = {}
        self.model_decision_factors = {"base_stock"}
        self.factors = fixed_factors
        self.specifications = {
            "initial_solution": {
                "description": "Initial solution.",
                "datatype": tuple,
                "default": tuple([0 for _ in range(56)])
            },
            "budget": {
                "description": "Max # of replications for a solver to take.",
                "datatype": int,
                "default": 70
            }
        }
        self.check_factor_list = {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget
        }
        super().__init__(fixed_factors, model_fixed_factors)
        # Instantiate model with fixed factors and over-riden defaults.
        self.model = Hotel(self.model_fixed_factors)
        self.dim = self.model.factors["num_products"]
        self.lower_bounds = tuple(np.zeros(self.dim))
        self.upper_bounds = tuple(self.model.factors["num_rooms"] * np.ones(self.dim))

    def check_initial_solution(self):
        return len(self.factors["initial_solution"]) == self.dim

    def check_budget(self):
        return self.factors["budget"] > 0

    def check_simulatable_factors(self):
        if len(self.lower_bounds) != self.dim:
            return False
        elif len(self.upper_bounds) != self.dim:
            return False
        else:
            return True

    def vector_to_factor_dict(self, vector):
        """
        Convert a vector of variables to a dictionary with factor keys

        Arguments
        ---------
        vector : tuple
            vector of values associated with decision variables

        Returns
        -------
        factor_dict : dictionary
            dictionary with factor keys and associated values
        """
        factor_dict = {
            "booking_limits": vector[:]
        }
        return factor_dict

    def factor_dict_to_vector(self, factor_dict):
        """
        Convert a dictionary with factor keys to a vector
        of variables.

        Arguments
        ---------
        factor_dict : dictionary
            dictionary with factor keys and associated values

        Returns
        -------
        vector : tuple
            vector of values associated with decision variables
        """
        vector = tuple(factor_dict["booking_limits"])
        return vector

    def response_dict_to_objectives(self, response_dict):
        """
        Convert a dictionary with response keys to a vector
        of objectives.

        Arguments
        ---------
        response_dict : dictionary
            dictionary with response keys and associated values

        Returns
        -------
        objectives : tuple
            vector of objectives
        """
        objectives = (response_dict["revenue"],)
        return objectives

    def response_dict_to_stoch_constraints(self, response_dict):
        """
        Convert a dictionary with response keys to a vector
        of left-hand sides of stochastic constraints: E[Y] >= 0

        Arguments
        ---------
        response_dict : dictionary
            dictionary with response keys and associated values

        Returns
        -------
        stoch_constraints : tuple
            vector of LHSs of stochastic constraint
        """
        stoch_constraints = None
        return stoch_constraints

    def deterministic_stochastic_constraints_and_gradients(self, x):
        """
        Compute deterministic components of stochastic constraints for a solution `x`.

        Arguments
        ---------
        x : tuple
            vector of decision variables

        Returns
        -------
        det_stoch_constraints : tuple
            vector of deterministic components of stochastic constraints
        det_stoch_constraints_gradients : tuple
            vector of gradients of deterministic components of stochastic constraints
        """
        det_stoch_constraints = None
        det_stoch_constraints_gradients = None
        return det_stoch_constraints, det_stoch_constraints_gradients

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
        det_objectives = (0,)
        det_objectives_gradients = ((0,) * self.dim,)
        return det_objectives, det_objectives_gradients

    def check_deterministic_constraints(self, x):
        """
        Check if a solution `x` satisfies the problem's deterministic constraints.

        Arguments
        ---------
        x : tuple
            vector of decision variables

        Returns
        -------
        satisfies : bool
            indicates if solution `x` satisfies the deterministic constraints.
        """
        return True

    def get_random_solution(self, rand_sol_rng):
        """
        Generate a random solution for starting or restarting solvers.

        Arguments
        ---------
        rand_sol_rng : rng.MRG32k3a object
            random-number generator used to sample a new random solution

        Returns
        -------
        x : tuple
            vector of decision variables
        """
        x = tuple([rand_sol_rng.randint(0, self.model.factors["num_rooms"]) for _ in range(self.dim)])
        return x
