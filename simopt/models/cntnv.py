"""
Summary
-------
Simulate a day's worth of sales for a newsvendor.
A detailed description of the model/problem can be found `here <https://simopt.readthedocs.io/en/latest/cntnv.html>`_.
"""
import numpy as np

from base import Model, Problem


class CntNV(Model):
    """
    A model that simulates a day's worth of sales for a newsvendor
    with a Burr Type XII demand distribution. Returns the profit, after
    accounting for order costs and salvage.

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
        details of each factor (for GUI, data validation, and defaults)
    check_factor_list : dict
        switch case for checking factor simulatability

    Arguments
    ---------
    fixed_factors : dict
        fixed_factors of the simulation model

    See also
    --------
    base.Model
    """
    def __init__(self, fixed_factors=None):
        if fixed_factors is None:
            fixed_factors = {}
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
        # Set factors of the simulation model.
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
        Simulate a single replication for the current model factors.

        Arguments
        ---------
        rng_list : list of rng.MRG32k3a objects
            rngs for model to use when simulating a replication

        Returns
        -------
        responses : dict
            performance measures of interest
            "profit" = profit in this scenario
            "stockout_qty" = amount by which demand exceeded supply
            "stockout" = was there unmet demand? (Y/N)
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
        stockout_qty = max(demand - self.factors["order_quantity"], 0)
        stockout = int(stockout_qty > 0)
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
        responses = {"profit": profit, "stockout_qty": stockout_qty, "stockout": stockout}
        gradients = {response_key:
                     {factor_key: np.nan for factor_key in self.specifications}
                     for response_key in responses
                     }
        gradients["profit"]["order_quantity"] = grad_profit_order_quantity
        return responses, gradients


"""
Summary
-------
Maximize the expected profit for the continuous newsvendor problem.
"""


class CntNVMaxProfit(Problem):
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
            initial_solution : tuple
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
    def __init__(self, name="CNTNEWS-1", fixed_factors=None, model_fixed_factors=None):
        if fixed_factors is None:
            fixed_factors = {}
        if model_fixed_factors is None:
            model_fixed_factors = {}
        self.name = name
        self.dim = 1
        self.n_objectives = 1
        self.n_stochastic_constraints = 0
        self.minmax = (1,)
        self.constraint_type = "box"
        self.variable_type = "continuous"
        self.lower_bounds = (0,)
        self.upper_bounds = (np.inf,)
        self.gradient_available = True
        self.optimal_value = None
        self.optimal_solution = (0.1878,)  # TO DO: Generalize to function of factors.
        self.model_default_factors = {
            "purchase_price": 5.0,
            "sales_price": 9.0,
            "salvage_price": 1.0,
            "Burr_c": 2.0,
            "Burr_k": 20.0
            }
        self.model_decision_factors = {"order_quantity"}
        self.factors = fixed_factors
        self.specifications = {
            "initial_solution": {
                "description": "Initial solution from which solvers start.",
                "datatype": tuple,
                "default": (0,)
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
        self.model = CntNV(self.model_fixed_factors)

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
            "order_quantity": vector[0]
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
        vector = (factor_dict["order_quantity"],)
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
        objectives = (response_dict["profit"],)
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
        det_objectives_gradients = ((0,),)
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
        return x[0] > 0

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
        # Generate an Exponential(rate = 1) r.v.
        x = (rand_sol_rng.expovariate(1),)
        return x
