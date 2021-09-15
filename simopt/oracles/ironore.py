"""
Summary
-------
Simulate multiple periods of production and sales for an inventory problem
with integer-ordered variables.
"""
import numpy as np

import math

from base import Oracle, Problem


class IronOre(Oracle):
    """
    An oracle that simulates multiple periods of production and sales for an
    inventory problem with integer-ordered variables, infinite demand, and
    stochastic price determined by a mean-reverting random walk.
    Returns

    Attributes
    ----------
    name : str
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

    Parameters
    ----------
    fixed_factors : dict
        fixed_factors of the simulation model

        Questions:
        1. do you need a variable for number of replications and market price? market price would be defined 
        in the replicate function
        2. What are the responses for this problem? Total Revenue

        Questions (Joe):
        1. what is integer-ordered variables?
        2. why price_sell should be larger than price_stop?

        ``mean_price``
            Mean ironore price per unit (`flt`)
        ``max_price``
            Maximum ironore price per unit (`flt`)
        ``min_price``
            Minimum ironore price per unit (`flt`)
        ``capacity (K)``
            Maximum holding capacity (`int`)
        ``st_dev``
            Standard deviation of random walk steps for price (`flt`)
        ``holding_cost``
            Holding cost per unit per period (`flt`)
        ``prod_cost``
            Production cost per unit (`flt`)
        ``max_prod_perday``
            Maximum units produced per day (`int`)
        ``price_prod (x1)`` 
            Price level to start production (`flt`)
        ``inven_stop (x2)``
            Inventory level to cease production (`int`)
        ``price_stop (x3)``
            Price level to stop production (`flt`)
        ``price_sell (x4)``
            Price level to sell all stock (`flt`)
        ``n_days``
            Number of days to simulate (`int`)


    See also
    --------
    base.Oracle
    """
    def __init__(self, fixed_factors={}):
        self.name = "IRONORE"
        self.n_rngs = 1
        self.n_responses =1
        self.factors = fixed_factors
        self.specifications = {

            "mean_price": {
                "description": "Mean ironore price per unit.",
                "datatype": float,
                "default": 100.0
            },
            "max_price": {
                "description": "Maximum ironore price per unit.",
                "datatype": float,
                "default": 200.0
            },
            "min_price": {
                "description": "Minimum ironore price per unit.",
                "datatype": float,
                "default": 0.0
            },
            "capacity": {
                "description": "Maximum holding capacity.",
                "datatype": int,
                "default": 10000
            },
            "st_dev": {
                "description": "Standard deviation of random walk steps for price.",
                "datatype": float,
                "default": 7.5
            },
            "holding_cost": {
                "description": "Holding cost per unit per period.",
                "datatype": float,
                "default": 1.0
            },
            "prod_cost": {
                "description": "Production cost per unit.",
                "datatype": float,
                "default": 100.0
            },
            "max_prod_perday": {
                "description": "Maximum units produced per day.",
                "datatype": int,
                "default": 100
            },
            "price_prod": {
                "description": "Price level to start production.",
                "datatype": float,
                "default": 80.0
            },
            "inven_stop": {
                "description": "Inventory level to cease production.",
                "datatype": int,
                "default": 7000
            },
            "price_stop": {
                "description": "Price level to stop production.",
                "datatype": float,
                "default": 40
            },
            "price_sell": {
                "description": "Price level to sell all stock.",
                "datatype": float,
                "default": 100
            },
            "n_days": {
                "description": "Number of days to simulate.",
                "datatype": int,
                "default": 1000
            },

        }

        self.check_factor_list = {
            "mean_price":self.check_mean_price,
            "max_price":self.check_max_price,
            "min_price":self.check_min_price,
            "capacity":self.check_capacity,
            "st_dev":self.check_st_dev,
            "holding_cost":self.check_holding_cost,
            "prod_cost":self.check_prod_cost,
            "max_prod_perday":self.check_max_prod_perday,
            "price_prod":self.check_price_prod,
            "inven_stop":self.check_inven_stop,
            "price_stop":self.check_price_stop,
            "price_sell":self.check_price_sell,
            "n_days":self.check_n_days,
        }
        # Set factors of the simulation oracle.
        super().__init__(fixed_factors)

    # Check for simulatable factors
    def check_mean_price(self):
        return self.factors["mean_price"] > 0

    def check_max_price(self):
        return self.factors["max_price"] > 0

    def check_min_price(self):
        return self.factors["min_price"] >= 0

    def check_capacity(self):
        return self.factors["capacity"] >= 0

    def check_st_dev(self):
        return self.factors["st_dev"] > 0

    def check_holding_cost(self):
        return self.factors["holding_cost"] > 0

    def check_prod_cost(self):
        return self.factors["prod_cost"] > 0

    def check_max_prod_perday(self):
        return self.factors["max_prod_perday"] > 0

    def check_price_prod(self):
        return self.factors["price_prod"] > 0

    def check_inven_stop(self):
        return self.factors["inven_stop"] > 0

    def check_price_stop(self):
        return self.factors["price_stop"] > 0

    def check_price_sell(self):
        return self.factors["price_sell"] > 0

    def check_n_days(self):
        return self.factors["n_days"] >= 1

    def check_simulatable_factors(self):
        return self.factors["price_stop"] < self.factors["price_sell"]


    def replicate(self, rng_list):
        """
        Simulate a single replication for the current oracle factors.

        Arguments
        ---------
        rng_list : [list]  [rng.mrg32k3a.MRG32k3a]
            rngs for oracle to use when simulating a replication

        Returns
        -------
        responses : dict
            performance measures of interest

            ``total_revenue``
                The total revenue over the time period
        """
        # Designate random number generators.
        price_rng = rng_list[0]
        # Initialize quantities to track:
        #   - Market Price in each period (Pt).
        #   - Stock in each period (st).
        #   - Whether currently producing or not.
        #   - Revenue in each period.
        mkt_price = np.zeros(self.factors['n_days'])
        mkt_price[0] = self.factors['mean_price'] # P1 = mu0
        stock = np.zeros(self.factors['n_days'])
        producing = np.zeros(self.factors['n_days'])
        revenue = np.zeros(self.factors["n_days"])

        #Run simulation over time horizon.
        for day in range(self.factors['n_days']):
            # Determine new price, mean-reverting random walk, Pt = trunc(Pt−1 +Nt(μt,σ))
            #   - μt, mean at period t, where μt = sgn(μ0 − Pt−1) ∗ |μ0 − Pt−1|^(1/4)
            mean_val = math.sqrt(math.sqrt(abs(self.factors['mean_price'] - mkt_price[day])))
            mean_dir = math.copysign(1, self.factors['mean_price'] - mkt_price[day])
            mean_move = mean_val *  mean_dir
            move = price_rng.normalvariate(mean_move, self.factors['st_dev'])
            mkt_price[day + 1] = max(min(mkt_price[day] + move, self.factors['max_price']), self.factors['min_price'])
            # If production is underway
            if producing == 1:
                # cease production if price goes too low or inventory is too much
                if ((mkt_price[day] <= self.factors['price_stop']) | (stock[day] >= self.factors['inven_stop'])):
                    producing = 0
                else:
                    prod = min(self.factors['max_prod_perday'], self.factors['capacity'] - stock[day])
                    stock[day + 1] = stock[day] + prod
                    revenue[day + 1] = revenue[day] - prod * self.factors['prod_cost']
            # if production iss not currently underway
            else:
                if ((mkt_price[day] >= self.factors["price_prod"]) & (stock[day] < self.factors['inven_stop'])):
                    producing = 1
                    prod = min(self.factors['max_prod_perday'], self.factors['capacity'] - stock[day])
                    stock[day + 1] = stock[day] + prod
                    revenue[day + 1] = revenue[day] - prod * self.factors['prod_cost']
            # Sell if price is high enough
            if (mkt_price[day] >= self.factors['price_sell']):
                revenue[day + 1] = revenue[day] + stock[day] * mkt_price[day]
                stock[day + 1] = 0
            # Charge holding cost
            revenue[day + 1] = revenue[day] - stock[day] * self.factors['holding_cost']
        
        total_revenue = np.sum(revenue)
        responses = {"total_revenue": total_revenue
                     }
        gradients = {response_key: {factor_key: np.nan for factor_key in self.specifications} for response_key in responses}
        return responses, gradients


"""
Summary
-------
Minimize the expected total cost for (s, S) inventory system.
"""


class IronOreMaxRev(Problem):
    """
    Class to make (s,S) inventory simulation-optimization problems.

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
    gradient_available : bool
        indicates if gradient of objective function is available
    optimal_value : float
        optimal objective function value
    optimal_solution : tuple
        optimal solution
    oracle : base.Oracle
        associated simulation oracle that generates replications
    oracle_default_factors : dict
        default values for overriding oracle-level default factors
    oracle_fixed_factors : dict
        combination of overriden oracle-level factors and defaults
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
    oracle_fixed factors : dict
        subset of user-specified non-decision factors to pass through to the oracle

    See also
    --------
    base.Problem
    """
    def __init__(self, name="SSCONT-1", fixed_factors={}, oracle_fixed_factors={}):
        self.name = name
        self.dim = 2
        self.n_objectives = 1
        self.n_stochastic_constraints = 0
        self.minmax = (-1,)
        self.constraint_type = "box"
        self.variable_type = "continuous"
        self.lowerbound = 0
        self.upperbound = np.inf
        self.gradient_available = False
        self.optimal_value = None
        self.optimal_solution = None
        self.oracle_default_factors = {}
        self.factors = fixed_factors
        self.specifications = {
            "initial_solution": {
                "description": "Initial solution from which solvers start.",
                "datatype": tuple,
                "default": (600, 600)
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
        super().__init__(fixed_factors, oracle_fixed_factors)
        # Instantiate oracle with fixed factors and overwritten defaults.
        self.oracle = IronOre(self.oracle_fixed_factors)

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
            "s": vector[0],
            "S": vector[0] + vector[1]
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
        vector = (factor_dict["s"], factor_dict["S"] - factor_dict["s"])
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
        return (x[0] >= 0 and x[1] >= 0)

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
        x = (rand_sol_rng.expovariate(1/200), rand_sol_rng.expovariate(1/200))
        return x
