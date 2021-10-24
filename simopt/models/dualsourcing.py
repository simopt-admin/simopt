"""
Summary
-------
Simulate multiple periods of ordering and sales for a dual sourcing inventory problem.
"""
import numpy as np

import math

from base import Model, Problem


class DuralSourcing(Model):
    """
    An model that simulates multiple periods of ordering and sales for a single-staged, 
    dual sourcing inventory problem with stochastic demand. Returns average holding cost, 
    average penalty cost, and average ordering cost per period.

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

        ``n_days``
            Number of days to simulate (`int`)
        ``initial_inv``
            Initial inventory (`int`)
        ``cost_reg``
            Regular ordering cost per unit (`flt`)
        ``cost_exp``
            Expedited ordering cost per unit (`flt`)
        ``lead_reg``
            Lead time for regular orders in days (`int`)
        ``lead_exp``
            Lead time for expedited orders in days (`int`)
        ``holding_cost``
            Holding cost per unit per period (`flt`)
        ``penalty_cost``
            Penalty cost per unit per period for backlogging(`flt`)
        ``distribution``
            Demand distribution(`str`)
        ``st_dev``
            Standard deviation of demand distribution (`flt`)
        ``mu``
            Mean of demand distribution (`flt`)
        ``order_level_reg``
            Order-up-to level for regular orders (`int`)
        ``order_level_exp``
            Order-up-to level for expedited orders (`int`)
        
        -inv_position_reg
        -inv_position_exp
        -order_reg
        -order_exp


    See also
    --------
    base.Model
    """
    def __init__(self, fixed_factors={}):
        self.name = "DUALSOURCING"
        self.n_rngs = 1
        self.n_responses = 3
        self.factors = fixed_factors
        self.specifications = {

            "n_days": {
                "description": "Number of days to simulate.",
                "datatype": int,
                "default": 1000
            },
            "initial_inv": {
                "description": "Initial inventory.",
                "datatype": int,
                "default": 40
            },
            "mean_price": {
                "description": "Mean iron ore price per unit.",
                "datatype": float,
                "default": 100.0
            },
            "max_price": {
                "description": "Maximum iron ore price per unit.",
                "datatype": float,
                "default": 200.0
            },
            "min_price": {
                "description": "Minimum iron ore price per unit.",
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
            

        }

        self.check_factor_list = {
            "mean_price": self.check_mean_price,
            "max_price": self.check_max_price,
            "min_price": self.check_min_price,
            "capacity": self.check_capacity,
            "st_dev": self.check_st_dev,
            "holding_cost": self.check_holding_cost,
            "prod_cost": self.check_prod_cost,
            "max_prod_perday": self.check_max_prod_perday,
            "price_prod": self.check_price_prod,
            "inven_stop": self.check_inven_stop,
            "price_stop": self.check_price_stop,
            "price_sell": self.check_price_sell,
            "n_days": self.check_n_days,
        }
        # Set factors of the simulation model
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
        return (self.factors["min_price"] <= self.factors["mean_price"]) & (self.factors["mean_price"] <= self.factors["max_price"]) & (self.factors["min_price"] <= self.factors["max_price"])

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

            ``total_revenue``
                The total revenue over the time period
            ``frac_producing``
                The fraction of days spent producing iron ore
            ``mean_stock``
                The average stocks over the time period
        """
        # Designate random number generators.
        price_rng = rng_list[0]
        # Initialize quantities to track:
        #   - Market price in each period (Pt).
        #   - Starting stock in each period.
        #   - Ending stock in each period.
        #   - Revenue in each period.
        #   - Whether producing or not in each period.
        #   - Production in each period.
        mkt_price = np.zeros(self.factors["n_days"])
        mkt_price[0] = self.factors["mean_price"]
        stock = np.zeros(self.factors["n_days"])
        revenue = np.zeros(self.factors["n_days"])
        producing = np.zeros(self.factors["n_days"])
        prod = np.zeros(self.factors["n_days"])

        #Run simulation over time horizon.
        for day in range(1, self.factors["n_days"]):
            # Determine new price, mean-reverting random walk, Pt = trunc(Pt−1 + Nt(μt,σ))
            # Run μt, mean at period t, where μt = sgn(μ0 − Pt−1) ∗ |μ0 − Pt−1|^(1/4)
            mean_val = math.sqrt(math.sqrt(abs(self.factors["mean_price"] - mkt_price[day])))
            mean_dir = math.copysign(1, self.factors["mean_price"] - mkt_price[day])
            mean_move = mean_val *  mean_dir
            move = price_rng.normalvariate(mean_move, self.factors["st_dev"])
            mkt_price[day] = max(min(mkt_price[day - 1] + move, self.factors["max_price"]), self.factors["min_price"])
            # If production is underway
            if producing[day] == 1:
                # cease production if price goes too low or inventory is too much
                if ((mkt_price[day] <= self.factors["price_stop"]) | (stock[day] >= self.factors["inven_stop"])):
                    producing[day] = 0
                else:
                    prod[day] = min(self.factors["max_prod_perday"], self.factors["capacity"] - stock[day])
                    stock[day] = stock[day] + prod[day]
                    revenue[day] = revenue[day] - prod[day] * self.factors["prod_cost"]
            # if production is not currently underway
            else:
                if ((mkt_price[day] >= self.factors["price_prod"]) & (stock[day] < self.factors["inven_stop"])):
                    producing[day] = 1
                    prod[day] = min(self.factors["max_prod_perday"], self.factors["capacity"] - stock[day])
                    stock[day] = stock[day] + prod[day]
                    revenue[day] = revenue[day] - prod[day] * self.factors["prod_cost"]
            # Sell if price is high enough
            if (mkt_price[day] >= self.factors["price_sell"]):
                revenue[day] = revenue[day] + stock[day] * mkt_price[day]
                stock[day] = 0
            # Charge holding cost
            revenue[day] = revenue[day] - stock[day] * self.factors["holding_cost"]
            # Calculate starting quantities for next period
            if day < self.factors["n_days"] - 1:
                revenue[day + 1] = revenue[day]
                stock[day + 1] = stock[day]
                mkt_price[day + 1] = mkt_price[day] 
                producing[day + 1] = producing[day]
        # Calculate responses from simulation data.
        responses = {"total_revenue": revenue[self.factors["n_days"] - 1],
                     "frac_producing": np.mean(producing),
                     "mean_stock": np.mean(stock)
                     }
        gradients = {response_key: {factor_key: np.nan for factor_key in self.specifications} for response_key in responses}
        return responses, gradients


"""
Summary
-------
Maximize the expected total revenue for iron ore inventory system.
"""


class IronOreMaxRev(Problem):
    """
    Class to make iron ore inventory simulation-optimization problems.

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
    def __init__(self, name="IRONORE-1", fixed_factors={}, model_fixed_factors={}):
        self.name = name
        self.dim = 4
        self.n_objectives = 1
        self.n_stochastic_constraints = 0
        self.minmax = (1,)
        self.constraint_type = "box"
        self.variable_type = "discrete"
        self.lowerbound = (0)
        self.upperbound = (np.inf)
        self.gradient_available = False
        self.optimal_value = None
        self.optimal_solution = None
        self.model_default_factors = {}
        self.factors = fixed_factors
        self.specifications = {
            "initial_solution": {
                "description": "Initial solution from which solvers start.",
                "datatype": tuple,
                "default": (80, 7000, 40, 100)
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
        self.model = IronOre(self.model_fixed_factors)

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
            "price_prod": vector[0],
            "inven_stop": vector[1],
            "price_stop": vector[2],
            "price_sell": vector[3],
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
        vector = (factor_dict["price_prod"], factor_dict["inven_stop"], factor_dict["price_stop"], factor_dict["price_sell"])
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
        objectives = (response_dict["total_revenue"],)
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
        return (x[0] >= 0 and x[1] >= 0 and x[2] >= 0 and x[3] >= 0)

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
        x = (rand_sol_rng.randint(70, 90), rand_sol_rng.randint(2000, 8000), rand_sol_rng.randint(30, 50), rand_sol_rng.randint(90, 110))
        return x
