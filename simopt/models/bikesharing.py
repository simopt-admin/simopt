"""
Summary
-------
Simulate the expected cost of a bike sharing system in different days.
A detailed description of the model/problem can be found
`here <TODO: no documentation>`_.
"""
import numpy as np

from ..base import Model, Problem


class BikeShare(Model):
    """
    A model that simulates a day of bike sharing program. Returns the
    total cost of distribution, total penalty incurred during the operation hours.

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

    Arguments
    ----------
    fixed_factors : dict
        fixed_factors of the simulation model

    See also
    --------
    base.Model
    """
    def __init__(self, fixed_factors=None):
        if fixed_factors is None:
            fixed_factors = {}
        self.name = "BIKESHARING"
        self.n_rngs = 1 # TODO: number of rng used in the model
        self.n_responses = 2 # TODO: modify if more responses are added
        self.factors = fixed_factors 
        self.specifications = {
            "num_bikes": {
                "description": "total number of bikes in the city",
                "datatype": int,
                "default": 3200
            },
            "num_stations": {
                "description": "total number of stations in the city",
                "datatype": int,
                "default": 225
            },
            "day_length": {
                "description": "the length of a day in operation in hours",
                "datatype": int,
                "default": 16
            },
            "station_capacities": {
                "description": "the capacity of each corresponding stations",
                "datatype": list,
                "default": [30] * 225 # TODO: how to set?
            },
            "empty_penalty_constant": {
                "description": "the penalty constant for when a station has no bike",
                "datatype": float,
                "default": 50.0
            },
            "full_penalty_constant": {
                "description": "the penalty constant for when a station is full",
                "datatype": float,
                "default": 50.0
            },
            "arrival_rates": {
                "description": "user arrival rates to each corresponding stations (in this model, we assume a homogeneous Poisson process for each station)",
                "datatype": list,
                "default": [] # TODO
            },
            "gamma_mean_const": {
                "description": "scalar for the mean time it takes the user to return the bike",
                "datatype": float,
                "default": 1/3
            },
            "gamma_variance_const": {
                "description": "scalar for the variance of time it takes the user to return the bike",
                "datatype": float,
                "default": 1/144
            },
            "gamma_mean_const_s": {
                "description": "mean time it takes the user to return bike to the same station",
                "datatype": float,
                "default": 3/4
            },
            "gamma_variance_const_s": {
                "description": "variance for time it takes the user to return bike to the same station",
                "datatype": float,
                "default": 49/60
            },
            "rebalancing_constant": {
                "description": "constant multiple for the cost of rebalancing bikes",
                "datatype": float,
                "default": 5
            },
            "distance": {
                "description": "An s x s matrix containing distance from each pair of stations",
                "datatype": list,
                "default": [[]] # TODO
            }
        }

        self.check_factor_list = {
            "num_bikes": self.check_num_bikes,
            "num_stations": self.check_num_stations,
            "day_length": self.check_day_length,
            "station_capacities": self.check_station_capacities,
            "empty_penalty_constant": self.check_empty_penalty_constant,
            "full_penalty_constant": self.check_full_penalty_constant,
            "arrival_rates": self.check_arrival_rates,
            "gamma_mean_const": self.check_gamma_mean_const,
            "gamma_variance_const": self.check_gamma_variance_const,
            "gamma_mean_const_s": self.check_gamma_mean_const_s,
            "gamma_variance_const_s": self.check_gamma_variance_const_s,
            "rebalancing_constant": self.check_rebalancing_constant,
            "distance": self.check_distance,
        }
        # Set factors of the simulation model
        super().__init__(fixed_factors)

    # Check for simulatable factors
    def check_num_bikes(self):
        return self.factors["num_bikes"] > 0

    def check_num_stations(self):
        return self.factors["num_stations"] > 0

    def check_day_length(self):
        return self.factors["day_length"] >= 0 and self.factors["day_length"] <= 24

    def check_station_capacities(self):
        return self.factors["station_capacities"] >= 0

    def check_empty_penalty_constant(self):
        return self.factors["empty_penalty_constant"] > 0

    def check_full_penalty_constant(self):
        return self.factors["full_penalty_constant"] > 0

    def check_arrival_rates(self):
        return all(rates > 0 for rates in self.factors["arrival_rates"])

    def check_gamma_mean_const(self):
        return self.factors["gamma_mean_const"] > 0

    def check_gamma_variance_const(self):
        return self.factors["gamma_variance_const"] > 0

    def check_gamma_mean_const_s(self):
        return self.factors["gamma_mean_const_s"] > 0

    def check_gamma_variance_const_s(self):
        return self.factors["gamma_variance_const_s"] > 0

    def check_rebalancing_constant(self):
        return self.factors["rebalancing_constant"] > 0

    def check_distance(self):
        return True # TODO: check distances

    def replicate(self, rng_list):
        """
        Simulate a single replication for the current model factors.

        Arguments
        ---------
        rng_list : [list]  [mrg32k3a.mrg32k3a.MRG32k3a]
            rngs for model to use when simulating a replication

        Returns
        -------
        responses : dict
            performance measures of interest
            "total_profit" = The total profit over the time period
            "frac_producing" = The fraction of days spent producing iron ore
            "mean_stock" = The average stocks over the time period
        """
        # Designate random number generators.
        arrival = rng_list[0]
        # Initialize quantities to track:
        #   - Market price in each period (Pt).
        #   - Starting stock in each period.
        #   - Ending stock in each period.
        #   - Profit in each period.
        #   - Whether producing or not in each period.
        #   - Production in each period.
        mkt_price = np.zeros(self.factors["n_days"])
        mkt_price[0] = self.factors["mean_price"]
        stock = np.zeros(self.factors["n_days"])
        profit = np.zeros(self.factors["n_days"])
        producing = np.zeros(self.factors["n_days"])
        prod = np.zeros(self.factors["n_days"])

        # Run simulation over time horizon.
        for day in range(1, self.factors["n_days"]):
            # Determine new price, mean-reverting random walk, Pt = trunc(Pt−1 + Nt(μt,σ)).
            # Run μt, mean at period t, where μt = sgn(μ0 − Pt−1) ∗ |μ0 − Pt−1|^(1/4).
            mean_val = sqrt(sqrt(abs(self.factors["mean_price"] - mkt_price[day])))
            mean_dir = copysign(1, self.factors["mean_price"] - mkt_price[day])
            mean_move = mean_val * mean_dir
            move = price_rng.normalvariate(mean_move, self.factors["st_dev"])
            mkt_price[day] = max(min(mkt_price[day - 1] + move, self.factors["max_price"]), self.factors["min_price"])
            # If production is underway...
            if producing[day] == 1:
                # ... cease production if price goes too low or inventory is too high.
                if ((mkt_price[day] <= self.factors["price_stop"]) | (stock[day] >= self.factors["inven_stop"])):
                    producing[day] = 0
                else:
                    prod[day] = min(self.factors["max_prod_perday"], self.factors["capacity"] - stock[day])
                    stock[day] = stock[day] + prod[day]
                    profit[day] = profit[day] - prod[day] * self.factors["prod_cost"]
            # If production is not currently underway...
            else:
                if ((mkt_price[day] >= self.factors["price_prod"]) & (stock[day] < self.factors["inven_stop"])):
                    producing[day] = 1
                    prod[day] = min(self.factors["max_prod_perday"], self.factors["capacity"] - stock[day])
                    stock[day] = stock[day] + prod[day]
                    profit[day] = profit[day] - prod[day] * self.factors["prod_cost"]
            # Sell if price is high enough.
            if (mkt_price[day] >= self.factors["price_sell"]):
                profit[day] = profit[day] + stock[day] * mkt_price[day]
                stock[day] = 0
            # Charge holding cost.
            profit[day] = profit[day] - stock[day] * self.factors["holding_cost"]
            # Calculate starting quantities for next period.
            if day < self.factors["n_days"] - 1:
                profit[day + 1] = profit[day]
                stock[day + 1] = stock[day]
                mkt_price[day + 1] = mkt_price[day]
                producing[day + 1] = producing[day]
        # Calculate responses from simulation data.
        responses = {"total_profit": profit[self.factors["n_days"] - 1],
                     "frac_producing": np.mean(producing),
                     "mean_stock": np.mean(stock)
                     }
        gradients = {response_key: {factor_key: np.nan for factor_key in self.specifications} for response_key in responses}
        return responses, gradients


"""
Summary
-------
Maximize the expected total profit for iron ore inventory system.
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
    rng_list : [list]  [mrg32k3a.mrg32k3a.MRG32k3a]
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
    def __init__(self, name="IRONORE-1", fixed_factors=None, model_fixed_factors=None):
        if fixed_factors is None:
            fixed_factors = {}
        if model_fixed_factors is None:
            model_fixed_factors = {}
        self.name = name
        self.dim = 4
        self.n_objectives = 1
        self.n_stochastic_constraints = 0
        self.minmax = (1,)
        self.constraint_type = "box"
        self.variable_type = "mixed"
        self.lower_bounds = (0, 0, 0, 0)
        self.upper_bounds = (np.inf, np.inf, np.inf, np.inf)
        self.gradient_available = False
        self.optimal_value = None
        self.optimal_solution = None
        self.model_default_factors = {}
        self.model_decision_factors = {"price_prod", "inven_stop", "price_stop", "price_sell"}
        self.factors = fixed_factors
        self.specifications = {
            "initial_solution": {
                "description": "initial solution",
                "datatype": tuple,
                "default": (80, 7000, 40, 100)
            },
            "budget": {
                "description": "max # of replications for a solver to take",
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
        objectives = (response_dict["total_profit"],)
        return objectives

    def response_dict_to_stoch_constraints(self, response_dict):
        """
        Convert a dictionary with response keys to a vector
        of left-hand sides of stochastic constraints: E[Y] <= 0

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
        det_objectives_gradients = ((0, 0, 0, 0),)
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
        # Check box constraints.
        box_feasible = super().check_deterministic_constraints(x)
        return box_feasible

    def get_random_solution(self, rand_sol_rng):
        """
        Generate a random solution for starting or restarting solvers.

        Arguments
        ---------
        rand_sol_rng : mrg32k3a.mrg32k3a.MRG32k3a
            random-number generator used to sample a new random solution

        Returns
        -------
        x : tuple
            vector of decision variables
        """
        # x = (rand_sol_rng.randint(70, 90), rand_sol_rng.randint(2000, 8000), rand_sol_rng.randint(30, 50), rand_sol_rng.randint(90, 110))
        x = (rand_sol_rng.lognormalvariate(10, 200), rand_sol_rng.lognormalvariate(1000, 10000), rand_sol_rng.lognormalvariate(10, 200), rand_sol_rng.lognormalvariate(10, 200))
        return x


"""
Summary
-------
Continuous version of the Maximization of the expected total profit for iron ore inventory system (removing the inven_stop from decision variables).
"""


class IronOreMaxRevCnt(Problem):
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
    rng_list : [list]  [mrg32k3a.mrg32k3a.MRG32k3a]
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
    def __init__(self, name="IRONORECONT-1", fixed_factors=None, model_fixed_factors=None):
        if fixed_factors is None:
            fixed_factors = {}
        if model_fixed_factors is None:
            model_fixed_factors = {}
        self.name = name
        self.dim = 3
        self.n_objectives = 1
        self.n_stochastic_constraints = 0
        self.minmax = (1,)
        self.constraint_type = "box"
        self.variable_type = "continuous"
        self.lower_bounds = (0., 0., 0.)
        self.upper_bounds = (np.inf, np.inf, np.inf)
        self.gradient_available = False
        self.optimal_value = None
        self.optimal_solution = None
        self.model_default_factors = {}
        self.model_decision_factors = {"price_prod", "price_stop", "price_sell"}
        self.factors = fixed_factors
        self.specifications = {
            "initial_solution": {
                "description": "initial solution",
                "datatype": tuple,
                "default": (80, 40, 100)
            },
            "budget": {
                "description": "max # of replications for a solver to take",
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
            "price_stop": vector[1],
            "price_sell": vector[2],
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
        vector = (factor_dict["price_prod"], factor_dict["price_stop"], factor_dict["price_sell"])
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
        objectives = (response_dict["total_profit"],)
        return objectives

    def response_dict_to_stoch_constraints(self, response_dict):
        """
        Convert a dictionary with response keys to a vector
        of left-hand sides of stochastic constraints: E[Y] <= 0

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
        det_objectives_gradients = ((0, 0, 0),)
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
        return (x[0] >= 0 and x[1] >= 0 and x[2] >= 0)

    def get_random_solution(self, rand_sol_rng):
        """
        Generate a random solution for starting or restarting solvers.

        Arguments
        ---------
        rand_sol_rng : mrg32k3a.mrg32k3a.MRG32k3a
            random-number generator used to sample a new random solution

        Returns
        -------
        x : tuple
            vector of decision variables
        """
        # x = (rand_sol_rng.randint(70, 90), rand_sol_rng.randint(30, 50), rand_sol_rng.randint(90, 110))
        
        x = (rand_sol_rng.lognormalvariate(10,1000),rand_sol_rng.lognormalvariate(10,1000),rand_sol_rng.lognormalvariate(10,1000))
        return x
