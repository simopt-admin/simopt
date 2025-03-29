"""
Summary
-------
Simulate multiple periods of production and sales for an iron ore inventory problem.
A detailed description of the model/problem can be found
`here <https://simopt.readthedocs.io/en/latest/ironore.html>`__.

Changed get_random_solution quantiles
    from 10 and 200 => mean=59.887, sd=53.338, p(X>100)=0.146
    to 10 and 1000 => mean=199.384, sd=343.925, p(X>100)=0.5
"""

from __future__ import annotations

from math import copysign, sqrt
from typing import Callable

import numpy as np

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.base import ConstraintType, Model, Problem, VariableType
from simopt.utils import classproperty


class IronOre(Model):
    """
    A model that simulates multiple periods of production and sales for an
    inventory problem with stochastic price determined by a mean-reverting
    random walk. Returns total profit, fraction of days producing iron, and
    mean stock.

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

    @classproperty
    def n_rngs(cls) -> int:
        return 1

    @classproperty
    def n_responses(cls) -> int:
        return 3

    @classproperty
    def specifications(cls) -> dict[str, dict]:
        return {
            "mean_price": {
                "description": "mean iron ore price per unit",
                "datatype": float,
                "default": 100.0,
            },
            "max_price": {
                "description": "maximum iron ore price per unit",
                "datatype": float,
                "default": 200.0,
            },
            "min_price": {
                "description": "minimum iron ore price per unit",
                "datatype": float,
                "default": 0.0,
            },
            "capacity": {
                "description": "maximum holding capacity",
                "datatype": int,
                "default": 10000,
            },
            "st_dev": {
                "description": "standard deviation of random walk steps for price",
                "datatype": float,
                "default": 7.5,
            },
            "holding_cost": {
                "description": "holding cost per unit per period",
                "datatype": float,
                "default": 1.0,
            },
            "prod_cost": {
                "description": "production cost per unit",
                "datatype": float,
                "default": 100.0,
            },
            "max_prod_perday": {
                "description": "maximum units produced per day",
                "datatype": int,
                "default": 100,
            },
            "price_prod": {
                "description": "price level to start production",
                "datatype": float,
                "default": 80.0,
            },
            "inven_stop": {
                "description": "inventory level to cease production",
                "datatype": int,
                "default": 7000,
            },
            "price_stop": {
                "description": "price level to stop production",
                "datatype": float,
                "default": 40.0,
            },
            "price_sell": {
                "description": "price level to sell all stock",
                "datatype": float,
                "default": 100.0,
            },
            "n_days": {
                "description": "number of days to simulate",
                "datatype": int,
                "default": 365,
                "isDatafarmable": False,
            },
        }

    @property
    def check_factor_list(self) -> dict[str, Callable]:
        return {
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

    def __init__(self, fixed_factors: dict | None = None) -> None:
        # Let the base class handle default arguments.
        super().__init__(fixed_factors)

    # Check for simulatable factors
    def check_mean_price(self) -> bool:
        if self.factors["mean_price"] <= 0:
            raise ValueError(
                "Mean iron ore price per unit must be greater than 0."
            )
        return True

    def check_max_price(self) -> None:
        if self.factors["max_price"] <= 0:
            raise ValueError("max_price must be greater than 0.")

    def check_min_price(self) -> None:
        if self.factors["min_price"] < 0:
            raise ValueError("min_price must be greater than or equal to 0.")

    def check_capacity(self) -> None:
        if self.factors["capacity"] < 0:
            raise ValueError("capacity must be greater than or equal to 0.")

    def check_st_dev(self) -> None:
        if self.factors["st_dev"] <= 0:
            raise ValueError("st_dev must be greater than 0.")

    def check_holding_cost(self) -> None:
        if self.factors["holding_cost"] <= 0:
            raise ValueError("holding_cost must be greater than 0.")

    def check_prod_cost(self) -> None:
        if self.factors["prod_cost"] <= 0:
            raise ValueError("prod_cost must be greater than 0.")

    def check_max_prod_perday(self) -> None:
        if self.factors["max_prod_perday"] <= 0:
            raise ValueError("max_prod_perday must be greater than 0.")

    def check_price_prod(self) -> None:
        if self.factors["price_prod"] <= 0:
            raise ValueError("price_prod must be greater than 0.")

    def check_inven_stop(self) -> None:
        if self.factors["inven_stop"] <= 0:
            raise ValueError("inven_stop must be greater than 0.")

    def check_price_stop(self) -> None:
        if self.factors["price_stop"] <= 0:
            raise ValueError("price_stop must be greater than 0.")

    def check_price_sell(self) -> None:
        if self.factors["price_sell"] <= 0:
            raise ValueError("price_sell must be greater than 0.")

    def check_n_days(self) -> None:
        if self.factors["n_days"] < 1:
            raise ValueError("n_days must be greater than or equal to 1.")

    def check_simulatable_factors(self) -> bool:
        if (self.factors["min_price"] > self.factors["mean_price"]) or (
            self.factors["mean_price"] > self.factors["max_price"]
        ):
            raise ValueError(
                "mean_price must be greater than or equal to min_price and less than or equal to max_price."
            )
        return True

    def replicate(self, rng_list: list[MRG32k3a]) -> tuple[dict, dict]:
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
        n_days: int = self.factors["n_days"]
        min_price: float = self.factors["min_price"]
        mean_price: float = self.factors["mean_price"]
        max_price: float = self.factors["max_price"]
        st_dev: float = self.factors["st_dev"]
        price_stop: float = self.factors["price_stop"]
        inven_stop: int = self.factors["inven_stop"]
        max_prod_perday: int = self.factors["max_prod_perday"]
        capacity: int = self.factors["capacity"]
        prod_cost: float = self.factors["prod_cost"]
        price_prod: float = self.factors["price_prod"]
        price_sell: float = self.factors["price_sell"]
        holding_cost: float = self.factors["holding_cost"]
        # Designate random number generators.
        price_rng = rng_list[0]
        # Initialize quantities to track:
        #   - Market price in each period (Pt).
        #   - Starting stock in each period.
        #   - Ending stock in each period.
        #   - Profit in each period.
        #   - Whether producing or not in each period.
        #   - Production in each period.
        mkt_price = np.zeros(n_days)
        mkt_price[0] = mean_price
        stock = np.zeros(n_days)
        producing = np.zeros(n_days)
        prod_costs = np.zeros(n_days)
        hold_costs = np.zeros(n_days)
        sell_profit = np.zeros(n_days)

        # Run simulation over time horizon.
        for day in range(1, n_days):
            # === Initializatize values ===
            # Initialize today with values from yesterday
            prior_day = day - 1
            # Stock doesn't reset between days
            prev_stock = stock[prior_day]
            stock[day] = prev_stock
            # The market price is a random walk, but it's based off of the
            # previous day's price.
            prev_price = mkt_price[prior_day]
            mkt_price[day] = prev_price
            # We just need yesterday's producing status to help detetmine
            # if we should produce today.
            prev_producing = producing[prior_day]

            # === Price Update: mean-reverting random walk ===
            delta = mean_price - prev_price
            mean_move = copysign(sqrt(sqrt(abs(delta))), delta)
            move = price_rng.normalvariate(mean_move, st_dev)
            price_today = max(min(prev_price + move, max_price), min_price)
            mkt_price[day] = price_today

            # === Production Logic ===
            # If stock is below the inventory stop and either:
            # - if producing, price is above the price stop
            # - if not producing, price is above the price prod
            # then produce the maximum amount possible.
            if prev_stock < inven_stop and (
                (prev_producing and price_today >= price_stop)
                or (not prev_producing and price_today >= price_prod)
            ):
                missing_stock = capacity - prev_stock
                production_amount = min(max_prod_perday, missing_stock)
                stock[day] += production_amount
                prod_costs[day] = production_amount * prod_cost
                producing[day] = 1

            # === Selling Logic ===
            if price_today >= price_sell:
                sell_profit[day] = stock[day] * price_today
                stock[day] = 0

            # === Holding Cost ===
            hold_costs[day] = stock[day] * holding_cost

        # Calculate total profit
        profits = sell_profit - prod_costs - hold_costs
        net_profit = np.sum(profits)

        # Calculate responses from simulation data.
        responses = {
            "total_profit": net_profit,
            "frac_producing": np.mean(producing),
            "mean_stock": np.mean(stock),
        }
        gradients = {
            response_key: {
                factor_key: np.nan for factor_key in self.specifications
            }
            for response_key in responses
        }
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

    @classproperty
    def class_name_abbr(cls) -> str:
        return "IRONORE-1"

    @classproperty
    def class_name(cls) -> str:
        return "Max Revenue for Iron Ore"

    @classproperty
    def n_objectives(cls) -> int:
        return 1

    @classproperty
    def n_stochastic_constraints(cls) -> int:
        return 0

    @classproperty
    def minmax(cls) -> tuple[int]:
        return (1,)

    @classproperty
    def constraint_type(cls) -> ConstraintType:
        return ConstraintType.BOX

    @classproperty
    def variable_type(cls) -> VariableType:
        return VariableType.MIXED

    @classproperty
    def gradient_available(cls) -> bool:
        return False

    @classproperty
    def optimal_value(cls) -> float | None:
        return None

    @classproperty
    def optimal_solution(cls) -> tuple | None:
        return None

    @classproperty
    def model_default_factors(cls) -> dict:
        return {}

    @classproperty
    def model_decision_factors(cls) -> set[str]:
        return {"price_prod", "inven_stop", "price_stop", "price_sell"}

    @classproperty
    def specifications(cls) -> dict[str, dict]:
        return {
            "initial_solution": {
                "description": "initial solution",
                "datatype": tuple,
                "default": (80, 7000, 40, 100),
            },
            "budget": {
                "description": "max # of replications for a solver to take",
                "datatype": int,
                "default": 1000,
                "isDatafarmable": False,
            },
        }

    @property
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget,
        }

    @classproperty
    def dim(cls) -> int:
        return 4

    @classproperty
    def lower_bounds(cls) -> tuple:
        return (0,) * cls.dim

    @classproperty
    def upper_bounds(cls) -> tuple:
        return (np.inf,) * cls.dim

    def __init__(
        self,
        name: str = "IRONORE-1",
        fixed_factors: dict | None = None,
        model_fixed_factors: dict | None = None,
    ) -> None:
        # Let the base class handle default arguments.
        super().__init__(
            name=name,
            fixed_factors=fixed_factors,
            model_fixed_factors=model_fixed_factors,
            model=IronOre,
        )

    def vector_to_factor_dict(self, vector: tuple) -> dict:
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

    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:
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
        vector = (
            factor_dict["price_prod"],
            factor_dict["inven_stop"],
            factor_dict["price_stop"],
            factor_dict["price_sell"],
        )
        return vector

    def response_dict_to_objectives(self, response_dict: dict) -> tuple:
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

    def response_dict_to_stoch_constraints(self, response_dict: dict) -> tuple:
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
        stoch_constraints = ()
        return stoch_constraints

    def deterministic_objectives_and_gradients(
        self, x: tuple
    ) -> tuple[tuple, tuple]:
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

    def deterministic_stochastic_constraints_and_gradients(
        self, x: tuple
    ) -> tuple[tuple, tuple]:
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
        det_stoch_constraints = ()
        det_stoch_constraints_gradients = ()
        return det_stoch_constraints, det_stoch_constraints_gradients

    def check_deterministic_constraints(self, x: tuple) -> bool:
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

    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
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
        x = (
            rand_sol_rng.lognormalvariate(10, 200),
            rand_sol_rng.lognormalvariate(1000, 10000),
            rand_sol_rng.lognormalvariate(10, 200),
            rand_sol_rng.lognormalvariate(10, 200),
        )
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
    optimal_value : tuple
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

    @classproperty
    def class_name_abbr(cls) -> str:
        return "IRONORECONT-1"

    @classproperty
    def class_name(cls) -> str:
        return "Max Revenue for Continuous Iron Ore"

    @classproperty
    def n_objectives(cls) -> int:
        return 1

    @classproperty
    def n_stochastic_constraints(cls) -> int:
        return 0

    @classproperty
    def minmax(cls) -> tuple[int]:
        return (1,)

    @classproperty
    def constraint_type(cls) -> ConstraintType:
        return ConstraintType.BOX

    @classproperty
    def variable_type(cls) -> VariableType:
        return VariableType.CONTINUOUS

    @classproperty
    def gradient_available(cls) -> bool:
        return False

    @classproperty
    def optimal_value(cls) -> float | None:
        return None

    @classproperty
    def optimal_solution(cls) -> tuple | None:
        return None

    @classproperty
    def model_default_factors(cls) -> dict:
        return {}

    @classproperty
    def model_decision_factors(cls) -> set[str]:
        return {"price_prod", "price_stop", "price_sell"}

    @classproperty
    def specifications(cls) -> dict[str, dict]:
        return {
            "initial_solution": {
                "description": "initial solution",
                "datatype": tuple,
                "default": (80, 40, 100),
            },
            "budget": {
                "description": "max # of replications for a solver to take",
                "datatype": int,
                "default": 1000,
            },
        }

    @property
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget,
        }

    @classproperty
    def dim(cls) -> int:
        return 3

    @classproperty
    def lower_bounds(cls) -> tuple:
        return (0.0,) * cls.dim

    @classproperty
    def upper_bounds(cls) -> tuple:
        return (np.inf,) * cls.dim

    def __init__(
        self,
        name: str = "IRONORECONT-1",
        fixed_factors: dict | None = None,
        model_fixed_factors: dict | None = None,
    ) -> None:
        # Let the base class handle default arguments.
        super().__init__(
            name=name,
            fixed_factors=fixed_factors,
            model_fixed_factors=model_fixed_factors,
            model=IronOre,
        )

    def vector_to_factor_dict(self, vector: tuple) -> dict:
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

    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:
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
        vector = (
            factor_dict["price_prod"],
            factor_dict["price_stop"],
            factor_dict["price_sell"],
        )
        return vector

    def response_dict_to_objectives(self, response_dict: dict) -> tuple:
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

    def response_dict_to_stoch_constraints(self, response_dict: dict) -> tuple:
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
        stoch_constraints = ()
        return stoch_constraints

    def deterministic_objectives_and_gradients(
        self, x: tuple
    ) -> tuple[tuple, tuple]:
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

    def deterministic_stochastic_constraints_and_gradients(
        self, x: tuple
    ) -> tuple[tuple, tuple]:
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
        det_stoch_constraints = ()
        det_stoch_constraints_gradients = ()
        return det_stoch_constraints, det_stoch_constraints_gradients

    def check_deterministic_constraints(self, x: tuple) -> bool:
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
        return x[0] >= 0 and x[1] >= 0 and x[2] >= 0

    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
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

        x = (
            rand_sol_rng.lognormalvariate(10, 1000),
            rand_sol_rng.lognormalvariate(10, 1000),
            rand_sol_rng.lognormalvariate(10, 1000),
        )
        return x
