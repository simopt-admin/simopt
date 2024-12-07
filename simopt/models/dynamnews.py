"""
Summary
-------
Simulate a day's worth of sales for a newsvendor under dynamic consumer substitution.
A detailed description of the model/problem can be found
`here <https://simopt.readthedocs.io/en/latest/dynamnews.html>`__.
"""

from __future__ import annotations

from typing import Callable, Final

import numpy as np
from mrg32k3a.mrg32k3a import MRG32k3a

from simopt.base import Model, Problem

NUM_PRODUCTS: Final[int] = 10


class DynamNews(Model):
    """
    A model that simulates a day's worth of sales for a newsvendor
    with dynamic consumer substitution. Returns the profit and the
    number of products that stock out.

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

    @property
    def name(self) -> str:
        return "DYNAMNEWS"

    @property
    def n_rngs(self) -> int:
        return 1

    @property
    def n_responses(self) -> int:
        return 4

    @property
    def specifications(self) -> dict[str, dict]:
        return {
            "num_prod": {
                "description": "number of products",
                "datatype": int,
                "default": NUM_PRODUCTS,
            },
            "num_customer": {
                "description": "number of customers",
                "datatype": int,
                "default": 30,
            },
            "c_utility": {
                "description": "constant of each product's utility",
                "datatype": list,
                "default": [6 + j for j in range(NUM_PRODUCTS)],
            },
            "mu": {
                "description": "mu for calculating Gumbel random variable",
                "datatype": float,
                "default": 1.0,
            },
            "init_level": {
                "description": "initial inventory level",
                "datatype": list,
                "default": [3] * NUM_PRODUCTS,
            },
            "price": {
                "description": "sell price of products",
                "datatype": list,
                "default": [9] * NUM_PRODUCTS,
            },
            "cost": {
                "description": "cost of products",
                "datatype": list,
                "default": [5] * NUM_PRODUCTS,
            },
        }

    @property
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "num_prod": self.check_num_prod,
            "num_customer": self.check_num_customer,
            "c_utility": self.check_c_utility,
            "mu": self.check_mu,
            "init_level": self.check_init_level,
            "price": self.check_price,
            "cost": self.check_cost,
        }

    def __init__(self, fixed_factors: dict | None = None) -> None:
        # Let the base class handle default arguments.
        super().__init__(fixed_factors)

    def check_num_prod(self) -> bool:
        return self.factors["num_prod"] > 0

    def check_num_customer(self) -> bool:
        return self.factors["num_customer"] > 0

    def check_c_utility(self) -> bool:
        return len(self.factors["c_utility"]) == self.factors["num_prod"]

    def check_init_level(self) -> bool:
        return all(np.array(self.factors["init_level"]) >= 0) & (
            len(self.factors["init_level"]) == self.factors["num_prod"]
        )

    def check_mu(self) -> bool:
        return True

    def check_price(self) -> bool:
        return all(np.array(self.factors["price"]) >= 0) & (
            len(self.factors["price"]) == self.factors["num_prod"]
        )

    def check_cost(self) -> bool:
        return all(np.array(self.factors["cost"]) >= 0) & (
            len(self.factors["cost"]) == self.factors["num_prod"]
        )

    def check_simulatable_factors(self) -> bool:
        return all(
            np.subtract(self.factors["price"], self.factors["cost"]) >= 0
        )

    def replicate(self, rng_list: list[MRG32k3a]) -> tuple[dict, dict]:
        """
        Simulate a single replication for the current model factors.

        Arguments
        ---------
        rng_list : list of mrg32k3a.mrg32k3a.MRG32k3a objects
            rngs for model to use when simulating a replication

        Returns
        -------
        responses : dict
            performance measures of interest
            "profit" = profit in this scenario
            "n_prod_stockout" = number of products which are out of stock
            "n_missed_orders" = number of unmet customer orders
            "fill_rate" = fraction of customer orders fulfilled
        """
        # Designate random number generator for generating a Gumbel random variable.
        gumbel_rng = rng_list[0]
        # Compute Gumbel rvs for the utility of the products.
        gumbel = np.zeros(
            (self.factors["num_customer"], self.factors["num_prod"])
        )
        for t in range(self.factors["num_customer"]):
            for j in range(self.factors["num_prod"]):
                gumbel[t][j] = gumbel_rng.gumbelvariate(
                    -self.factors["mu"] * np.euler_gamma, self.factors["mu"]
                )
        # Compute utility for each product and each customer.
        utility = np.zeros(
            (self.factors["num_customer"], self.factors["num_prod"] + 1)
        )
        for t in range(self.factors["num_customer"]):
            for j in range(self.factors["num_prod"] + 1):
                if j == 0:
                    utility[t][j] = 0
                else:
                    utility[t][j] = (
                        self.factors["c_utility"][j - 1] + gumbel[t][j - 1]
                    )

        # Initialize inventory.
        inventory = np.copy(self.factors["init_level"])
        itembought = np.zeros(self.factors["num_customer"])

        # Loop through customers
        for t in range(self.factors["num_customer"]):
            instock = np.where(inventory > 0)[0]
            # Initialize the purchase option to be no-purchase.
            itembought[t] = 0
            # Assign the purchase option to be the product that maximizes the utility.
            for j in instock:
                if utility[t][j + 1] > utility[t][int(itembought[t])]:
                    itembought[t] = j + 1
            # print("item bought", int(itembought[t]))
            if itembought[t] != 0:
                inventory[int(itembought[t] - 1)] -= 1

        # Calculate profit.
        numsold = self.factors["init_level"] - inventory
        revenue = numsold * np.array(self.factors["price"])
        cost = self.factors["init_level"] * np.array(self.factors["cost"])
        profit = revenue - cost
        unmet_demand = self.factors["num_customer"] - sum(numsold)
        order_fill_rate = sum(numsold) / self.factors["num_customer"]

        # Compose responses and gradients.
        responses = {
            "profit": np.sum(profit),
            "n_prod_stockout": np.sum(inventory == 0),
            "n_missed_orders": unmet_demand,
            "fill_rate": order_fill_rate,
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
Maximize the expected profit for the continuous newsvendor problem.
"""


class DynamNewsMaxProfit(Problem):
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
    optimal_value : tuple
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
    rng_list : list of mrg32k3a.mrg32k3a.MRG32k3a objects
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

    def __init__(
        self,
        name: str = "DYNAMNEWS-1",
        fixed_factors: dict | None = None,
        model_fixed_factors: dict | None = None,
    ) -> None:
        # Handle default arguments.
        if fixed_factors is None:
            fixed_factors = {}
        if model_fixed_factors is None:
            model_fixed_factors = {}
        # Set problem attributes.
        self.name = name
        self.n_objectives = 1
        self.n_stochastic_constraints = 0
        self.minmax = (1,)
        self.constraint_type = "box"
        self.variable_type = "continuous"
        self.gradient_available = False
        self.optimal_value = None
        self.optimal_solution = None
        self.model_default_factors = {}
        self.model_fixed_factors = {}
        self.model_decision_factors = {"init_level"}
        self.factors = fixed_factors
        self.specifications = {
            "initial_solution": {
                "description": "initial solution",
                "datatype": tuple,
                "default": (3,) * NUM_PRODUCTS,
            },
            "budget": {
                "description": "max # of replications for a solver to take",
                "datatype": int,
                "default": 1000,
            },
        }
        self.check_factor_list = {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget,
        }
        super().__init__(fixed_factors, model_fixed_factors)
        # Instantiate model with fixed factors and overwritten defaults.
        self.model = DynamNews(self.model_fixed_factors)
        self.dim = self.model.factors["num_prod"]
        self.lower_bounds = (0,) * self.dim
        self.upper_bounds = (np.inf,) * self.dim

    def vector_to_factor_dict(self, vector: tuple) -> dict:
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
        factor_dict = {"init_level": vector[:]}
        return factor_dict

    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:
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
        vector = tuple(factor_dict["init_level"])
        return vector

    def response_dict_to_objectives(self, response_dict: dict) -> tuple:
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

    def response_dict_to_stoch_constraints(self, response_dict: dict) -> tuple:
        """
        Convert a dictionary with response keys to a vector
        of left-hand sides of stochastic constraints: E[Y] <= 0

        Arguments
        ---------
        response_dict : dictionary
            dictionary with response keys and associated values

        Returns
        -------
        stoch_constraints : tuple
            vector of LHSs of stochastic constraint
        """
        stoch_constraints = ()
        return stoch_constraints

    def deterministic_objectives_and_gradients(self, x: tuple) -> tuple:
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
        greater_than_zero: list[bool] = [x[j] > 0 for j in range(self.dim)]
        return all(greater_than_zero)

    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
        """
        Generate a random solution for starting or restarting solvers.

        Arguments
        ---------
        rand_sol_rng : mrg32k3a.mrg32k3a.MRG32k3a object
            random-number generator used to sample a new random solution

        Returns
        -------
        x : tuple
            vector of decision variables
        """
        x = tuple([rand_sol_rng.uniform(0, 10) for _ in range(self.dim)])
        return x
