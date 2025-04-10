"""Dynamic Newsvendor Problem.

Simulate a day's worth of sales for a newsvendor under dynamic consumer substitution.
A detailed description of the model/problem can be found
`here <https://simopt.readthedocs.io/en/latest/dynamnews.html>`__.
"""

from __future__ import annotations

from typing import Callable, Final

import numpy as np

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.base import ConstraintType, Model, Problem, VariableType
from simopt.utils import classproperty, override

NUM_PRODUCTS: Final[int] = 10


class DynamNews(Model):
    """Dynamic Newsvendor Model.

    A model that simulates a day's worth of sales for a newsvendor
    with dynamic consumer substitution. Returns the profit and the
    number of products that stock out.

    Attributes:
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

    Arguments:
    ---------
    fixed_factors : dict
        fixed_factors of the simulation model

    See Also:
    --------
    base.Model
    """

    @classproperty
    @override
    def class_name(cls) -> str:
        return "Dynamic Newsvendor"

    @classproperty
    @override
    def n_rngs(cls) -> int:
        return 1

    @classproperty
    @override
    def n_responses(cls) -> int:
        return 4

    @classproperty
    @override
    def specifications(cls) -> dict[str, dict]:
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
    @override
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "num_prod": self._check_num_prod,
            "num_customer": self._check_num_customer,
            "c_utility": self._check_c_utility,
            "mu": lambda: None,
            "init_level": self._check_init_level,
            "price": self._check_price,
            "cost": self._check_cost,
        }

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize the model.

        Args:
            fixed_factors (dict, optional): Fixed factors for the model.
                Defaults to None.
        """
        # Let the base class handle default arguments.
        super().__init__(fixed_factors)

    def _check_num_prod(self) -> None:
        if self.factors["num_prod"] <= 0:
            raise ValueError("num_prod must be greater than 0.")

    def _check_num_customer(self) -> None:
        if self.factors["num_customer"] <= 0:
            raise ValueError("num_customer must be greater than 0.")

    def _check_c_utility(self) -> None:
        if len(self.factors["c_utility"]) != self.factors["num_prod"]:
            raise ValueError("The length of c_utility must be equal to num_prod.")

    def _check_init_level(self) -> None:
        if any(np.array(self.factors["init_level"]) < 0) or (
            len(self.factors["init_level"]) != self.factors["num_prod"]
        ):
            raise ValueError(
                "The length of init_level must be equal to num_prod and every element "
                "in init_level must be greater than or equal to zero."
            )

    def _check_price(self) -> None:
        if any(np.array(self.factors["price"]) < 0) or (
            len(self.factors["price"]) != self.factors["num_prod"]
        ):
            raise ValueError(
                "The length of price must be equal to num_prod and every element in "
                "price must be greater than or equal to zero."
            )

    def _check_cost(self) -> None:
        if any(np.array(self.factors["cost"]) < 0) or (
            len(self.factors["cost"]) != self.factors["num_prod"]
        ):
            raise ValueError(
                "The length of cost must be equal to num_prod and every element in "
                "cost must be greater than or equal to 0."
            )

    @override
    def check_simulatable_factors(self) -> bool:
        if any(np.subtract(self.factors["price"], self.factors["cost"]) < 0):
            raise ValueError(
                "Each element in price must be greater than its corresponding element "
                "in cost."
            )
        return True

    def replicate(self, rng_list: list[MRG32k3a]) -> tuple[dict, dict]:
        """Simulate a single replication for the current model factors.

        Arguments:
        ---------
        rng_list : list of mrg32k3a.mrg32k3a.MRG32k3a objects
            rngs for model to use when simulating a replication

        Returns:
        -------
        responses : dict
            performance measures of interest
            "profit" = profit in this scenario
            "n_prod_stockout" = number of products which are out of stock
            "n_missed_orders" = number of unmet customer orders
            "fill_rate" = fraction of customer orders fulfilled
        """
        num_customer: int = self.factors["num_customer"]
        num_prod: int = self.factors["num_prod"]
        mu: float = self.factors["mu"]
        init_level: list = self.factors["init_level"]
        c_utility: list = self.factors["c_utility"]
        price: list = self.factors["price"]
        cost: list = self.factors["cost"]

        # Designate random number generator for generating a Gumbel random variable.
        gumbel_rng = rng_list[0]
        # Compute Gumbel rvs for the utility of the products.
        gumbel_mu = -mu * np.euler_gamma
        gumbel_beta = mu
        gumbel_flat = [
            gumbel_rng.gumbelvariate(gumbel_mu, gumbel_beta)
            for _ in range(num_customer * num_prod)
        ]
        gumbel = np.reshape(gumbel_flat, (num_customer, num_prod))

        # Compute utility for each product and each customer.
        utility = np.zeros((num_customer, num_prod + 1))
        # Keep the first column of utility as 0, which indicates no purchase.
        utility[:, 1:] = np.array(c_utility) + gumbel

        # Initialize inventory.
        inventory = np.copy(init_level)
        itembought = np.zeros(num_customer)

        # Loop through customers
        for t in range(num_customer):
            # Figure out which producs are in stock
            instock = np.where(inventory > 0)[0]

            # If no products are in stock, no purchase is made.
            if len(instock) == 0:
                itembought[t] = 0
                continue

            # Shift indices to match utility (1-based product indices)
            utility_options = utility[t, instock + 1]

            # Pick index of max utility
            best_idx = np.argmax(utility_options)
            best_product = instock[best_idx] + 1

            # Record it and decrement inventory.
            itembought[t] = best_product
            inventory[best_product - 1] -= 1

        # Calculate profit.
        numsold = init_level - inventory
        total_sold = sum(numsold)
        revenue = numsold * np.array(price)
        costs = init_level * np.array(cost)
        profit = revenue - costs
        unmet_demand = num_customer - total_sold
        order_fill_rate = total_sold / num_customer

        # Compose responses and gradients.
        responses = {
            "profit": np.sum(profit),
            "n_prod_stockout": np.sum(inventory == 0),
            "n_missed_orders": unmet_demand,
            "fill_rate": order_fill_rate,
        }
        gradients = {
            response_key: dict.fromkeys(self.specifications, np.nan)
            for response_key in responses
        }
        return responses, gradients


"""
Summary
-------
Maximize the expected profit for the continuous newsvendor problem.
"""


class DynamNewsMaxProfit(Problem):
    """Base class to implement simulation-optimization problems.

    Attributes:
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

    Arguments:
    ---------
    name : str
        user-specified name for problem
    fixed_factors : dict
        dictionary of user-specified problem factors
    model_fixed factors : dict
        subset of user-specified non-decision factors to pass through to the model

    See Also:
    --------
    base.Problem
    """

    @classproperty
    @override
    def class_name_abbr(cls) -> str:
        return "DYNAMNEWS-1"

    @classproperty
    @override
    def class_name(cls) -> str:
        return "Max Profit for Dynamic Newsvendor"

    @classproperty
    @override
    def n_objectives(cls) -> int:
        return 1

    @classproperty
    @override
    def n_stochastic_constraints(cls) -> int:
        return 0

    @classproperty
    @override
    def minmax(cls) -> tuple[int]:
        return (1,)

    @classproperty
    @override
    def constraint_type(cls) -> ConstraintType:
        return ConstraintType.BOX

    @classproperty
    @override
    def variable_type(cls) -> VariableType:
        return VariableType.CONTINUOUS

    @classproperty
    @override
    def gradient_available(cls) -> bool:
        return False

    @classproperty
    @override
    def optimal_value(cls) -> float | None:
        return None

    @classproperty
    @override
    def optimal_solution(cls) -> tuple | None:
        return None

    @classproperty
    @override
    def model_default_factors(cls) -> dict:
        return {}

    @classproperty
    @override
    def model_decision_factors(cls) -> set[str]:
        return {"init_level"}

    @classproperty
    @override
    def specifications(cls) -> dict[str, dict]:
        return {
            "initial_solution": {
                "description": "initial solution",
                "datatype": tuple,
                "default": (3,) * NUM_PRODUCTS,
            },
            "budget": {
                "description": "max # of replications for a solver to take",
                "datatype": int,
                "default": 1000,
                "isDatafarmable": False,
            },
        }

    @property
    @override
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget,
        }

    @property
    @override
    def dim(self) -> int:
        return self.model.factors["num_prod"]

    @property
    @override
    def lower_bounds(self) -> tuple:
        return (0,) * self.dim

    @property
    @override
    def upper_bounds(self) -> tuple:
        return (np.inf,) * self.dim

    def __init__(
        self,
        name: str = "DYNAMNEWS-1",
        fixed_factors: dict | None = None,
        model_fixed_factors: dict | None = None,
    ) -> None:
        """Initialize the problem.

        Args:
            name (str, optional): Name of the problem. Defaults to "DYNAMNEWS-1".
            fixed_factors (dict, optional): Fixed factors for the problem.
                Defaults to None.
            model_fixed_factors (dict, optional): Fixed factors for the model.
                Defaults to None.
        """
        # Let the base class handle default arguments.
        super().__init__(
            name=name,
            fixed_factors=fixed_factors,
            model_fixed_factors=model_fixed_factors,
            model=DynamNews,
        )

    @override
    def vector_to_factor_dict(self, vector: tuple) -> dict:
        return {"init_level": vector[:]}

    @override
    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:
        return tuple(factor_dict["init_level"])

    @override
    def response_dict_to_objectives(self, response_dict: dict) -> tuple:
        return (response_dict["profit"],)

    def deterministic_objectives_and_gradients(self, _x: tuple) -> tuple:
        """Compute deterministic components of objectives for a solution `x`.

        Arguments:
        ---------
        x : tuple
            vector of decision variables

        Returns:
        -------
        det_objectives : tuple
            vector of deterministic components of objectives
        det_objectives_gradients : tuple
            vector of gradients of deterministic components of objectives
        """
        det_objectives = (0,)
        det_objectives_gradients = ((0,),)
        return det_objectives, det_objectives_gradients

    def check_deterministic_constraints(self, x: tuple) -> bool:
        """Check if a solution `x` satisfies the problem's deterministic constraints.

        Arguments:
        ---------
        x : tuple
            vector of decision variables

        Returns:
        -------
        satisfies : bool
            indicates if solution `x` satisfies the deterministic constraints.
        """
        greater_than_zero: list[bool] = [x[j] > 0 for j in range(self.dim)]
        return all(greater_than_zero)

    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
        """Generate a random solution for starting or restarting solvers.

        Arguments:
        ---------
        rand_sol_rng : mrg32k3a.mrg32k3a.MRG32k3a object
            random-number generator used to sample a new random solution

        Returns:
        -------
        x : tuple
            vector of decision variables
        """
        return tuple([rand_sol_rng.uniform(0, 10) for _ in range(self.dim)])
