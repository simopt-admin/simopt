"""Simulate sales for a newsvendor under dynamic consumer substitution."""

from __future__ import annotations

from typing import Callable, Final

import numpy as np

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.base import ConstraintType, Model, Problem, VariableType
from simopt.input_models import InputModel
from simopt.utils import classproperty, override

NUM_PRODUCTS: Final[int] = 10


class Utility(InputModel):
    def set_rng(self, rng: random.Random) -> None:
        self.rng = rng

    def unset_rng(self) -> None:
        self.rng = None

    def random(
        self, mu: float, num_customer: int, num_prod: int, c_utility: list
    ) -> np.ndarray:
        # Compute Gumbel rvs for the utility of the products.
        gumbel_mu = -mu * np.euler_gamma
        gumbel_beta = mu
        gumbel_flat = [
            self.rng.gumbelvariate(gumbel_mu, gumbel_beta)
            for _ in range(num_customer * num_prod)
        ]
        gumbel = np.reshape(gumbel_flat, (num_customer, num_prod))

        # Compute utility for each product and each customer.
        utility = np.zeros((num_customer, num_prod + 1))
        # Keep the first column of utility as 0, which indicates no purchase.
        utility[:, 1:] = np.array(c_utility) + gumbel
        return utility


class DynamNews(Model):
    """Dynamic Newsvendor Model.

    A model that simulates a day's worth of sales for a newsvendor
    with dynamic consumer substitution. Returns the profit and the
    number of products that stock out.
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

        self.utility_model = Utility()

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

    def before_replicate(self, rng_list):
        self.utility_model.set_rng(rng_list[0])

    def replicate(self) -> tuple[dict, dict]:
        """Simulate a single replication for the current model factors.

        Args:
            rng_list (list[MRG32k3a]): Random number generators used to simulate
                the replication.

        Returns:
            tuple[dict, dict]: A tuple containing:
                - responses (dict): Performance measures of interest, including:
                    - "profit": Profit in this scenario.
                    - "n_prod_stockout": Number of products that are out of stock.
                    - "n_missed_orders": Number of unmet customer orders.
                    - "fill_rate": Fraction of customer orders fulfilled.
                - gradients (dict): A dictionary of gradient estimates for
                    each response.
        """
        num_customer: int = self.factors["num_customer"]
        num_prod: int = self.factors["num_prod"]
        mu: float = self.factors["mu"]
        init_level: list = self.factors["init_level"]
        c_utility: list = self.factors["c_utility"]
        price: list = self.factors["price"]
        cost: list = self.factors["cost"]

        utility = self.utility_model.random(
            mu,
            num_customer,
            num_prod,
            c_utility,
        )

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


class DynamNewsMaxProfit(Problem):
    """Base class to implement simulation-optimization problems."""

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

    @override
    def deterministic_objectives_and_gradients(self, _x: tuple) -> tuple:
        det_objectives = (0,)
        det_objectives_gradients = ((0,),)
        return det_objectives, det_objectives_gradients

    @override
    def check_deterministic_constraints(self, x: tuple) -> bool:
        return all(x[j] > 0 for j in range(self.dim))

    @override
    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
        return tuple([rand_sol_rng.uniform(0, 10) for _ in range(self.dim)])
