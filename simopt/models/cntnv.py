"""Simulate a day's worth of sales for a newsvendor."""

from __future__ import annotations

from typing import Callable

import numpy as np

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.base import ConstraintType, Model, Problem, VariableType
from simopt.input_models import InputModel
from simopt.utils import classproperty, override


class DemandInputModel(InputModel):
    def set_rng(self, rng: random.Random) -> None:
        self.rng = rng

    def unset_rng(self) -> None:
        self.rng = None

    def random(self, burr_c: float, burr_k: float) -> float:
        # Generate random demand according to Burr Type XII distribution.
        # If U ~ Uniform(0,1) and the Burr Type XII has parameters c and k,
        #   X = ((1-U)**(-1/k) - 1)**(1/c) has the desired distribution.
        # https://en.wikipedia.org/wiki/Burr_distribution
        def nth_root(x: float, n: float) -> float:
            """Return the nth root of x."""
            return x ** (1 / n)

        u = self.rng.random()
        demand = nth_root(nth_root(1 - u, -burr_k) - 1, burr_c)
        return demand


class CntNV(Model):
    """Continuous Newsvendor Model with a Burr Type XII demand distribution.

    A model that simulates a day's worth of sales for a newsvendor with a Burr Type XII
    demand distribution. Returns the profit, after accounting for order costs and
    salvage.
    """

    @classproperty
    @override
    def class_name_abbr(cls) -> str:
        return "CNTNEWS"

    @classproperty
    @override
    def class_name(cls) -> str:
        return "Continuous Newsvendor"

    @classproperty
    @override
    def n_rngs(cls) -> int:
        return 1

    @classproperty
    @override
    def n_responses(cls) -> int:
        return 1

    @classproperty
    @override
    def specifications(cls) -> dict[str, dict]:
        return {
            "purchase_price": {
                "description": "purchasing cost per unit",
                "datatype": float,
                "default": 5.0,
            },
            "sales_price": {
                "description": "sales price per unit",
                "datatype": float,
                "default": 9.0,
            },
            "salvage_price": {
                "description": "salvage cost per unit",
                "datatype": float,
                "default": 1.0,
            },
            "order_quantity": {
                "description": "order quantity",
                "datatype": float,  # or int
                "default": 0.5,
            },
            "Burr_c": {
                "description": "Burr Type XII cdf shape parameter",
                "datatype": float,
                "default": 2.0,
            },
            "Burr_k": {
                "description": "Burr Type XII cdf shape parameter",
                "datatype": float,
                "default": 20.0,
            },
        }

    @property
    @override
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "purchase_price": self._check_purchase_price,
            "sales_price": self._check_sales_price,
            "salvage_price": self._check_salvage_price,
            "order_quantity": self._check_order_quantity,
            "Burr_c": self._check_burr_c,
            "Burr_k": self._check_burr_k,
        }

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize the Continuous Newsvendor model.

        Args:
            fixed_factors (dict, optional): Fixed factors for the model.
                Defaults to None.
        """
        # Let the base class handle default arguments.
        super().__init__(fixed_factors)

        self.demand_model = DemandInputModel()

    def _check_purchase_price(self) -> None:
        if self.factors["purchase_price"] <= 0:
            raise ValueError("Purchasing cost per unit must be greater than 0.")

    def _check_sales_price(self) -> None:
        if self.factors["sales_price"] <= 0:
            raise ValueError("Sales price per unit must be greater than 0.")

    def _check_salvage_price(self) -> None:
        if self.factors["salvage_price"] <= 0:
            raise ValueError("Salvage cost per unit must be greater than 0.")

    def _check_order_quantity(self) -> None:
        if self.factors["order_quantity"] <= 0:
            raise ValueError("Order quantity must be greater than 0.")

    def _check_burr_c(self) -> None:
        if self.factors["Burr_c"] <= 0:
            raise ValueError(
                "Burr Type XII cdf shape parameter must be greater than 0."
            )

    def _check_burr_k(self) -> None:
        if self.factors["Burr_k"] <= 0:
            raise ValueError(
                "Burr Type XII cdf shape parameter must be greater than 0."
            )

    @override
    def check_simulatable_factors(self) -> bool:
        if (
            self.factors["salvage_price"]
            < self.factors["purchase_price"]
            < self.factors["sales_price"]
        ):
            return True
        raise ValueError(
            "The salvage cost per unit must be greater than the purchasing cost per "
            "unit, which must be greater than the sales price per unit."
        )

    def before_replicate(self, rng_list):
        self.demand_model.set_rng(rng_list[0])

    def replicate(self) -> tuple[dict, dict]:
        """Simulate a single replication for the current model factors.

        Args:
            rng_list (list[MRG32k3a]): Random number generators used to simulate the
                replication.

        Returns:
            tuple[dict, dict]: A tuple containing:
                - responses (dict): Performance measures of interest, including:
                    - "profit": Profit in this scenario.
                    - "stockout_qty": Amount by which demand exceeded supply.
                    - "stockout": Whether there was unmet demand ("Y" or "N").
                - gradients (dict): Gradient estimates for each response.
        """
        ord_quant: float = self.factors["order_quantity"]
        purch_price: float = self.factors["purchase_price"]
        sales_price: float = self.factors["sales_price"]
        salvage_price: float = self.factors["salvage_price"]
        burr_k: float = self.factors["Burr_k"]
        burr_c: float = self.factors["Burr_c"]
        # Designate random number generator for demand variability.
        demand = self.demand_model.random(burr_c, burr_k)

        # Calculate units sold, as well as unsold/stockout
        units_sold = min(demand, ord_quant)
        order_diff = ord_quant - demand
        units_unsold = max(order_diff, 0)
        stockout_qty = max(-order_diff, 0)

        # Compute revenue and cost components
        order_cost = purch_price * ord_quant
        sales_revenue = units_sold * sales_price
        salvage_revenue = units_unsold * salvage_price

        # Build profit
        profit = sales_revenue + salvage_revenue - order_cost

        # Determine if there was a stockout.
        stockout = int(stockout_qty > 0)

        # Calculate gradient of profit w.r.t. order quantity.
        if order_diff < 0:
            grad_profit_order_quantity = sales_price - purch_price
        elif order_diff > 0:
            grad_profit_order_quantity = salvage_price - purch_price
        else:
            grad_profit_order_quantity = np.nan

        # Compose responses and gradients.
        responses = {
            "profit": profit,
            "stockout_qty": stockout_qty,
            "stockout": stockout,
        }
        gradients = {
            response_key: dict.fromkeys(self.specifications, np.nan)
            for response_key in responses
        }
        gradients["profit"]["order_quantity"] = grad_profit_order_quantity
        return responses, gradients


class CntNVMaxProfit(Problem):
    """Base class to implement simulation-optimization problems."""

    @classproperty
    @override
    def class_name_abbr(cls) -> str:
        return "CNTNEWS-1"

    @classproperty
    @override
    def class_name(cls) -> str:
        return "Max Profit for Continuous Newsvendor"

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
        return True

    @classproperty
    @override
    def optimal_value(cls) -> float | None:
        return None

    @classproperty
    @override
    def optimal_solution(cls) -> tuple | None:
        # TODO: Generalize to function of factors.
        # return (0.1878,)
        return None

    @classproperty
    @override
    def model_default_factors(cls) -> dict:
        return {
            "purchase_price": 5.0,
            "sales_price": 9.0,
            "salvage_price": 1.0,
            "Burr_c": 2.0,
            "Burr_k": 20.0,
        }

    @classproperty
    @override
    def model_decision_factors(cls) -> set[str]:
        return {"order_quantity"}

    @classproperty
    @override
    def specifications(cls) -> dict[str, dict]:
        return {
            "initial_solution": {
                "description": "initial solution",
                "datatype": tuple,
                "default": (0,),
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

    @classproperty
    @override
    def dim(cls) -> int:
        return 1

    @classproperty
    @override
    def lower_bounds(cls) -> tuple:
        return (0,)

    @classproperty
    @override
    def upper_bounds(cls) -> tuple:
        return (np.inf,)

    def __init__(
        self,
        name: str = "CNTNEWS-1",
        fixed_factors: dict | None = None,
        model_fixed_factors: dict | None = None,
    ) -> None:
        """Initialize the Continuous Newsvendor problem.

        Args:
            name (str, optional): Name of the problem. Defaults to "CNTNEWS-1".
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
            model=CntNV,
        )

    @override
    def vector_to_factor_dict(self, vector: tuple) -> dict:
        return {"order_quantity": vector[0]}

    @override
    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:
        return (factor_dict["order_quantity"],)

    @override
    def response_dict_to_objectives(self, response_dict: dict) -> tuple:
        return (response_dict["profit"],)

    @override
    def deterministic_objectives_and_gradients(self, _x: tuple) -> tuple[tuple, tuple]:
        det_objectives = (0,)
        det_objectives_gradients = ((0,),)
        return det_objectives, det_objectives_gradients

    @override
    def check_deterministic_constraints(self, x: tuple) -> bool:
        return x[0] > 0

    @override
    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
        # Generate an Exponential(rate = 1) r.v.
        return (rand_sol_rng.expovariate(1),)
