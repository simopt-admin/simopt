"""Simulate periods of ordering and sales for a dual sourcing inventory problem."""

from __future__ import annotations

from typing import Callable

import numpy as np

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.base import ConstraintType, Model, Problem, VariableType
from simopt.utils import classproperty, override


class DemandInputModel:
    def set_rng(self, rng) -> None:
        self.rng = rng

    def unset_rng(self) -> None:
        self.rng = None

    def random(self, mu: float, sigma: float) -> float:
        def round_and_clamp_non_neg(x: float | int) -> int:
            return round(max(0, x))

        return round_and_clamp_non_neg(self.rng.normalvariate(mu, sigma))


class DualSourcing(Model):
    """Dual Sourcing Inventory Model.

    A model that simulates multiple periods of ordering and sales for a single-staged,
    dual sourcing inventory problem with stochastic demand. Returns average holding
    cost, average penalty cost, and average ordering cost per period.
    """

    @classproperty
    @override
    def n_rngs(cls) -> int:
        return 1

    @classproperty
    @override
    def n_responses(cls) -> int:
        return 3

    @classproperty
    @override
    def specifications(cls) -> dict[str, dict]:
        return {
            "n_days": {
                "description": "number of days to simulate",
                "datatype": int,
                "default": 1000,
                "isDatafarmable": False,
            },
            "initial_inv": {
                "description": "initial inventory",
                "datatype": int,
                "default": 40,
            },
            "cost_reg": {
                "description": "regular ordering cost per unit",
                "datatype": float,
                "default": 100.00,
            },
            "cost_exp": {
                "description": "expedited ordering cost per unit",
                "datatype": float,
                "default": 110.00,
            },
            "lead_reg": {
                "description": "lead time for regular orders in days",
                "datatype": int,
                "default": 2,
            },
            "lead_exp": {
                "description": "lead time for expedited orders in days",
                "datatype": int,
                "default": 0,
            },
            "holding_cost": {
                "description": "holding cost per unit per period",
                "datatype": float,
                "default": 5.00,
            },
            "penalty_cost": {
                "description": "penalty cost per unit per period for backlogging",
                "datatype": float,
                "default": 495.00,
            },
            "st_dev": {
                "description": "standard deviation of demand distribution",
                "datatype": float,
                "default": 10.0,
            },
            "mu": {
                "description": "mean of demand distribution",
                "datatype": float,
                "default": 30.0,
            },
            "order_level_reg": {
                "description": "order-up-to level for regular orders",
                "datatype": int,
                "default": 80,
            },
            "order_level_exp": {
                "description": "order-up-to level for expedited orders",
                "datatype": int,
                "default": 50,
            },
        }

    @property
    @override
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "n_days": self._check_n_days,
            "initial_inv": self._check_initial_inv,
            "cost_reg": self._check_cost_reg,
            "cost_exp": self._check_cost_exp,
            "lead_reg": self._check_lead_reg,
            "lead_exp": self._check_lead_exp,
            "holding_cost": self._check_holding_cost,
            "penalty_cost": self._check_penalty_cost,
            "st_dev": self._check_st_dev,
            "mu": self._check_mu,
            "order_level_reg": self._check_order_level_reg,
            "order_level_exp": self._check_order_level_exp,
        }

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize the DualSourcing model.

        Args:
            fixed_factors (dict, optional): Fixed factors for the model.
                Defaults to None.
        """
        # Let the base class handle default arguments.
        super().__init__(fixed_factors)

        self.demand_model = DemandInputModel()

    # Check for simulatable factors
    def _check_n_days(self) -> None:
        if self.factors["n_days"] < 1:
            raise ValueError("n_days must be greater than or equal to 1.")

    def _check_initial_inv(self) -> None:
        if self.factors["initial_inv"] < 0:
            raise ValueError("initial_inv must be greater than or equal to 0.")

    def _check_cost_reg(self) -> None:
        if self.factors["cost_reg"] <= 0:
            raise ValueError("cost_reg must be greater than 0.")

    def _check_cost_exp(self) -> None:
        if self.factors["cost_exp"] <= 0:
            raise ValueError("cost_exp must be greater than 0.")

    def _check_lead_reg(self) -> None:
        if self.factors["lead_reg"] < 0:
            raise ValueError("lead_reg must be greater than or equal to 0.")

    def _check_lead_exp(self) -> None:
        if self.factors["lead_exp"] < 0:
            raise ValueError("lead_exp must be greater than or equal to 0.")

    def _check_holding_cost(self) -> None:
        if self.factors["holding_cost"] <= 0:
            raise ValueError("holding_cost must be greater than 0.")

    def _check_penalty_cost(self) -> None:
        if self.factors["penalty_cost"] <= 0:
            raise ValueError("penalty_cost must be greater than 0.")

    def _check_st_dev(self) -> None:
        if self.factors["st_dev"] <= 0:
            raise ValueError("st-dev must be greater than 0.")

    def _check_mu(self) -> None:
        if self.factors["mu"] <= 0:
            raise ValueError("mu must be greater than 0.")

    def _check_order_level_reg(self) -> None:
        if self.factors["order_level_reg"] < 0:
            raise ValueError("order_level_reg must be greater than or equal to 0.")

    def _check_order_level_exp(self) -> None:
        if self.factors["order_level_exp"] < 0:
            raise ValueError("order_level_exp must be greater than or equal to 0.")

    @override
    def check_simulatable_factors(self) -> bool:
        if (self.factors["lead_exp"] > self.factors["lead_reg"]) or (
            self.factors["cost_exp"] < self.factors["cost_reg"]
        ):
            raise ValueError(
                "lead_exp must be less than lead_reg and cost_exp must be greater than "
                "cost_reg"
            )
        return True

    def before_replicate(self, rng_list: list[MRG32k3a]) -> None:
        """Set the random number generator for the demand input model."""
        self.demand_model.set_rng(rng_list[0])

    def replicate(self) -> tuple[dict, dict]:
        """Simulate a single replication for the current model factors.

        Args:
            rng_list (list[MRG32k3a]): Random number generators used to simulate
                the replication.

        Returns:
            tuple[dict, dict]: A tuple containing:
                - responses (dict): Performance measures of interest:
                    - "average_holding_cost": The average holding cost over the
                        time period.
                    - "average_penalty_cost": The average penalty cost over the
                        time period.
                    - "average_ordering_cost": The average ordering cost over the
                        time period.
                - gradients (dict): A dictionary of gradient estimates for
                    each response.
        """
        n_days: int = self.factors["n_days"]
        n_days_range = range(n_days)
        lead_reg: int = self.factors["lead_reg"]
        lead_exp: int = self.factors["lead_exp"]
        order_level_reg: int = self.factors["order_level_reg"]
        order_level_exp: int = self.factors["order_level_exp"]
        mu: float = self.factors["mu"]
        st_dev: float = self.factors["st_dev"]
        initial_inv: int = self.factors["initial_inv"]
        cost_exp: float = self.factors["cost_exp"]
        cost_reg: float = self.factors["cost_reg"]
        penalty_cost: float = self.factors["penalty_cost"]
        holding_cost: float = self.factors["holding_cost"]

        def round_and_clamp_non_neg(x: float | int) -> int:
            return round(max(0, x))

        # Vectors of regular orders to be received in periods n through n + lr - 1.
        orders_reg = [0] * lead_reg
        # Vectors of expedited orders to be received in periods n through n + le - 1.
        orders_exp = [0] * lead_exp

        # Generate demand.
        demand = [self.demand_model.random(mu, st_dev) for _ in n_days_range]

        # Track total expenses.
        total_holding_cost = np.zeros(n_days)
        total_penalty_cost = np.zeros(n_days)
        total_ordering_cost = np.zeros(n_days)
        inv = initial_inv

        # Run simulation over time horizon.
        for day in n_days_range:
            # Calculate inventory positions.
            inv_order_exp_sum = inv + sum(orders_exp)
            inv_position_exp = round(inv_order_exp_sum + sum(orders_reg[:lead_exp]))
            inv_position_reg = round(inv_order_exp_sum + sum(orders_reg))
            # Calculate how much to order.
            order_exp = round_and_clamp_non_neg(
                order_level_exp - inv_position_exp - orders_reg[lead_exp]
            )
            orders_exp.append(order_exp)
            order_reg = round_and_clamp_non_neg(
                order_level_reg - inv_position_reg - orders_exp[lead_exp]
            )
            orders_reg.append(order_reg)
            # Charge ordering cost.
            daily_cost_exp = cost_exp * order_exp
            daily_cost_reg = cost_reg * order_reg
            total_ordering_cost[day] = daily_cost_exp + daily_cost_reg
            # Orders arrive, update on-hand inventory.
            inv += orders_exp.pop(0) + orders_reg.pop(0)
            # Satisfy or backorder demand.
            # dn = max(0, demand[day]) THIS IS DONE TWICE
            # inv = inv - dn
            inv -= demand[day]
            # Calculate holding and penalty costs.
            total_penalty_cost[day] = -penalty_cost * min(0, inv)
            total_holding_cost[day] = holding_cost * max(0, inv)

        # Calculate responses from simulation data.
        responses = {
            "average_ordering_cost": np.mean(total_ordering_cost),
            "average_penalty_cost": np.mean(total_penalty_cost),
            "average_holding_cost": np.mean(total_holding_cost),
        }
        gradients = {
            response_key: dict.fromkeys(self.specifications, np.nan)
            for response_key in responses
        }
        return responses, gradients


class DualSourcingMinCost(Problem):
    """Class to make dual-sourcing inventory simulation-optimization problems."""

    @classproperty
    @override
    def class_name_abbr(cls) -> str:
        return "DUALSOURCING-1"

    @classproperty
    @override
    def class_name(cls) -> str:
        return "Min Cost for Dual Sourcing"

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
        return (-1,)

    @classproperty
    @override
    def constraint_type(cls) -> ConstraintType:
        return ConstraintType.BOX

    @classproperty
    @override
    def variable_type(cls) -> VariableType:
        return VariableType.DISCRETE

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
        return {"order_level_exp", "order_level_reg"}

    @classproperty
    @override
    def specifications(cls) -> dict[str, dict]:
        return {
            "initial_solution": {
                "description": "initial solution",
                "datatype": tuple,
                "default": (50, 80),
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
        return 2

    @classproperty
    @override
    def lower_bounds(cls) -> tuple:
        return (0, 0)

    @classproperty
    @override
    def upper_bounds(cls) -> tuple:
        return (np.inf, np.inf)

    def __init__(
        self,
        name: str = "DUALSOURCING-1",
        fixed_factors: dict | None = None,
        model_fixed_factors: dict | None = None,
    ) -> None:
        """Initialize the DualSourcingMinCost problem.

        Args:
            name (str, optional): Name of the problem. Defaults to "DUALSOURCING-1".
            fixed_factors (dict, optional): Fixed factors for the problem.
                Defaults to None.
            model_fixed_factors (dict, optional): Model fixed factors. Defaults to None.
        """
        # Let the base class handle default arguments.
        super().__init__(
            name=name,
            fixed_factors=fixed_factors,
            model_fixed_factors=model_fixed_factors,
            model=DualSourcing,
        )

    @override
    def vector_to_factor_dict(self, vector: tuple) -> dict:
        return {
            "order_level_exp": vector[0],
            "order_level_reg": vector[1],
        }

    @override
    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:
        return (
            factor_dict["order_level_exp"],
            factor_dict["order_level_reg"],
        )

    @override
    def response_dict_to_objectives(self, response_dict: dict) -> tuple:
        return (
            response_dict["average_ordering_cost"]
            + response_dict["average_penalty_cost"]
            + response_dict["average_holding_cost"],
        )

    @override
    def deterministic_objectives_and_gradients(self, _x: tuple) -> tuple[tuple, tuple]:
        det_objectives = (0,)
        det_objectives_gradients = ((0, 0),)
        return det_objectives, det_objectives_gradients

    @override
    def check_deterministic_constraints(self, x: tuple) -> bool:
        return x[0] >= 0 and x[1] >= 0

    @override
    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
        return (rand_sol_rng.randint(40, 60), rand_sol_rng.randint(70, 90))
