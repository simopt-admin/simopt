"""Simulate sales for a (s,S) inventory problem with continuous inventory."""

from __future__ import annotations

from math import sqrt
from typing import Callable

import numpy as np

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.base import ConstraintType, Model, Problem, VariableType
from simopt.input_models import Exp, Poisson
from simopt.utils import classproperty, override


class SSCont(Model):
    """(s,S) Inventory Simulation Model.

    A model that simulates multiple periods' worth of sales for a (s,S)
    inventory problem with continuous inventory, exponentially distributed
    demand, and poisson distributed lead time. Returns the various types of
    average costs per period, order rate, stockout rate, fraction of demand
    met with inventory on hand, average amount backordered given a stockout
    occured, and average amount ordered given an order occured.
    """

    @classproperty
    @override
    def class_name(cls) -> str:
        return "(s, S) Inventory"

    @classproperty
    @override
    def n_rngs(cls) -> int:
        return 2

    @classproperty
    @override
    def n_responses(cls) -> int:
        return 7

    @classproperty
    @override
    def specifications(cls) -> dict[str, dict]:
        return {
            "demand_mean": {
                "description": (
                    "mean of exponentially distributed demand in each period"
                ),
                "datatype": float,
                "default": 100.0,
            },
            "lead_mean": {
                "description": "mean of Poisson distributed order lead time",
                "datatype": float,
                "default": 6.0,
            },
            "backorder_cost": {
                "description": (
                    "cost per unit of demand not met with in-stock inventory"
                ),
                "datatype": float,
                "default": 4.0,
            },
            "holding_cost": {
                "description": "holding cost per unit per period",
                "datatype": float,
                "default": 1.0,
            },
            "fixed_cost": {
                "description": "order fixed cost",
                "datatype": float,
                "default": 36.0,
            },
            "variable_cost": {
                "description": "order variable cost per unit",
                "datatype": float,
                "default": 2.0,
            },
            "s": {
                "description": "inventory threshold for placing order",
                "datatype": float,
                "default": 1000.0,
            },
            "S": {
                "description": "max inventory",
                "datatype": float,
                "default": 2000.0,
            },
            "n_days": {
                "description": "number of periods to simulate",
                "datatype": int,
                "default": 100,
                "isDatafarmable": False,
            },
            "warmup": {
                "description": (
                    "number of periods as warmup before collecting statistics"
                ),
                "datatype": int,
                "default": 20,
                "isDatafarmable": False,
            },
        }

    @property
    @override
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "demand_mean": self._check_demand_mean,
            "lead_mean": self._check_lead_mean,
            "backorder_cost": self._check_backorder_cost,
            "holding_cost": self._check_holding_cost,
            "fixed_cost": self._check_fixed_cost,
            "variable_cost": self._check_variable_cost,
            "s": self._check_s,
            "S": self._check_S,
            "n_days": self._check_n_days,
            "warmup": self._check_warmup,
        }

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize the (s,S) inventory simulation model.

        Args:
            fixed_factors (dict, optional): Fixed factors of the simulation model.
                Defaults to None.
        """
        # Let the base class handle default arguments.
        super().__init__(fixed_factors)

        self.demand_model = Exp()
        self.lead_model = Poisson()

    # Check for simulatable factors
    def _check_demand_mean(self) -> None:
        if self.factors["demand_mean"] <= 0:
            raise ValueError("demand_mean must be greater than 0.")

    def _check_lead_mean(self) -> None:
        if self.factors["lead_mean"] <= 0:
            raise ValueError("lead_mean must be greater than 0.")

    def _check_backorder_cost(self) -> None:
        if self.factors["backorder_cost"] <= 0:
            raise ValueError("backorder_cost must be greater than 0.")

    def _check_holding_cost(self) -> None:
        if self.factors["holding_cost"] <= 0:
            raise ValueError("holding_cost must be greater than 0.")

    def _check_fixed_cost(self) -> None:
        if self.factors["fixed_cost"] <= 0:
            raise ValueError("fixed_cost must be greater than 0.")

    def _check_variable_cost(self) -> None:
        if self.factors["variable_cost"] <= 0:
            raise ValueError("variable_cost must be greater than 0.")

    def _check_s(self) -> None:
        if self.factors["s"] <= 0:
            raise ValueError("s must be greater than 0.")

    def _check_S(self) -> None:  # noqa: N802
        if self.factors["S"] <= 0:
            raise ValueError("S must be greater than 0.")

    def _check_n_days(self) -> None:
        if self.factors["n_days"] < 1:
            raise ValueError("n_days must be greater than or equal to 1.")

    def _check_warmup(self) -> None:
        if self.factors["warmup"] < 0:
            raise ValueError("warmup must be greater than or equal to 0.")

    @override
    def check_simulatable_factors(self) -> bool:
        if self.factors["s"] >= self.factors["S"]:
            raise ValueError("s must be less than S.")
        return True

    def before_replicate(self, rng_list):
        self.demand_model.set_rng(rng_list[0])
        self.lead_model.set_rng(rng_list[1])

    def replicate(self) -> tuple[dict, dict]:
        """Simulate a single replication for the current model factors.

        Args:
            rng_list (list[MRG32k3a]): Random number generators used to simulate
                the replication.

        Returns:
            tuple[dict, dict]: A tuple containing:
                - responses (dict): Performance measures of interest, including:
                    - "avg_backorder_costs": Average backorder costs per period.
                    - "avg_order_costs": Average order costs per period.
                    - "avg_holding_costs": Average holding costs per period.
                    - "on_time_rate": Fraction of demand met with stock on hand
                        in store.
                    - "order_rate": Fraction of periods in which an order was made.
                    - "stockout_rate": Fraction of periods with a stockout.
                    - "avg_stockout": Mean amount of product backordered given a
                        stockout occurred.
                    - "avg_order": Mean amount of product ordered given an
                        order occurred.
                - gradients (dict): A dictionary of gradient estimates for
                    each response.
        """
        demand_mean = self.factors["demand_mean"]
        n_days = self.factors["n_days"]
        warmup = self.factors["warmup"]
        fac_s = self.factors["s"]
        fac_S = self.factors["S"]  # noqa: N806
        lead_mean = self.factors["lead_mean"]
        fixed_cost = self.factors["fixed_cost"]
        variable_cost = self.factors["variable_cost"]
        holding_cost = self.factors["holding_cost"]
        backorder_cost = self.factors["backorder_cost"]

        periods = n_days + warmup
        # Generate exponential random demands.
        inv_demand_mean = 1 / demand_mean
        demands = np.array(
            [self.demand_model.random(inv_demand_mean) for _ in range(periods)]
        )
        # Initialize starting and ending inventories for each period.
        start_inv = np.zeros(periods)
        start_inv[0] = fac_s  # Start with s units at period 0.
        end_inv = np.zeros(periods)
        # Initialize other quantities to track:
        #   - Amount of product to be received in each period.
        #   - Inventory position each period.
        #   - Amount of product ordered in each period.
        #   - Amount of product outstanding in each period.
        orders_received = np.zeros(periods)
        inv_pos = np.zeros(periods)
        orders_placed = np.zeros(periods)
        orders_outstanding = np.zeros(periods)
        # Run simulation over time horizon.
        for day in range(periods):
            next_day = day + 1

            # Inventory position
            end_inv[day] = start_inv[day] - demands[day]
            inv_pos[day] = end_inv[day] + orders_outstanding[day]

            if inv_pos[day] < fac_s:
                order_qty = fac_S - inv_pos[day]
                orders_placed[day] = order_qty

                lead = self.lead_model.random(lead_mean)
                delivery_day = next_day + lead

                if delivery_day < periods:
                    orders_received[delivery_day] += order_qty

                # Track future outstanding orders
                if next_day < periods:
                    orders_outstanding[next_day : min(delivery_day, periods)] += (
                        order_qty
                    )

            if next_day < periods:
                start_inv[next_day] = end_inv[day] + orders_received[next_day]

        # Calculate responses from simulation data.
        orders_post_warmup = orders_placed[warmup:]
        pos_orders_post_warmup_mask = orders_post_warmup > 0
        inv_post_warmup = end_inv[warmup:]
        neg_inv_post_warmup_mask = inv_post_warmup < 0
        pos_inv_post_warmup_mask = inv_post_warmup > 0

        order_rate = np.mean(pos_orders_post_warmup_mask)
        stockout_rate = np.mean(neg_inv_post_warmup_mask)

        fixed_costs = fixed_cost * pos_orders_post_warmup_mask
        variable_costs = variable_cost * orders_post_warmup
        avg_order_costs = np.mean(fixed_costs + variable_costs)

        avg_holding_costs = np.mean(
            holding_cost * inv_post_warmup * pos_inv_post_warmup_mask
        )
        demands_post_warmup = demands[warmup:]
        demand_start_inv_diff = demands_post_warmup - start_inv[warmup:]

        shortage = np.minimum(demands_post_warmup, demand_start_inv_diff)
        shortage[demand_start_inv_diff <= 0] = 0
        on_time_rate = 1 - shortage.sum() / np.sum(demands_post_warmup)

        avg_backorder_costs = (
            backorder_cost * (1 - on_time_rate) * np.sum(demands_post_warmup) / n_days
        )
        # Calculate average stockout costs.
        neg_inv_post_warmup_mask = np.where(neg_inv_post_warmup_mask)
        if len(neg_inv_post_warmup_mask[0]) == 0:
            avg_stockout = 0
        else:
            avg_stockout = -np.mean(inv_post_warmup[neg_inv_post_warmup_mask])
        # Calculate average backorder costs.
        pos_orders_placed_post_warmup = np.where(pos_orders_post_warmup_mask)
        if len(pos_orders_placed_post_warmup[0]) == 0:
            avg_order = 0
        else:
            avg_order = np.mean(orders_post_warmup[pos_orders_placed_post_warmup])
        # Compose responses and gradients.
        responses = {
            "avg_backorder_costs": avg_backorder_costs,
            "avg_order_costs": avg_order_costs,
            "avg_holding_costs": avg_holding_costs,
            "on_time_rate": on_time_rate,
            "order_rate": order_rate,
            "stockout_rate": stockout_rate,
            "avg_stockout": avg_stockout,
            "avg_order": avg_order,
        }
        gradients = {
            response_key: dict.fromkeys(self.specifications, np.nan)
            for response_key in responses
        }
        return responses, gradients


class SSContMinCost(Problem):
    """Class to make (s,S) inventory simulation-optimization problems."""

    @classproperty
    @override
    def class_name_abbr(cls) -> str:
        return "SSCONT-1"

    @classproperty
    @override
    def class_name(cls) -> str:
        return "Min Total Cost for (s, S) Inventory"

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
        return {"demand_mean": 100.0, "lead_mean": 6.0}

    @classproperty
    @override
    def model_decision_factors(cls) -> set[str]:
        return {"s", "S"}

    @classproperty
    @override
    def specifications(cls) -> dict[str, dict]:
        return {
            "initial_solution": {
                "description": "initial solution from which solvers start",
                "datatype": tuple,
                "default": (600, 600),
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
        return (0,) * cls.dim

    @classproperty
    @override
    def upper_bounds(cls) -> tuple:
        return (np.inf,) * cls.dim

    def __init__(
        self,
        name: str = "SSCONT-1",
        fixed_factors: dict | None = None,
        model_fixed_factors: dict | None = None,
    ) -> None:
        """Initialize the (s,S) inventory simulation-optimization problem.

        Args:
            name (str, optional): name of problem. Defaults to "SSCONT-1".
            fixed_factors (dict, optional): fixed factors of the simulation model.
                Defaults to None.
            model_fixed_factors (dict, optional): fixed factors for the simulation
                model. Defaults to None.
        """
        # Let the base class handle default arguments.
        super().__init__(
            name=name,
            fixed_factors=fixed_factors,
            model_fixed_factors=model_fixed_factors,
            model=SSCont,
        )

    @override
    def vector_to_factor_dict(self, vector: tuple) -> dict:
        return {"s": vector[0], "S": vector[0] + vector[1]}

    @override
    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:
        return (factor_dict["s"], factor_dict["S"] - factor_dict["s"])

    @override
    def response_dict_to_objectives(self, response_dict: dict) -> tuple:
        return (
            response_dict["avg_backorder_costs"]
            + response_dict["avg_order_costs"]
            + response_dict["avg_holding_costs"],
        )

    @override
    def deterministic_objectives_and_gradients(self, _x: tuple) -> tuple[tuple, tuple]:
        det_objectives = (0,)
        det_objectives_gradients = ((0,),)
        return det_objectives, det_objectives_gradients

    @override
    def check_deterministic_constraints(self, x: tuple) -> bool:
        return x[0] >= 0 and x[1] >= 0

    @override
    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
        # x = (rand_sol_rng.expovariate(1 / 300), rand_sol_rng.expovariate(1 / 300))
        # x = tuple(
        #     sorted(
        #         [
        #             rand_sol_rng.lognormalvariate(600, 1),
        #             rand_sol_rng.lognormalvariate(600, 1),
        #         ],
        #         key=float,
        #     )
        # )
        mu_d = self.model_default_factors["demand_mean"]
        mu_l = self.model_default_factors["lead_mean"]
        return (
            rand_sol_rng.lognormalvariate(
                mu_d * mu_l / 3, mu_d * mu_l + 2 * sqrt(2 * mu_d**2 * mu_l)
            ),
            rand_sol_rng.lognormalvariate(
                mu_d * mu_l / 3, mu_d * mu_l + 2 * sqrt(2 * mu_d**2 * mu_l)
            ),
        )


# If T is lead time and X is a single demand, then:
#   var(sum_{i=1}^T X_i) = E(T) var(X) + (E X))^2 var T
# var(S) = E var(S|T) + var E(S|T)
