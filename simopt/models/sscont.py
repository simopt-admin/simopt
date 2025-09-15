"""Simulate sales for a (s,S) inventory problem with continuous inventory."""

from __future__ import annotations

from math import sqrt
from typing import Annotated, ClassVar, Self

import numpy as np
from pydantic import BaseModel, Field, model_validator

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.base import (
    ConstraintType,
    Model,
    Objective,
    Problem,
    RepResult,
    VariableType,
)
from simopt.input_models import Exp, Poisson


class SSContConfig(BaseModel):
    """Configuration for the (s, S) continuous inventory model."""

    demand_mean: Annotated[
        float,
        Field(
            default=100.0,
            description="mean of exponentially distributed demand in each period",
            gt=0,
        ),
    ]
    lead_mean: Annotated[
        float,
        Field(
            default=6.0,
            description="mean of Poisson distributed order lead time",
            gt=0,
        ),
    ]
    backorder_cost: Annotated[
        float,
        Field(
            default=4.0,
            description="cost per unit of demand not met with in-stock inventory",
            gt=0,
        ),
    ]
    holding_cost: Annotated[
        float,
        Field(
            default=1.0,
            description="holding cost per unit per period",
            gt=0,
        ),
    ]
    fixed_cost: Annotated[
        float,
        Field(
            default=36.0,
            description="order fixed cost",
            gt=0,
        ),
    ]
    variable_cost: Annotated[
        float,
        Field(
            default=2.0,
            description="order variable cost per unit",
            gt=0,
        ),
    ]
    s: Annotated[
        float,
        Field(
            default=1000.0,
            description="inventory threshold for placing order",
            gt=0,
        ),
    ]
    S: Annotated[
        float,
        Field(
            default=2000.0,
            description="max inventory",
            gt=0,
        ),
    ]
    n_days: Annotated[
        int,
        Field(
            default=100,
            description="number of periods to simulate",
            ge=1,
            json_schema_extra={"isDatafarmable": False},
        ),
    ]
    warmup: Annotated[
        int,
        Field(
            default=20,
            description="number of periods as warmup before collecting statistics",
            ge=0,
            json_schema_extra={"isDatafarmable": False},
        ),
    ]

    @model_validator(mode="after")
    def _validate_model(self) -> Self:
        if self.s >= self.S:
            raise ValueError("s must be less than S.")
        return self


class SSContMinCostConfig(BaseModel):
    """Configuration model for SSCont Min Cost Problem.

    Min Total Cost for (s, S) Inventory simulation-optimization problem.
    """

    initial_solution: Annotated[
        tuple[float, ...],
        Field(
            default=(600, 600),
            description="initial solution from which solvers start",
        ),
    ]
    budget: Annotated[
        int,
        Field(
            default=1000,
            description="max # of replications for a solver to take",
            gt=0,
            json_schema_extra={"isDatafarmable": False},
        ),
    ]


class SSCont(Model):
    """(s,S) Inventory Simulation Model.

    A model that simulates multiple periods' worth of sales for a (s,S)
    inventory problem with continuous inventory, exponentially distributed
    demand, and poisson distributed lead time. Returns the various types of
    average costs per period, order rate, stockout rate, fraction of demand
    met with inventory on hand, average amount backordered given a stockout
    occured, and average amount ordered given an order occured.
    """

    class_name_abbr: ClassVar[str] = "SSCONT"
    class_name: ClassVar[str] = "(s, S) Inventory"
    config_class: ClassVar[type[BaseModel]] = SSContConfig
    n_rngs: ClassVar[int] = 2
    n_responses: ClassVar[int] = 7

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

    def before_replicate(self, rng_list: list[MRG32k3a]) -> None:  # noqa: D102
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
        return responses, {}


class SSContMinCost(Problem):
    """Class to make (s,S) inventory simulation-optimization problems."""

    class_name_abbr: ClassVar[str] = "SSCONT-1"
    class_name: ClassVar[str] = "Min Total Cost for (s, S) Inventory"
    config_class: ClassVar[type[BaseModel]] = SSContMinCostConfig
    model_class: ClassVar[type[Model]] = SSCont
    n_objectives: ClassVar[int] = 1
    n_stochastic_constraints: ClassVar[int] = 0
    minmax: ClassVar[tuple[int, ...]] = (-1,)
    constraint_type: ClassVar[ConstraintType] = ConstraintType.BOX
    variable_type: ClassVar[VariableType] = VariableType.CONTINUOUS
    gradient_available: ClassVar[bool] = False
    optimal_value: ClassVar[float | None] = None
    optimal_solution: tuple | None = None
    model_default_factors: ClassVar[dict] = {"demand_mean": 100.0, "lead_mean": 6.0}
    model_decision_factors: ClassVar[set[str]] = {"s", "S"}

    @property
    def dim(self) -> int:  # noqa: D102
        return 2

    @property
    def lower_bounds(self) -> tuple:  # noqa: D102
        return (0,) * self.dim

    @property
    def upper_bounds(self) -> tuple:  # noqa: D102
        return (np.inf,) * self.dim

    def vector_to_factor_dict(self, vector: tuple) -> dict:  # noqa: D102
        return {"s": vector[0], "S": vector[0] + vector[1]}

    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:  # noqa: D102
        return (factor_dict["s"], factor_dict["S"] - factor_dict["s"])

    def replicate(self, _x: tuple) -> RepResult:  # noqa: D102
        responses, _ = self.model.replicate()
        objectives = [
            Objective(
                stochastic=(
                    responses["avg_backorder_costs"]
                    + responses["avg_order_costs"]
                    + responses["avg_holding_costs"]
                )
            )
        ]
        return RepResult(objectives=objectives)

    def check_deterministic_constraints(self, x: tuple) -> bool:  # noqa: D102
        return x[0] >= 0 and x[1] >= 0

    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:  # noqa: D102
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
