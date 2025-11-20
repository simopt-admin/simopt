"""Simulate periods of ordering and sales for a dual sourcing inventory problem."""

from __future__ import annotations

from random import Random
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
from simopt.input_models import InputModel


class DualSourcingConfig(BaseModel):
    """Configuration model for Dual Sourcing Inventory simulation.

    A model that simulates multiple periods of ordering and sales for a single-staged,
    dual sourcing inventory problem with stochastic demand. Returns average holding
    cost, average penalty cost, and average ordering cost per period.
    """

    n_days: Annotated[
        int,
        Field(
            default=1000,
            description="number of days to simulate",
            ge=1,
            json_schema_extra={"isDatafarmable": False},
        ),
    ]
    initial_inv: Annotated[
        int,
        Field(
            default=40,
            description="initial inventory",
            ge=0,
        ),
    ]
    cost_reg: Annotated[
        float,
        Field(
            default=100.00,
            description="regular ordering cost per unit",
            gt=0,
        ),
    ]
    cost_exp: Annotated[
        float,
        Field(
            default=110.00,
            description="expedited ordering cost per unit",
            gt=0,
        ),
    ]
    lead_reg: Annotated[
        int,
        Field(
            default=2,
            description="lead time for regular orders in days",
            ge=0,
        ),
    ]
    lead_exp: Annotated[
        int,
        Field(
            default=0,
            description="lead time for expedited orders in days",
            ge=0,
        ),
    ]
    holding_cost: Annotated[
        float,
        Field(
            default=5.00,
            description="holding cost per unit per period",
            gt=0,
        ),
    ]
    penalty_cost: Annotated[
        float,
        Field(
            default=495.00,
            description="penalty cost per unit per period for backlogging",
            gt=0,
        ),
    ]
    st_dev: Annotated[
        float,
        Field(
            default=10.0,
            description="standard deviation of demand distribution",
            gt=0,
        ),
    ]
    mu: Annotated[
        float,
        Field(
            default=30.0,
            description="mean of demand distribution",
            gt=0,
        ),
    ]
    order_level_reg: Annotated[
        int,
        Field(
            default=80,
            description="order-up-to level for regular orders",
            ge=0,
        ),
    ]
    order_level_exp: Annotated[
        int,
        Field(
            default=50,
            description="order-up-to level for expedited orders",
            ge=0,
        ),
    ]

    @model_validator(mode="after")
    def _validate_model(self) -> Self:
        # Cross-validation: check lead time and cost constraints
        if (self.lead_exp > self.lead_reg) or (self.cost_exp < self.cost_reg):
            raise ValueError(
                "lead_exp must be less than lead_reg and cost_exp must be greater than "
                "cost_reg"
            )

        return self


class DualSourcingMinCostConfig(BaseModel):
    """Configuration model for Dual Sourcing Min Cost Problem.

    A problem configuration that minimizes total cost for dual sourcing inventory
    by optimizing order levels for regular and expedited orders.
    """

    initial_solution: Annotated[
        tuple[int, int],
        Field(
            default=(50, 80),
            description="initial solution",
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


class DemandInputModel(InputModel):
    """Input model for daily demand."""

    rng: Random | None = None

    def random(self, mu: float, sigma: float) -> int:  # noqa: D102
        def round_and_clamp_non_neg(x: float | int) -> int:
            return round(max(0.0, float(x)))

        assert self.rng is not None
        return round_and_clamp_non_neg(self.rng.normalvariate(mu, sigma))


class DualSourcing(Model):
    """Dual Sourcing Inventory Model.

    A model that simulates multiple periods of ordering and sales for a single-staged,
    dual sourcing inventory problem with stochastic demand. Returns average holding
    cost, average penalty cost, and average ordering cost per period.
    """

    class_name_abbr: ClassVar[str] = "DUALSOURCING"
    class_name: ClassVar[str] = "Dual Sourcing"
    config_class: ClassVar[type[BaseModel]] = DualSourcingConfig
    n_rngs: ClassVar[int] = 1
    n_responses: ClassVar[int] = 3

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize the DualSourcing model.

        Args:
            fixed_factors (dict, optional): Fixed factors for the model.
                Defaults to None.
        """
        # Let the base class handle default arguments.
        super().__init__(fixed_factors)

        self.demand_model = DemandInputModel()

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
            return round(max(0.0, float(x)))

        # Vectors of regular orders to be received in periods n through n + lr - 1.
        orders_reg: list[int] = [0] * lead_reg
        # Vectors of expedited orders to be received in periods n through n + le - 1.
        orders_exp: list[int] = [0] * lead_exp

        # Generate demand.
        demand = [self.demand_model.random(mu, st_dev) for _ in n_days_range]

        # Track total expenses.
        total_holding_cost = np.zeros(n_days)
        total_penalty_cost = np.zeros(n_days)
        total_ordering_cost = np.zeros(n_days)
        inv: int = initial_inv

        # Run simulation over time horizon.
        for day in n_days_range:
            # Calculate inventory positions.
            inv_order_exp_sum = inv + sum(orders_exp)
            inv_position_exp = round(inv_order_exp_sum + sum(orders_reg[:lead_exp]))
            inv_position_reg = round(inv_order_exp_sum + sum(orders_reg))
            # Calculate how much to order.
            order_exp: int = round_and_clamp_non_neg(
                order_level_exp - inv_position_exp - orders_reg[lead_exp]
            )
            orders_exp.append(order_exp)
            order_reg: int = round_and_clamp_non_neg(
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
            total_penalty_cost[day] = -penalty_cost * min(0, int(inv))
            total_holding_cost[day] = holding_cost * max(0, int(inv))

        # Calculate responses from simulation data.
        responses = {
            "average_ordering_cost": np.mean(total_ordering_cost),
            "average_penalty_cost": np.mean(total_penalty_cost),
            "average_holding_cost": np.mean(total_holding_cost),
        }
        return responses, {}


class DualSourcingMinCost(Problem):
    """Class to make dual-sourcing inventory simulation-optimization problems."""

    class_name_abbr: ClassVar[str] = "DUALSOURCING-1"
    class_name: ClassVar[str] = "Min Cost for Dual Sourcing"
    config_class: ClassVar[type[BaseModel]] = DualSourcingMinCostConfig
    model_class: ClassVar[type[Model]] = DualSourcing
    n_objectives: ClassVar[int] = 1
    n_stochastic_constraints: ClassVar[int] = 0
    minmax: ClassVar[tuple[int, ...]] = (-1,)
    constraint_type: ClassVar[ConstraintType] = ConstraintType.BOX
    variable_type: ClassVar[VariableType] = VariableType.DISCRETE
    gradient_available: ClassVar[bool] = False
    optimal_value: ClassVar[float | None] = None
    optimal_solution: tuple | None = None
    model_default_factors: ClassVar[dict] = {}
    model_decision_factors: ClassVar[set[str]] = {"order_level_exp", "order_level_reg"}

    @property
    def dim(self) -> int:  # noqa: D102
        return 2

    @property
    def lower_bounds(self) -> tuple:  # noqa: D102
        return (0, 0)

    @property
    def upper_bounds(self) -> tuple:  # noqa: D102
        return (np.inf, np.inf)

    def vector_to_factor_dict(self, vector: tuple) -> dict:  # noqa: D102
        return {
            "order_level_exp": vector[0],
            "order_level_reg": vector[1],
        }

    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:  # noqa: D102
        return (
            factor_dict["order_level_exp"],
            factor_dict["order_level_reg"],
        )

    def replicate(self, _x: tuple) -> RepResult:  # noqa: D102
        responses, _ = self.model.replicate()
        return RepResult(
            objectives=[
                Objective(
                    stochastic=responses["average_ordering_cost"]
                    + responses["average_penalty_cost"]
                    + responses["average_holding_cost"],
                )
            ],
        )

    def check_deterministic_constraints(self, x: tuple) -> bool:  # noqa: D102
        return x[0] >= 0 and x[1] >= 0

    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:  # noqa: D102
        return (rand_sol_rng.randint(40, 60), rand_sol_rng.randint(70, 90))
