"""Simulate a multi-stage revenue management system with inter-temporal dependence."""

from __future__ import annotations

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


class RMITDConfig(BaseModel):
    """Configuration for the RMITD model."""

    time_horizon: Annotated[
        int,
        Field(
            default=3,
            description="time horizon",
            gt=0,
        ),
    ]
    prices: Annotated[
        list[float],
        Field(
            default=[100, 300, 400],
            description="prices for each period",
        ),
    ]
    demand_means: Annotated[
        list[float],
        Field(
            default=[50, 20, 30],
            description="mean demand for each period",
        ),
    ]
    cost: Annotated[
        float,
        Field(
            default=80.0,
            description="cost per unit of capacity at t = 0",
            gt=0,
        ),
    ]
    gamma_shape: Annotated[
        float,
        Field(
            default=1.0,
            description="shape parameter of gamma distribution",
            gt=0,
        ),
    ]
    gamma_scale: Annotated[
        float,
        Field(
            default=1.0,
            description="scale parameter of gamma distribution",
            gt=0,
        ),
    ]
    initial_inventory: Annotated[
        int,
        Field(
            default=100,
            description="initial inventory",
            gt=0,
        ),
    ]
    reservation_qtys: Annotated[
        list[int],
        Field(
            default=[50, 30],
            description="inventory to reserve going into periods 2, 3, ..., T",
        ),
    ]

    def _check_prices(self) -> None:
        if any(price <= 0 for price in self.prices):
            raise ValueError("All elements in prices must be greater than 0.")

    def _check_demand_means(self) -> None:
        if any(demand_mean <= 0 for demand_mean in self.demand_means):
            raise ValueError("All elements in demand_means must be greater than 0.")

    def _check_reservation_qtys(self) -> None:
        if any(reservation_qty <= 0 for reservation_qty in self.reservation_qtys):
            raise ValueError("All elements in reservation_qtys must be greater than 0.")

    @model_validator(mode="after")
    def _validate_model(self) -> Self:
        self._check_prices()
        self._check_demand_means()
        self._check_reservation_qtys()

        if len(self.prices) != self.time_horizon:
            raise ValueError("The length of prices must be equal to time_horizon.")
        if len(self.demand_means) != self.time_horizon:
            raise ValueError(
                "The length of demand_means must be equal to time_horizon."
            )
        if len(self.reservation_qtys) != self.time_horizon - 1:
            raise ValueError(
                "The length of reservation_qtys must be equal to the time_horizon "
                "minus 1."
            )

        if self.initial_inventory < self.reservation_qtys[0]:
            raise ValueError(
                "The initial_inventory must be greater than or equal to the first "
                "element in reservation_qtys."
            )

        if any(
            self.reservation_qtys[idx] < self.reservation_qtys[idx + 1]
            for idx in range(self.time_horizon - 2)
        ):
            raise ValueError(
                "Each value in reservation_qtys must be greater than the next value "
                "in the list."
            )

        if not np.isclose(self.gamma_shape * self.gamma_scale, 1):
            raise ValueError("gamma_shape times gamma_scale should be close to 1.")

        return self


class RMITDMaxRevenueConfig(BaseModel):
    """Configuration model for RMITD Max Revenue Problem.

    Max Revenue for Revenue Management Temporal Demand simulation-optimization problem.
    """

    initial_solution: Annotated[
        tuple[int, ...],
        Field(
            default=(100, 50, 30),
            description="initial solution",
        ),
    ]
    budget: Annotated[
        int,
        Field(
            default=10000,
            description="max # of replications for a solver to take",
            gt=0,
            json_schema_extra={"isDatafarmable": False},
        ),
    ]


class DemandInputModel(InputModel):
    """Input model for temporally dependent demand components."""

    x_rng: MRG32k3a | None = None
    y_rng: MRG32k3a | None = None

    def set_rng(self, rng: list[MRG32k3a] | tuple[MRG32k3a, MRG32k3a]) -> None:  # noqa: D102 # type: ignore
        self.x_rng = rng[0]
        self.y_rng = rng[1]

    def unset_rng(self) -> None:  # noqa: D102
        self.x_rng = None
        self.y_rng = None

    def random(  # noqa: D102
        self, demand_means: np.ndarray, gamma_shape: float, gamma_scale: float
    ) -> np.ndarray:
        assert self.x_rng is not None and self.y_rng is not None
        x_demand = self.x_rng.gammavariate(
            alpha=gamma_shape,
            beta=1.0 / gamma_scale,
        )
        y_demand = np.array(
            [self.y_rng.expovariate(1) for _ in range(len(demand_means))]
        )
        return demand_means * x_demand * y_demand


class RMITD(Model):
    """Multi-stage Revenue Management with Inter-temporal Dependence (RMITD).

    A model that simulates a multi-stage revenue management system with
    inter-temporal dependence. Returns the total revenue.
    """

    class_name_abbr: ClassVar[str] = "RMITD"
    class_name: ClassVar[str] = "Revenue Management Temporal Demand"
    config_class: ClassVar[type[BaseModel]] = RMITDConfig
    n_rngs: ClassVar[int] = 2
    n_responses: ClassVar[int] = 1

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize the RMITD model.

        Args:
            fixed_factors (dict, optional): Dictionary of fixed factors for the model.
                Defaults to None.
        """
        # Let the base class handle default arguments.
        super().__init__(fixed_factors)

        self.demand_model = DemandInputModel()

    def before_replicate(self, rng_list: list[MRG32k3a]) -> None:  # noqa: D102
        self.demand_model.set_rng(rng_list)

    def replicate(self) -> tuple[dict, dict]:
        """Simulate a single replication for the current model factors.

        Args:
            rng_list (list[MRG32k3a]): Random number generators used to simulate
                the replication.

        Returns:
            tuple[dict, dict]: A tuple containing:
                - responses (dict): Performance measures of interest, including:
                    - "revenue": Total revenue.
                - gradients (dict): A dictionary of gradient estimates for
                    each response.
        """
        gamma_shape = self.factors["gamma_shape"]
        gamma_scale = self.factors["gamma_scale"]
        initial_inventory = self.factors["initial_inventory"]
        reservation_qtys: list = self.factors["reservation_qtys"]
        demand_means = np.array(self.factors["demand_means"])
        prices = self.factors["prices"]
        cost = self.factors["cost"]
        # Generate X and Y (to use for computing demand).
        # random.gammavariate takes two inputs: alpha and beta.
        #     alpha = k = gamma_shape
        #     beta = 1/theta = 1/gamma_scale
        reservations = [*reservation_qtys, 0]
        demand_vec = self.demand_model.random(demand_means, gamma_shape, gamma_scale)

        # Set initial inventory and revenue
        remaining_inventory = initial_inventory
        revenue = 0.0

        # Compute revenue for each period.
        for reservation, demand, price in zip(
            reservations, list(demand_vec), prices, strict=False
        ):
            available = max(remaining_inventory - reservation, 0)
            sell = min(available, demand)
            remaining_inventory -= sell
            revenue += sell * price

        revenue -= cost * initial_inventory

        # Compose responses and gradients.
        responses = {"revenue": revenue}
        return responses, {}


class RMITDMaxRevenue(Problem):
    """Base class to implement simulation-optimization problems."""

    class_name_abbr: ClassVar[str] = "RMITD-1"
    class_name: ClassVar[str] = "Max Revenue for Revenue Management Temporal Demand"
    config_class: ClassVar[type[BaseModel]] = RMITDMaxRevenueConfig
    model_class: ClassVar[type[Model]] = RMITD
    n_objectives: ClassVar[int] = 1
    n_stochastic_constraints: ClassVar[int] = 0
    minmax: ClassVar[tuple[int, ...]] = (1,)
    constraint_type: ClassVar[ConstraintType] = ConstraintType.DETERMINISTIC
    variable_type: ClassVar[VariableType] = VariableType.DISCRETE
    gradient_available: ClassVar[bool] = False
    optimal_value: ClassVar[float | None] = None
    optimal_solution: tuple | None = None
    model_default_factors: ClassVar[dict] = {}
    model_decision_factors: ClassVar[set[str]] = {
        "initial_inventory",
        "reservation_qtys",
    }

    @property
    def dim(self) -> int:  # noqa: D102
        return 3

    @property
    def lower_bounds(self) -> tuple:  # noqa: D102
        return (0,) * self.dim

    @property
    def upper_bounds(self) -> tuple:  # noqa: D102
        return (np.inf,) * self.dim

    def vector_to_factor_dict(self, vector: tuple) -> dict:  # noqa: D102
        return {
            "initial_inventory": vector[0],
            "reservation_qtys": list(vector[0:]),
        }

    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:  # noqa: D102
        return (
            factor_dict["initial_inventory"],
            *tuple(factor_dict["reservation_qtys"]),
        )

    def replicate(self, _x: tuple) -> RepResult:  # noqa: D102
        responses, _ = self.model.replicate()
        objectives = [Objective(stochastic=responses["revenue"])]
        return RepResult(objectives=objectives)

    def check_deterministic_constraints(self, x: tuple) -> bool:  # noqa: D102
        return all(x[idx] >= x[idx + 1] for idx in range(self.dim - 1))

    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:  # noqa: D102
        # Generate random solution using acceptable/rejection.
        while True:
            x = tuple([200 * rand_sol_rng.random() for _ in range(self.dim)])
            if self.check_deterministic_constraints(x):
                break
        return x
