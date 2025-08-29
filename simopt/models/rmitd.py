"""Simulate a multi-stage revenue management system with inter-temporal dependence."""

from __future__ import annotations

from typing import Annotated, ClassVar, Self

import numpy as np
from pydantic import BaseModel, Field, model_validator

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.base import ConstraintType, Model, Problem, VariableType
from simopt.input_models import InputModel
from simopt.utils import override


class RMITDConfig(BaseModel):
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

    def set_rng(self, rng: MRG32k3a) -> None:  # noqa: D102
        self.x_rng = rng[0]
        self.y_rng = rng[1]

    def unset_rng(self) -> None:  # noqa: D102
        self.x_rng = None
        self.y_rng = None

    def random(
        self, demand_means: np.ndarray, gamma_shape: float, gamma_scale: float
    ) -> float:
        """Sample period demand vector given means and gamma parameters."""
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

    config_class: ClassVar[type[BaseModel]] = RMITDConfig
    class_name: str = "Revenue Management Temporal Demand"
    n_rngs: int = 2
    n_responses: int = 1

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize the RMITD model.

        Args:
            fixed_factors (dict, optional): Dictionary of fixed factors for the model.
                Defaults to None.
        """
        # Let the base class handle default arguments.
        super().__init__(fixed_factors)

        self.demand_model = DemandInputModel()

    @override
    def check_simulatable_factors(self) -> bool:
        # Check for matching number of periods.
        if len(self.factors["prices"]) != self.factors["time_horizon"]:
            raise ValueError("The length of prices must be equal to time_horizon.")
        if len(self.factors["demand_means"]) != self.factors["time_horizon"]:
            raise ValueError(
                "The length of demand_means must be equal to time_horizon."
            )
        if len(self.factors["reservation_qtys"]) != self.factors["time_horizon"] - 1:
            raise ValueError(
                "The length of reservation_qtys must be equal to the time_horizon "
                "minus 1."
            )
        # Check that first reservation level is less than initial inventory.
        if self.factors["initial_inventory"] < self.factors["reservation_qtys"][0]:
            raise ValueError(
                "The initial_inventory must be greater than or equal to the first "
                "element in reservation_qtys."
            )
        # Check for non-increasing reservation levels.
        if any(
            self.factors["reservation_qtys"][idx]
            < self.factors["reservation_qtys"][idx + 1]
            for idx in range(self.factors["time_horizon"] - 2)
        ):
            raise ValueError(
                "Each value in reservation_qtys must be greater than the next value "
                "in the list."
            )
        # Check that gamma_shape*gamma_scale = 1.
        if (
            np.isclose(self.factors["gamma_shape"] * self.factors["gamma_scale"], 1)
            is False
        ):
            raise ValueError("gamma_shape times gamma_scale should be close to 1.")
        return True

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
        for reservation, demand, price in zip(reservations, demand_vec, prices):
            available = max(remaining_inventory - reservation, 0)
            sell = min(available, demand)
            remaining_inventory -= sell
            revenue += sell * price

        revenue -= cost * initial_inventory

        # Compose responses and gradients.
        responses = {"revenue": revenue}
        gradients = {
            response_key: dict.fromkeys(self.specifications, np.nan)
            for response_key in responses
        }
        return responses, gradients


class RMITDMaxRevenue(Problem):
    """Base class to implement simulation-optimization problems."""

    config_class: ClassVar[type[BaseModel]] = RMITDMaxRevenueConfig
    model_class: ClassVar[type[Model]] = RMITD
    class_name_abbr: str = "RMITD-1"
    class_name: str = "Max Revenue for Revenue Management Temporal Demand"
    n_objectives: int = 1
    n_stochastic_constraints: int = 0
    minmax: tuple[int] = (1,)
    constraint_type: ConstraintType = ConstraintType.DETERMINISTIC
    variable_type: VariableType = VariableType.DISCRETE
    gradient_available: bool = False
    optimal_value: float | None = None
    optimal_solution: tuple | None = None
    model_default_factors: dict = {}
    model_decision_factors: set[str] = {"initial_inventory", "reservation_qtys"}
    dim: int = 3
    lower_bounds: tuple = (0,) * dim
    upper_bounds: tuple = (np.inf,) * dim

    @override
    def vector_to_factor_dict(self, vector: tuple) -> dict:
        return {
            "initial_inventory": vector[0],
            "reservation_qtys": list(vector[0:]),
        }

    @override
    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:
        return (
            factor_dict["initial_inventory"],
            *tuple(factor_dict["reservation_qtys"]),
        )

    @override
    def response_dict_to_objectives(self, response_dict: dict) -> tuple:
        return (response_dict["revenue"],)

    @override
    def check_deterministic_constraints(self, x: tuple) -> bool:
        return all(x[idx] >= x[idx + 1] for idx in range(self.dim - 1))

    @override
    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
        # Generate random solution using acceptable/rejection.
        while True:
            x = tuple([200 * rand_sol_rng.random() for _ in range(self.dim)])
            if self.check_deterministic_constraints(x):
                break
        return x
