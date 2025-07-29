"""Simulate a multi-stage revenue management system with inter-temporal dependence."""

from __future__ import annotations

from typing import Callable

import numpy as np

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.base import ConstraintType, Model, Problem, VariableType
from simopt.input_models import InputModel
from simopt.utils import classproperty, override


class DemandInputModel(InputModel):
    def set_rng(self, rng: MRG32k3a) -> None:
        self.x_rng = rng[0]
        self.y_rng = rng[1]

    def unset_rng(self) -> None:
        self.x_rng = None
        self.y_rng = None

    def random(self, demand_means, gamma_shape, gamma_scale) -> float:
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

    @classproperty
    @override
    def class_name(cls) -> str:
        return "Revenue Management Temporal Demand"

    @classproperty
    @override
    def n_rngs(cls) -> int:
        return 2

    @classproperty
    @override
    def n_responses(cls) -> int:
        return 1

    @classproperty
    @override
    def specifications(cls) -> dict[str, dict]:
        return {
            "time_horizon": {
                "description": "time horizon",
                "datatype": int,
                "default": 3,
            },
            "prices": {
                "description": "prices for each period",
                "datatype": list,
                "default": [100, 300, 400],
            },
            "demand_means": {
                "description": "mean demand for each period",
                "datatype": list,
                "default": [50, 20, 30],
            },
            "cost": {
                "description": "cost per unit of capacity at t = 0",
                "datatype": float,
                "default": 80.0,
            },
            "gamma_shape": {
                "description": "shape parameter of gamma distribution",
                "datatype": float,
                "default": 1.0,
            },
            "gamma_scale": {
                "description": "scale parameter of gamma distribution",
                "datatype": float,
                "default": 1.0,
            },
            "initial_inventory": {
                "description": "initial inventory",
                "datatype": int,
                "default": 100,
            },
            "reservation_qtys": {
                "description": "inventory to reserve going into periods 2, 3, ..., T",
                "datatype": list,
                "default": [50, 30],
            },
        }

    @property
    @override
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "time_horizon": self._check_time_horizon,
            "prices": self._check_prices,
            "demand_means": self._check_demand_means,
            "cost": self._check_cost,
            "gamma_shape": self._check_gamma_shape,
            "gamma_scale": self._check_gamma_scale,
            "initial_inventory": self._check_initial_inventory,
            "reservation_qtys": self._check_reservation_qtys,
        }

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize the RMITD model.

        Args:
            fixed_factors (dict, optional): Dictionary of fixed factors for the model.
                Defaults to None.
        """
        # Let the base class handle default arguments.
        super().__init__(fixed_factors)

        self.demand_model = DemandInputModel()

    def _check_time_horizon(self) -> None:
        if self.factors["time_horizon"] <= 0:
            raise ValueError("time_horizon must be greater than 0.")

    def _check_prices(self) -> None:
        if any(price <= 0 for price in self.factors["prices"]):
            raise ValueError("All elements in prices must be greater than 0.")

    def _check_demand_means(self) -> None:
        if any(demand_mean <= 0 for demand_mean in self.factors["demand_means"]):
            raise ValueError("All elements in demand_means must be greater than 0.")

    def _check_cost(self) -> None:
        if self.factors["cost"] <= 0:
            raise ValueError("cost must be greater than 0.")

    def _check_gamma_shape(self) -> None:
        if self.factors["gamma_shape"] <= 0:
            raise ValueError("gamma_shape must be greater than 0.")

    def _check_gamma_scale(self) -> None:
        if self.factors["gamma_scale"] <= 0:
            raise ValueError("gamma_scale must be greater than 0.")

    def _check_initial_inventory(self) -> None:
        if self.factors["initial_inventory"] <= 0:
            raise ValueError("initial_inventory must be greater than 0.")

    def _check_reservation_qtys(self) -> None:
        if any(
            reservation_qty <= 0 for reservation_qty in self.factors["reservation_qtys"]
        ):
            raise ValueError("All elements in reservation_qtys must be greater than 0.")

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

    def before_replicate(self, rng_list):
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
        time_horizon = self.factors["time_horizon"]
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

    @classproperty
    @override
    def class_name_abbr(cls) -> str:
        return "RMITD-1"

    @classproperty
    @override
    def class_name(cls) -> str:
        return "Max Revenue for Revenue Management Temporal Demand"

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
        return ConstraintType.DETERMINISTIC

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
    def optimal_solution(cls) -> None:
        return None

    @classproperty
    @override
    def model_default_factors(cls) -> dict:
        return {}

    @classproperty
    @override
    def model_decision_factors(cls) -> set[str]:
        return {"initial_inventory", "reservation_qtys"}

    @classproperty
    @override
    def specifications(cls) -> dict[str, dict]:
        return {
            "initial_solution": {
                "description": "initial solution",
                "datatype": tuple,
                "default": (100, 50, 30),
            },
            "budget": {
                "description": "max # of replications for a solver to take",
                "datatype": int,
                "default": 10000,
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
        return 3

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
        name: str = "RMITD-1",
        fixed_factors: dict | None = None,
        model_fixed_factors: dict | None = None,
    ) -> None:
        """Initialize the RMITDMaxRevenue problem.

        Args:
            name (str): Name of the problem.
            fixed_factors (dict, optional): Dictionary of fixed factors for the model.
                Defaults to None.
            model_fixed_factors (dict, optional): Dictionary of fixed factors for the
                model. Defaults to None.
        """
        # Let the base class handle default arguments.
        super().__init__(
            name=name,
            fixed_factors=fixed_factors,
            model_fixed_factors=model_fixed_factors,
            model=RMITD,
        )

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
