"""Simulate demand at facilities."""

from __future__ import annotations

from random import Random
from typing import Annotated, ClassVar, Final, Self

import numpy as np
from pydantic import BaseModel, Field, model_validator

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.base import (
    ConstraintType,
    Model,
    Objective,
    Problem,
    RepResult,
    StochasticConstraint,
    VariableType,
)
from simopt.input_models import InputModel

NUM_FACILITIES: Final[int] = 3


class FacilitySizeConfig(BaseModel):
    """Configuration model for Facility Sizing simulation.

    A model that simulates a facility size problem with a multi-variate normal
    distribution. Returns the probability of violating demand in each scenario.
    """

    mean_vec: Annotated[
        list[float],
        Field(
            default_factory=lambda: [100] * NUM_FACILITIES,
            description=("location parameters of the multivariate normal distribution"),
        ),
    ]
    cov: Annotated[
        list[list[float]],
        Field(
            default_factory=lambda: [
                [2000, 1500, 500],
                [1500, 2000, 750],
                [500, 750, 2000],
            ],
            description="covariance of multivariate normal distribution",
        ),
    ]
    capacity: Annotated[
        list[float],
        Field(
            default=[150, 300, 400],
            description="capacity",
        ),
    ]
    n_fac: Annotated[
        int,
        Field(
            default=NUM_FACILITIES,
            description="number of facilities",
            gt=0,
            json_schema_extra={"isDatafarmable": False},
        ),
    ]

    def _check_mean_vec(self) -> None:
        if any(mean <= 0 for mean in self.mean_vec):
            raise ValueError("All elements in mean_vec must be greater than 0.")

    def _check_cov(self) -> None:
        try:
            np.linalg.cholesky(np.array(self.cov))
        except np.linalg.LinAlgError as err:
            if "Matrix is not positive definite" in str(err):
                raise ValueError("Covariance matrix is not positive definite.") from err

    def _check_capacity(self) -> None:
        if len(self.capacity) != self.n_fac:
            raise ValueError("The length of capacity must equal n_fac.")

    @model_validator(mode="after")
    def _validate_model(self) -> Self:
        self._check_mean_vec()
        self._check_cov()
        self._check_capacity()

        # Cross-validation: check dimensions match n_fac
        if len(self.capacity) != self.n_fac:
            raise ValueError("The length of capacity must be equal to n_fac.")
        if len(self.mean_vec) != self.n_fac:
            raise ValueError("The length of mean_vec must be equal to n_fac.")
        if len(self.cov) != self.n_fac:
            raise ValueError("The length of cov must be equal to n_fac.")
        if len(self.cov[0]) != self.n_fac:
            raise ValueError("The length of cov[0] must be equal to n_fac.")

        return self


class FacilitySizingMaxServiceConfig(BaseModel):
    """Configuration model for Facility Sizing Max Service Problem.

    Max Service for Facility Sizing simulation-optimization problem.
    """

    initial_solution: Annotated[
        tuple[float, ...],
        Field(
            default_factory=lambda: (100,) * NUM_FACILITIES,
            description="Initial solution from which solvers start.",
        ),
    ]
    budget: Annotated[
        int,
        Field(
            default=10000,
            description="Max # of replications for a solver to take.",
            gt=0,
        ),
    ]
    installation_costs: Annotated[
        tuple[float, ...],
        Field(
            default_factory=lambda: (1,) * NUM_FACILITIES,
            description="Cost to install a unit of capacity at each facility.",
        ),
    ]
    installation_budget: Annotated[
        float,
        Field(
            default=500.0,
            description="Total budget for installation costs.",
            gt=0,
        ),
    ]

    def _check_installation_costs(self) -> None:
        if len(self.installation_costs) != NUM_FACILITIES:
            raise ValueError("The length of installation_costs must equal n_fac.")
        if any(elem < 0 for elem in self.installation_costs):
            raise ValueError("All elements in installation_costs must be non-negative.")

    @model_validator(mode="after")
    def _validate_problem(self) -> Self:
        self._check_installation_costs()
        return self


class FacilitySizingTotalCostConfig(BaseModel):
    """Configuration model for Facility Sizing Total Cost Problem.

    Min Total Cost for Facility Sizing simulation-optimization problem.
    """

    initial_solution: Annotated[
        tuple[float, ...],
        Field(
            default_factory=lambda: (300,) * NUM_FACILITIES,
            description="Initial solution from which solvers start.",
        ),
    ]
    budget: Annotated[
        int,
        Field(
            default=10000,
            description="Max # of replications for a solver to take.",
            gt=0,
            json_schema_extra={"isDatafarmable": False},
        ),
    ]
    installation_costs: Annotated[
        tuple[float, ...],
        Field(
            default_factory=lambda: (1,) * NUM_FACILITIES,
            description="Cost to install a unit of capacity at each facility.",
        ),
    ]
    epsilon: Annotated[
        float,
        Field(
            default=0.05,
            description="Maximum allowed probability of stocking out.",
            ge=0,
            le=1,
        ),
    ]

    def _check_installation_costs(self) -> None:
        if len(self.installation_costs) != NUM_FACILITIES:
            raise ValueError("The length of installation_costs must equal n_fac.")
        if any(elem < 0 for elem in self.installation_costs):
            raise ValueError(
                "All elements in installation_costs must be greater than or equal to 0."
            )

    @model_validator(mode="after")
    def _validate_problem(self) -> Self:
        self._check_installation_costs()
        return self


class DemandInputModel(InputModel):
    """Input model for multivariate normal demand at facilities."""

    rng: Random | None = None

    def _mvnormalvariate(
        self,
        mean_vec: np.ndarray,
        cov: np.ndarray,
        factorized: bool = False,
    ) -> np.ndarray:
        chol = np.linalg.cholesky(cov) if not factorized else cov
        assert self.rng is not None
        observations = [self.rng.normalvariate(0, 1) for _ in range(len(cov))]
        return np.dot(chol, observations).transpose() + mean_vec

    def random(self, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:  # noqa: D102
        while True:
            demand = np.array(self._mvnormalvariate(mean, cov))
            if np.all(demand >= 0):
                return demand


class FacilitySize(Model):
    """Facility Sizing Model.

    A model that simulates a facilitysize problem with a multi-variate normal
    distribution. Returns the probability of violating demand in each scenario.
    """

    class_name_abbr: ClassVar[str] = "FACSIZE"
    class_name: ClassVar[str] = "Facility Sizing"
    config_class: ClassVar[type[BaseModel]] = FacilitySizeConfig
    n_rngs: ClassVar[int] = 1
    n_responses: ClassVar[int] = 3

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize the FacilitySize model.

        Args:
            fixed_factors (dict | None): Fixed factors for the model.
                If None, default values are used.
        """
        # Let the base class handle default arguments.
        super().__init__(fixed_factors)

        self.demand_model = DemandInputModel()

    def before_replicate(self, rng_list: list[MRG32k3a]) -> None:  # noqa: D102
        self.demand_model.set_rng(rng_list[0])

    def replicate(self) -> tuple[dict, dict]:
        """Simulate a single replication using the current model factors.

        Args:
            rng_list (list[MRG32k3a]): Random number generators for the model to use
                when simulating a replication.

        Returns:
            tuple: A tuple containing:
                - dict: The responses dictionary, with keys:
                    - "stockout_flag" (bool): True if at least one facility failed to satisfy demand;
                        False otherwise.
                    - "n_fac_stockout" (int): Number of facilities that could not satisfy demand.
                    - "n_cut" (int): Total number of demand units that could not be satisfied.
                - dict: Gradient estimates for each response.
        """  # noqa: E501
        mean_vec = np.array(self.factors["mean_vec"])
        cov = np.array(self.factors["cov"])
        capacity = np.array(self.factors["capacity"])
        demand = self.demand_model.random(mean_vec, cov)
        extra_demand = demand - capacity
        pos_excess_mask = extra_demand > 0
        n_fac_stockout = np.sum(pos_excess_mask).astype(int)
        n_cut = np.sum(extra_demand[pos_excess_mask]).astype(int)
        # Compose responses and gradients.
        responses = {
            "stockout_flag": int(n_fac_stockout > 0),
            "n_fac_stockout": n_fac_stockout,
            "n_cut": n_cut,
        }
        return responses, {}


class FacilitySizingTotalCost(Problem):
    """Base class to implement simulation-optimization problems."""

    class_name_abbr: ClassVar[str] = "FACSIZE-1"
    class_name: ClassVar[str] = "Min Total Cost for Facility Sizing"
    config_class: ClassVar[type[BaseModel]] = FacilitySizingTotalCostConfig
    model_class: ClassVar[type[Model]] = FacilitySize
    n_objectives: ClassVar[int] = 1
    n_stochastic_constraints: ClassVar[int] = 1
    minmax: ClassVar[tuple[int, ...]] = (-1,)
    constraint_type: ClassVar[ConstraintType] = ConstraintType.STOCHASTIC
    variable_type: ClassVar[VariableType] = VariableType.CONTINUOUS
    gradient_available: ClassVar[bool] = True
    optimal_value: ClassVar[float | None] = None
    optimal_solution: tuple | None = None
    model_default_factors: ClassVar[dict] = {}
    model_decision_factors: ClassVar[set[str]] = {"capacity"}

    @property
    def dim(self) -> int:  # noqa: D102
        return self.model.factors["n_fac"]

    @property
    def lower_bounds(self) -> tuple:  # noqa: D102
        return (0,) * self.dim

    @property
    def upper_bounds(self) -> tuple:  # noqa: D102
        return (np.inf,) * self.dim

    def vector_to_factor_dict(self, vector: tuple) -> dict:  # noqa: D102
        return {"capacity": vector[:]}

    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:  # noqa: D102
        return tuple(factor_dict["capacity"])

    def replicate(self, x: tuple) -> RepResult:  # noqa: D102
        responses, _ = self.model.replicate()
        objectives = [
            Objective(
                stochastic=0.0,
                deterministic=np.dot(self.factors["installation_costs"], x),
                deterministic_gradients=self.factors["installation_costs"],
            )
        ]
        stochastic_constraints = [
            StochasticConstraint(
                stochastic=responses["stockout_flag"],
                deterministic=-self.factors["epsilon"],
            )
        ]
        return RepResult(
            objectives=objectives, stochastic_constraints=stochastic_constraints
        )

    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:  # noqa: D102
        cov_matrix = np.diag([x**2 for x in self.factors["initial_solution"]])
        x = rand_sol_rng.mvnormalvariate(
            self.factors["initial_solution"], cov_matrix.tolist(), factorized=False
        )
        while any(elem < 0 for elem in x):
            x = rand_sol_rng.mvnormalvariate(
                self.factors["initial_solution"], cov_matrix.tolist(), factorized=False
            )
        return tuple(x)


class FacilitySizingMaxService(Problem):
    """Base class to implement simulation-optimization problems."""

    class_name_abbr: ClassVar[str] = "FACSIZE-2"
    class_name: ClassVar[str] = "Max Service for Facility Sizing"
    config_class: ClassVar[type[BaseModel]] = FacilitySizingMaxServiceConfig
    model_class: ClassVar[type[Model]] = FacilitySize
    n_objectives: ClassVar[int] = 1
    n_stochastic_constraints: ClassVar[int] = 0
    minmax: ClassVar[tuple[int, ...]] = (1,)
    constraint_type: ClassVar[ConstraintType] = ConstraintType.DETERMINISTIC
    variable_type: ClassVar[VariableType] = VariableType.CONTINUOUS
    gradient_available: ClassVar[bool] = False
    optimal_value: ClassVar[float | None] = None
    optimal_solution: tuple | None = None
    model_default_factors: ClassVar[dict] = {}
    model_decision_factors: ClassVar[set[str]] = {"capacity"}

    @property
    def dim(self) -> int:  # noqa: D102
        return self.model.factors["n_fac"]

    @property
    def lower_bounds(self) -> tuple:  # noqa: D102
        return (0,) * self.dim

    @property
    def upper_bounds(self) -> tuple:  # noqa: D102
        return (np.inf,) * self.dim

    def vector_to_factor_dict(self, vector: tuple) -> dict:  # noqa: D102
        return {"capacity": vector[:]}

    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:  # noqa: D102
        return tuple(factor_dict["capacity"])

    def replicate(self, _x: tuple) -> RepResult:  # noqa: D102
        responses, _ = self.model.replicate()
        service_value = 1 - responses["stockout_flag"]
        objectives = [Objective(stochastic=service_value)]
        return RepResult(objectives=objectives)

    def check_deterministic_constraints(self, x: tuple) -> bool:  # noqa: D102
        # Check budget constraint
        budget_feasible = (
            np.dot(self.factors["installation_costs"], x)
            <= self.factors["installation_budget"]
        )
        if not budget_feasible:
            return False

        # Check box constraints from the base class
        return super().check_deterministic_constraints(x)

    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:  # noqa: D102
        # Generate random vector of length # of facilities of continuous values
        # summing to less than or equal to installation budget.
        x = rand_sol_rng.continuous_random_vector_from_simplex(
            n_elements=self.model.factors["n_fac"],
            summation=self.factors["installation_budget"],
            exact_sum=False,
        )
        return tuple(x)
