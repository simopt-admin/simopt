"""Simulate demand at facilities."""

from __future__ import annotations

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
from simopt.utils import override

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

    def set_rng(self, rng: random.Random) -> None:  # noqa: D102
        self.rng = rng

    def unset_rng(self) -> None:  # noqa: D102
        self.rng = None

    def random(self, mean: list, cov: list) -> float:  # noqa: D102
        while True:
            demand = np.array(self.rng.mvnormalvariate(mean, cov))
            if np.all(demand >= 0):
                return demand


class FacilitySize(Model):
    """Facility Sizing Model.

    A model that simulates a facilitysize problem with a multi-variate normal
    distribution. Returns the probability of violating demand in each scenario.
    """

    config_class: ClassVar[type[BaseModel]] = FacilitySizeConfig
    class_name_abbr: str = "FACSIZE"
    class_name: str = "Facility Sizing"
    n_rngs: int = 1
    n_responses: int = 3

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize the FacilitySize model.

        Args:
            fixed_factors (dict | None): Fixed factors for the model.
                If None, default values are used.
        """
        # Let the base class handle default arguments.
        super().__init__(fixed_factors)

        self.demand_model = DemandInputModel()

    @override
    def check_simulatable_factors(self) -> bool:
        if len(self.factors["capacity"]) != self.factors["n_fac"]:
            raise ValueError("The length of capacity must be equal to n_fac.")
        if len(self.factors["mean_vec"]) != self.factors["n_fac"]:
            raise ValueError("The length of mean_vec must be equal to n_fac.")
        if len(self.factors["cov"]) != self.factors["n_fac"]:
            raise ValueError("The length of cov must be equal to n_fac.")
        if len(self.factors["cov"][0]) != self.factors["n_fac"]:
            raise ValueError("The length of cov[0] must be equal to n_fac.")
        return True

    def before_replicate(self, rngs: list[MRG32k3a]) -> None:  # noqa: D102
        self.demand_model.set_rng(rngs[0])

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
        mean_vec: list[float | int] = self.factors["mean_vec"]
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
        return responses, None


class FacilitySizingTotalCost(Problem):
    """Base class to implement simulation-optimization problems."""

    config_class: ClassVar[type[BaseModel]] = FacilitySizingTotalCostConfig
    model_class: ClassVar[type[Model]] = FacilitySize
    class_name_abbr: str = "FACSIZE-1"
    class_name: str = "Min Total Cost for Facility Sizing"
    n_objectives: int = 1
    n_stochastic_constraints: int = 1
    minmax: tuple[int] = (-1,)
    constraint_type: ConstraintType = ConstraintType.STOCHASTIC
    variable_type: VariableType = VariableType.CONTINUOUS
    gradient_available: bool = True
    optimal_value: float | None = None
    optimal_solution: tuple | None = None
    model_default_factors: dict = {}
    model_decision_factors: set[str] = {"capacity"}

    @property
    @override
    def dim(self) -> int:
        return self.model.factors["n_fac"]

    @property
    @override
    def lower_bounds(self) -> tuple:
        return (0,) * self.dim

    @property
    @override
    def upper_bounds(self) -> tuple:
        return (np.inf,) * self.dim

    @override
    def vector_to_factor_dict(self, vector: tuple) -> dict:
        return {"capacity": vector[:]}

    @override
    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:
        return tuple(factor_dict["capacity"])

    @override
    def factor_dict_to_vector_gradients(self, factor_dict: dict) -> tuple:  # noqa: ARG002
        return (np.nan * len(self.model.factors["capacity"]),)

    def replicate(self, x: tuple) -> RepResult:
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

    @override
    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
        cov_matrix = np.diag([x**2 for x in self.factors["initial_solution"]])
        x = rand_sol_rng.mvnormalvariate(
            self.factors["initial_solution"], cov_matrix, factorized=False
        )
        while any(elem < 0 for elem in x):
            x = rand_sol_rng.mvnormalvariate(
                self.factors["initial_solution"], cov_matrix, factorized=False
            )
        return tuple(x)


class FacilitySizingMaxService(Problem):
    """Base class to implement simulation-optimization problems."""

    config_class: ClassVar[type[BaseModel]] = FacilitySizingMaxServiceConfig
    model_class: ClassVar[type[Model]] = FacilitySize
    class_name_abbr: str = "FACSIZE-2"
    class_name: str = "Max Service for Facility Sizing"
    n_objectives: int = 1
    n_stochastic_constraints: int = 0
    minmax: tuple[int] = (1,)
    constraint_type: ConstraintType = ConstraintType.DETERMINISTIC
    variable_type: VariableType = VariableType.CONTINUOUS
    gradient_available: bool = False
    optimal_value: float | None = None
    optimal_solution: tuple | None = None
    model_default_factors: dict = {}
    model_decision_factors: set[str] = {"capacity"}

    @property
    @override
    def dim(self) -> int:
        return self.model.factors["n_fac"]

    @property
    @override
    def lower_bounds(self) -> tuple:
        return (0,) * self.dim

    @property
    @override
    def upper_bounds(self) -> tuple:
        return (np.inf,) * self.dim

    @override
    def vector_to_factor_dict(self, vector: tuple) -> dict:
        return {"capacity": vector[:]}

    @override
    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:
        return tuple(factor_dict["capacity"])

    def replicate(self, x: tuple) -> RepResult:
        responses, _ = self.model.replicate()
        service_value = 1 - responses["stockout_flag"]
        objectives = [Objective(stochastic=service_value)]
        return RepResult(objectives=objectives)

    @override
    def check_deterministic_constraints(self, x: tuple) -> bool:
        # Check budget constraint
        budget_feasible = (
            np.dot(self.factors["installation_costs"], x)
            <= self.factors["installation_budget"]
        )
        if not budget_feasible:
            return False

        # Check box constraints from the base class
        return super().check_deterministic_constraints(x)

    @override
    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
        # Generate random vector of length # of facilities of continuous values
        # summing to less than or equal to installation budget.
        x = rand_sol_rng.continuous_random_vector_from_simplex(
            n_elements=self.model.factors["n_fac"],
            summation=self.factors["installation_budget"],
            exact_sum=False,
        )
        return tuple(x)
