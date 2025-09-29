"""Simulate contamination rates."""

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
from simopt.input_models import Beta

NUM_STAGES: Final[int] = 5


class ContaminationConfig(BaseModel):
    """Configuration model for Contamination simulation.

    A model that simulates a contamination problem with a beta distribution.
    Returns the probability of violating contamination upper limit in each level of
    supply chain.
    """

    contam_rate_alpha: Annotated[
        float,
        Field(
            default=1.0,
            description=(
                "alpha parameter of beta distribution for growth rate of "
                "contamination at each stage"
            ),
            gt=0,
        ),
    ]
    contam_rate_beta: Annotated[
        float,
        Field(
            default=round(17 / 3, 2),
            description=(
                "beta parameter of beta distribution for growth rate of "
                "contamination at each stage"
            ),
            gt=0,
        ),
    ]
    restore_rate_alpha: Annotated[
        float,
        Field(
            default=1.0,
            description=(
                "alpha parameter of beta distribution for rate that contamination "
                "decreases by after prevention effort"
            ),
            gt=0,
        ),
    ]
    restore_rate_beta: Annotated[
        float,
        Field(
            default=round(3 / 7, 3),
            description=(
                "beta parameter of beta distribution for rate that contamination "
                "decreases by after prevention effort"
            ),
            gt=0,
        ),
    ]
    initial_rate_alpha: Annotated[
        float,
        Field(
            default=1.0,
            description=(
                "alpha parameter of beta distribution for initial contamination "
                "fraction"
            ),
            gt=0,
        ),
    ]
    initial_rate_beta: Annotated[
        float,
        Field(
            default=30.0,
            description=(
                "beta parameter of beta distribution for initial contamination fraction"
            ),
            gt=0,
        ),
    ]
    stages: Annotated[
        int,
        Field(
            default=NUM_STAGES,
            description="stage of food supply chain",
            gt=0,
        ),
    ]
    prev_decision: Annotated[
        tuple[float, ...],
        Field(
            default=(0,) * NUM_STAGES,
            description="prevention decision",
        ),
    ]

    def _check_prev_decision(self) -> None:
        if not all(0 <= u <= 1 for u in self.prev_decision):
            raise ValueError(
                "All elements in prev_decision must be in the range [0, 1]."
            )

    @model_validator(mode="after")
    def _validate_model(self) -> Self:
        self._check_prev_decision()

        # Cross-validation: check for matching number of stages
        if len(self.prev_decision) != self.stages:
            raise ValueError(
                "The number of stages must be equal to the length of the previous "
                "decision tuple."
            )

        return self


class ContaminationTotalCostContConfig(BaseModel):
    """Configuration model for Contamination Total Cost Continuous Problem.

    A problem configuration that minimizes total cost for continuous contamination
    control decisions.
    """

    initial_solution: Annotated[
        tuple[float, ...],
        Field(
            default_factory=lambda: (1,) * NUM_STAGES,
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
    prev_cost: Annotated[
        list[float],
        Field(
            default_factory=lambda: [1] * NUM_STAGES,
            description="cost of prevention",
        ),
    ]
    error_prob: Annotated[
        list[float],
        Field(
            default_factory=lambda: [0.2] * NUM_STAGES,
            description="error probability",
        ),
    ]
    upper_thres: Annotated[
        list[float],
        Field(
            default_factory=lambda: [0.1] * NUM_STAGES,
            description="upper limit of amount of contamination",
        ),
    ]

    def _check_prev_cost(self) -> None:
        if len(self.prev_cost) != NUM_STAGES:
            raise ValueError(f"prev_cost must have length {NUM_STAGES}.")

        if any(cost <= 0 for cost in self.prev_cost):
            raise ValueError("All costs in prev_cost must be greater than 0.")

    def _check_error_prob(self) -> None:
        if len(self.error_prob) != NUM_STAGES:
            raise ValueError(f"error_prob must have length {NUM_STAGES}.")

        if any(prob < 0 for prob in self.error_prob):
            raise ValueError("All error probabilities must be non-negative.")

    def _check_upper_thres(self) -> None:
        if len(self.upper_thres) != NUM_STAGES:
            raise ValueError(f"upper_thres must have length {NUM_STAGES}.")

    @model_validator(mode="after")
    def _validate_model(self) -> Self:
        self._check_prev_cost()
        self._check_error_prob()
        self._check_upper_thres()

        if any(u < 0 or u > 1 for u in self.initial_solution):
            raise ValueError(
                "All elements in initial_solution must be in the range [0, 1]."
            )

        return self


class ContaminationTotalCostDiscConfig(BaseModel):
    """Configuration model for Contamination Total Cost Discrete Problem.

    A problem configuration that minimizes total cost for discrete contamination
    control decisions.
    """

    initial_solution: Annotated[
        tuple[int, ...],
        Field(
            default_factory=lambda: (1,) * NUM_STAGES,
            description="initial solution",
        ),
    ]
    budget: Annotated[
        int,
        Field(
            default=10000,
            description="max # of replications for a solver to take",
            gt=0,
        ),
    ]
    prev_cost: Annotated[
        list[float],
        Field(
            default_factory=lambda: [1] * NUM_STAGES,
            description="cost of prevention",
        ),
    ]
    error_prob: Annotated[
        list[float],
        Field(
            default_factory=lambda: [0.2] * NUM_STAGES,
            description="error probability",
        ),
    ]
    upper_thres: Annotated[
        list[float],
        Field(
            default_factory=lambda: [0.1] * NUM_STAGES,
            description="upper limit of amount of contamination",
        ),
    ]

    def _check_prev_cost(self) -> None:
        if len(self.prev_cost) != NUM_STAGES:
            raise ValueError(f"prev_cost must have length {NUM_STAGES}.")

        if any(cost <= 0 for cost in self.prev_cost):
            raise ValueError("All costs in prev_cost must be greater than 0.")

    def _check_error_prob(self) -> None:
        if len(self.error_prob) != NUM_STAGES:
            raise ValueError(f"error_prob must have length {NUM_STAGES}.")

        if any(prob < 0 for prob in self.error_prob):
            raise ValueError("All error probabilities must be non-negative.")

    def _check_upper_thres(self) -> None:
        if len(self.upper_thres) != NUM_STAGES:
            raise ValueError(f"upper_thres must have length {NUM_STAGES}.")

    @model_validator(mode="after")
    def _validate_model(self) -> Self:
        self._check_prev_cost()
        self._check_error_prob()
        self._check_upper_thres()

        return self


class Contamination(Model):
    """Contamination model with contamination and restoration rates.

    A model that simulates a contamination problem with a beta distribution.
    Returns the probability of violating contamination upper limit in each level of
    supply chain.
    """

    class_name_abbr: ClassVar[str] = "CONTAM"
    class_name: ClassVar[str] = "Contamination"
    config_class: ClassVar[type[BaseModel]] = ContaminationConfig
    n_rngs: ClassVar[int] = 2
    n_responses: ClassVar[int] = 1

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize the Contamination model.

        Args:
            fixed_factors (dict, optional): Fixed factors for the model.
                Defaults to None.
        """
        # Let the base class handle default arguments.
        super().__init__(fixed_factors)
        self.contam_model = Beta()
        self.restore_model = Beta()

    def before_replicate(self, rng_list: list[MRG32k3a]) -> None:  # noqa: D102
        self.contam_model.set_rng(rng_list[0])
        self.restore_model.set_rng(rng_list[1])

    def replicate(self) -> tuple[dict, dict]:
        """Simulate a single replication for the current model factors.

        Args:
            rng_list (list[MRG32k3a]): Random number generators used to simulate
                the replication.

        Returns:
            tuple[dict, dict]: A tuple containing:
                - responses (dict): Performance measures of interest, including:
                    - "level": A list of contamination levels over time.
                - gradients (dict): A dictionary of gradient estimates for each
                    response.
        """
        stages: int = self.factors["stages"]
        init_alpha: float = self.factors["initial_rate_alpha"]
        init_beta: float = self.factors["initial_rate_beta"]
        contam_alpha: float = self.factors["contam_rate_alpha"]
        contam_beta: float = self.factors["contam_rate_beta"]
        restore_alpha: float = self.factors["restore_rate_alpha"]
        restore_beta: float = self.factors["restore_rate_beta"]
        u: tuple = self.factors["prev_decision"]

        # Initialize levels with beta distribution.
        levels = np.zeros(stages)
        levels[0] = self.restore_model.random(init_alpha, init_beta)

        # Generate contamination and restoration values with beta distribution.
        rand_range = range(stages - 1)
        contamination_rates = [
            self.contam_model.random(contam_alpha, contam_beta) for _ in rand_range
        ]
        restoration_rates = [
            self.restore_model.random(restore_alpha, restore_beta) for _ in rand_range
        ]

        # Calculate contamination and restoration levels.
        # Start from stage 1; stage 0 was initialized separately.
        for i in range(1, stages):
            c = contamination_rates[i - 1]
            r = restoration_rates[i - 1]
            u_i = u[i]
            prev = levels[i - 1]

            contamination_change = c * (1 - u_i) * (1 - prev)
            restoration_change = (1 - r * u_i) * prev
            levels[i] = contamination_change + restoration_change
        # Compose responses and gradients.
        responses = {"level": levels}
        return responses, {}


class ContaminationTotalCostDisc(Problem):
    """Base class to implement simulation-optimization problems."""

    class_name_abbr: ClassVar[str] = "CONTAM-1"
    class_name: ClassVar[str] = "Min Total Cost for Discrete Contamination"
    config_class: ClassVar[type[BaseModel]] = ContaminationTotalCostDiscConfig
    model_class: ClassVar[type[Model]] = Contamination
    n_objectives: ClassVar[int] = 1
    n_stochastic_constraints: ClassVar[int] = NUM_STAGES
    minmax: ClassVar[tuple[int, ...]] = (-1,)
    constraint_type: ClassVar[ConstraintType] = ConstraintType.STOCHASTIC
    variable_type: ClassVar[VariableType] = VariableType.DISCRETE
    gradient_available: ClassVar[bool] = True
    optimal_value: float | None = None
    optimal_solution: tuple | None = None
    model_default_factors: ClassVar[dict] = {}
    model_decision_factors: ClassVar[set[str]] = {"prev_decision"}

    @property
    def dim(self) -> int:  # noqa: D102
        return self.model.factors["stages"]

    @property
    def lower_bounds(self) -> tuple:  # noqa: D102
        return (0,) * self.model.factors["stages"]

    @property
    def upper_bounds(self) -> tuple:  # noqa: D102
        return (1,) * self.model.factors["stages"]

    def vector_to_factor_dict(self, vector: tuple) -> dict:  # noqa: D102
        return {"prev_decision": vector[:]}

    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:  # noqa: D102
        return tuple(factor_dict["prev_decision"])

    def replicate(self, x: tuple) -> RepResult:  # noqa: D102
        responses, _ = self.model.replicate()
        objectives = [
            Objective(
                stochastic=0.0,
                deterministic=np.dot(self.factors["prev_cost"], x),
                deterministic_gradients=self.factors["prev_cost"],
            )
        ]
        under_control = responses["level"] <= self.factors["upper_thres"]
        error_prob = self.factors["error_prob"]
        stochastic_constraints = [
            StochasticConstraint(
                stochastic=-1 * under_control[i], deterministic=1 - error_prob[i]
            )
            for i in range(len(under_control))
        ]
        return RepResult(
            objectives=objectives, stochastic_constraints=stochastic_constraints
        )

    def check_deterministic_constraints(self, x: tuple) -> bool:  # noqa: D102
        return all(0 <= u <= 1 for u in x)

    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:  # noqa: D102
        return tuple([rand_sol_rng.randint(0, 1) for _ in range(self.dim)])


class ContaminationTotalCostCont(Problem):
    """Base class to implement simulation-optimization problems."""

    class_name_abbr: ClassVar[str] = "CONTAM-2"
    class_name: ClassVar[str] = "Min Total Cost for Continuous Contamination"
    config_class: ClassVar[type[BaseModel]] = ContaminationTotalCostContConfig
    model_class: ClassVar[type[Model]] = Contamination
    n_objectives: ClassVar[int] = 1
    n_stochastic_constraints: ClassVar[int] = NUM_STAGES
    minmax: ClassVar[tuple[int, ...]] = (-1,)
    constraint_type: ClassVar[ConstraintType] = ConstraintType.STOCHASTIC
    variable_type: ClassVar[VariableType] = VariableType.CONTINUOUS
    gradient_available: ClassVar[bool] = True
    optimal_value: ClassVar[float | None] = None
    optimal_solution: tuple | None = None
    model_default_factors: ClassVar[dict] = {}
    model_decision_factors: ClassVar[set[str]] = {"prev_decision"}

    @property
    def dim(self) -> int:  # noqa: D102
        return self.model.factors["stages"]

    @property
    def lower_bounds(self) -> tuple:  # noqa: D102
        return (0,) * self.model.factors["stages"]

    @property
    def upper_bounds(self) -> tuple:  # noqa: D102
        return (1,) * self.model.factors["stages"]

    # # TODO: figure out how Problem.check_simulatable_factors() works
    # def check_simulatable_factors(self) -> bool:
    #     lower_len = len(self.lower_bounds)
    #     upper_len = len(self.upper_bounds)
    #     if lower_len != upper_len or lower_len != self.dim:
    #         error_msg = (
    #             f"Lower bounds: {lower_len}, "
    #             f"Upper bounds: {upper_len}, "
    #             f"Dim: {self.dim}"
    #         )
    #         raise ValueError(error_msg)
    #     return True

    def vector_to_factor_dict(self, vector: tuple) -> dict:  # noqa: D102
        return {"prev_decision": vector[:]}

    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:  # noqa: D102
        return tuple(factor_dict["prev_decision"])

    def replicate(self, x: tuple) -> RepResult:  # noqa: D102
        responses, _ = self.model.replicate()
        deterministic_cost = np.dot(self.factors["prev_cost"], x)
        objectives = [
            Objective(
                stochastic=0.0,
                deterministic=deterministic_cost,
                deterministic_gradients=self.factors["prev_cost"],
            )
        ]
        under_control = responses["level"] <= self.factors["upper_thres"]
        error_prob = self.factors["error_prob"]
        stochastic_constraints = [
            StochasticConstraint(
                stochastic=-1 * under_control[i], deterministic=1 - error_prob[i]
            )
            for i in range(len(under_control))
        ]
        return RepResult(
            objectives=objectives, stochastic_constraints=stochastic_constraints
        )

    def check_deterministic_constraints(self, x: tuple) -> bool:  # noqa: D102
        return all(0 <= u <= 1 for u in x)

    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:  # noqa: D102
        return tuple([rand_sol_rng.random() for _ in range(self.dim)])
