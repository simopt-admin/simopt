"""Simulate contamination rates."""

from __future__ import annotations

from typing import Annotated, ClassVar, Final, Self

import numpy as np
from pydantic import BaseModel, Field, model_validator

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt import dsl
from simopt.base import (
    ConstraintType,
    Model,
    Problem,
    VariableType,
)
from simopt.input_models import Beta
from simopt.simulations.contam import ContaminationConfig, replicate
from simopt.utils import override

NUM_STAGES: Final[int] = 5


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
            raise ValueError("All elements in initial_solution must be in the range [0, 1].")

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


class Contamination(Model[ContaminationConfig]):
    """Contamination model with contamination and restoration rates.

    A model that simulates a contamination problem with a beta distribution.
    Returns the probability of violating contamination upper limit in each level of
    supply chain.
    """

    class_name_abbr: ClassVar[str] = "CONTAM"
    class_name: ClassVar[str] = "Contamination"
    config_class: ClassVar[type[ContaminationConfig]] = ContaminationConfig
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

    def replicate(self, factors: ContaminationConfig, rngs: list[MRG32k3a]) -> tuple[dict, dict]:
        """Simulate a single replication for the current model factors.

        Args:
            rngs (list[MRG32k3a]): Random number generators used to simulate
                the replication.

        Returns:
            tuple[dict, dict]: A tuple containing:
                - responses (dict): Performance measures of interest, including:
                    - "level": A list of contamination levels over time.
                - gradients (dict): A dictionary of gradient estimates for each
                    response.
        """
        levels = replicate(factors, rngs, self.contam_model, self.restore_model)
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
    optimal_value: ClassVar[float | None] = None
    optimal_solution: tuple | None = None
    model_default_factors: ClassVar[dict] = {}
    model_decision_factors: ClassVar[set[str]] = {"prev_decision"}

    @override
    def build(self) -> dsl.Model:
        problem = dsl.Model()
        stages = self.model.factors["stages"]
        prev_decision = problem.add_integer_vector(
            lb=0, ub=1, shape=(stages,), initial=tuple(self.factors["initial_solution"])
        )

        def run(
            decisions: dict[str, float | tuple[float, ...]],
            rngs: list[MRG32k3a],
        ) -> tuple[dict, dict]:
            prevention = decisions["prev_decision"]
            if not isinstance(prevention, tuple):
                raise TypeError("prev_decision must be a vector")
            decision_factors = {"prev_decision": prevention}
            factors = self.model.config.model_copy(update=decision_factors)
            responses, _ = self.model.replicate(factors, rngs)
            under_control = np.asarray(responses["level"]) <= np.asarray(
                self.factors["upper_thres"]
            )
            return {"under_control": under_control.astype(float)}, {}

        simulation = problem.add_simulation(
            run=run, decisions={"prev_decision": prev_decision}, n_rngs=self.model.n_rngs
        )
        problem.minimize(
            sum(
                cost * prevention
                for cost, prevention in zip(self.factors["prev_cost"], prev_decision, strict=True)
            )
        )
        for stage, error_probability in enumerate(self.factors["error_prob"]):
            problem.add_stochastic_constraint(
                (1.0 - error_probability) - simulation.metric("under_control")[stage] <= 0.0
            )
        return problem

    def vector_to_factor_dict(self, vector: tuple) -> dict:
        return {"prev_decision": vector[:]}

    def check_deterministic_constraints(self, x: tuple) -> bool:
        return all(0 <= u <= 1 for u in x)

    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
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

    @override
    def build(self) -> dsl.Model:
        problem = dsl.Model()
        stages = self.model.factors["stages"]
        prev_decision = problem.add_continuous_vector(
            lb=0.0, ub=1.0, shape=(stages,), initial=tuple(self.factors["initial_solution"])
        )

        def run(
            decisions: dict[str, float | tuple[float, ...]],
            rngs: list[MRG32k3a],
        ) -> tuple[dict, dict]:
            prevention = decisions["prev_decision"]
            if not isinstance(prevention, tuple):
                raise TypeError("prev_decision must be a vector")
            decision_factors = {"prev_decision": prevention}
            factors = self.model.config.model_copy(update=decision_factors)
            responses, _ = self.model.replicate(factors, rngs)
            under_control = np.asarray(responses["level"]) <= np.asarray(
                self.factors["upper_thres"]
            )
            return {"under_control": under_control.astype(float)}, {}

        simulation = problem.add_simulation(
            run=run, decisions={"prev_decision": prev_decision}, n_rngs=self.model.n_rngs
        )
        problem.minimize(
            sum(
                cost * prevention
                for cost, prevention in zip(self.factors["prev_cost"], prev_decision, strict=True)
            )
        )
        for stage, error_probability in enumerate(self.factors["error_prob"]):
            problem.add_stochastic_constraint(
                (1.0 - error_probability) - simulation.metric("under_control")[stage] <= 0.0
            )
        return problem

    def vector_to_factor_dict(self, vector: tuple) -> dict:
        return {"prev_decision": vector[:]}

    def check_deterministic_constraints(self, x: tuple) -> bool:
        return all(0 <= u <= 1 for u in x)

    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
        return tuple([rand_sol_rng.random() for _ in range(self.dim)])
