"""Simulate contamination rates."""

from __future__ import annotations

from typing import Annotated, ClassVar, Final, Self

import numpy as np
from pydantic import BaseModel, Field, model_validator

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.base import ConstraintType, Model, Problem, VariableType
from simopt.input_models import Beta
from simopt.utils import override

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
                "beta parameter of beta distribution for initial contamination "
                "fraction"
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

    config_class: ClassVar[type[BaseModel]] = ContaminationConfig
    class_name_abbr: str = "CONTAM"
    n_rngs: int = 2
    n_responses: int = 1

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

    @override
    def check_simulatable_factors(self) -> bool:
        # Check for matching number of stages.
        if len(self.factors["prev_decision"]) != self.factors["stages"]:
            raise ValueError(
                "The number of stages must be equal to the length of the previous "
                "decision tuple."
            )
        return True

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
        gradients = {
            response_key: dict.fromkeys(self.specifications, np.nan)
            for response_key in responses
        }
        return responses, gradients


class ContaminationTotalCostDisc(Problem):
    """Base class to implement simulation-optimization problems."""

    config_class: ClassVar[type[BaseModel]] = ContaminationTotalCostDiscConfig
    model_class: ClassVar[type[Model]] = Contamination
    class_name_abbr: str = "CONTAM-1"
    class_name: str = "Min Total Cost for Discrete Contamination"
    n_objectives: int = 1

    @property
    @override
    def n_stochastic_constraints(self) -> int:
        return self.model.factors["stages"]

    minmax: tuple[int] = (-1,)
    constraint_type: ConstraintType = ConstraintType.STOCHASTIC
    variable_type: VariableType = VariableType.DISCRETE
    gradient_available: bool = True
    optimal_value: float | None = None
    optimal_solution: tuple | None = None
    model_default_factors: dict = {}
    model_decision_factors: set[str] = {"prev_decision"}

    @property
    @override
    def dim(self) -> int:
        return self.model.factors["stages"]

    @property
    @override
    def lower_bounds(self) -> tuple:
        return (0,) * self.model.factors["stages"]

    @property
    @override
    def upper_bounds(self) -> tuple:
        return (1,) * self.model.factors["stages"]

    @override
    def vector_to_factor_dict(self, vector: tuple) -> dict:
        return {"prev_decision": vector[:]}

    @override
    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:
        return tuple(factor_dict["prev_decision"])

    @override
    def factor_dict_to_vector_gradients(self, factor_dict: dict) -> tuple:  # noqa: ARG002
        return (np.nan * len(self.model.factors["prev_decision"]),)

    @override
    def response_dict_to_objectives(self, response_dict: dict) -> tuple:  # noqa: ARG002
        return (0,)

    @override
    def response_dict_to_objectives_gradients(self, _response_dict: dict) -> tuple:
        return ((0,) * len(self.model.factors["prev_decision"]),)

    def response_dict_to_stoch_constraints(self, response_dict: dict) -> tuple:
        """Convert a response dictionary to a vector of stochastic constraint values.

        Each returned value represents the left-hand side of a constraint of the form
        E[Y] ≤ 0.

        Args:
            response_dict (dict): A dictionary containing response keys and their
                associated values.

        Returns:
            tuple: A tuple representing the left-hand sides of the stochastic
                constraints.
        """
        under_control = response_dict["level"] <= self.factors["upper_thres"]
        return tuple([-1 * z for z in under_control])

    def deterministic_stochastic_constraints_and_gradients(self) -> tuple[tuple, tuple]:
        """Compute deterministic components of stochastic constraints.

        Returns:
            tuple:
                - tuple: The deterministic components of the stochastic constraints.
                - tuple: The gradients of those deterministic components.
        """
        det_stoch_constraints = tuple(np.ones(self.dim) - self.factors["error_prob"])
        det_stoch_constraints_gradients = ((0,),)
        return det_stoch_constraints, det_stoch_constraints_gradients

    @override
    def deterministic_objectives_and_gradients(self, x: tuple) -> tuple[tuple, tuple]:
        det_objectives = (np.dot(self.factors["prev_cost"], x),)
        det_objectives_gradients = (tuple(self.factors["prev_cost"]),)
        return det_objectives, det_objectives_gradients

    @override
    def check_deterministic_constraints(self, x: tuple) -> bool:
        return all(0 <= u <= 1 for u in x)

    @override
    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
        return tuple([rand_sol_rng.randint(0, 1) for _ in range(self.dim)])


class ContaminationTotalCostCont(Problem):
    """Base class to implement simulation-optimization problems."""

    config_class: ClassVar[type[BaseModel]] = ContaminationTotalCostContConfig
    model_class: ClassVar[type[Model]] = Contamination
    class_name_abbr: str = "CONTAM-2"
    class_name: str = "Min Total Cost for Continuous Contamination"
    n_objectives: int = 1

    @property
    @override
    def n_stochastic_constraints(self) -> int:
        return self.model.factors["stages"]

    minmax: tuple[int] = (-1,)
    constraint_type: ConstraintType = ConstraintType.STOCHASTIC
    variable_type: VariableType = VariableType.CONTINUOUS
    gradient_available: bool = True
    optimal_value: float | None = None
    optimal_solution: tuple | None = None
    model_default_factors: dict = {}
    model_decision_factors: set[str] = {"prev_decision"}

    @property
    @override
    def dim(self) -> int:
        return self.model.factors["stages"]

    @property
    @override
    def lower_bounds(self) -> tuple:
        return (0,) * self.model.factors["stages"]

    @property
    @override
    def upper_bounds(self) -> tuple:
        return (1,) * self.model.factors["stages"]

    @override
    def check_initial_solution(self) -> bool:
        return not (
            len(self.factors["initial_solution"]) != self.dim
            or all(u < 0 or u > 1 for u in self.factors["initial_solution"])
        )

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

    @override
    def vector_to_factor_dict(self, vector: tuple) -> dict:
        return {"prev_decision": vector[:]}

    @override
    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:
        return tuple(factor_dict["prev_decision"])

    @override
    def factor_dict_to_vector_gradients(self, factor_dict: dict) -> tuple:  # noqa: ARG002
        return (np.nan * len(self.model.factors["prev_decision"]),)

    @override
    def response_dict_to_objectives(self, response_dict: dict) -> tuple:  # noqa: ARG002
        return (0,)

    @override
    def response_dict_to_objectives_gradients(self, _response_dict: dict) -> tuple:
        return ((0,) * len(self.model.factors["prev_decision"]),)

    def response_dict_to_stoch_constraints(self, response_dict: dict) -> tuple:
        """Convert a response dictionary to a vector of stochastic constraint values.

        Each returned value represents the left-hand side of a constraint of the form
        E[Y] ≤ 0.

        Args:
            response_dict (dict): A dictionary containing response keys and their
                associated values.

        Returns:
            tuple: A tuple representing the left-hand sides of the stochastic
                constraints.
        """
        under_control = response_dict["level"] <= self.factors["upper_thres"]
        return tuple([-1 * z for z in under_control])

    def deterministic_stochastic_constraints_and_gradients(self) -> tuple[tuple, tuple]:
        """Compute deterministic components of stochastic constraints.

        Returns:
            tuple:
                - tuple: The deterministic components of the stochastic constraints.
                - tuple: The gradients of those deterministic components.
        """
        det_stoch_constraints = tuple(np.ones(self.dim) - self.factors["error_prob"])
        det_stoch_constraints_gradients = (
            (0,),
        )  # tuple of tuples - of sizes self.dim by self.dim, full of zeros
        return det_stoch_constraints, det_stoch_constraints_gradients

    @override
    def deterministic_objectives_and_gradients(self, x: tuple) -> tuple[tuple, tuple]:
        det_objectives = (np.dot(self.factors["prev_cost"], x),)
        det_objectives_gradients = (tuple(self.factors["prev_cost"]),)
        return det_objectives, det_objectives_gradients

    @override
    def check_deterministic_constraints(self, x: tuple) -> bool:
        return all(0 <= u <= 1 for u in x)

    @override
    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
        return tuple([rand_sol_rng.random() for _ in range(self.dim)])
