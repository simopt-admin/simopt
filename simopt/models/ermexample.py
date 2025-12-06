"""Example problem of deterministic function with noise.

Simulate a synthetic problem with a deterministic objective function
evaluated with noise.
"""

from __future__ import annotations

from typing import Annotated, ClassVar

import numpy as np
from pydantic import BaseModel, Field

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


class ERMExampleModelConfig(BaseModel):
    """Configuration model for ERMExample simulation.

    An empirical risk minimization model for linear regression.
    """

    beta: Annotated[
        tuple[float, ...],
        Field(
            default=(0.0, 0.0),
            description="(intercept, slope) coefficients",
        ),
    ]


class ERMExampleProblemConfig(BaseModel):
    """Configuration model for ERMExample Problem.

    Base class to implement simulation-optimization problems.
    """

    initial_solution: Annotated[
        tuple[float, ...],
        Field(
            default=(0.0, 0.0),
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


class FileInputModel(InputModel):
    def __init__(self, filename):
        self.data = np.load(filename)

    def set_rng(self, rng: random.Random) -> None:
        self.rng = rng

    def unset_rng(self) -> None:
        self.rng = None

    def random(self) -> float:
        n_rows = np.shape(self.data)[0]
        resample_idx = np.random.choice(n_rows, size=1, replace=True)
        resample_x = self.data[resample_idx, 0].item()
        resample_y = self.data[resample_idx, 1].item()
        return resample_x, resample_y


class ERMExampleModel(Model):
    """A model that for the empirical risk of a linear regression model."""

    class_name_abbr: ClassVar[str] = "ERMEXAMPLE"
    class_name: ClassVar[str] = "Linear Regression ERM"
    config_class: ClassVar[type[BaseModel]] = ERMExampleModelConfig
    n_rngs: ClassVar[int] = 1
    n_responses: ClassVar[int] = 1

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize the model.

        Args:
            fixed_factors (dict | None): fixed factors of the model.
                If None, use default values.
        """
        # Let the base class handle default arguments.
        super().__init__(fixed_factors)
        self.resample_model = FileInputModel("workshop/erm_data.npy")

    def before_replicate(self, rng_list: list[MRG32k3a]) -> None:  # noqa: D102
        self.resample_model.set_rng(rng_list[0])

    def replicate(self) -> tuple[dict, dict]:
        """Evaluate the squared error loss of a single observation.

        Returns:
            tuple[dict, dict]: A tuple containing:
                - responses (dict): Performance measures of interest, including:
                    - "sq_error_loss": Squared error loss of a single observation.
                - gradients (dict): A dictionary of gradient estimates for
                    each response.
        """
        beta0, beta1 = self.factors["beta"]
        x, y = self.resample_model.random()
        sq_error_loss = (y - beta0 - beta1 * x) ** 2
        error_loss = y - beta0 - beta1 * x
        # gradients wrt beta0 and beta1
        grad_sq_error_loss = (-2 * error_loss, -2 * x * error_loss)

        # Compose responses and gradients.
        responses = {"sq_error_loss": sq_error_loss}
        gradients = {"sq_error_loss": {"beta": grad_sq_error_loss}}
        return responses, gradients


class ERMExampleProblem(Problem):
    """Base class to implement simulation-optimization problems."""

    class_name_abbr: ClassVar[str] = "ERM-EXAMPLE-1"
    class_name: ClassVar[str] = "Min Empirical Risk"
    config_class: ClassVar[type[BaseModel]] = ERMExampleProblemConfig
    model_class: ClassVar[type[Model]] = ERMExampleModel
    n_objectives: ClassVar[int] = 1
    n_stochastic_constraints: ClassVar[int] = 0
    minmax: ClassVar[tuple[int, ...]] = (-1,)
    constraint_type: ClassVar[ConstraintType] = ConstraintType.UNCONSTRAINED
    variable_type: ClassVar[VariableType] = VariableType.CONTINUOUS
    gradient_available: ClassVar[bool] = True
    model_default_factors: ClassVar[dict] = {}
    model_decision_factors: ClassVar[set[str]] = {"beta"}

    @property
    def optimal_value(self) -> float | None:  # noqa: D102
        # Compute optimal beta0 and beta1
        all_data = np.load("workshop/erm_data.npy")
        x = all_data[:, 0]
        y = all_data[:, 1]
        optbeta1, optbeta0 = np.polyfit(x, y, 1)
        opttrainingmse = np.mean(
            [(yy - optbeta0 - optbeta1 * xx) ** 2 for (xx, yy) in zip(x, y)]
        )
        return opttrainingmse

    @property
    def optimal_solution(self) -> tuple | None:  # noqa: D102
        # Compute optimal beta0 and beta1
        all_data = np.load("workshop/erm_data.npy")
        x = all_data[:, 0]
        y = all_data[:, 1]
        optbeta1, optbeta0 = np.polyfit(x, y, 1)
        return (optbeta0, optbeta1)

    @property
    def dim(self) -> int:  # noqa: D102
        return 2

    @property
    def lower_bounds(self) -> tuple:  # noqa: D102
        return (-np.inf,) * self.dim

    @property
    def upper_bounds(self) -> tuple:  # noqa: D102
        return (np.inf,) * self.dim

    def vector_to_factor_dict(self, vector: tuple) -> dict:  # noqa: D102
        return {"beta": vector[:]}

    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:  # noqa: D102
        return tuple(factor_dict["beta"])

    def replicate(self, _x: tuple) -> RepResult:  # noqa: D102
        responses, gradients = self.model.replicate()
        objectives = [
            Objective(
                stochastic=responses["sq_error_loss"],
                stochastic_gradients=gradients["sq_error_loss"]["beta"],
            )
        ]
        return RepResult(objectives=objectives)

    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:  # noqa: D102
        # beta = tuple([rand_sol_rng.uniform(-2, 2) for _ in range(self.dim)])
        beta = tuple(
            rand_sol_rng.mvnormalvariate(
                mean_vec=[1.0] * self.dim,
                cov=np.eye(self.dim).tolist(),
                factorized=False,
            )
        )
        return beta
