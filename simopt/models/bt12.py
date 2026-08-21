# -*- coding: utf-8 -*-

from __future__ import annotations
"""
Created on Sun Jul 26 16:30:45 2026

@author: nikki
"""

"""Simulate a noisy quadratic objective subject to nonlinear equality/inequality constraints.

Source (deterministic) model, in AMPL:

    var x{1..3} >= 0;
    minimize obj: (x[1] + 3*x[2] + x[3])^2 + 4*(x[1] - x[2])^2;
    subject to constr1: 6*x[2] + 4*x[3] - x[1]^3 >= 3;
    subject to constr2: x[1] + x[2] + x[3] = 1;

The objective is augmented with additive Gaussian noise to turn this into a
simulation-optimization problem; the constraints remain deterministic.
"""



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
    VariableType,
)

NUM_VARS: Final[int] = 5


class BT12Config(BaseModel):
    """Configuration for the noisy quadratic model."""

    noise_std: Annotated[
        float,
        Field(
            default=1,
            description="standard deviation of additive Gaussian noise on the objective value",
            ge=0,
        ),
    ]


class BT12(Model):
    """CUTEst problem "BT12". https://github.com/jacobwilliams/schittkowski-test-problems/

    Simulates a single replication of the objective
    f(x) = 0.01x1**2 + x2**2
    with additive Gaussian noise on the value. No gradient information
    is produced (DFO use case).
    """

    class_name_abbr: ClassVar[str] = "BT12"
    class_name: ClassVar[str] = "CUTEst BT12"
    config_class: ClassVar[type[BaseModel]] = BT12Config
    n_rngs: ClassVar[int] = 1
    n_responses: ClassVar[int] = 1

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize the noisy quadratic model.

        Args:
            fixed_factors : dict
                fixed factors of the simulation model
        """
        super().__init__(fixed_factors)

    def before_replicate(self, rng_list: list[MRG32k3a]) -> None:  # noqa: D102
        self.rng = rng_list[0]

    def replicate(self) -> tuple[dict, dict]:
        """Simulate a single replication for the current model factors.

        Returns:
            tuple[dict, dict]: A tuple containing:
                - responses (dict): Performance measures of interest, including:
                    - "obj_value": Noisy objective value.
                - gradients (dict): Empty; not used (DFO).
        """
        x = np.array(self.factors["x"], dtype=float)
        noise_std = self.factors["noise_std"]

        # true_obj = 0.01*x[0]**2 + x[1]**2
        # z = self.rng.normalvariate(0, 1) 
        
        # noisy_obj = true_obj +  noise_std * z  # same z reused across x via CRN stream state
        true_obj = 0.01 * x[0]**2 + x[1]**2

        z = self.rng.normalvariate(0, 1)

        noise = noise_std * (0.1 * x[0] + x[1]) * z

        noisy_obj = true_obj + noise

        responses = {"obj_value": noisy_obj}
        gradients = {"obj_value": {}}
        return responses, gradients


class BT12ConstrainedConfig(BaseModel):
    """Configuration model for the noisy constrained quadratic problem."""

    initial_solution: Annotated[
        tuple[float, ...],
        Field(
            default= (15.811, 1.5811, 0, 15.083, 3.7164) ,
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

    @model_validator(mode="after")
    def _validate_model(self) -> Self:
        if len(self.initial_solution) != NUM_VARS:
            raise ValueError(f"initial_solution must be of length {NUM_VARS}.")
        return self


class BT12Constrained(Problem):
    """Minimize a noisy quadratic objective subject to nonlinear constraints.

    minimize   (x1 + 3*x2 + x3)^2 + 4*(x1 - x2)^2
    subject to 6*x2 + 4*x3 - x1^3 >= 3   (nonlinear inequality)
               x1 + x2 + x3 = 1          (linear equality)
               x >= 0
    """

    class_name_abbr: ClassVar[str] = "BT12-1"
    class_name: ClassVar[str] = "CUTEst BT12 with Nonlinear Constraints"
    config_class: ClassVar[type[BaseModel]] = BT12ConstrainedConfig
    model_class: ClassVar[type[Model]] = BT12
    n_objectives: ClassVar[int] = 1
    n_stochastic_constraints: ClassVar[int] = 0
    minmax: ClassVar[tuple[int, ...]] = (-1,)
    constraint_type: ClassVar[ConstraintType] = ConstraintType.DETERMINISTIC
    variable_type: ClassVar[VariableType] = VariableType.CONTINUOUS
    gradient_available: ClassVar[bool] = False
    optimal_value: ClassVar[float | None] = None
    optimal_solution: tuple | None = None
    model_default_factors: ClassVar[dict] = {}
    model_decision_factors: ClassVar[set[str]] = {"x"}

    @property
    def dim(self) -> int:  # noqa: D102
        return NUM_VARS

    @property
    def lower_bounds(self) -> tuple:  # noqa: D102
        return (-np.inf,) * self.dim

    @property
    def upper_bounds(self) -> tuple:  # noqa: D102
        return (np.inf,) * self.dim

    def vector_to_factor_dict(self, vector: tuple) -> dict:  # noqa: D102
        return {"x": vector[:]}

    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:  # noqa: D102
        return factor_dict["x"]

    def replicate(self, x: tuple) -> RepResult:  # noqa: D102
        responses, gradients = self.model.replicate()
        objectives = [
            Objective(
                stochastic=responses["obj_value"],
                stochastic_gradients=None,
                deterministic=0,
                deterministic_gradients=None,
            )
        ]
        return RepResult(objectives=objectives)

    def check_deterministic_constraints(self, x: tuple) -> bool:  # noqa: D102
        x1, x2, x3, x4, x5 = x
        ok = (x1 + x2 -x3**2== 25 and x1**2+x2**2 - x4**2 == 25 and x1 - x5**2 == 2 )
        return ok

    # get lhs value of deterministic equality constraint(s), c(x) == 0 form
    def get_deterministic_equality_constraints(self, x: tuple) -> float:
        x1, x2, x3, x4, x5 = x
        c1 = x1 + x2 -x3**2 -25
        c2 =   x1**2+x2**2 - x4**2 -25
        c3 = x1 - x5**2 -2
        return [c1,c2,c3]

    # get lhs value of deterministic inequality constraint(s), h(x) <= 0 form
    def get_deterministic_inequality_constraints(self, x: tuple) -> float:
        return None

    # jacobian of the equality constraint(s)
    def get_deterministic_equality_constraints_gradients(self, x: tuple) -> np.ndarray:
        x1, x2, x3, x4,x5 = x
        c1 = np.array([1,1,-2.0*x3,0,0])
        c2 = np.array([2.0*x1,2.0*x2, 0, -2.0*x4,0] )
        c3 = np.array([1,0, 0, 0,-2.0*x5] )
        return np.vstack([c1,c2,c3])

    # jacobian of the inequality constraint(s)
    def get_deterministic_inequality_constraints_gradients(self, x: tuple) -> np.ndarray:
        return None

    # constraint Hessians, ordered equality constraints first then inequality constraints
    def get_deterministic_constraints_hessian(self, x: tuple) -> np.ndarray:
        #intialize all zero Hessian
        H = np.zeros((3, 5, 5))
        # c1
        H[0, 2, 2] = -2.0
        # c2
        H[1, 0, 0] = 2.0
        H[1, 1, 1] = 2.0
        H[1, 3, 3] = -2.0
        
        #c3
        H[2, 4, 4] = -2.0
        return H

    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:  # noqa: D102
        return tuple(
            [rand_sol_rng.lognormalvariate(lq=0.1, uq=10) for _ in range(self.dim)]
        )