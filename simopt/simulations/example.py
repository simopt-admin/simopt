"""Standalone Python simulations for the synthetic example models."""

from __future__ import annotations

from typing import Annotated

import numpy as np
from pydantic import BaseModel, Field

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt._markers import simulation
from simopt.input_models import InputModel


class ExampleModelConfig(BaseModel):
    """Configuration model for Example simulation.

    A model that is a deterministic function evaluated with noise.
    """

    x: Annotated[
        tuple[float, ...],
        Field(
            default=(2.0, 2.0),
            description="point to evaluate",
        ),
    ]


class Example2ModelConfig(BaseModel):
    """Configuration model for Example-2 simulation.

    A model that is a deterministic quadratic function evaluated with noise.
    """

    x: Annotated[
        tuple[int, ...],
        Field(
            default=(0, 0, 0, 0),
            description="point to evaluate",
        ),
    ]


@simulation
def replicate(
    factors: ExampleModelConfig,
    rngs: list[MRG32k3a],
    noise_model: InputModel,
) -> tuple[float, tuple[float, ...]]:
    """Return the noisy function value and its gradient for one replication."""
    x = np.array(factors.x)
    fn_eval_at_x = np.linalg.norm(x) ** 2 + noise_model.random(rngs[0])

    return fn_eval_at_x, tuple(2 * x)


@simulation
def replicate_discrete(
    factors: Example2ModelConfig,
    rngs: list[MRG32k3a],
    noise_model: InputModel,
) -> float:
    """Return the noisy discrete quadratic value for one replication."""
    x = np.array(factors.x)
    target = np.array([1, 2, 3, 4])
    fn_eval_at_x = np.sum((x - target) ** 2) + noise_model.random(rngs[0])

    return fn_eval_at_x  # noqa: RET504
