"""Standalone Python simulation for ambulance dispatch."""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
from typing import Annotated, Self, cast

import numpy as np
import simpy
from pydantic import BaseModel, Field, model_validator

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt._markers import simulation
from simopt.input_models import InputModel


class AmbulanceConfig(BaseModel):
    """Configuration for the Ambulance simulation model."""

    fixed_base_count: Annotated[int, Field(default=3, ge=0, description="Number of fixed bases")]
    variable_base_count: Annotated[
        int, Field(default=2, gt=0, description="Number of variable bases")
    ]
    fixed_locs: Annotated[
        list[float],
        Field(
            default=[15, 15, 5, 15, 5, 5],
            description="Fixed base coordinates [x0, y0, x1, y1, ...]",
        ),
    ]
    variable_locs: Annotated[
        list[float],
        Field(
            default=[6, 6, 6, 6],
            description="Variable base coordinates [x0, y0, x1, y1, ...]",
        ),
    ]
    call_loc_beta_x: Annotated[
        tuple[float, float],
        Field(default=(2.0, 1.0), description="Beta distribution params for x-axis"),
    ]
    call_loc_beta_y: Annotated[
        tuple[float, float],
        Field(default=(2.0, 1.0), description="Beta distribution params for y-axis"),
    ]

    @model_validator(mode="after")
    def _validate_model(self) -> Self:
        # Check fixed locations length
        expected_fixed_len = 2 * self.fixed_base_count
        if len(self.fixed_locs) != expected_fixed_len:
            raise ValueError(
                f"The length of fixed_locs must be {expected_fixed_len} (2 * fixed_base_count)."
            )

        # Check variable locations length
        expected_var_len = 2 * self.variable_base_count
        if len(self.variable_locs) != expected_var_len:
            raise ValueError(
                f"The length of variable_locs must be {expected_var_len} (2 * variable_base_count)."
            )

        # Check variable locations bounds (Simulatable check)
        if not all(0 <= loc <= 20 for loc in self.variable_locs):
            raise ValueError("All variable_locs must be between 0 and 20.")

        for factor_name, beta_params in (
            ("call_loc_beta_x", self.call_loc_beta_x),
            ("call_loc_beta_y", self.call_loc_beta_y),
        ):
            if not all(param > 0 for param in beta_params):
                raise ValueError(f"All parameters in {factor_name} must be greater than 0.")

        return self


@dataclass
class State:
    """Response statistics shared by call processes within one replication."""

    total_response_time: float = 0.0
    num_calls: int = 0


@simulation
def replicate(
    factors: AmbulanceConfig,
    rngs: list[MRG32k3a],
    # Ensure RNGs are available
    arrival_time_model: InputModel,
    scene_time_model: InputModel,
    # Use input models for random generation
    beta_x_model: InputModel,
    beta_y_model: InputModel,
) -> tuple[float, np.ndarray]:
    """Return average response time and its base-location gradients for one replication."""
    # ------------------------------
    # Setup base locations and system parameters
    # ------------------------------
    fixed_base_count = factors.fixed_base_count
    variable_base_count = factors.variable_base_count
    fixed_locs = factors.fixed_locs
    variable_locs = factors.variable_locs

    # Beta parameters
    alpha_x, beta_x = factors.call_loc_beta_x
    alpha_y, beta_y = factors.call_loc_beta_y

    fixed_base_positions = [
        [fixed_locs[2 * i], fixed_locs[2 * i + 1]] for i in range(fixed_base_count)
    ]
    variable_bases = [
        [variable_locs[2 * i], variable_locs[2 * i + 1]] for i in range(variable_base_count)
    ]

    bases = fixed_base_positions + variable_bases
    variable_base_start_index = len(fixed_base_positions)

    n_ambulances = fixed_base_count + variable_base_count
    sqaure_width = 20.0
    amb_speed = 1.0
    utilization = 0.6
    # est travel time for an ambulance to reach a call
    est_travel_time = 10.0  # Should be close to
    mean_scene_time = 10.0
    mean_interval = (2 * est_travel_time + mean_scene_time) / n_ambulances / utilization
    sim_length = 60 * 24.0 * 1  # Simulate 1 day

    state = State()
    grad_total = np.zeros((variable_base_count, 2))

    # per-variable-base carry for waiting-time derivative to use at next queued call
    # carry is used if the next call for this ambulance has to wait
    carry_next = np.zeros((variable_base_count, 2))

    env = simpy.Environment()
    available_ambulances = simpy.FilterStore(env, capacity=n_ambulances)
    for i in range(n_ambulances):
        available_ambulances.put(i)

    def call(
        arrival_time: float,
        x_coord: float,
        y_coord: float,
        service_time: float,
    ) -> Generator[simpy.Event, object, None]:
        queued = not available_ambulances.items

        if queued:
            i = cast(int, (yield available_ambulances.get()))
        else:
            times = [
                (
                    np.sum(np.abs(np.array(bases[i]) - [x_coord, y_coord])) / amb_speed
                    if i in available_ambulances.items
                    else float("inf")
                )
                for i in range(n_ambulances)
            ]
            i = int(np.argmin(times))
            yield available_ambulances.get(lambda ambulance: ambulance == i)

        travel = np.sum(np.abs(np.array(bases[i]) - [x_coord, y_coord])) / amb_speed
        queue_delay = env.now - arrival_time
        state.total_response_time += travel + queue_delay
        state.num_calls += 1

        if i >= variable_base_start_index and i - variable_base_start_index < variable_base_count:
            j = i - variable_base_start_index
            dx = np.sign(bases[i][0] - x_coord) / amb_speed
            dy = np.sign(bases[i][1] - y_coord) / amb_speed
            dd = np.array([dx, dy])
            if queued:
                grad_total[j] += carry_next[j] + dd
                carry_next[j] = carry_next[j] + 2.0 * dd
            else:
                grad_total[j] += dd
                carry_next[j] = 2.0 * dd

        yield env.timeout(2 * travel)
        yield env.timeout(service_time)
        yield available_ambulances.put(i)

    def call_arrivals() -> Generator[simpy.Event, object, None]:
        while True:
            # Draw Beta-based coordinates in [0, SQUARE_WIDTH]
            x_coord = beta_x_model.random(rngs[2], alpha_x, beta_x) * sqaure_width
            y_coord = beta_y_model.random(rngs[3], alpha_y, beta_y) * sqaure_width
            interarrival_time = arrival_time_model.random(rngs[0], 1.0 / mean_interval)
            service_time = scene_time_model.random(rngs[1], 1.0 / mean_scene_time)

            yield env.timeout(interarrival_time)
            env.process(call(env.now, x_coord, y_coord, service_time))

    env.process(call_arrivals())
    env.run(until=sim_length)

    if state.num_calls:
        avg_time = state.total_response_time / state.num_calls
        grad_avg = grad_total / state.num_calls
    else:
        avg_time = float("inf")
        grad_avg = np.full((variable_base_count, 2), float("nan"))

    return avg_time, grad_avg
