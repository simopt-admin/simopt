"""Simulation of the average response time in a multi-base ambulance dispatch system."""

from __future__ import annotations

from collections.abc import Generator
from typing import Annotated, ClassVar, Self, cast

import numpy as np
import simpy
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
from simopt.input_models import Beta, Exp
from simopt.utils import override


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


class AmbBaseAllocationConfig(BaseModel):
    """Configuration for the Ambulance optimization problem."""

    initial_solution: Annotated[
        tuple[float, ...],
        Field(
            default=(6, 6, 6, 6),
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


class Ambulance(Model):
    """Simulate the average response time in a multi-base ambulance dispatch system.

    The system includes a set of fixed ambulance bases and a set of variable bases
    with decision-variable coordinates. The objective is to minimize the expected
    response time by optimizing the locations of the variable bases.
    """

    class_name_abbr: ClassVar[str] = "AMBULANCE"
    class_name: ClassVar[str] = "Ambulance Base Allocation"
    config_class: ClassVar[type[BaseModel]] = AmbulanceConfig
    n_rngs: ClassVar[int] = 4
    n_responses: ClassVar[int] = 1

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize the ambulance simulation model.

        Args:
            fixed_factors : dict
                fixed factors of the simulation model
        """
        super().__init__(fixed_factors)

        # Instantiate Input Models
        # 1. Exponential distributions for times
        self.arrival_time_model = Exp()
        self.scene_time_model = Exp()

        # 2. Beta distributions for locations
        self.beta_x_model = Beta()
        self.beta_y_model = Beta()

    def before_replicate(self, rng_list: list[MRG32k3a]) -> None:
        # Assign RNGs to input models
        # RNG 0 -> Arrival Times
        self.arrival_time_model.set_rng(rng_list[0])
        # RNG 1 -> Scene Times
        self.scene_time_model.set_rng(rng_list[1])
        # RNG 2 -> X coordinates
        self.beta_x_model.set_rng(rng_list[2])
        # RNG 3 -> Y coordinates
        self.beta_y_model.set_rng(rng_list[3])

    def replicate(self) -> tuple[dict, dict]:
        """Run one replication of the ambulance dispatch simulation."""
        # ------------------------------
        # Setup base locations and system parameters
        # ------------------------------
        fixed_base_count = self.factors["fixed_base_count"]
        variable_base_count = self.factors["variable_base_count"]
        fixed_locs = self.factors["fixed_locs"]
        variable_locs = self.factors["variable_locs"]

        # Beta parameters
        alpha_x, beta_x = self.factors["call_loc_beta_x"]
        alpha_y, beta_y = self.factors["call_loc_beta_y"]

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

        # Ensure RNGs are available
        arrival_time_model = self.arrival_time_model
        scene_time_model = self.scene_time_model

        # Use input models for random generation
        beta_x_model = self.beta_x_model
        beta_y_model = self.beta_y_model

        total_response_time = 0.0
        num_calls = 0
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
            nonlocal num_calls, total_response_time
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
            total_response_time += travel + queue_delay
            num_calls += 1

            if (
                i >= variable_base_start_index
                and i - variable_base_start_index < variable_base_count
            ):
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
                x_coord = beta_x_model.random(alpha_x, beta_x) * sqaure_width
                y_coord = beta_y_model.random(alpha_y, beta_y) * sqaure_width
                interarrival_time = arrival_time_model.random(1.0 / mean_interval)
                service_time = scene_time_model.random(1.0 / mean_scene_time)

                yield env.timeout(interarrival_time)
                env.process(call(env.now, x_coord, y_coord, service_time))

        env.process(call_arrivals())
        env.run(until=sim_length)

        if num_calls:
            avg_time = total_response_time / num_calls
            grad_avg = grad_total / num_calls
        else:
            avg_time = float("inf")
            grad_avg = np.full((variable_base_count, 2), float("nan"))

        responses = {"avg_response_time": avg_time}
        gradients = {"avg_response_time": {"variable_locs": grad_avg.flatten().tolist()}}
        return responses, gradients


class AmbulanceMinAvgResponse(Problem):
    """Base class to implement simulation-optimization problems."""

    config_class: ClassVar[type[BaseModel]] = AmbBaseAllocationConfig
    model_class: ClassVar[type[Model]] = Ambulance
    class_name_abbr: ClassVar[str] = "AMBULANCE-1"
    class_name: ClassVar[str] = "Minimum Average Waiting Time for Ambulance Dispatch"
    config_class: ClassVar[type[BaseModel]] = AmbBaseAllocationConfig
    model_class: ClassVar[type[Model]] = Ambulance
    n_objectives: ClassVar[int] = 1
    n_stochastic_constraints: ClassVar[int] = 0
    minmax: ClassVar[tuple[int]] = (-1,)
    constraint_type: ClassVar[ConstraintType] = ConstraintType.BOX
    variable_type: ClassVar[VariableType] = VariableType.CONTINUOUS
    gradient_available: ClassVar[bool] = True
    optimal_value: ClassVar[float | None] = None
    optimal_solution: tuple | None = None

    # Define default factors used for problem instantiation
    model_default_factors: ClassVar[dict] = {
        "fixed_base_count": 3,
        "variable_base_count": 2,
        "fixed_locs": [15, 15, 5, 15, 5, 5],
        "variable_locs": [6, 6, 6, 6],
    }
    model_decision_factors: ClassVar[set[str]] = {"variable_locs"}

    @property
    @override
    def dim(self) -> int:
        return int(2 * self.model.factors["variable_base_count"])

    @property
    @override
    def lower_bounds(self) -> tuple:
        return tuple(0.0 for _ in range(self.dim))

    @property
    @override
    def upper_bounds(self) -> tuple:
        return tuple(20.0 for _ in range(self.dim))

    @override
    def vector_to_factor_dict(self, vector: tuple) -> dict:
        return {"variable_locs": list(vector)}

    @override
    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:
        return tuple(factor_dict["variable_locs"])

    def replicate(self, _x: tuple) -> RepResult:
        # 1. Run the simulation
        responses, gradients = self.model.replicate()

        # 2. Construct the Objective
        # Since this problem has no deterministic cost component,
        # deterministic values are 0.
        objectives = [
            Objective(
                stochastic=responses["avg_response_time"],
                stochastic_gradients=gradients["avg_response_time"]["variable_locs"],
                deterministic=0.0,
                deterministic_gradients=(0.0,) * self.dim,
            )
        ]

        # 3. Return result
        return RepResult(objectives=objectives)

    @override
    def check_deterministic_constraints(self, _x: tuple) -> bool:
        return len(_x) == self.dim and all(0 <= xi <= 20 for xi in _x)

    @override
    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
        return tuple(rand_sol_rng.uniform(0, 20) for _ in range(self.dim))
