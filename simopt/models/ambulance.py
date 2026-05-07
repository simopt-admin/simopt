"""Simulation of the average response time in a multi-base ambulance dispatch system."""

from __future__ import annotations

from typing import Annotated, ClassVar, Self

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
from simopt.input_models import Beta, Exp
from simopt.utils import override

AVAILABLE = 0
BUSY = 1


class AmbulanceConfig(BaseModel):
    """Configuration for the Ambulance simulation model."""

    fixed_base_count: Annotated[
        int, Field(default=3, ge=0, description="Number of fixed bases")
    ]
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
                f"The length of fixed_locs must be {expected_fixed_len} "
                f"(2 * fixed_base_count)."
            )

        # Check variable locations length
        expected_var_len = 2 * self.variable_base_count
        if len(self.variable_locs) != expected_var_len:
            raise ValueError(
                f"The length of variable_locs must be {expected_var_len} "
                f"(2 * variable_base_count)."
            )

        # Check variable locations bounds (Simulatable check)
        if not all(0 <= loc <= 20 for loc in self.variable_locs):
            raise ValueError("All variable_locs must be between 0 and 20.")

        for factor_name, beta_params in (
            ("call_loc_beta_x", self.call_loc_beta_x),
            ("call_loc_beta_y", self.call_loc_beta_y),
        ):
            if not all(param > 0 for param in beta_params):
                raise ValueError(
                    f"All parameters in {factor_name} must be greater than 0."
                )

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

    def before_replicate(self, rng_list: list[MRG32k3a]) -> None:  # noqa: D102
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
            [variable_locs[2 * i], variable_locs[2 * i + 1]]
            for i in range(variable_base_count)
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
        mean_interval = (
            (2 * est_travel_time + mean_scene_time) / n_ambulances / utilization
        )
        sim_length = 60 * 24.0 * 1  # Simulate 1 day

        # Ensure RNGs are available
        arrival_time_model = self.arrival_time_model
        scene_time_model = self.scene_time_model

        # Use input models for random generation
        beta_x_model = self.beta_x_model
        beta_y_model = self.beta_y_model

        def next_arrival(curr: float) -> list:
            """Generate the next arrival event.

            - Arrival inter-time: exponential.
            - Call location (x, y): drawn from scaled Beta distributions.
            """
            # Draw Beta-based coordinates in [0, SQUARE_WIDTH]
            x_coord = beta_x_model.random(alpha_x, beta_x) * sqaure_width
            y_coord = beta_y_model.random(alpha_y, beta_y) * sqaure_width

            # Generate times using Exp input model
            interarrival_time = arrival_time_model.random(1.0 / mean_interval)
            service_time = scene_time_model.random(1.0 / mean_scene_time)

            return [
                curr + interarrival_time,  # interarrival time
                1,  # event type: arrival
                x_coord,  # x from Beta
                y_coord,  # y from Beta
                service_time,  # scene time
            ]

        # ------------------------------
        # Event list:
        # For type 0: end: [time, 0, 0, 0, 0]
        # For type 1: arrival: [time, 1, x_coord, y_coord, service_time]
        # For type 2: service completion: [time, 2, assigned_amb_index, 0, 0]
        # ------------------------------
        event_list = []
        current_time = 0.0

        # Schedule termination and first arrival
        event_list.append([sim_length, 0, 0, 0, 0])
        event_list.append(next_arrival(0))

        # Ambulance state: [x, y, status]
        ambs = np.array([[bx, by, AVAILABLE] for bx, by in bases])
        queued_calls = []
        active_calls = 0

        total_response_time = 0.0
        num_calls = 0
        grad_total = np.zeros((variable_base_count, 2))

        # per-variable-base carry for waiting-time derivative to use at next queued call
        # carry is used if the next call for this ambulance has to wait
        carry_next = np.zeros((variable_base_count, 2))

        # ------------------------------
        # Main event loop
        # ------------------------------
        while event_list:
            event_list.sort(key=lambda e: e[0])
            event = event_list.pop(0)
            current_time = event[0]
            etype = event[1]

            if etype == 0:
                # End
                break

            if etype == 1:
                # Arrival
                active_calls += 1
                if active_calls > len(bases):
                    queued_calls.append(event)
                else:
                    # Find nearest available ambulance
                    times = [
                        (
                            np.sum(np.abs(amb[:2] - event[2:4])) / amb_speed
                            if amb[2] == AVAILABLE
                            else float("inf")
                        )
                        for amb in ambs
                    ]
                    i = int(np.argmin(times))
                    ambs[i, 2] = BUSY
                    response_time = times[i]
                    total_response_time += response_time
                    num_calls += 1

                    # Gradient update
                    # no waiting: Response time (R) = Driving time (D)
                    if (
                        i >= variable_base_start_index
                        and i - variable_base_start_index < variable_base_count
                    ):
                        j = i - variable_base_start_index
                        # Compute travel gradient dD wrt base position
                        dx = np.sign(ambs[i, 0] - event[2]) / amb_speed
                        dy = np.sign(ambs[i, 1] - event[3]) / amb_speed
                        dd = np.array([dx, dy])
                        # No waiting time
                        grad_total[j] += dd
                        # Set carry for next potential queued call for this ambulance
                        # dW(i)+2dD(i) = 0 + 2dD
                        carry_next[j] = 2.0 * dd

                    done_time = current_time + 2 * response_time + event[4]
                    event_list.append([done_time, 2, i, 0, 0])

                event_list.append(next_arrival(current_time))

            elif etype == 2:
                # service completion
                i = int(event[2])
                ambs[i, 2] = AVAILABLE
                active_calls -= 1

                if queued_calls:
                    # dispatch first queued call
                    qevent = queued_calls.pop(0)
                    ambs[i, 2] = BUSY  # mark ambulance as busy again

                    travel = np.sum(np.abs(ambs[i, 0:2] - qevent[2:4])) / amb_speed
                    queue_delay = current_time - qevent[0]
                    total_response_time += travel + queue_delay
                    num_calls += 1

                    # Gradient update (queued: R = W + D)
                    # If ambulance is from a variable base:
                    if (
                        i >= variable_base_start_index
                        and i - variable_base_start_index < variable_base_count
                    ):
                        j = i - variable_base_start_index
                        # Compute travel gradient dD wrt base position
                        dx = np.sign(ambs[i, 0] - qevent[2]) / amb_speed
                        dy = np.sign(ambs[i, 1] - qevent[3]) / amb_speed
                        dd = np.array([dx, dy])
                        # For queued calls:
                        # Response time (R) = Waiting time (W) + Driving time (D)
                        grad_total[j] += carry_next[j] + dd
                        # Update carry for the next call assigned to this ambulance
                        # new carry = old carry + 2 * dD
                        carry_next[j] = carry_next[j] + 2.0 * dd

                    done_time = current_time + 2 * travel + qevent[4]
                    event_list.append([done_time, 2, i, 0, 0])

                # if no queue, we do not need to change carry_next here
                # the next non-queued arrival will overwrite it with 2*dD

        if num_calls:
            avg_time = total_response_time / num_calls
            grad_avg = grad_total / num_calls
        else:
            avg_time = float("inf")
            grad_avg = np.full((variable_base_count, 2), float("nan"))

        responses = {"avg_response_time": avg_time}
        gradients = {
            "avg_response_time": {"variable_locs": grad_avg.flatten().tolist()}
        }
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

    def replicate(self, _x: tuple) -> RepResult:  # noqa: D102
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
