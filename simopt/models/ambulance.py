"""Simulation of the average response time in a multi-base ambulance dispatch system."""

from __future__ import annotations

from typing import Callable, Final

import numpy as np

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.base import ConstraintType, Model, Problem, VariableType
from simopt.utils import classproperty, override

AVAILABLE = 0
BUSY = 1

NUM_FIXED: Final[int] = 3
NUM_VARIABLE: Final[int] = 2

class Ambulance(Model):
    """Simulate the average response time in a multi-base ambulance dispatch system.
    
    The system includes a set of fixed ambulance bases and a set of variable bases
    with decision-variable coordinates. The objective is to minimize the expected 
    response time by optimizing the locations of the variable bases.
    """ 
    @classproperty
    @override
    def class_name_abbr(cls) -> str:
        return "AMBULANCE"
    
    @classproperty
    @override
    def class_name(cls) -> str:
        return "Ambulance Dispatch"

    @classproperty
    @override
    def n_rngs(cls) -> int:
        return 4

    @classproperty
    @override
    def n_responses(cls) -> int:
        return 1

    @classproperty
    @override
    def specifications(cls) -> dict[str, dict]:
        # In specifications, use NUM_FIXED and NUM_VARIABLE to define default locs.
        return {
            "fixed_base_count": {
            "description": "Number of fixed bases",
            "datatype": int,
            "default": NUM_FIXED,
            },
            "variable_base_count": {
            "description": "Number of variable bases",
            "datatype": int,
            "default": NUM_VARIABLE,
            },
            "fixed_locs": {
            "description": "Fixed base coordinates [x0, y0, x1, y1, ...]",
            "datatype": list,
            "default": [15, 15, 5, 15, 5, 5],
            },
            "variable_locs": {
            "description": "Variable base coordinates [x0, y0, x1, y1, ...]",
            "datatype": list,
            "default": [6, 6, 6, 6],
            },
            "call_loc_beta_x": {
            "description": "Beta distribution params for x-axis",
            "datatype": tuple,
            "default": (2.0, 1.0),
            },
            "call_loc_beta_y": {
            "description": "Beta distribution params for y-axis",
            "datatype": tuple,
            "default": (2.0, 1.0),
            },
        }

    @property
    @override
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "fixed_base_count": self._check_fixed_base_count,
            "variable_base_count": self._check_variable_base_count,
            "fixed_locs": self._check_fixed_locs,
            "variable_locs": self._check_variable_locs,
            "call_loc_beta_x": self._check_call_loc_beta_x,
            "call_loc_beta_y": self._check_call_loc_beta_y,
        }

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """First call parent constructor, this will populate self.factors."""
        super().__init__(fixed_factors)

    def _check_fixed_base_count(self) -> None:
        if self.factors["fixed_base_count"] < 0:
            raise ValueError("fixed_base_count must be >= 0.")

    def _check_variable_base_count(self) -> None:
        if self.factors["variable_base_count"] <= 0:
            raise ValueError("variable_base_count must be > 0.")

    def _check_fixed_locs(self) -> None:
        expected_length = 2 * self.factors["fixed_base_count"]
        if len(self.factors["fixed_locs"]) != expected_length:
            raise ValueError(
                f"The length of fixed_locs must be {expected_length} "
                f"(2 * fixed_base_count)."
            )

    def _check_variable_locs(self) -> None:
        expected_length = 2 * self.factors["variable_base_count"]
        if len(self.factors["variable_locs"]) != expected_length:
            raise ValueError(
                f"The length of variable_locs must be {expected_length} "
                f"(2 * variable_base_count)."
            )
            
    def _check_call_loc_beta_x(self) -> None:
        if not isinstance(self.factors["call_loc_beta_x"], tuple) or \
           len(self.factors["call_loc_beta_x"]) != 2:
            raise ValueError(
            "call_loc_beta_x must be a tuple of (alpha, beta)."
            )

    def _check_call_loc_beta_y(self) -> None:
        if not isinstance(self.factors["call_loc_beta_y"], tuple) or \
           len(self.factors["call_loc_beta_y"]) != 2:
            raise ValueError(
            "call_loc_beta_y must be a tuple of (alpha, beta)."
            )
            
    @override
    def check_simulatable_factors(self) -> bool:
        variable_locs = self.factors["variable_locs"]
        if not all(0 <= loc <= 20 for loc in variable_locs):
            raise ValueError("All variable_locs must be between 0 and 20.")
        return True

    def replicate(self, rng_list: list[MRG32k3a]) -> tuple[dict, dict]:
        """Run one replication of the ambulance dispatch simulation."""
        # ------------------------------
        # Setup base locations and system parameters
        # ------------------------------
        fixed_base_count = self.factors["fixed_base_count"]
        variable_base_count = self.factors["variable_base_count"]
        fixed_locs = self.factors["fixed_locs"]
        variable_locs = self.factors["variable_locs"]

        fixed_base_positions = [
            [fixed_locs[2 * i], fixed_locs[2 * i + 1]] 
            for i in range(fixed_base_count)
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
            (2 * est_travel_time + mean_scene_time) 
            / n_ambulances 
            / utilization
        )
        sim_length = 60 * 24.0 * 1  # Simulate 1 day

        # Random number streams
        rng_arrival, rng_scene, rng_x, rng_y = rng_list

        def beta_sample_with_rng(rng: MRG32k3a, alpha: float, beta: float) -> float:
            """Draw a Beta(alpha, beta) variate via the Gamma ratio.
            
            Beta(a,b) = Gamma(a,1) / (Gamma(a,1) + Gamma(b,1)).
            This keeps all randomness on the same MRG32k3a stream.
            """
            gx = rng.gammavariate(alpha, 1.0)
            gy = rng.gammavariate(beta, 1.0)
            return gx / (gx + gy)

        def next_arrival(curr: float) -> list:
            """Generate the next arrival event.

            - Arrival inter-time: still exponential.
            - Call location (x, y): drawn from scaled Beta distributions.
            Tune alpha/beta to shape spatial hot spots.
            """
            # Draw Beta-based coordinates in [0, SQUARE_WIDTH]
            x_coord = beta_sample_with_rng(rng_x, alpha=2.0, beta=1.0) * sqaure_width
            y_coord = beta_sample_with_rng(rng_y, alpha=1.0, beta=2.0) * sqaure_width

            return [
            curr + rng_arrival.expovariate(1.0 / mean_interval), # interarrival time
            1,                                                   # event type: arrival
            x_coord,                                             # x from Beta
            y_coord,                                             # y from Beta
            rng_scene.expovariate(1.0 / mean_scene_time),        # scene time
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
                    if (i >= variable_base_start_index and 
                        i - variable_base_start_index < variable_base_count):
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
                    travel = np.sum(np.abs(ambs[i, 0:2] - qevent[2:4])) / amb_speed
                    queue_delay = current_time - qevent[0]
                    total_response_time += travel + queue_delay
                    num_calls += 1

                    # Gradient update (queued: R = W + D)
                    # If ambulance is from a variable base:
                    if (i >= variable_base_start_index and 
                        i - variable_base_start_index < variable_base_count):
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
            "avg_response_time": {
                "variable_locs": grad_avg.flatten().tolist()
            }
        }
        return responses, gradients

class AmbulanceMinAvgResponse(Problem):
    """Base class to implement simulation-optimization problems."""
    @classproperty
    @override
    def class_name_abbr(cls) -> str:
        return "AMBULANCE-1"

    @classproperty
    @override
    def class_name(cls) -> str:
        return "Minimum Average Waiting Time for Ambulance Dispatch"
    
    @classproperty
    @override
    def n_objectives(cls) -> int:
        return 1

    @classproperty
    @override
    def n_stochastic_constraints(cls) -> int:
        return 0

    @classproperty
    @override
    def minmax(cls) -> tuple[int]:
        return (-1,)

    @classproperty
    @override
    def constraint_type(cls) -> ConstraintType:
        return ConstraintType.BOX

    @classproperty
    @override
    def variable_type(cls) -> VariableType:
        return VariableType.CONTINUOUS

    @classproperty
    @override
    def gradient_available(cls) -> bool:
        return True
    
    @classproperty
    @override
    def optimal_value(cls) -> float | None:
        return None

    @classproperty
    @override
    def optimal_solution(cls) -> tuple | None:
        return None

    @classproperty
    @override
    def model_default_factors(cls) -> dict:
        return {
            "fixed_base_count": NUM_FIXED,
            "variable_base_count": NUM_VARIABLE,
            "fixed_locs": [15, 15, 5, 15, 5, 5],
            "variable_locs": [6, 6, 6, 6],
        }

    @classproperty
    @override
    def model_decision_factors(cls) -> set[str]:
        return {"variable_locs"}

    @classproperty
    @override
    def specifications(cls) -> dict[str, dict]:
        return {
            "initial_solution": {
                "description": "initial solution",
                "datatype": tuple,
                "default": (6, 6, 6, 6),
            },
            "budget": {
                "description": "max # of replications for a solver to take",
                "datatype": int,
                "default": 1000,
                "isDatafarmable": False,
            },
        }

    @property
    @override
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget,
        }

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

    def __init__(
        self,
        name: str = "AMBULANCE-1",
        fixed_factors: dict | None = None,
        model_fixed_factors: dict | None = None,
    ) -> None:
        """Initialize the problem instance."""
        super().__init__(name=name,
                         fixed_factors=fixed_factors,
                         model_fixed_factors=model_fixed_factors,
                         model=Ambulance)


    @override
    def vector_to_factor_dict(self, vector: tuple) -> dict:
        return {"variable_locs": vector[:]}
    
    @override
    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:
        return tuple(factor_dict["variable_locs"])

    @override
    def response_dict_to_objectives(self, response_dict: dict) -> tuple:
        return (response_dict["avg_response_time"],)

    @override
    def check_deterministic_constraints(self, _x: tuple) -> bool:
        return all(0 <= xi <= 20 for xi in _x)

    @override
    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
        return tuple(rand_sol_rng.uniform(0, 20) for _ in range(self.dim))
