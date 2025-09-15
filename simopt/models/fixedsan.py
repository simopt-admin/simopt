"""Simulate duration of a stochastic activity network (SAN)."""

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
    VariableType,
)
from simopt.input_models import Exp

# TODO: figure out if this should ever be anything other than 13
NUM_ARCS: Final[int] = 13


class FixedSANConfig(BaseModel):
    """Configuration model for Fixed Stochastic Activity Network simulation.

    A model that simulates a stochastic activity network problem with tasks
    that have exponentially distributed durations, and the selected means
    come with a cost.
    """

    num_arcs: Annotated[
        int,
        Field(
            default=NUM_ARCS,
            description="number of arcs",
            gt=0,
        ),
    ]
    num_nodes: Annotated[
        int,
        Field(
            default=9,
            description="number of nodes",
            gt=0,
        ),
    ]
    arc_means: Annotated[
        tuple[float, ...],
        Field(
            default=(1,) * NUM_ARCS,
            description="mean task durations for each arc",
        ),
    ]

    def _check_arc_means(self) -> None:
        if not all(x > 0 for x in list(self.arc_means)):
            raise ValueError("All arc means must be greater than 0.")

    @model_validator(mode="after")
    def _validate_model(self) -> Self:
        self._check_arc_means()

        return self


class FixedSANLongestPathConfig(BaseModel):
    """Configuration model for Fixed SAN Longest Path Problem.

    Min Mean Longest Path for Fixed Stochastic Activity Network problem.
    """

    initial_solution: Annotated[
        tuple[float, ...],
        Field(
            default=(10,) * 13,
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
    arc_costs: Annotated[
        tuple[float, ...],
        Field(
            default=(1,) * 13,
            description="cost associated to each arc",
        ),
    ]

    def _check_arc_costs(self) -> None:
        if len(self.arc_costs) != NUM_ARCS:
            raise ValueError(f"arc_costs must be of length {NUM_ARCS}.")

        if not all(x > 0 for x in list(self.arc_costs)):
            raise ValueError("All arc costs must be greater than 0.")

    @model_validator(mode="after")
    def _validate_model(self) -> Self:
        self._check_arc_costs()
        return self


class FixedSAN(Model):
    """Fixed Stochastic Activity Network (SAN) Model.

    A model that simulates a stochastic activity network problem with tasks
    that have exponentially distributed durations, and the selected means
    come with a cost.
    """

    class_name_abbr: ClassVar[str] = "FIXEDSAN"
    class_name: ClassVar[str] = "Fixed Stochastic Activity Network"
    config_class: ClassVar[type[BaseModel]] = FixedSANConfig
    n_rngs: ClassVar[int] = 1
    n_responses: ClassVar[int] = 1

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize the Fixed Stochastic Activity Network model.

        Args:
            fixed_factors (dict, optional): Fixed factors for the model.
                Defaults to None.
        """
        # Let the base class handle default arguments.
        super().__init__(fixed_factors)

        self.time_model = Exp()

    def before_replicate(self, rng_list: list[MRG32k3a]) -> None:  # noqa: D102
        self.time_model.set_rng(rng_list[0])

    def replicate(self) -> tuple[dict, dict]:
        """Simulate a single replication for the current model factors.

        Args:
            rng_list (list[MRG32k3a]): Random number generators used to simulate
                the replication.

        Returns:
            tuple[dict, dict]: A tuple containing:
                - responses (dict): Performance measures of interest, including:
                    - "longest_path_length": The length or duration of the longest path.
                - gradients (dict): A dictionary of gradient estimates for
                    each response.
        """
        num_nodes: int = self.factors["num_nodes"]
        num_arcs: int = self.factors["num_arcs"]
        thetas = list(self.factors["arc_means"])

        # Make sure we're not going to index out of bounds.
        if num_nodes < 9 or num_arcs < 13:
            raise ValueError(
                "This model only supports 9 nodes and 13 arcs. "
                f"num_nodes: {num_nodes}, num_arcs: {num_arcs}"
            )
        # Generate arc lengths.
        nodes = np.zeros(num_nodes)
        time_deriv = np.zeros((num_nodes, num_arcs))
        arcs = [self.time_model.random(1 / x) for x in thetas]

        def get_time(prev_node_idx: int, arc_idx: int) -> float:
            return nodes[prev_node_idx] + arcs[arc_idx]

        def update_node(target_node_idx: int, segments: list[tuple[int, int]]) -> None:
            """Update the target node with the maximum time from the segments.

            Args:
                target_node_idx (int): Index of the target node to be updated.
                segments (list[tuple[int, int]]): List of (previous_node_idx, arc_idx)
                    tuples representing the segments leading to the target node.
            """
            # Get the time for the first segment in the list
            best_prev, best_arc = segments[0]
            max_time = get_time(best_prev, best_arc)
            # Iterate through the rest of the segments (if any) to find the
            # maximum time
            for seg_prev, seg_arc in segments[1:]:
                t = get_time(seg_prev, seg_arc)
                if t > max_time:
                    max_time = t
                    best_prev, best_arc = seg_prev, seg_arc

            # Update the target node with the maximum time and the
            # time derivative
            nodes[target_node_idx] = max_time
            time_deriv[target_node_idx, :] = time_deriv[best_prev, :].copy()
            time_deriv[target_node_idx, best_arc] += arcs[best_arc] / thetas[best_arc]

        # node 1 = node 0 + arc 0
        update_node(1, [(0, 0)])
        # node 2 = max(node0+arc1, node1+arc2)
        update_node(2, [(0, 1), (1, 2)])
        # node 3 = node1 + arc3
        update_node(3, [(1, 3)])
        # node 4 = node3 + arc6
        update_node(4, [(3, 6)])
        # node 5 = max(node1+arc4, node2+arc5, node4+arc8)
        update_node(5, [(1, 4), (2, 5), (4, 8)])
        # node 6 = node3 + arc7
        update_node(6, [(3, 7)])
        # node 7 = max(node6+arc11, node4+arc9)
        update_node(7, [(6, 11), (4, 9)])
        # node 8 = max(node5+arc10, node7+arc12)
        update_node(8, [(5, 10), (7, 12)])

        longest_path = float(nodes[8])
        longest_path_gradient = time_deriv[8, :]

        # Compose responses and gradients.
        responses = {"longest_path_length": longest_path}
        gradients = {
            response_key: {
                factor_key: np.zeros(len(self.specifications))
                for factor_key in self.specifications
            }
            for response_key in responses
        }
        gradients["longest_path_length"]["arc_means"] = longest_path_gradient

        return responses, gradients


class FixedSANLongestPath(Problem):
    """Base class to implement simulation-optimization problems."""

    class_name_abbr: ClassVar[str] = "FIXEDSAN-1"
    class_name: ClassVar[str] = (
        "Min Mean Longest Path for Fixed Stochastic Activity Network"
    )
    config_class: ClassVar[type[BaseModel]] = FixedSANLongestPathConfig
    model_class: ClassVar[type[Model]] = FixedSAN
    n_objectives: ClassVar[int] = 1
    n_stochastic_constraints: ClassVar[int] = 0
    minmax: ClassVar[tuple[int, ...]] = (-1,)
    constraint_type: ClassVar[ConstraintType] = ConstraintType.BOX
    variable_type: ClassVar[VariableType] = VariableType.CONTINUOUS
    gradient_available: ClassVar[bool] = True
    optimal_value: ClassVar[float | None] = None
    optimal_solution: tuple | None = None
    model_default_factors: ClassVar[dict] = {}
    model_decision_factors: ClassVar[set[str]] = {"arc_means"}

    @property
    def dim(self) -> int:  # noqa: D102
        return self.model.factors["num_arcs"]

    @property
    def lower_bounds(self) -> tuple:  # noqa: D102
        return (1e-2,) * self.dim

    @property
    def upper_bounds(self) -> tuple:  # noqa: D102
        return (np.inf,) * self.dim

    def check_arc_costs(self) -> bool:
        """Check if all arc costs are positive and match the number of arcs."""
        return len(self.factors["arc_costs"]) == self.model.factors["num_arcs"] and all(
            x > 0 for x in self.factors["arc_costs"]
        )

    def vector_to_factor_dict(self, vector: tuple) -> dict:  # noqa: D102
        return {"arc_means": vector[:]}

    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:  # noqa: D102
        return tuple(factor_dict["arc_means"])

    def replicate(self, x: tuple) -> RepResult:  # noqa: D102
        responses, gradients = self.model.replicate()
        objectives = [
            Objective(
                stochastic=responses["longest_path_length"],
                stochastic_gradients=gradients["longest_path_length"]["arc_means"],
                deterministic=np.sum(np.array(self.factors["arc_costs"]) / np.array(x)),
                deterministic_gradients=-np.array(self.factors["arc_costs"])
                / (np.array(x) ** 2),
            )
        ]
        return RepResult(objectives=objectives)

    def check_deterministic_constraints(self, x: tuple) -> bool:  # noqa: D102
        return all(x_i >= 0 for x_i in x)

    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:  # noqa: D102
        return tuple(
            [rand_sol_rng.lognormalvariate(lq=0.1, uq=10) for _ in range(self.dim)]
        )
