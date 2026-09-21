"""Standalone Python simulation for the fixed stochastic activity network."""

from __future__ import annotations

from typing import Annotated, Final, Self

import numpy as np
from pydantic import BaseModel, Field, model_validator

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt._markers import simulation
from simopt.input_models import InputModel

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


@simulation
def replicate(
    factors: FixedSANConfig,
    rngs: list[MRG32k3a],
    time_model: InputModel,
) -> tuple[float, np.ndarray]:
    """Return longest-path length and its arc-mean gradient for one replication."""
    num_nodes: int = factors.num_nodes
    num_arcs: int = factors.num_arcs
    thetas = list(factors.arc_means)

    # Make sure we're not going to index out of bounds.
    if num_nodes < 9 or num_arcs < 13:
        raise ValueError(
            "This model only supports 9 nodes and 13 arcs. "
            f"num_nodes: {num_nodes}, num_arcs: {num_arcs}"
        )
    # Generate arc lengths.
    nodes = np.zeros(num_nodes)
    time_deriv = np.zeros((num_nodes, num_arcs))
    arcs = [time_model.random(rngs[0], 1 / x) for x in thetas]

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

    return longest_path, longest_path_gradient
