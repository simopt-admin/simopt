"""Standalone Python simulation for a stochastic activity network."""

from __future__ import annotations

from collections import deque
from typing import Annotated, Final, Self

import numpy as np
from pydantic import BaseModel, Field, model_validator

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt._markers import simulation
from simopt.input_models import InputModel

NUM_ARCS: Final[int] = 13


class SANConfig(BaseModel):
    """Configuration for the Stochastic Activity Network model."""

    num_nodes: Annotated[
        int,
        Field(
            default=9,
            description="number of nodes",
            gt=0,
            json_schema_extra={"isDatafarmable": False},
        ),
    ]
    arcs: Annotated[
        list[tuple[int, int]],
        Field(
            default=[
                (1, 2),
                (1, 3),
                (2, 3),
                (2, 4),
                (2, 6),
                (3, 6),
                (4, 5),
                (4, 7),
                (5, 6),
                (5, 8),
                (6, 9),
                (7, 8),
                (8, 9),
            ],
            description="list of arcs",
            min_length=1,
        ),
    ]
    arc_means: Annotated[
        tuple[float, ...],
        Field(
            default=(1.0,) * NUM_ARCS,
            description="mean task durations for each arc",
        ),
    ]

    def __dfs(self, graph: dict[int, set], start: int, visited: set | None = None) -> set:
        if visited is None:
            visited = set()
        visited.add(start)

        for next_point in graph[start] - visited:
            self.__dfs(graph, next_point, visited)
        return visited

    def _check_arcs(self) -> None:
        if len(self.arcs) <= 0:
            raise ValueError("The length of arcs must be greater than 0.")
        # Check graph is connected.
        graph = {node: set() for node in range(1, self.num_nodes + 1)}
        for a in self.arcs:
            graph[a[0]].add(a[1])
        visited = self.__dfs(graph, 1)

        if self.num_nodes not in visited:
            raise ValueError("Graph must be connected from node 1 to the final node.")

    def _check_arc_means(self) -> None:
        positive = True
        for x in list(self.arc_means):
            positive = positive and (x > 0)
        if not positive:
            raise ValueError("All elements in arc_means must be greater than 0.")

    @model_validator(mode="after")
    def _validate_model(self) -> Self:
        self._check_arcs()
        self._check_arc_means()
        if len(self.arc_means) != len(self.arcs):
            raise ValueError("The length of arc_means must be equal to the length of arcs.")
        return self


@simulation
def replicate(
    factors: SANConfig,
    rngs: list[MRG32k3a],
    time_model: InputModel,
) -> tuple[float, np.ndarray, list[int], np.ndarray]:
    """Return longest-path metrics and arc-mean gradients for one replication."""
    num_nodes: int = factors.num_nodes
    arcs: list[tuple[int, int]] = factors.arcs
    arc_means: tuple[float, ...] = factors.arc_means

    # Topological sort.
    node_range = range(1, num_nodes + 1)
    graph_in = {node: set() for node in node_range}
    graph_out = {node: set() for node in node_range}
    for start, end in arcs:
        graph_in[end].add(start)
        graph_out[start].add(end)

    indegrees = [len(graph_in[n]) for n in node_range]
    # outdegrees = [len(graph_out[n]) for n in node_range]
    queue = deque(n for n in node_range if indegrees[n - 1] == 0)
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in graph_out[u]:
            indegrees[v - 1] -= 1
            if indegrees[v - 1] == 0:
                queue.append(v)

    # Arc lengths
    arc_length = {arc: time_model.random(rngs[0], 1 / arc_means[i]) for i, arc in enumerate(arcs)}

    # Longest path
    path_length = np.zeros(num_nodes)
    prev = np.full(num_nodes, -1)
    for vi in topo_order:
        for j in graph_out[vi]:
            new_len = path_length[vi - 1] + arc_length[(vi, j)]
            if new_len > path_length[j - 1]:
                path_length[j - 1] = new_len
                prev[j - 1] = vi

    longest_path = path_length[-1]

    # Calculate the IPA gradient w.r.t. arc means.
    # If an arc is on the longest path, the component of the gradient
    # is the length of the length of that arc divided by its mean.
    # If an arc is not on the longest path, the component of the gradient is zero.
    arc_to_index = {arc: i for i, arc in enumerate(arcs)}

    grads = np.zeros((num_nodes, len(arcs)))
    for node in topo_order:
        gradient = np.zeros(len(arcs))
        current = node
        backtrack = int(prev[node - 1])

        while current != topo_order[0]:
            arc = (backtrack, current)
            idx = arc_to_index[arc]
            gradient[idx] = arc_length[arc] / arc_means[idx]
            current = backtrack
            backtrack = int(prev[backtrack - 1])

        grads[node - 1] = gradient

    return longest_path, path_length, topo_order, grads
