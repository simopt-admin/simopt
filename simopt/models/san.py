"""Simulate duration of a stochastic activity network (SAN)."""

from __future__ import annotations

from collections import deque
from typing import Callable, Final

import numpy as np

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.base import ConstraintType, Model, Problem, VariableType
from simopt.input_models import Exp
from simopt.utils import classproperty, override

NUM_ARCS: Final[int] = 13


class SAN(Model):
    """Stochastic Activity Network (SAN) Model.

    A model that simulates a stochastic activity network problem with
    tasks that have exponentially distributed durations, and the selected
    means come with a cost.
    """

    @classproperty
    @override
    def class_name(cls) -> str:
        return "Stochastic Activity Network"

    @classproperty
    @override
    def n_rngs(cls) -> int:
        return 1

    @classproperty
    @override
    def n_responses(cls) -> int:
        return 1

    @classproperty
    @override
    def specifications(cls) -> dict[str, dict]:
        return {
            "num_nodes": {
                "description": "number of nodes",
                "datatype": int,
                "default": 9,
                "isDatafarmable": False,
            },
            "arcs": {
                "description": "list of arcs",
                "datatype": list,
                "default": [
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
            },
            "arc_means": {
                "description": "mean task durations for each arc",
                "datatype": tuple,
                "default": (1,) * NUM_ARCS,
            },
        }

    @property
    @override
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "num_nodes": self._check_num_nodes,
            "arcs": self._check_arcs,
            "arc_means": self._check_arc_means,
        }

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize the SAN model.

        Args:
            fixed_factors : dict
                fixed factors of the simulation model
        """
        # Let the base class handle default arguments.
        super().__init__(fixed_factors)

        self.time_model = Exp()

    def _check_num_nodes(self) -> None:
        if self.factors["num_nodes"] <= 0:
            raise ValueError("num_nodes must be greater than 0.")

    def __dfs(
        self, graph: dict[int, set], start: int, visited: set | None = None
    ) -> set:
        if visited is None:
            visited = set()
        visited.add(start)

        for next_point in graph[start] - visited:
            self.__dfs(graph, next_point, visited)
        return visited

    def _check_arcs(self) -> bool:
        if len(self.factors["arcs"]) <= 0:
            raise ValueError("The length of arcs must be greater than 0.")
        # Check graph is connected.
        graph = {node: set() for node in range(1, self.factors["num_nodes"] + 1)}
        for a in self.factors["arcs"]:
            graph[a[0]].add(a[1])
        visited = self.__dfs(graph, 1)
        return self.factors["num_nodes"] in visited

    def _check_arc_means(self) -> bool:
        positive = True
        for x in list(self.factors["arc_means"]):
            positive = positive and (x > 0)
        return positive

    @override
    def check_simulatable_factors(self) -> bool:
        if len(self.factors["arc_means"]) != len(self.factors["arcs"]):
            raise ValueError(
                "The length of arc_means must be equal to the length of arcs."
            )
        return True

    def before_replicate(self, rng_list):
        self.time_model.set_rng(rng_list[0])

    def replicate(self) -> tuple[dict, dict]:
        """Simulate a single replication for the current model factors.

        Args:
            rng_list (list[MRG32k3a]): Random number generators used to simulate
                the replication.

        Returns:
            tuple[dict, dict]: A tuple containing:
                - responses (dict): Performance measures of interest, including:
                    - "longest_path_length": Length or duration of the longest path.
                - gradients (dict): A dictionary of gradient estimates for
                    each response.
        """
        num_nodes: int = self.factors["num_nodes"]
        arcs: list[tuple[int, int]] = self.factors["arcs"]
        arc_means: tuple[int, ...] = self.factors["arc_means"]

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
        arc_length = {
            arc: self.time_model.random(1 / arc_means[i]) for i, arc in enumerate(arcs)
        }

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
        gradient = np.zeros(len(arcs))
        current = topo_order[-1]
        backtrack = int(prev[-1])

        while current != topo_order[0]:
            arc = (backtrack, current)
            idx = arc_to_index[arc]
            gradient[idx] = arc_length[arc] / arc_means[idx]
            current = backtrack
            backtrack = int(prev[backtrack - 1])

        # Compose responses and gradients.
        responses = {"longest_path_length": longest_path}
        gradients = {
            response_key: {
                factor_key: np.zeros(len(self.specifications))
                for factor_key in self.specifications
            }
            for response_key in responses
        }
        gradients["longest_path_length"]["arc_means"] = gradient
        return responses, gradients


class SANLongestPath(Problem):
    """Base class to implement simulation-optimization problems."""

    @classproperty
    @override
    def class_name_abbr(cls) -> str:
        return "SAN-1"

    @classproperty
    @override
    def class_name(cls) -> str:
        return "Min Mean Longest Path for Stochastic Activity Network"

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
    def optimal_value(cls) -> None:
        return None

    @classproperty
    @override
    def optimal_solution(cls) -> None:
        return None

    @classproperty
    @override
    def model_default_factors(cls) -> dict:
        return {}

    @classproperty
    @override
    def model_decision_factors(cls) -> set[str]:
        return {"arc_means"}

    @classproperty
    @override
    def specifications(cls) -> dict[str, dict]:
        return {
            "initial_solution": {
                "description": "initial solution",
                "datatype": tuple,
                "default": (8,) * NUM_ARCS,
            },
            "budget": {
                "description": "max # of replications for a solver to take",
                "datatype": int,
                "default": 10000,
                "isDatafarmable": False,
            },
            "arc_costs": {
                "description": "Cost associated to each arc.",
                "datatype": tuple,
                "default": (1,) * NUM_ARCS,
            },
        }

    @property
    @override
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget,
            "arc_costs": self._check_arc_costs,
        }

    @property
    @override
    def dim(self) -> int:
        return len(self.model.factors["arcs"])

    @property
    @override
    def lower_bounds(self) -> tuple:
        return (1e-2,) * self.dim

    @property
    @override
    def upper_bounds(self) -> tuple:
        return (np.inf,) * self.dim

    def __init__(
        self,
        name: str = "SAN-1",
        fixed_factors: dict | None = None,
        model_fixed_factors: dict | None = None,
    ) -> None:
        """Initialize the SANLongestPath problem.

        Args:
            name : str
                user-specified name for problem
            fixed_factors : dict
                dictionary of user-specified problem factors
            model_fixed_factors : dict
                subset of user-specified non-decision factors to pass through to the
                model
        """
        # Let the base class handle default arguments.
        super().__init__(
            name=name,
            fixed_factors=fixed_factors,
            model_fixed_factors=model_fixed_factors,
            model=SAN,
        )

    def _check_arc_costs(self) -> bool:
        positive = True
        for x in list(self.factors["arc_costs"]):
            positive = positive and x > 0
        matching_len = len(self.factors["arc_costs"]) == len(self.model.factors["arcs"])
        return positive and matching_len

    @override
    def vector_to_factor_dict(self, vector: tuple) -> dict:
        return {"arc_means": vector[:]}

    @override
    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:
        return factor_dict["arc_means"]

    @override
    def response_dict_to_objectives(self, response_dict: dict) -> tuple:
        return (response_dict["longest_path_length"],)

    def deterministic_stochastic_constraints_and_gradients(self) -> tuple[tuple, tuple]:
        """Compute deterministic components of stochastic constraints.

        Returns:
            tuple:
                - tuple: The deterministic components of the stochastic constraints.
                - tuple: The gradients of those deterministic components.
        """
        det_stoch_constraints = ()
        det_stoch_constraints_gradients = (
            (0,) * self.dim,
        )  # tuple of tuples - of sizes self.dim by self.dim, full of zeros
        return det_stoch_constraints, det_stoch_constraints_gradients

    @override
    def deterministic_objectives_and_gradients(self, x: tuple) -> tuple[tuple, tuple]:
        det_objectives = (np.sum(np.array(self.factors["arc_costs"]) / np.array(x)),)
        det_objectives_gradients = (
            -np.array(self.factors["arc_costs"]) / (np.array(x) ** 2),
        )
        return det_objectives, det_objectives_gradients

    @override
    def check_deterministic_constraints(self, x: tuple) -> bool:
        return all(x_i >= 0 for x_i in x)

    @override
    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
        return tuple(
            [rand_sol_rng.lognormalvariate(lq=0.1, uq=10) for _ in range(self.dim)]
        )
