"""Simulate duration of a stochastic activity network (SAN)."""

from __future__ import annotations

from typing import Callable, Final

import numpy as np

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.base import ConstraintType, Model, Problem, VariableType
from simopt.input_models import Exp
from simopt.utils import classproperty, override

# TODO: figure out if this should ever be anything other than 13
NUM_ARCS: Final[int] = 13


class FixedSAN(Model):
    """Fixed Stochastic Activity Network (SAN) Model.

    A model that simulates a stochastic activity network problem with tasks
    that have exponentially distributed durations, and the selected means
    come with a cost.
    """

    @classproperty
    @override
    def class_name(cls) -> str:
        return "Fixed Stochastic Activity Network"

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
            "num_arcs": {
                "description": "number of arcs",
                "datatype": int,
                "default": NUM_ARCS,
            },
            "num_nodes": {
                "description": "number of nodes",
                "datatype": int,
                "default": 9,
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
            "num_arcs": self._check_num_arcs,
            "num_nodes": self._check_num_nodes,
            "arc_means": self._check_arc_means,
        }

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize the Fixed Stochastic Activity Network model.

        Args:
            fixed_factors (dict, optional): Fixed factors for the model.
                Defaults to None.
        """
        # Let the base class handle default arguments.
        super().__init__(fixed_factors)

        self.time_model = Exp()

    def _check_num_arcs(self) -> None:
        if self.factors["num_arcs"] <= 0:
            raise ValueError("num_arcs must be greater than 0.")

    def _check_num_nodes(self) -> None:
        if self.factors["num_nodes"] <= 0:
            raise ValueError("num_nodes must be greater than 0.")

    def _check_arc_means(self) -> bool:
        return all(x > 0 for x in list(self.factors["arc_means"]))

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

    @classproperty
    @override
    def class_name_abbr(cls) -> str:
        return "FIXEDSAN-1"

    @classproperty
    @override
    def class_name(cls) -> str:
        return "Min Mean Longest Path for Fixed Stochastic Activity Network"

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
                "default": (10,) * 13,
            },
            "budget": {
                "description": "max # of replications for a solver to take",
                "datatype": int,
                "default": 10000,
                "isDatafarmable": False,
            },
            "arc_costs": {
                "description": "cost associated to each arc",
                "datatype": tuple,
                "default": (1,) * 13,
            },
        }

    @property
    @override
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget,
            "arc_costs": self.check_arc_costs,
        }

    @property
    @override
    def dim(self) -> int:
        return self.model.factors["num_arcs"]

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
        name: str = "FIXEDSAN-1",
        fixed_factors: dict | None = None,
        model_fixed_factors: dict | None = None,
    ) -> None:
        """Initialize the Fixed Stochastic Activity Network problem.

        Args:
            name (str, optional): Name of the problem. Defaults to "FIXEDSAN-1".
            fixed_factors (dict, optional): Fixed factors for the problem.
                Defaults to None.
            model_fixed_factors (dict, optional): Fixed factors for the model.
                Defaults to None.
        """
        # Let the base class handle default arguments.
        super().__init__(
            name=name,
            fixed_factors=fixed_factors,
            model_fixed_factors=model_fixed_factors,
            model=FixedSAN,
        )

    def check_arc_costs(self) -> bool:
        """Check if all arc costs are positive and match the number of arcs."""
        return len(self.factors["arc_costs"]) == self.model.factors["num_arcs"] and all(
            x > 0 for x in self.factors["arc_costs"]
        )

    @override
    def vector_to_factor_dict(self, vector: tuple) -> dict:
        return {"arc_means": vector[:]}

    @override
    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:
        return tuple(factor_dict["arc_means"])

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
