"""
Summary
-------
Simulate duration of a stochastic activity network (SAN).
A detailed description of the model/problem can be found
`here <https://simopt.readthedocs.io/en/latest/san.html>`__.
"""

from __future__ import annotations

from typing import Callable, Final, NamedTuple

import numpy as np

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.base import ConstraintType, Model, Problem, VariableType
from simopt.utils import classproperty

# TODO: figure out if this should ever be anything other than 13
NUM_ARCS: Final[int] = 13


class FixedSAN(Model):
    """
    A model that simulates a stochastic activity network problem with tasks
    that have exponentially distributed durations, and the selected means
    come with a cost.

    Attributes
    ----------
    name : string
        name of model
    n_rngs : int
        number of random-number generators used to run a simulation replication
    n_responses : int
        number of responses (performance measures)
    factors : dict
        changeable factors of the simulation model
    specifications : dict
        details of each factor (for GUI and data validation)
    check_factor_list : dict
        switch case for checking factor simulatability

    Arguments
    ---------
    fixed_factors : nested dict
        fixed factors of the simulation model

    See also
    --------
    base.Model
    """

    @classproperty
    def class_name(cls) -> str:
        return "Fixed Stochastic Activity Network"

    @classproperty
    def n_rngs(cls) -> int:
        return 1

    @classproperty
    def n_responses(cls) -> int:
        return 1

    @classproperty
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
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "num_arcs": self.check_num_arcs,
            "num_nodes": self.check_num_nodes,
            "arc_means": self.check_arc_means,
        }

    def __init__(self, fixed_factors: dict | None = None) -> None:
        # Let the base class handle default arguments.
        super().__init__(fixed_factors)

    def check_num_arcs(self) -> None:
        if self.factors["num_arcs"] <= 0:
            raise ValueError("num_arcs must be greater than 0.")

    def check_num_nodes(self) -> None:
        if self.factors["num_nodes"] <= 0:
            raise ValueError("num_nodes must be greater than 0.")

    def check_arc_means(self) -> bool:
        for x in list(self.factors["arc_means"]):
            if x <= 0:
                return False
        return True

    def replicate(self, rng_list: list[MRG32k3a]) -> tuple[dict, dict]:
        """
        Simulate a single replication for the current model factors.

        Arguments
        ---------
        rng_list : list of mrg32k3a.mrg32k3a.MRG32k3a objects
            rngs for model to use when simulating a replication

        Returns
        -------
        responses : dict
            performance measures of interest
            "longest_path_length" = length/duration of longest path
        gradients : dict of dicts
            gradient estimates for each response
        """
        num_nodes: int = self.factors["num_nodes"]
        num_arcs: int = self.factors["num_arcs"]
        thetas = list(self.factors["arc_means"])

        # Designate separate random number generators.
        exp_rng = rng_list[0]

        # Generate arc lengths.
        nodes = np.zeros(num_nodes)
        time_deriv = np.zeros((num_nodes, num_arcs))
        arcs = [exp_rng.expovariate(1 / x) for x in thetas]

        class PathSegment(NamedTuple):
            prev_node_idx: int
            arc_idx: int

            def get_time(self) -> float:
                return nodes[self.prev_node_idx] + arcs[self.arc_idx]

            def to_tuple(self) -> tuple[int, int]:
                return self.prev_node_idx, self.arc_idx

        def update_node(target_node_idx: int, segment: PathSegment) -> None:
            prev = segment.prev_node_idx
            arc = segment.arc_idx
            nodes[target_node_idx] = segment.get_time()
            time_deriv[target_node_idx, :] = time_deriv[prev, :].copy()
            time_deriv[target_node_idx, arc] += arcs[arc] / thetas[arc]

        def max_update_node(
            target_node_idx: int, segments: list[PathSegment]
        ) -> None:
            seg_times = [seg.get_time() for seg in segments]
            max_segment = segments[np.argmax(seg_times)]
            update_node(target_node_idx, max_segment)

        def max_update_node_shifted(
            target_node_idx: int, segments: list[PathSegment]
        ) -> None:
            seg_times = [seg.get_time() for seg in segments]
            nodes[target_node_idx] = max(seg_times)
            ind = np.argmax(seg_times)
            # TODO: The nodes/arcs seem to be off by one, investigate if this
            # is due to a copying error from Matlab (1-based indexing).
            # If this is an error, this function can be removed and replaced
            # with max_update_node.
            segment = segments[(ind - 1) % len(segments)]
            arc = segment.arc_idx
            node = segment.prev_node_idx
            time_deriv[target_node_idx, :] = time_deriv[node, :]
            time_deriv[target_node_idx, arc] += arcs[arc] / thetas[arc]

        # node 1 = node 0 + arc 0
        update_node(1, PathSegment(0, 0))
        # node 2 = max(node0+arc1, node1+arc2)
        max_update_node(2, [PathSegment(0, 1), PathSegment(1, 2)])
        # node 3 = node1 + arc3
        update_node(3, PathSegment(1, 3))
        # node 4 = node3 + arc6
        update_node(4, PathSegment(3, 6))
        # node 5 = max(node1+arc4, node2+arc5, node4+arc8)
        max_update_node_shifted(
            5, [PathSegment(1, 4), PathSegment(2, 5), PathSegment(4, 8)]
        )
        # node 6 = node3 + arc7
        update_node(6, PathSegment(3, 7))
        # node 7 = max(node6+arc11, node4+arc9)
        max_update_node(7, [PathSegment(6, 11), PathSegment(4, 9)])
        # node 8 = max(node5+arc10, node7+arc12)
        max_update_node(8, [PathSegment(5, 10), PathSegment(7, 12)])

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


"""
Summary
-------
Minimize the duration of the longest path from a to i plus cost.
"""


class FixedSANLongestPath(Problem):
    """
    Base class to implement simulation-optimization problems.

    Attributes
    ----------
    name : string
        name of problem
    dim : int
        number of decision variables
    n_objectives : int
        number of objectives
    n_stochastic_constraints : int
        number of stochastic constraints
    minmax : tuple of int (+/- 1)
        indicator of maximization (+1) or minimization (-1) for each objective
    constraint_type : string
        description of constraints types:
            "unconstrained", "box", "deterministic", "stochastic"
    variable_type : string
        description of variable types:
            "discrete", "continuous", "mixed"
    lower_bounds : tuple
        lower bound for each decision variable
    upper_bounds : tuple
        upper bound for each decision variable
    gradient_available : bool
        indicates if gradient of objective function is available
    optimal_value : tuple
        optimal objective function value
    optimal_solution : tuple
        optimal solution
    model : Model object
        associated simulation model that generates replications
    model_default_factors : dict
        default values for overriding model-level default factors
    model_fixed_factors : dict
        combination of overriden model-level factors and defaults
    model_decision_factors : set of str
        set of keys for factors that are decision variables
    rng_list : list of mrg32k3a.mrg32k3a.MRG32k3a objects
        list of RNGs used to generate a random initial solution
        or a random problem instance
    factors : dict
        changeable factors of the problem
            initial_solution : list
                default initial solution from which solvers start
            budget : int > 0
                max number of replications (fn evals) for a solver to take
    specifications : dict
        details of each factor (for GUI, data validation, and defaults)

    Arguments
    ---------
    name : str
        user-specified name for problem
    fixed_factors : dict
        dictionary of user-specified problem factors
    model_fixed factors : dict
        subset of user-specified non-decision factors to pass through to the model

    See also
    --------
    base.Problem
    """

    @classproperty
    def class_name_abbr(cls) -> str:
        return "FIXEDSAN-1"

    @classproperty
    def class_name(cls) -> str:
        return "Min Mean Longest Path for Fixed Stochastic Activity Network"

    @classproperty
    def n_objectives(cls) -> int:
        return 1

    @classproperty
    def n_stochastic_constraints(cls) -> int:
        return 0

    @classproperty
    def minmax(cls) -> tuple[int]:
        return (-1,)

    @classproperty
    def constraint_type(cls) -> ConstraintType:
        return ConstraintType.BOX

    @classproperty
    def variable_type(cls) -> VariableType:
        return VariableType.CONTINUOUS

    @classproperty
    def gradient_available(cls) -> bool:
        return True

    @classproperty
    def optimal_value(cls) -> float | None:
        return None

    @classproperty
    def optimal_solution(cls) -> tuple | None:
        return None

    @classproperty
    def model_default_factors(cls) -> dict:
        return {}

    @classproperty
    def model_decision_factors(cls) -> set[str]:
        return {"arc_means"}

    @classproperty
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
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget,
            "arc_costs": self.check_arc_costs,
        }

    @property
    def dim(self) -> int:
        return self.model.factors["num_arcs"]

    @property
    def lower_bounds(self) -> tuple:
        return (1e-2,) * self.dim

    @property
    def upper_bounds(self) -> tuple:
        return (np.inf,) * self.dim

    def __init__(
        self,
        name: str = "FIXEDSAN-1",
        fixed_factors: dict | None = None,
        model_fixed_factors: dict | None = None,
    ) -> None:
        # Let the base class handle default arguments.
        super().__init__(
            name=name,
            fixed_factors=fixed_factors,
            model_fixed_factors=model_fixed_factors,
            model=FixedSAN,
        )

    def check_arc_costs(self) -> bool:
        positive = True
        for x in list(self.factors["arc_costs"]):
            positive = positive and x > 0
        return (
            len(self.factors["arc_costs"]) != self.model.factors["num_arcs"]
        ) and positive

    def vector_to_factor_dict(self, vector: tuple) -> dict:
        """
        Convert a vector of variables to a dictionary with factor keys

        Arguments
        ---------
        vector : tuple
            vector of values associated with decision variables

        Returns
        -------
        factor_dict : dictionary
            dictionary with factor keys and associated values
        """
        factor_dict = {"arc_means": vector[:]}
        return factor_dict

    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:
        """
        Convert a dictionary with factor keys to a vector
        of variables.

        Arguments
        ---------
        factor_dict : dictionary
            dictionary with factor keys and associated values

        Returns
        -------
        vector : tuple
            vector of values associated with decision variables
        """
        vector = tuple(factor_dict["arc_means"])
        return vector

    def response_dict_to_objectives(self, response_dict: dict) -> tuple:
        """
        Convert a dictionary with response keys to a vector
        of objectives.

        Arguments
        ---------
        response_dict : dictionary
            dictionary with response keys and associated values

        Returns
        -------
        objectives : tuple
            vector of objectives
        """
        objectives = (response_dict["longest_path_length"],)
        return objectives

    def response_dict_to_stoch_constraints(self, response_dict: dict) -> tuple:
        """
        Convert a dictionary with response keys to a vector
        of left-hand sides of stochastic constraints: E[Y] <= 0

        Arguments
        ---------
        response_dict : dictionary
            dictionary with response keys and associated values

        Returns
        -------
        stoch_constraints : tuple
            vector of LHSs of stochastic constraint
        """
        stoch_constraints = ()
        return stoch_constraints

    def deterministic_stochastic_constraints_and_gradients(
        self, x: tuple
    ) -> tuple[tuple, tuple]:
        """
        Compute deterministic components of stochastic constraints for a solution `x`.

        Arguments
        ---------
        x : tuple
            vector of decision variables

        Returns
        -------
        det_stoch_constraints : tuple
            vector of deterministic components of stochastic constraints
        det_stoch_constraints_gradients : tuple
            vector of gradients of deterministic components of stochastic constraints
        """
        det_stoch_constraints = ()
        det_stoch_constraints_gradients = (
            (0,) * self.dim,
        )  # tuple of tuples - of sizes self.dim by self.dim, full of zeros
        return det_stoch_constraints, det_stoch_constraints_gradients

    def deterministic_objectives_and_gradients(
        self, x: tuple
    ) -> tuple[tuple, tuple]:
        """
        Compute deterministic components of objectives for a solution `x`.

        Arguments
        ---------
        x : tuple
            vector of decision variables

        Returns
        -------
        det_objectives : tuple
            vector of deterministic components of objectives
        det_objectives_gradients : tuple
            vector of gradients of deterministic components of objectives
        """
        det_objectives = (
            np.sum(np.array(self.factors["arc_costs"]) / np.array(x)),
        )
        det_objectives_gradients = (
            -np.array(self.factors["arc_costs"]) / (np.array(x) ** 2),
        )
        return det_objectives, det_objectives_gradients

    def check_deterministic_constraints(self, x: tuple) -> bool:
        """
        Check if a solution `x` satisfies the problem's deterministic constraints.

        Arguments
        ---------
        x : tuple
            vector of decision variables

        Returns
        -------
        satisfies : bool
            indicates if solution `x` satisfies the deterministic constraints.
        """
        is_positive: list[bool] = [x_i >= 0 for x_i in x]
        return all(is_positive)

    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
        """
        Generate a random solution for starting or restarting solvers.

        Arguments
        ---------
        rand_sol_rng : mrg32k3a.mrg32k3a.MRG32k3a object
            random-number generator used to sample a new random solution

        Returns
        -------
        x : tuple
            vector of decision variables
        """
        x = tuple(
            [
                rand_sol_rng.lognormalvariate(lq=0.1, uq=10)
                for _ in range(self.dim)
            ]
        )
        return x
