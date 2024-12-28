"""
Summary
-------
Simulate duration of a stochastic activity network (SAN).
A detailed description of the model/problem can be found
`here <https://simopt.readthedocs.io/en/latest/san.html>`__.
"""

from __future__ import annotations

from typing import Callable, Final

import numpy as np
from mrg32k3a.mrg32k3a import MRG32k3a

from simopt.base import ConstraintType, Model, Problem, VariableType

NUM_ARCS: Final[int] = 13


class SAN(Model):
    """
    A model that simulates a stochastic activity network problem with
    tasks that have exponentially distributed durations, and the selected
    means come with a cost.

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

    @property
    def name(self) -> str:
        return "SAN"

    @property
    def n_rngs(self) -> int:
        return 1

    @property
    def n_responses(self) -> int:
        return 1

    @property
    def specifications(self) -> dict[str, dict]:
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
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "num_nodes": self.check_num_nodes,
            "arcs": self.check_arcs,
            "arc_means": self.check_arc_means,
        }

    def __init__(self, fixed_factors: dict | None = None) -> None:
        # Let the base class handle default arguments.
        super().__init__(fixed_factors)

    def check_num_nodes(self) -> None:
        if self.factors["num_nodes"] <= 0:
            raise ValueError("num_nodes must be greater than 0.")

    def dfs(
        self, graph: dict[int, set], start: int, visited: set | None = None
    ) -> set:
        if visited is None:
            visited = set()
        visited.add(start)

        for next_point in graph[start] - visited:
            self.dfs(graph, next_point, visited)
        return visited

    def check_arcs(self) -> bool:
        if len(self.factors["arcs"]) <= 0:
            raise ValueError("The length of arcs must be greater than 0.")
        # Check graph is connected.
        graph = {
            node: set() for node in range(1, self.factors["num_nodes"] + 1)
        }
        for a in self.factors["arcs"]:
            graph[a[0]].add(a[1])
        visited = self.dfs(graph, 1)
        if self.factors["num_nodes"] in visited:
            return True
        return False

    def check_arc_means(self) -> bool:
        positive = True
        for x in list(self.factors["arc_means"]):
            positive = positive and (x > 0)
        return positive

    def check_simulatable_factors(self) -> bool:
        if len(self.factors["arc_means"]) != len(self.factors["arcs"]):
            raise ValueError(
                "The length of arc_means must be equal to the length of arcs."
            )
        return True

    def replicate(self, rng_list: list[MRG32k3a]) -> tuple[dict, dict]:
        """
        Simulate a single replication for the current model factors.

        Arguments
        ---------
        rng_list : list of mrg32k3a.mrg32k3a.MRG32k3a
            rngs for model to use when simulating a replication

        Returns
        -------
        responses : dict
            performance measures of interest
            "longest_path_length" = length/duration of longest path
        gradients : dict of dicts
            gradient estimates for each response
        """
        # Designate separate random number generators.
        exp_rng = rng_list[0]

        # Topological sort.
        graph_in = {
            node: set() for node in range(1, self.factors["num_nodes"] + 1)
        }
        graph_out = {
            node: set() for node in range(1, self.factors["num_nodes"] + 1)
        }
        for a in self.factors["arcs"]:
            graph_in[a[1]].add(a[0])
            graph_out[a[0]].add(a[1])
        indegrees = [
            len(graph_in[n]) for n in range(1, self.factors["num_nodes"] + 1)
        ]
        # outdegrees = [len(graph_out[n]) for n in range(1, self.factors["num_nodes"]+1)]
        queue = []
        topo_order = []
        for n in range(self.factors["num_nodes"]):
            if indegrees[n] == 0:
                queue.append(n + 1)
        while len(queue) != 0:
            u = queue.pop(0)
            topo_order.append(u)
            for n in graph_out[u]:
                indegrees[n - 1] -= 1
                if indegrees[n - 1] == 0:
                    queue.append(n)

        # Generate arc lengths.
        arc_length = {}
        for i in range(len(self.factors["arcs"])):
            arc_length[str(self.factors["arcs"][i])] = exp_rng.expovariate(
                1 / self.factors["arc_means"][i]
            )

        # Calculate the length of the longest path.
        path_length = np.zeros(self.factors["num_nodes"])
        prev = np.zeros(self.factors["num_nodes"])
        for i in range(1, self.factors["num_nodes"]):
            vi = topo_order[i - 1]
            for j in graph_out[vi]:
                if (
                    path_length[j - 1]
                    < path_length[vi - 1] + arc_length[str((vi, j))]
                ):
                    path_length[j - 1] = (
                        path_length[vi - 1] + arc_length[str((vi, j))]
                    )
                    prev[j - 1] = vi
        longest_path = path_length[self.factors["num_nodes"] - 1]

        # Calculate the IPA gradient w.r.t. arc means.
        # If an arc is on the longest path, the component of the gradient
        # is the length of the length of that arc divided by its mean.
        # If an arc is not on the longest path, the component of the gradient is zero.
        gradient = np.zeros(len(self.factors["arcs"]))
        current = topo_order[-1]
        backtrack = int(prev[self.factors["num_nodes"] - 1])
        while current != topo_order[0]:
            idx = self.factors["arcs"].index((backtrack, current))
            gradient[idx] = (
                arc_length[str((backtrack, current))]
                / (self.factors["arc_means"][idx])
            )
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


"""
Summary
-------
Minimize the duration of the longest path from a to i plus cost.
"""


class SANLongestPath(Problem):
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

    @property
    def n_objectives(self) -> int:
        return 1

    @property
    def n_stochastic_constraints(self) -> int:
        return 0

    @property
    def minmax(self) -> tuple[int]:
        return (-1,)

    @property
    def constraint_type(self) -> ConstraintType:
        return ConstraintType.BOX

    @property
    def variable_type(self) -> VariableType:
        return VariableType.CONTINUOUS

    @property
    def gradient_available(self) -> bool:
        return True

    @property
    def optimal_value(self) -> float | None:
        return None

    @property
    def optimal_solution(self) -> tuple | None:
        return None

    @property
    def model_default_factors(self) -> dict:
        return {}

    @property
    def model_decision_factors(self) -> set[str]:
        return {"arc_means"}

    @property
    def specifications(self) -> dict[str, dict]:
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
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget,
            "arc_costs": self.check_arc_costs,
        }

    @property
    def dim(self) -> int:
        return len(self.model.factors["arcs"])

    @property
    def lower_bounds(self) -> tuple:
        return (1e-2,) * self.dim

    @property
    def upper_bounds(self) -> tuple:
        return (np.inf,) * self.dim

    def __init__(
        self,
        name: str = "SAN-1",
        fixed_factors: dict | None = None,
        model_fixed_factors: dict | None = None,
    ) -> None:
        # Let the base class handle default arguments.
        super().__init__(
            name=name,
            fixed_factors=fixed_factors,
            model_fixed_factors=model_fixed_factors,
            model=SAN,
        )

    def check_arc_costs(self) -> bool:
        positive = True
        for x in list(self.factors["arc_costs"]):
            positive = positive and x > 0
        matching_len = len(self.factors["arc_costs"]) == len(
            self.model.factors["arcs"]
        )
        return positive and matching_len

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
        vector = factor_dict["arc_means"]
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
