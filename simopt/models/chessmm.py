"""Chess Matchmaking Model.

Simulate matching of chess players on an online platform.
A detailed description of the model/problem can be found
`here <https://simopt.readthedocs.io/en/latest/chessmm.html>`__.
"""

from __future__ import annotations

from typing import Callable, Final

import numpy as np
from scipy import special

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.base import ConstraintType, Model, Problem, VariableType
from simopt.utils import classproperty, override

MEAN_ELO: Final[int] = 1200
MAX_ALLOWABLE_DIFF: Final[int] = 150


class ChessMatchmaking(Model):
    """Matchmaking model following an Elo distribution.

    A model that simulates a matchmaking problem with a Elo (truncated normal)
    distribution of players and Poisson arrivals and returns the average difference
    between matched players.

    Attributes:
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

    Arguments:
    ---------
    fixed_factors : nested dict
        fixed factors of the simulation model

    See Also:
    --------
    base.Model
    """

    @classproperty
    @override
    def class_name_abbr(cls) -> str:
        return "CHESS"

    @classproperty
    @override
    def class_name(cls) -> str:
        return "Chess Matchmaking"

    @classproperty
    @override
    def n_rngs(cls) -> int:
        return 2

    @classproperty
    @override
    def n_responses(cls) -> int:
        return 2

    @classproperty
    @override
    def specifications(cls) -> dict[str, dict]:
        return {
            "elo_mean": {
                "description": "mean of normal distribution for Elo rating",
                "datatype": float,
                "default": MEAN_ELO,
            },
            "elo_sd": {
                "description": (
                    "standard deviation of normal distribution for Elo rating"
                ),
                "datatype": float,
                "default": round(MEAN_ELO / (np.sqrt(2) * special.erfcinv(1 / 50)), 1),
            },
            "poisson_rate": {
                "description": "rate of Poisson process for player arrivals",
                "datatype": float,
                "default": 1.0,
            },
            "num_players": {
                "description": "number of players",
                "datatype": int,
                "default": 1000,
            },
            "allowable_diff": {
                "description": "maximum allowable difference between Elo ratings",
                "datatype": float,
                "default": MAX_ALLOWABLE_DIFF,
            },
        }

    @property
    @override
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "elo_mean": self._check_elo_mean,
            "elo_sd": self._check_elo_sd,
            "poisson_rate": self._check_poisson_rate,
            "num_players": self._check_num_players,
            "allowable_diff": self._check_allowable_diff,
        }

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize the ChessMatchmaking model.

        Args:
            fixed_factors (dict, optional): Fixed factors for the model.
                Defaults to None.
        """
        # Let the base class handle default arguments.
        super().__init__(fixed_factors)

    def _check_elo_mean(self) -> None:
        if self.factors["elo_mean"] <= 0:
            raise ValueError(
                "Mean of normal distribution for Elo rating must be greater than 0."
            )

    def _check_elo_sd(self) -> None:
        if self.factors["elo_sd"] <= 0:
            raise ValueError(
                "Standard deviation of normal distribution for Elo rating must be "
                "greater than 0."
            )

    def _check_poisson_rate(self) -> None:
        if self.factors["poisson_rate"] <= 0:
            raise ValueError(
                "Rate of Poisson process for player arrivals must be greater than 0."
            )

    def _check_num_players(self) -> None:
        if self.factors["num_players"] <= 0:
            raise ValueError("Number of players must be greater than 0.")

    def _check_allowable_diff(self) -> None:
        if self.factors["allowable_diff"] <= 0:
            raise ValueError(
                "The maximum mallowable different between Elo ratings must be greater "
                "than 0."
            )

    @override
    def check_simulatable_factors(self) -> bool:
        # No factors need cross-checked
        return True

    def replicate(self, rng_list: list[MRG32k3a]) -> tuple[dict, dict[str, dict]]:
        """Simulate a single replication for the current model factors.

        Arguments:
        ---------
        rng_list : list of mrg32k3a.mrg32k3a.MRG32k3a objects
            rngs for model to use when simulating a replication

        Returns:
        -------
        dict
            performance measures of interest
            "avg_diff" = the average Elo difference between all pairs
            "avg_wait_time" = the average waiting time
        dict[str, dict]
            gradient estimates for each response
        """
        # Constants
        num_players = self.factors["num_players"]
        num_players_range = range(num_players)
        elo_mean = self.factors["elo_mean"]
        elo_sd = self.factors["elo_sd"]
        elo_min, elo_max = 0, 2400
        allowable_diff = self.factors["allowable_diff"]
        poisson_rate = self.factors["poisson_rate"]

        # Designate separate RNGs for Elo and arrival times.
        elo_rng = rng_list[0]
        arrival_rng = rng_list[1]

        def generate_elo() -> float:
            while True:
                rating = elo_rng.normalvariate(elo_mean, elo_sd)
                if elo_min <= rating <= elo_max:
                    return rating

        # Generate Elo ratings (normal distribution).
        player_ratings = [generate_elo() for _ in num_players_range]

        # Generate interarrival times (Poisson distribution).
        interarrival_times = [
            arrival_rng.poissonvariate(poisson_rate) for _ in num_players_range
        ]

        # Initialize statistics.
        # Incoming players are initialized with a wait time of 0.
        wait_times = np.zeros(num_players)
        waiting_players = []
        total_diff = 0  # TODO: make this do something
        elo_diffs = []

        # Simulate arrival and matching and players.
        for interarrival_time, player_rating in zip(interarrival_times, player_ratings):
            # Try to match the player
            for i, waiting_rating in enumerate(waiting_players):
                diff = abs(player_rating - waiting_rating)
                if diff <= allowable_diff:
                    total_diff += diff
                    elo_diffs.append(diff)
                    waiting_players.pop(i)
                    break
                wait_times[i] += interarrival_time
            # If break did not execute, then the player was not matched.
            else:
                waiting_players.append(player_rating)

        # If there weren't any matches, the elo_diffs list will be empty.
        avg_diff = np.mean(elo_diffs) if elo_diffs else np.nan
        # Compose responses and gradients.
        responses = {
            "avg_diff": avg_diff,
            "avg_wait_time": np.mean(wait_times),
        }
        gradients = {
            response_key: dict.fromkeys(self.specifications, np.nan)
            for response_key in responses
        }
        return responses, gradients


"""
Summary
-------
Minimize the expected Elo difference between all pairs of matched
players subject to the expected waiting time being sufficiently small.
"""


class ChessAvgDifference(Problem):
    """Base class to implement simulation-optimization problems.

    Attributes:
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
    rng_list : list of mrg32k3a.mrg32k3a.MRG32k3a objects
        list of RNGs used to generate a random initial solution
        or a random problem instance
    factors : dict
        changeable factors of the problem
            initial_solution : list
                default initial solution from which solvers start
            budget : int > 0
                max number of replications (fn evals) for a solver to take
            prev_cost : list
                cost of prevention
            upper_thres : float > 0
                upper limit of amount of contamination
    specifications : dict
        details of each factor (for GUI, data validation, and defaults)

    Arguments:
    ---------
    name : str
        user-specified name for problem
    fixed_factors : dict
        dictionary of user-specified problem factors
    model_fixed factors : dict
        subset of user-specified non-decision factors to pass through to the model

    See Also:
    --------
    base.Problem
    """

    @classproperty
    @override
    def class_name_abbr(cls) -> str:
        return "CHESS-1"

    @classproperty
    @override
    def class_name(cls) -> str:
        return "Min Avg Difference for Chess Matchmaking"

    @classproperty
    @override
    def n_objectives(cls) -> int:
        return 1

    @classproperty
    @override
    def n_stochastic_constraints(cls) -> int:
        return 1

    @classproperty
    @override
    def minmax(cls) -> tuple[int]:
        return (-1,)

    @classproperty
    @override
    def constraint_type(cls) -> ConstraintType:
        return ConstraintType.STOCHASTIC

    @classproperty
    @override
    def variable_type(cls) -> VariableType:
        return VariableType.CONTINUOUS

    @classproperty
    @override
    def gradient_available(cls) -> bool:
        return False

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
        return {}

    @classproperty
    @override
    def model_decision_factors(cls) -> set:
        return {"allowable_diff"}

    @classproperty
    @override
    def specifications(cls) -> dict[str, dict]:
        return {
            "initial_solution": {
                "description": "initial solution",
                "datatype": tuple,
                "default": (MAX_ALLOWABLE_DIFF,),
            },
            "budget": {
                "description": "max # of replications for a solver to take",
                "datatype": int,
                "default": 1000,
                "isDatafarmable": False,
            },
            "upper_time": {
                "description": "upper bound on wait time",
                "datatype": float,
                "default": 5.0,
            },
        }

    @property
    @override
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget,
            "upper_time": self._check_upper_time,
        }

    @classproperty
    @override
    def dim(cls) -> int:
        return 1

    @classproperty
    @override
    def lower_bounds(cls) -> tuple:
        return (0,)

    @classproperty
    @override
    def upper_bounds(cls) -> tuple:
        return (2400,)

    def __init__(
        self,
        name: str = "CHESS-1",
        fixed_factors: dict | None = None,
        model_fixed_factors: dict | None = None,
    ) -> None:
        """Initialize the ChessAvgDifference problem.

        Args:
            name (str, optional): User-specified name for the problem.
                Defaults to "CHESS-1".
            fixed_factors (dict, optional): Fixed factors for the problem.
                Defaults to None.
            model_fixed_factors (dict, optional): Fixed factors for the model.
                Defaults to None.
        """
        # Let the base class handle default arguments.
        super().__init__(name, fixed_factors, model_fixed_factors, ChessMatchmaking)

    def _check_upper_time(self) -> None:
        if self.factors["upper_time"] <= 0:
            raise ValueError("The upper bound on wait time must be greater than 0.")

    @override
    def vector_to_factor_dict(self, vector: tuple) -> dict:
        return {"allowable_diff": vector[0]}

    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:
        """Convert a dictionary with factor keys to a vector of variables.

        Arguments:
        ---------
        factor_dict : dictionary
            dictionary with factor keys and associated values

        Returns:
        -------
        vector : tuple
            vector of values associated with decision variables
        """
        return (factor_dict["allowable_diff"],)

    def response_dict_to_objectives(self, response_dict: dict) -> tuple:
        """Convert a dictionary with response keys to a vector of objectives.

        Arguments:
        ---------
        response_dict : dictionary
            dictionary with response keys and associated values

        Returns:
        -------
        objectives : tuple
            vector of objectives
        """
        return (response_dict["avg_diff"],)

    def response_dict_to_stoch_constraints(self, response_dict: dict) -> tuple:
        """Convert a response dictionary to a vector of stochastic constraint values.

        Each returned value represents the left-hand side of a constraint of the form
        E[Y] ≤ 0.

        Args:
            response_dict (dict): A dictionary containing response keys and their
                associated values.

        Returns:
            tuple: A tuple representing the left-hand sides of the stochastic
                constraints.
        """
        return (response_dict["avg_wait_time"],)

    def deterministic_stochastic_constraints_and_gradients(self) -> tuple[tuple, tuple]:
        """Compute deterministic components of stochastic constraints.

        Returns:
            tuple:
                - tuple: The deterministic components of the stochastic constraints.
                - tuple: The gradients of those deterministic components.
        """
        det_stoch_constraints = (-1 * self.factors["upper_time"],)
        det_stoch_constraints_gradients = ((0,),)
        return det_stoch_constraints, det_stoch_constraints_gradients

    def deterministic_objectives_and_gradients(self, _x: tuple) -> tuple[tuple, tuple]:
        """Compute deterministic components of objectives for a solution `x`.

        Arguments:
        ---------
        x : tuple
            vector of decision variables

        Returns:
        -------
        det_objectives : tuple
            vector of deterministic components of objectives
        det_objectives_gradients : tuple
            vector of gradients of deterministic components of objectives
        """
        det_objectives = (0,)
        det_objectives_gradients = ()
        return det_objectives, det_objectives_gradients

    def check_deterministic_constraints(self, x: tuple) -> bool:
        """Check if a solution `x` satisfies the problem's deterministic constraints.

        Arguments:
        ---------
        x : tuple
            vector of decision variables

        Returns:
        -------
        satisfies : bool
            indicates if solution `x` satisfies the deterministic constraints.
        """
        is_greater_than_zero: list[bool] = [x_val > 0 for x_val in x]
        return all(is_greater_than_zero)

    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
        """Generate a random solution for starting or restarting solvers.

        Arguments:
        ---------
        rand_sol_rng : mrg32k3a.mrg32k3a.MRG32k3a object
            random-number generator used to sample a new random solution

        Returns:
        -------
        x : tuple
            vector of decision variables
        """
        return (min(max(0, rand_sol_rng.normalvariate(150, 50)), 2400),)
