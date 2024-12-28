"""
Summary
-------
Simulate matching of chess players on an online platform.
A detailed description of the model/problem can be found
`here <https://simopt.readthedocs.io/en/latest/chessmm.html>`__.
"""

from __future__ import annotations

from typing import Callable, Final

import numpy as np
from mrg32k3a.mrg32k3a import MRG32k3a
from scipy import special

from simopt.base import ConstraintType, Model, Problem, VariableType

MEAN_ELO: Final[int] = 1200
MAX_ALLOWABLE_DIFF: Final[int] = 150


class ChessMatchmaking(Model):
    """
    A model that simulates a matchmaking problem with a
    Elo (truncated normal) distribution of players and Poisson arrivals.
    Returns the average difference between matched players.

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
        return "CHESS"

    @property
    def n_rngs(self) -> int:
        return 2

    @property
    def n_responses(self) -> int:
        return 2

    @property
    def specifications(self) -> dict[str, dict]:
        return {
            "elo_mean": {
                "description": "mean of normal distribution for Elo rating",
                "datatype": float,
                "default": MEAN_ELO,
            },
            "elo_sd": {
                "description": "standard deviation of normal distribution for Elo rating",
                "datatype": float,
                "default": round(
                    MEAN_ELO / (np.sqrt(2) * special.erfcinv(1 / 50)), 1
                ),
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
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "elo_mean": self.check_elo_mean,
            "elo_sd": self.check_elo_sd,
            "poisson_rate": self.check_poisson_rate,
            "num_players": self.check_num_players,
            "allowable_diff": self.check_allowable_diff,
        }

    def __init__(self, fixed_factors: dict | None = None) -> None:
        # Let the base class handle default arguments.
        super().__init__(fixed_factors)

    def check_elo_mean(self) -> None:
        if self.factors["elo_mean"] <= 0:
            raise ValueError(
                "Mean of normal distribution for Elo rating must be greater than 0."
            )

    def check_elo_sd(self) -> None:
        if self.factors["elo_sd"] <= 0:
            raise ValueError(
                "Standard deviation of normal distribution for Elo rating must be greater than 0."
            )

    def check_poisson_rate(self) -> None:
        if self.factors["poisson_rate"] <= 0:
            raise ValueError(
                "Rate of Poisson process for player arrivals must be greater than 0."
            )

    def check_num_players(self) -> None:
        if self.factors["num_players"] <= 0:
            raise ValueError("Number of players must be greater than 0.")

    def check_allowable_diff(self) -> None:
        if self.factors["allowable_diff"] <= 0:
            raise ValueError(
                "The maximum mallowable different between Elo ratings must be greater than 0."
            )

    def check_simulatable_factors(self) -> bool:
        # No factors need cross-checked
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
            "avg_diff" = the average Elo difference between all pairs
            "avg_wait_time" = the average waiting time
        gradients : dict of dicts
            gradient estimates for each response
        """
        # Designate separate random number generators.
        elo_rng = rng_list[0]
        arrival_rng = rng_list[1]
        # Initialize statistics.
        # Incoming players are initialized with a wait time of 0.
        wait_times = np.zeros(self.factors["num_players"])
        waiting_players = []
        total_diff = 0
        elo_diffs = []
        # Simulate arrival and matching and players.
        for _ in range(self.factors["num_players"]):
            # Generate interarrival time of the player.
            time = arrival_rng.poissonvariate(self.factors["poisson_rate"])
            # Generate rating of the player via acceptance/rejection (not truncation).
            player_rating = elo_rng.normalvariate(
                self.factors["elo_mean"], self.factors["elo_sd"]
            )
            while player_rating < 0 or player_rating > 2400:
                player_rating = elo_rng.normalvariate(
                    self.factors["elo_mean"], self.factors["elo_sd"]
                )
            # Attempt to match the incoming player with waiting players in FIFO manner.
            old_total = total_diff
            for p in range(len(waiting_players)):
                if (
                    abs(player_rating - waiting_players[p])
                    <= self.factors["allowable_diff"]
                ):
                    total_diff += abs(player_rating - waiting_players[p])
                    elo_diffs.append(abs(player_rating - waiting_players[p]))
                    del waiting_players[p]
                    break
                else:
                    wait_times[p] += time
            # If incoming player is not matched, add them to the waiting pool.
            if old_total == total_diff:
                waiting_players.append(player_rating)
        # If there weren't any matches, the elo_diffs list will be empty.
        # This throws some warnings, so we'll add a 0 to the list.
        # TODO: Check to see if there is a better way to handle this.
        if not elo_diffs:
            elo_diffs.append(0)
        # Compose responses and gradients.
        responses = {
            "avg_diff": np.mean(elo_diffs),
            "avg_wait_time": np.mean(wait_times),
        }
        gradients = {
            response_key: {
                factor_key: np.nan for factor_key in self.specifications
            }
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
        return 1

    @property
    def minmax(self) -> tuple[int]:
        return (-1,)

    @property
    def constraint_type(self) -> ConstraintType:
        return ConstraintType.STOCHASTIC

    @property
    def variable_type(self) -> VariableType:
        return VariableType.CONTINUOUS

    @property
    def gradient_available(self) -> bool:
        return False

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
    def model_decision_factors(self) -> set:
        return {"allowable_diff"}

    @property
    def specifications(self) -> dict[str, dict]:
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
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget,
            "upper_time": self.check_upper_time,
        }

    @property
    def dim(self) -> int:
        return 1

    @property
    def lower_bounds(self) -> tuple:
        return (0,)

    @property
    def upper_bounds(self) -> tuple:
        return (2400,)

    def __init__(
        self,
        name: str = "CHESS-1",
        fixed_factors: dict | None = None,
        model_fixed_factors: dict | None = None,
    ) -> None:
        # Let the base class handle default arguments.
        super().__init__(
            name, fixed_factors, model_fixed_factors, ChessMatchmaking
        )

    def check_upper_time(self) -> None:
        if self.factors["upper_time"] <= 0:
            raise ValueError(
                "The upper bound on wait time must be greater than 0."
            )

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
        factor_dict = {"allowable_diff": vector[0]}
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
        vector = (factor_dict["allowable_diff"],)
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
        objectives = (response_dict["avg_diff"],)
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
        stoch_constraints = (response_dict["avg_wait_time"],)
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
        det_stoch_constraints = (-1 * self.factors["upper_time"],)
        det_stoch_constraints_gradients = ((0,),)
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
        det_objectives = (0,)
        det_objectives_gradients = ()
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
        is_greater_than_zero: list[bool] = [x_val > 0 for x_val in x]
        return all(is_greater_than_zero)

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
        x = (min(max(0, rand_sol_rng.normalvariate(150, 50)), 2400),)
        return x
