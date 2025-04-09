"""Contamination Modeling.

Simulate contamination rates.
A detailed description of the model/problem can be found
`here <https://simopt.readthedocs.io/en/latest/contam.html>`__.
"""

from __future__ import annotations

from typing import Callable, Final

import numpy as np

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.base import ConstraintType, Model, Problem, VariableType
from simopt.utils import classproperty, override

NUM_STAGES: Final[int] = 5


class Contamination(Model):
    """Contamination model with contamination and restoration rates.

    A model that simulates a contamination problem with a beta distribution.
    Returns the probability of violating contamination upper limit in each level of
    supply chain.

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
        return "CONTAM"

    @classproperty
    @override
    def n_rngs(cls) -> int:
        return 2

    @classproperty
    @override
    def n_responses(cls) -> int:
        return 1

    @classproperty
    @override
    def specifications(cls) -> dict[str, dict]:
        return {
            "contam_rate_alpha": {
                "description": (
                    "alpha parameter of beta distribution for growth rate of "
                    "contamination at each stage"
                ),
                "datatype": float,
                "default": 1.0,
            },
            "contam_rate_beta": {
                "description": (
                    "beta parameter of beta distribution for growth rate of "
                    "contamination at each stage"
                ),
                "datatype": float,
                "default": round(17 / 3, 2),
            },
            "restore_rate_alpha": {
                "description": (
                    "alpha parameter of beta distribution for rate that contamination "
                    "decreases by after prevention effort"
                ),
                "datatype": float,
                "default": 1.0,
            },
            "restore_rate_beta": {
                "description": (
                    "beta parameter of beta distribution for rate that contamination "
                    "decreases by after prevention effort"
                ),
                "datatype": float,
                "default": round(3 / 7, 3),
            },
            "initial_rate_alpha": {
                "description": (
                    "alpha parameter of beta distribution for initial contamination "
                    "fraction"
                ),
                "datatype": float,
                "default": 1.0,
            },
            "initial_rate_beta": {
                "description": (
                    "beta parameter of beta distribution for initial contamination "
                    "fraction"
                ),
                "datatype": float,
                "default": 30.0,
            },
            "stages": {
                "description": "stage of food supply chain",
                "datatype": int,
                "default": NUM_STAGES,
            },
            "prev_decision": {
                "description": "prevention decision",
                "datatype": tuple,
                "default": (0,) * NUM_STAGES,
            },
        }

    @property
    @override
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "contam_rate_alpha": self._check_contam_rate_alpha,
            "contam_rate_beta": self._check_contam_rate_beta,
            "restore_rate_alpha": self._check_restore_rate_alpha,
            "restore_rate_beta": self._check_restore_rate_beta,
            "initial_rate_alpha": self._check_initial_rate_alpha,
            "initial_rate_beta": self._check_initial_rate_beta,
            "stages": self._check_stages,
            "prev_decision": self._check_prev_decision,
        }

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize the Contamination model.

        Args:
            fixed_factors (dict, optional): Fixed factors for the model.
                Defaults to None.
        """
        # Let the base class handle default arguments.
        super().__init__(fixed_factors)

    def _check_contam_rate_alpha(self) -> None:
        if self.factors["contam_rate_alpha"] <= 0:
            raise ValueError("contam_rate_alpha must be greater than 0.")

    def _check_contam_rate_beta(self) -> None:
        if self.factors["contam_rate_beta"] <= 0:
            raise ValueError("contam_rate_beta must be greater than 0.")

    def _check_restore_rate_alpha(self) -> None:
        if self.factors["restore_rate_alpha"] <= 0:
            raise ValueError("restore_rate_alpha must be greater than 0.")

    def _check_restore_rate_beta(self) -> None:
        if self.factors["restore_rate_beta"] <= 0:
            raise ValueError("restore_rate_beta must be greater than 0.")

    def _check_initial_rate_alpha(self) -> None:
        if self.factors["initial_rate_alpha"] <= 0:
            raise ValueError("initial_rate_alpha must be greater than 0.")

    def _check_initial_rate_beta(self) -> None:
        if self.factors["initial_rate_beta"] <= 0:
            raise ValueError("initial_rate_beta must be greater than 0.")

    def _check_prev_cost(self) -> None:
        if any(cost <= 0 for cost in self.factors["prev_cost"]):
            raise ValueError("All costs in prev_cost must be greater than 0.")

    def _check_stages(self) -> None:
        if self.factors["stages"] <= 0:
            raise ValueError("Stages must be greater than 0.")

    def _check_prev_decision(self) -> None:
        if not all(0 <= u <= 1 for u in self.factors["prev_decision"]):
            raise ValueError(
                "All elements in prev_decision must be in the range [0, 1]."
            )

    @override
    def check_simulatable_factors(self) -> bool:
        # Check for matching number of stages.
        if len(self.factors["prev_decision"]) != self.factors["stages"]:
            raise ValueError(
                "The number of stages must be equal to the length of the previous "
                "decision tuple."
            )
        return True

    def replicate(self, rng_list: list[MRG32k3a]) -> tuple[dict, dict]:
        """Simulate a single replication for the current model factors.

        Arguments:
        ---------
        rng_list : list of mrg32k3a.mrg32k3a.MRG32k3a objects
            rngs for model to use when simulating a replication

        Returns:
        -------
        responses : dict
            performance measures of interest
            "level" = a list of contamination levels over time
        gradients : dict of dicts
            gradient estimates for each response
        """
        stages: int = self.factors["stages"]
        init_alpha: float = self.factors["initial_rate_alpha"]
        init_beta: float = self.factors["initial_rate_beta"]
        contam_alpha: float = self.factors["contam_rate_alpha"]
        contam_beta: float = self.factors["contam_rate_beta"]
        restore_alpha: float = self.factors["restore_rate_alpha"]
        restore_beta: float = self.factors["restore_rate_beta"]
        u: tuple = self.factors["prev_decision"]

        # Designate separate random number generators.
        # Outputs will be coupled when generating demand.
        contam_rng = rng_list[0]
        restore_rng = rng_list[1]

        # Initialize levels with beta distribution.
        levels = np.zeros(stages)
        levels[0] = restore_rng.betavariate(alpha=init_alpha, beta=init_beta)

        # Generate contamination and restoration values with beta distribution.
        rand_range = range(stages - 1)
        contamination_rates = [
            contam_rng.betavariate(contam_alpha, contam_beta) for _ in rand_range
        ]
        restoration_rates = [
            restore_rng.betavariate(restore_alpha, restore_beta) for _ in rand_range
        ]

        # Calculate contamination and restoration levels.
        # Start from stage 1; stage 0 was initialized separately.
        for i in range(1, stages):
            c = contamination_rates[i - 1]
            r = restoration_rates[i - 1]
            u_i = u[i]
            prev = levels[i - 1]

            contamination_change = c * (1 - u_i) * (1 - prev)
            restoration_change = (1 - r * u_i) * prev
            levels[i] = contamination_change + restoration_change
        # Compose responses and gradients.
        responses = {"level": levels}
        gradients = {
            response_key: dict.fromkeys(self.specifications, np.nan)
            for response_key in responses
        }
        return responses, gradients


"""
Summary
-------
Minimize the (deterministic) total cost of prevention efforts.
"""


class ContaminationTotalCostDisc(Problem):
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
    optimal_value : float
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
        return "CONTAM-1"

    @classproperty
    @override
    def class_name(cls) -> str:
        return "Min Total Cost for Discrete Contamination"

    @classproperty
    @override
    def n_objectives(cls) -> int:
        return 1

    @property
    @override
    def n_stochastic_constraints(self) -> int:
        return self.model.factors["stages"]

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
        return VariableType.DISCRETE

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
        return {}

    @classproperty
    @override
    def model_decision_factors(cls) -> set[str]:
        return {"prev_decision"}

    @classproperty
    @override
    def specifications(cls) -> dict[str, dict]:
        return {
            "initial_solution": {
                "description": "initial solution",
                "datatype": tuple,
                "default": (1,) * NUM_STAGES,
            },
            "budget": {
                "description": "max # of replications for a solver to take",
                "datatype": int,
                "default": 10000,
            },
            "prev_cost": {
                "description": "cost of prevention",
                "datatype": list,
                "default": [1] * NUM_STAGES,
            },
            "error_prob": {
                "description": "error probability",
                "datatype": list,
                "default": [0.2] * NUM_STAGES,
            },
            "upper_thres": {
                "description": "upper limit of amount of contamination",
                "datatype": list,
                "default": [0.1] * NUM_STAGES,
            },
        }

    @property
    @override
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget,
            "prev_cost": self._check_prev_cost,
            "error_prob": self._check_error_prob,
            "upper_thres": self._check_upper_thres,
        }

    @property
    @override
    def dim(self) -> int:
        return self.model.factors["stages"]

    @property
    @override
    def lower_bounds(self) -> tuple:
        return (0,) * self.model.factors["stages"]

    @property
    @override
    def upper_bounds(self) -> tuple:
        return (1,) * self.model.factors["stages"]

    def __init__(
        self,
        name: str = "CONTAM-1",
        fixed_factors: dict | None = None,
        model_fixed_factors: dict | None = None,
    ) -> None:
        """Initialize the ContaminationTotalCostDisc problem.

        Args:
            name (str, optional): Name of the problem. Defaults to "CONTAM-1".
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
            model=Contamination,
        )

    def _check_prev_cost(self) -> bool:
        return not (
            len(self.factors["prev_cost"]) != self.dim
            or any(elem < 0 for elem in self.factors["prev_cost"])
        )

    def _check_error_prob(self) -> bool:
        return not (
            len(self.factors["error_prob"]) != self.dim
            or all(error < 0 for error in self.factors["error_prob"])
        )

    def _check_upper_thres(self) -> bool:
        return len(self.factors["upper_thres"]) == self.dim

    def vector_to_factor_dict(self, vector: tuple) -> dict:
        """Convert a vector of variables to a dictionary with factor keys.

        Arguments:
        ---------
        vector : tuple
            vector of values associated with decision variables

        Returns:
        -------
        factor_dict : dictionary
            dictionary with factor keys and associated values
        """
        return {"prev_decision": vector[:]}

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
        return tuple(factor_dict["prev_decision"])

    def factor_dict_to_vector_gradients(self, factor_dict: dict) -> tuple:  # noqa: ARG002
        """Convert a dictionary with factor keys to a gradient vector.

        Notes:
        -----
        A subclass of ``base.Problem`` can have its own custom
        ``factor_dict_to_vector_gradients`` method if the
        objective is deterministic.

        Parameters
        ----------
        factor_dict : dict
            Dictionary with factor keys and associated values.

        Returns:
        -------
        vector : tuple
            Vector of partial derivatives associated with decision variables.
        """
        return (np.nan * len(self.model.factors["prev_decision"]),)

    def response_dict_to_objectives(self, response_dict: dict) -> tuple:  # noqa: ARG002
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
        return (0,)

    def response_dict_to_objectives_gradients(self, response_dict: dict) -> tuple:  # noqa: ARG002
        """Convert a dictionary with response keys to a vector of gradients.

        Notes:
        -----
        A subclass of ``base.Problem`` can have its own custom
        ``response_dict_to_objectives_gradients`` method if the
        objective is deterministic.

        Parameters
        ----------
        response_dict : dict
            Dictionary with response keys and associated values.

        Returns:
        -------
        tuple
            Vector of gradients.
        """
        return ((0,) * len(self.model.factors["prev_decision"]),)

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
        under_control = response_dict["level"] <= self.factors["upper_thres"]
        return tuple([-1 * z for z in under_control])

    def deterministic_stochastic_constraints_and_gradients(self) -> tuple[tuple, tuple]:
        """Compute deterministic components of stochastic constraints.

        Returns:
            tuple:
                - tuple: The deterministic components of the stochastic constraints.
                - tuple: The gradients of those deterministic components.
        """
        det_stoch_constraints = tuple(np.ones(self.dim) - self.factors["error_prob"])
        det_stoch_constraints_gradients = ((0,),)
        return det_stoch_constraints, det_stoch_constraints_gradients

    def deterministic_objectives_and_gradients(self, x: tuple) -> tuple[tuple, tuple]:
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
        det_objectives = (np.dot(self.factors["prev_cost"], x),)
        det_objectives_gradients = (tuple(self.factors["prev_cost"]),)
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
        between_0_and_1: list[bool] = [0 <= u <= 1 for u in x]
        return all(between_0_and_1)

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
        return tuple([rand_sol_rng.randint(0, 1) for _ in range(self.dim)])


class ContaminationTotalCostCont(Problem):
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
        return "CONTAM-2"

    @classproperty
    @override
    def class_name(cls) -> str:
        return "Min Total Cost for Continuous Contamination"

    @classproperty
    @override
    def n_objectives(cls) -> int:
        return 1

    @property
    @override
    def n_stochastic_constraints(self) -> int:
        return self.model.factors["stages"]

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
        return {}

    @classproperty
    @override
    def model_decision_factors(cls) -> set[str]:
        return {"prev_decision"}

    @classproperty
    @override
    def specifications(cls) -> dict[str, dict]:
        return {
            "initial_solution": {
                "description": "initial solution",
                "datatype": tuple,
                "default": (1,) * NUM_STAGES,
            },
            "budget": {
                "description": "max # of replications for a solver to take",
                "datatype": int,
                "default": 10000,
                "isDatafarmable": False,
            },
            "prev_cost": {
                "description": "cost of prevention",
                "datatype": list,
                "default": [1] * NUM_STAGES,
            },
            "error_prob": {
                "description": "error probability",
                "datatype": list,
                "default": [0.2] * NUM_STAGES,
            },
            "upper_thres": {
                "description": "upper limit of amount of contamination",
                "datatype": list,
                "default": [0.1] * NUM_STAGES,
            },
        }

    @property
    @override
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget,
            "prev_cost": self._check_prev_cost,
            "error_prob": self._check_error_prob,
            "upper_thres": self._check_upper_thres,
        }

    @property
    @override
    def dim(self) -> int:
        return self.model.factors["stages"]

    @property
    @override
    def lower_bounds(self) -> tuple:
        return (0,) * self.model.factors["stages"]

    @property
    @override
    def upper_bounds(self) -> tuple:
        return (1,) * self.model.factors["stages"]

    def __init__(
        self,
        name: str = "CONTAM-2",
        fixed_factors: dict | None = None,
        model_fixed_factors: dict | None = None,
    ) -> None:
        """Initialize the ContaminationTotalCostCont problem.

        Args:
            name (str, optional): Name of the problem. Defaults to "CONTAM-2".
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
            model=Contamination,
        )

    @override
    def check_initial_solution(self) -> bool:
        return not (
            len(self.factors["initial_solution"]) != self.dim
            or all(u < 0 or u > 1 for u in self.factors["initial_solution"])
        )

    def _check_prev_cost(self) -> bool:
        return not (
            len(self.factors["prev_cost"]) != self.dim
            or any(elem < 0 for elem in self.factors["prev_cost"])
        )

    def _check_error_prob(self) -> bool:
        return not (
            len(self.factors["error_prob"]) != self.dim
            or all(error < 0 for error in self.factors["error_prob"])
        )

    def _check_upper_thres(self) -> bool:
        return len(self.factors["upper_thres"]) == self.dim

    # TODO: figure out how Problem.check_simulatable_factors() works
    def check_simulatable_factors(self) -> bool:
        lower_len = len(self.lower_bounds)
        upper_len = len(self.upper_bounds)
        if lower_len != upper_len or lower_len != self.dim:
            error_msg = (
                f"Lower bounds: {lower_len}, Upper bounds: {upper_len}, Dim: {self.dim}"
            )
            raise ValueError(error_msg)
        return True

    def vector_to_factor_dict(self, vector: tuple) -> dict:
        """Convert a vector of variables to a dictionary with factor keys.

        Arguments:
        ---------
        vector : tuple
            vector of values associated with decision variables

        Returns:
        -------
        factor_dict : dictionary
            dictionary with factor keys and associated values
        """
        return {"prev_decision": vector[:]}

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
        return tuple(factor_dict["prev_decision"])

    def factor_dict_to_vector_gradients(self, factor_dict: dict) -> tuple:  # noqa: ARG002
        """Convert a dictionary with factor keys to a gradient vector.

        Notes:
        -----
        A subclass of ``base.Problem`` can have its own custom
        ``factor_dict_to_vector_gradients`` method if the
        objective is deterministic.

        Parameters
        ----------
        factor_dict : dict
            Dictionary with factor keys and associated values.

        Returns:
        -------
        vector : tuple
            Vector of partial derivatives associated with decision variables.
        """
        return (np.nan * len(self.model.factors["prev_decision"]),)

    def response_dict_to_objectives(self, response_dict: dict) -> tuple:  # noqa: ARG002
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
        return (0,)

    def response_dict_to_objectives_gradients(self, response_dict: dict) -> tuple:  # noqa: ARG002
        """Convert a dictionary with response keys to a vector of gradients.

        Notes:
        -----
        A subclass of ``base.Problem`` can have its own custom
        ``response_dict_to_objectives_gradients`` method if the
        objective is deterministic.

        Parameters
        ----------
        response_dict : dict
            Dictionary with response keys and associated values.

        Returns:
        -------
        tuple
            Vector of gradients.
        """
        return ((0,) * len(self.model.factors["prev_decision"]),)

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
        under_control = response_dict["level"] <= self.factors["upper_thres"]
        return tuple([-1 * z for z in under_control])

    def deterministic_stochastic_constraints_and_gradients(self) -> tuple[tuple, tuple]:
        """Compute deterministic components of stochastic constraints.

        Returns:
            tuple:
                - tuple: The deterministic components of the stochastic constraints.
                - tuple: The gradients of those deterministic components.
        """
        det_stoch_constraints = tuple(np.ones(self.dim) - self.factors["error_prob"])
        det_stoch_constraints_gradients = (
            (0,),
        )  # tuple of tuples - of sizes self.dim by self.dim, full of zeros
        return det_stoch_constraints, det_stoch_constraints_gradients

    def deterministic_objectives_and_gradients(self, x: tuple) -> tuple[tuple, tuple]:
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
        det_objectives = (np.dot(self.factors["prev_cost"], x),)
        det_objectives_gradients = (tuple(self.factors["prev_cost"]),)
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
        between_0_and_1: list[bool] = [0 <= u <= 1 for u in x]
        return all(between_0_and_1)

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
        return tuple([rand_sol_rng.random() for _ in range(self.dim)])
