"""Example problem of deterministic function with noise.

Simulate a synthetic problem with a deterministic objective function
evaluated with noise.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.base import ConstraintType, Model, Problem, VariableType
from simopt.utils import classproperty, override


class ExampleModel(Model):
    """A model that is a deterministic function evaluated with noise.

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
        details of each factor (for GUI, data validation, and defaults)
    check_factor_list : dict
        switch case for checking factor simulatability

    Arguments:
    ---------
    fixed_factors : dict
        fixed_factors of the simulation model

    See Also:
    --------
    base.Model
    """

    @classproperty
    @override
    def class_name_abbr(cls) -> str:
        return "EXAMPLE"

    @classproperty
    @override
    def class_name(cls) -> str:
        return "Deterministic Function + Noise"

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
            "x": {
                "description": "point to evaluate",
                "datatype": tuple,
                "default": (2.0, 2.0),
            }
        }

    @property
    @override
    def check_factor_list(self) -> dict[str, Callable]:
        return {"x": lambda: True}

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize the model.

        Args:
            fixed_factors (dict | None): fixed factors of the model.
                If None, use default values.
        """
        # Let the base class handle default arguments.
        super().__init__(fixed_factors)

    def replicate(self, rng_list: list[MRG32k3a]) -> tuple[dict, dict]:
        """Evaluate a deterministic function f(x) with stochastic noise.

        Arguments:
        ---------
        rng_list : list of mrg32k3a.mrg32k3a.MRG32k3a objects
            rngs for model to use when simulating a replication

        Returns:
        -------
        responses : dict
            performance measures of interest
            "est_f(x)" = f(x) evaluated with stochastic noise
        """
        # Designate random number generator for stochastic noise.
        noise_rng = rng_list[0]
        x = np.array(self.factors["x"])
        fn_eval_at_x = np.linalg.norm(x) ** 2 + noise_rng.normalvariate()

        # Compose responses and gradients.
        responses = {"est_f(x)": fn_eval_at_x}
        gradients = {"est_f(x)": {"x": tuple(2 * x)}}
        return responses, gradients


"""
Summary
-------
Minimize f(x).
"""


class ExampleProblem(Problem):
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
    model_decision_factors : set of str
        set of keys for factors that are decision variables
    rng_list : list of mrg32k3a.mrg32k3a.MRG32k3a objects
        list of RNGs used to generate a random initial solution
        or a random problem instance
    factors : dict
        changeable factors of the problem
            initial_solution : tuple
                default initial solution from which solvers start
            budget : int > 0
                max number of replications (fn evals) for a solver to take
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
        return "EXAMPLE-1"

    @classproperty
    @override
    def class_name(cls) -> str:
        return "Min Deterministic Function + Noise"

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
        return ConstraintType.UNCONSTRAINED

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
        # Change if f is changed
        # TODO: figure out what f is
        return 0.0

    @property
    @override
    def optimal_solution(self) -> tuple:
        # Change if f is changed
        # TODO: figure out what f is
        return (0,) * self.dim

    @classproperty
    @override
    def model_default_factors(cls) -> dict:
        return {}

    @property
    @override
    def model_fixed_factors(self) -> dict:
        return {}

    @model_fixed_factors.setter
    def model_fixed_factors(self, value: dict | None) -> None:
        # TODO: figure out if fixed factors should change
        pass

    @classproperty
    @override
    def model_decision_factors(cls) -> set[str]:
        return {"x"}

    @classproperty
    @override
    def specifications(cls) -> dict[str, dict]:
        return {
            "initial_solution": {
                "description": "initial solution",
                "datatype": tuple,
                "default": (2.0, 2.0),
            },
            "budget": {
                "description": "max # of replications for a solver to take",
                "datatype": int,
                "default": 1000,
                "isDatafarmable": False,
            },
        }

    @property
    @override
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget,
        }

    @property
    @override
    def dim(self) -> int:
        return len(self.factors["initial_solution"])

    @property
    @override
    def lower_bounds(self) -> tuple:
        return (-np.inf,) * self.dim

    @property
    @override
    def upper_bounds(self) -> tuple:
        return (np.inf,) * self.dim

    def __init__(
        self,
        name: str = "EXAMPLE-1",
        fixed_factors: dict | None = None,
        model_fixed_factors: dict | None = None,  # noqa: ARG002
    ) -> None:
        """Initialize the problem.

        Args:
            name (str): user-specified name for problem
            fixed_factors (dict | None): fixed factors of the problem.
                If None, use default values.
            model_fixed_factors (dict | None): fixed factors of the model.
                If None, use default values.
        """
        # Let the base class handle default arguments.
        # TODO: check if model_fixed_factors should be passed to the model
        super().__init__(
            name=name,
            fixed_factors=fixed_factors,
            model_fixed_factors=None,
            model=ExampleModel,
        )

    @override
    def vector_to_factor_dict(self, vector: tuple) -> dict:
        return {"x": vector[:]}

    @override
    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:
        return tuple(factor_dict["x"])

    @override
    def response_dict_to_objectives(self, response_dict: dict) -> tuple:
        return (response_dict["est_f(x)"],)

    def deterministic_objectives_and_gradients(self, _x: tuple) -> tuple:
        """Compute deterministic components of objectives for a solution `x`.

        Arguments:
        ---------
        x : tuple
            vector of decision variables

        Returns:
        -------
        tuple
            vector of deterministic components of objectives
        tuple
            vector of gradients of deterministic components of objectives
        """
        det_objectives = (0,)
        det_objectives_gradients = ((0,) * self.dim,)
        return det_objectives, det_objectives_gradients

    def check_deterministic_constraints(self, x: tuple) -> bool:
        """Check if a solution `x` satisfies the problem's deterministic constraints.

        Arguments:
        ---------
        x : tuple
            vector of decision variables

        Returns:
        -------
        bool
            indicates if solution `x` satisfies the deterministic constraints.
        """
        # Superclass method will check box constraints.
        # Can add other constraints here.
        return super().check_deterministic_constraints(x)

    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
        """Generate a random solution for starting or restarting solvers.

        Arguments:
        ---------
        rand_sol_rng : mrg32k3a.mrg32k3a.MRG32k3a object
            random-number generator used to sample a new random solution

        Returns:
        -------
        tuple
            vector of decision variables
        """
        # x = tuple([rand_sol_rng.uniform(-2, 2) for _ in range(self.dim)])
        return tuple(
            rand_sol_rng.mvnormalvariate(
                mean_vec=[0] * self.dim,
                cov=np.eye(self.dim),
                factorized=False,
            )
        )
