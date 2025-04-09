"""Facility Sizing Problem.

Simulate demand at facilities.
A detailed description of the model/problem can be found
`here <https://simopt.readthedocs.io/en/latest/facilitysizing.html>`__.
"""

from __future__ import annotations

from typing import Callable, Final

import numpy as np

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.base import ConstraintType, Model, Problem, VariableType
from simopt.utils import classproperty, override

NUM_FACILITIES: Final[int] = 3


class FacilitySize(Model):
    """Facility Sizing Model.

    A model that simulates a facilitysize problem with a
    multi-variate normal distribution.
    Returns the probability of violating demand in each scenario.

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
        return "FACSIZE"

    @classproperty
    @override
    def class_name(cls) -> str:
        return "Facility Sizing"

    @classproperty
    @override
    def n_rngs(cls) -> int:
        return 1

    @classproperty
    @override
    def n_responses(cls) -> int:
        return 3

    @classproperty
    @override
    def specifications(cls) -> dict[str, dict]:
        return {
            "mean_vec": {
                "description": (
                    "location parameters of the multivariate normal distribution"
                ),
                "datatype": list,
                "default": [100] * NUM_FACILITIES,
            },
            "cov": {
                "description": "covariance of multivariate normal distribution",
                "datatype": list,
                "default": [
                    [2000, 1500, 500],
                    [1500, 2000, 750],
                    [500, 750, 2000],
                ],
            },
            "capacity": {
                "description": "capacity",
                "datatype": list,
                "default": [150, 300, 400],
            },
            "n_fac": {
                "description": "number of facilities",
                "datatype": int,
                "default": NUM_FACILITIES,
                "isDatafarmable": False,
            },
        }

    @property
    @override
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "mean_vec": self._check_mean_vec,
            "cov": self._check_cov,
            "capacity": self._check_capacity,
            "n_fac": self._check_n_fac,
        }

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize the FacilitySize model.

        Args:
            fixed_factors (dict | None): Fixed factors for the model.
                If None, default values are used.
        """
        # Let the base class handle default arguments.
        super().__init__(fixed_factors)

    def _check_mean_vec(self) -> None:
        if any(mean <= 0 for mean in self.factors["mean_vec"]):
            raise ValueError("All elements in mean_vec must be greater than 0.")

    def _check_cov(self) -> bool:
        try:
            np.linalg.cholesky(np.array(self.factors["cov"]))
            return True
        except np.linalg.LinAlgError as err:
            if "Matrix is not positive definite" in str(err):
                return False
            raise

    def _check_capacity(self) -> None:
        if len(self.factors["capacity"]) != self.factors["n_fac"]:
            raise ValueError("The length of capacity must equal n_fac.")

    def _check_n_fac(self) -> None:
        if self.factors["n_fac"] <= 0:
            raise ValueError("n_fac must be greater than 0.")

    @override
    def check_simulatable_factors(self) -> bool:
        if len(self.factors["capacity"]) != self.factors["n_fac"]:
            raise ValueError("The length of capacity must be equal to n_fac.")
        if len(self.factors["mean_vec"]) != self.factors["n_fac"]:
            raise ValueError("The length of mean_vec must be equal to n_fac.")
        if len(self.factors["cov"]) != self.factors["n_fac"]:
            raise ValueError("The length of cov must be equal to n_fac.")
        if len(self.factors["cov"][0]) != self.factors["n_fac"]:
            raise ValueError("The length of cov[0] must be equal to n_fac.")
        return True

    def replicate(self, rng_list: list[MRG32k3a]) -> tuple[dict, dict]:
        """Simulate a single replication using the current model factors.

        Args:
            rng_list (list[MRG32k3a]): Random number generators for the model to use
                when simulating a replication.

        Returns:
            tuple: A tuple containing:
                - dict: The responses dictionary, with keys:
                    - "stockout_flag" (bool): True if at least one facility failed to satisfy demand;
                    False otherwise.
                    - "n_fac_stockout" (int): Number of facilities that could not satisfy demand.
                    - "n_cut" (int): Total number of demand units that could not be satisfied.
                - dict: Gradient estimates for each response.
        """  # noqa: E501
        mean_vec: list[float | int] = self.factors["mean_vec"]
        cov = np.array(self.factors["cov"])
        capacity = np.array(self.factors["capacity"])
        # Designate RNG for demands.
        demand_rng = rng_list[0]
        # Generate random demands at facilities from truncated mv normal distribution.
        while True:
            demand = np.array(demand_rng.mvnormalvariate(mean_vec, cov))
            # Only leave if all demands are non-negative.
            if np.all(demand >= 0):
                break
        # Check for stockouts.
        extra_demand = demand - capacity
        pos_excess_mask = extra_demand > 0
        n_fac_stockout = np.sum(pos_excess_mask).astype(int)
        n_cut = np.sum(extra_demand[pos_excess_mask]).astype(int)
        # Compose responses and gradients.
        responses = {
            "stockout_flag": int(n_fac_stockout > 0),
            "n_fac_stockout": n_fac_stockout,
            "n_cut": n_cut,
        }
        gradients = {
            response_key: dict.fromkeys(self.specifications, np.nan)
            for response_key in responses
        }
        return responses, gradients


"""
Summary
-------
Minimize the (deterministic) total cost of installing capacity at
facilities subject to a chance constraint on stockout probability.
"""


class FacilitySizingTotalCost(Problem):
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
        return "FACSIZE-1"

    @classproperty
    @override
    def class_name(cls) -> str:
        return "Min Total Cost for Facility Sizing"

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
        return True

    @classproperty
    @override
    def optimal_value(cls) -> None:
        return None

    @classproperty
    @override
    def optimal_solution(cls) -> None:
        # return (185, 185, 185)
        return None

    @classproperty
    @override
    def model_default_factors(cls) -> dict:
        return {}

    @classproperty
    @override
    def model_decision_factors(cls) -> set[str]:
        return {"capacity"}

    @classproperty
    @override
    def specifications(cls) -> dict[str, dict]:
        return {
            "initial_solution": {
                "description": "Initial solution from which solvers start.",
                "datatype": tuple,
                "default": (300,) * NUM_FACILITIES,
            },
            "budget": {
                "description": "Max # of replications for a solver to take.",
                "datatype": int,
                "default": 10000,
                "isDatafarmable": False,
            },
            "installation_costs": {
                "description": "Cost to install a unit of capacity at each facility.",
                "datatype": tuple,
                "default": (1,) * NUM_FACILITIES,
            },
            "epsilon": {
                "description": "Maximum allowed probability of stocking out.",
                "datatype": float,
                "default": 0.05,
            },
        }

    @property
    @override
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget,
            "installation_costs": self._check_installation_costs,
            "epsilon": self._check_epsilon,
        }

    @property
    @override
    def dim(self) -> int:
        return self.model.factors["n_fac"]

    @property
    @override
    def lower_bounds(self) -> tuple:
        return (0,) * self.dim

    @property
    @override
    def upper_bounds(self) -> tuple:
        return (np.inf,) * self.dim

    def __init__(
        self,
        name: str = "FACSIZE-1",
        fixed_factors: dict | None = None,
        model_fixed_factors: dict | None = None,
    ) -> None:
        """Initialize the FacilitySizingTotalCost problem.

        Args:
            name (str): User-specified name for the problem.
            fixed_factors (dict | None): User-specified problem factors.
                If None, default values are used.
            model_fixed_factors (dict | None): Subset of user-specified
                non-decision factors to pass through to the model.
        """
        # Let the base class handle default arguments.
        super().__init__(
            name=name,
            fixed_factors=fixed_factors,
            model_fixed_factors=model_fixed_factors,
            model=FacilitySize,
        )

    def _check_installation_costs(self) -> None:
        if len(self.factors["installation_costs"]) != self.model.factors["n_fac"]:
            raise ValueError("The length of installation_costs must equal n_fac.")
        if any(elem < 0 for elem in self.factors["installation_costs"]):
            raise ValueError(
                "All elements in installation_costs must be greater than or equal to 0."
            )

    def _check_epsilon(self) -> None:
        if self.factors["epsilon"] < 0 or self.factors["epsilon"] > 1:
            raise ValueError(
                "epsilon must be greater than or equal to 0 and less than or equal "
                "to 1."
            )

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
        return {"capacity": vector[:]}

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
        return tuple(factor_dict["capacity"])

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
        return (np.nan * len(self.model.factors["capacity"]),)

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
        return ((0,) * len(self.model.factors["capacity"]),)

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
        return (response_dict["stockout_flag"],)

    def deterministic_stochastic_constraints_and_gradients(self) -> tuple[tuple, tuple]:
        """Compute deterministic components of stochastic constraints.

        Returns:
            tuple:
                - tuple: The deterministic components of the stochastic constraints.
                - tuple: The gradients of those deterministic components.
        """
        det_stoch_constraints = (-self.factors["epsilon"],)
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
        det_objectives = (np.dot(self.factors["installation_costs"], x),)
        det_objectives_gradients = (tuple(self.factors["installation_costs"]),)
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
        # Check box constraints.
        return super().check_deterministic_constraints(x)

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
        cov_matrix = np.diag([x**2 for x in self.factors["initial_solution"]])
        x = rand_sol_rng.mvnormalvariate(
            self.factors["initial_solution"], cov_matrix, factorized=False
        )
        while any(elem < 0 for elem in x):
            x = rand_sol_rng.mvnormalvariate(
                self.factors["initial_solution"], cov_matrix, factorized=False
            )
        return tuple(x)


"""
Summary
-------
Maximize the probability of not stocking out subject to a budget
constraint on the total cost of installing capacity.
"""


class FacilitySizingMaxService(Problem):
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
        return "FACSIZE-2"

    @classproperty
    @override
    def class_name(cls) -> str:
        return "Max Service for Facility Sizing"

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
        return (1,)

    @classproperty
    @override
    def constraint_type(cls) -> ConstraintType:
        return ConstraintType.DETERMINISTIC

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
    def optimal_solution(cls) -> None:
        # return (175, 179, 143)
        return None

    @classproperty
    @override
    def model_default_factors(cls) -> dict:
        return {}

    @classproperty
    @override
    def model_decision_factors(cls) -> set[str]:
        return {"capacity"}

    @classproperty
    @override
    def specifications(cls) -> dict[str, dict]:
        return {
            "initial_solution": {
                "description": "Initial solution from which solvers start.",
                "datatype": tuple,
                "default": (100,) * NUM_FACILITIES,
            },
            "budget": {
                "description": "Max # of replications for a solver to take.",
                "datatype": int,
                "default": 10000,
            },
            "installation_costs": {
                "description": "Cost to install a unit of capacity at each facility.",
                "datatype": tuple,
                "default": (1,) * NUM_FACILITIES,
            },
            "installation_budget": {
                "description": "Total budget for installation costs.",
                "datatype": float,
                "default": 500.0,
            },
        }

    @property
    @override
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget,
            "installation_costs": self._check_installation_costs,
            "installation_budget": self._check_installation_budget,
        }

    @property
    @override
    def dim(self) -> int:
        return self.model.factors["n_fac"]

    @property
    @override
    def lower_bounds(self) -> tuple:
        return (0,) * self.dim

    @property
    @override
    def upper_bounds(self) -> tuple:
        return (np.inf,) * self.dim

    def __init__(
        self,
        name: str = "FACSIZE-2",
        fixed_factors: dict | None = None,
        model_fixed_factors: dict | None = None,
    ) -> None:
        """Initialize the FacilitySizingMaxService problem.

        Args:
            name (str): User-specified name for the problem.
            fixed_factors (dict | None): User-specified problem factors.
                If None, default values are used.
            model_fixed_factors (dict | None): Subset of user-specified
                non-decision factors to pass through to the model.
        """
        # Let the base class handle default arguments.
        super().__init__(
            name=name,
            fixed_factors=fixed_factors,
            model_fixed_factors=model_fixed_factors,
            model=FacilitySize,
        )

    def _check_installation_costs(self) -> bool:
        return not (
            len(self.factors["installation_costs"]) != self.model.factors["n_fac"]
            or any(elem < 0 for elem in self.factors["installation_costs"])
        )

    def _check_installation_budget(self) -> bool:
        return self.factors["installation_budget"] > 0

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
        return {"capacity": vector[:]}

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
        return tuple(factor_dict["capacity"])

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
        return (1 - response_dict["stockout_flag"],)

    def deterministic_objectives_and_gradients(self, x: tuple) -> tuple[tuple, tuple]:  # noqa: ARG002
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
        det_objectives_gradients = ((0, 0, 0),)
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
        # Check budget constraint.
        budget_feasible = (
            np.dot(self.factors["installation_costs"], x)
            <= self.factors["installation_budget"]
        )
        # Check box constraints.
        box_feasible = super().check_deterministic_constraints(x)
        return budget_feasible * box_feasible

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
        # Generate random vector of length # of facilities of continuous values
        # summing to less than or equal to installation budget.
        x = rand_sol_rng.continuous_random_vector_from_simplex(
            n_elements=self.model.factors["n_fac"],
            summation=self.factors["installation_budget"],
            exact_sum=False,
        )
        return tuple(x)
