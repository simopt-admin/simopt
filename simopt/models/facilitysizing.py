"""
Summary
-------
Simulate demand at facilities.
A detailed description of the model/problem can be found
`here <https://simopt.readthedocs.io/en/latest/facilitysizing.html>`__.
"""

from __future__ import annotations

from typing import Callable, Final

import numpy as np
from mrg32k3a.mrg32k3a import MRG32k3a

from simopt.base import ConstraintType, Model, Problem, VariableType

NUM_FACILITIES: Final[int] = 3


class FacilitySize(Model):
    """
    A model that simulates a facilitysize problem with a
    multi-variate normal distribution.
    Returns the probability of violating demand in each scenario.

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
        return "FACSIZE"

    @property
    def n_rngs(self) -> int:
        return 1

    @property
    def n_responses(self) -> int:
        return 3

    @property
    def specifications(self) -> dict[str, dict]:
        return {
            "mean_vec": {
                "description": "location parameters of the multivariate normal distribution",
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
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "mean_vec": self.check_mean_vec,
            "cov": self.check_cov,
            "capacity": self.check_capacity,
            "n_fac": self.check_n_fac,
        }

    def __init__(self, fixed_factors: dict | None = None) -> None:
        # Let the base class handle default arguments.
        super().__init__(fixed_factors)

    def check_mean_vec(self) -> None:
        if any(mean <= 0 for mean in self.factors["mean_vec"]):
            raise ValueError("All elements in mean_vec must be greater than 0.")

    def check_cov(self) -> bool:
        try:
            np.linalg.cholesky(np.matrix(self.factors["cov"]))
            return True
        except np.linalg.linalg.LinAlgError as err:
            if "Matrix is not positive definite" in err.message:
                return False
            else:
                raise

    def check_capacity(self) -> None:
        if len(self.factors["capacity"]) != self.factors["n_fac"]:
            raise ValueError("The length of capacity must equal n_fac.")

    def check_n_fac(self) -> None:
        if self.factors["n_fac"] <= 0:
            raise ValueError("n_fac must be greater than 0.")

    def check_simulatable_factors(self) -> bool:
        if len(self.factors["capacity"]) != self.factors["n_fac"]:
            raise ValueError("The length of capacity must be equal to n_fac.")
        elif len(self.factors["mean_vec"]) != self.factors["n_fac"]:
            raise ValueError("The length of mean_vec must be equal to n_fac.")
        elif len(self.factors["cov"]) != self.factors["n_fac"]:
            raise ValueError("The length of cov must be equal to n_fac.")
        elif len(self.factors["cov"][0]) != self.factors["n_fac"]:
            raise ValueError("The length of cov[0] must be equal to n_fac.")
        else:
            return True

    def replicate(self, rng_list: list[MRG32k3a]) -> tuple[dict, dict]:
        """
        Simulate a single replication for the current model factors.

        Args:
            rng_list : list of mrg32k3a.mrg32k3a.MRG32k3a objects
                rngs for model to use when simulating a replication

        Returns:
            A tuple containing both a dictionary of responses and a dictionary
            of gradient estimates for each response.
            The responses dictionary contains the following keys:
            stockout_flag : boolean
            false - all facilities satisfy the demand, true - at least one of the facilities did not satisfy the demand
            n_fac_stockout : integer
            the number of facilities which cannot satisfy the demand
            n_cut : integer
            the number of toal demand which cannot be satisfied
        """
        # Designate RNG for demands.
        demand_rng = rng_list[0]
        stockout_flag = 0
        n_fac_stockout = 0
        n_cut = 0
        # Generate random demands at facilities from truncated multivariate normal distribution.
        demand = demand_rng.mvnormalvariate(
            self.factors["mean_vec"], self.factors["cov"], factorized=False
        )
        while np.any(demand < 0):
            demand = demand_rng.mvnormalvariate(
                self.factors["mean_vec"], self.factors["cov"], factorized=False
            )
        # Check for stockouts.
        for i in range(self.factors["n_fac"]):
            if demand[i] > self.factors["capacity"][i]:
                n_fac_stockout = n_fac_stockout + 1
                stockout_flag = 1
                n_cut += demand[i] - self.factors["capacity"][i]
        # Compose responses and gradients.
        responses = {
            "stockout_flag": stockout_flag,
            "n_fac_stockout": n_fac_stockout,
            "n_cut": n_cut,
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
Minimize the (deterministic) total cost of installing capacity at
facilities subject to a chance constraint on stockout probability.
"""


class FacilitySizingTotalCost(Problem):
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
        return True

    @property
    def optimal_value(self) -> float | None:
        return None

    @property
    def optimal_solution(self) -> tuple | None:
        # return (185, 185, 185)
        return None

    @property
    def model_default_factors(self) -> dict:
        return {}

    @property
    def model_decision_factors(self) -> set[str]:
        return {"capacity"}

    @property
    def specifications(self) -> dict[str, dict]:
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
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget,
            "installation_costs": self.check_installation_costs,
            "epsilon": self.check_epsilon,
        }

    @property
    def dim(self) -> int:
        return self.model.factors["n_fac"]

    @property
    def lower_bounds(self) -> tuple:
        return (0,) * self.dim

    @property
    def upper_bounds(self) -> tuple:
        return (np.inf,) * self.dim

    def __init__(
        self,
        name: str = "FACSIZE-1",
        fixed_factors: dict | None = None,
        model_fixed_factors: dict | None = None,
    ) -> None:
        # Let the base class handle default arguments.
        super().__init__(
            name=name,
            fixed_factors=fixed_factors,
            model_fixed_factors=model_fixed_factors,
            model=FacilitySize,
        )

    def check_installation_costs(self) -> None:
        if (
            len(self.factors["installation_costs"])
            != self.model.factors["n_fac"]
        ):
            raise ValueError(
                "The length of installation_costs must equal n_fac."
            )
        elif any([elem < 0 for elem in self.factors["installation_costs"]]):
            raise ValueError(
                "All elements in installation_costs must be greater than or equal to 0."
            )

    def check_epsilon(self) -> None:
        if 0 > self.factors["epsilon"] or self.factors["epsilon"] > 1:
            raise ValueError(
                "epsilon must be greater than or equal to 0 and less than or equal to 1."
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
        factor_dict = {"capacity": vector[:]}
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
        vector = tuple(factor_dict["capacity"])
        return vector

    def factor_dict_to_vector_gradients(self, factor_dict: dict) -> tuple:
        """Convert a dictionary with factor keys to a gradient vector.

        Notes
        -----
        A subclass of ``base.Problem`` can have its own custom
        ``factor_dict_to_vector_gradients`` method if the
        objective is deterministic.

        Parameters
        ----------
        factor_dict : dict
            Dictionary with factor keys and associated values.

        Returns
        -------
        vector : tuple
            Vector of partial derivatives associated with decision variables.
        """
        vector = (np.nan * len(self.model.factors["capacity"]),)
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
        objectives = (0,)
        return objectives

    def response_dict_to_objectives_gradients(
        self, response_dict: dict
    ) -> tuple:
        """Convert a dictionary with response keys to a vector
        of gradients.

        Notes
        -----
        A subclass of ``base.Problem`` can have its own custom
        ``response_dict_to_objectives_gradients`` method if the
        objective is deterministic.

        Parameters
        ----------
        response_dict : dict
            Dictionary with response keys and associated values.

        Returns
        -------
        tuple
            Vector of gradients.
        """
        return ((0,) * len(self.model.factors["capacity"]),)

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
        stoch_constraints = (response_dict["stockout_flag"],)
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
        det_stoch_constraints = (-self.factors["epsilon"],)
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
        det_objectives = (np.dot(self.factors["installation_costs"], x),)
        det_objectives_gradients = (tuple(self.factors["installation_costs"]),)
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
        # Check box constraints.
        box_feasible = super().check_deterministic_constraints(x)
        return box_feasible

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
        cov_matrix = np.diag([x**2 for x in self.factors["initial_solution"]])
        x = rand_sol_rng.mvnormalvariate(
            self.factors["initial_solution"], cov_matrix, factorized=False
        )
        while np.any(x < 0):
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
            initial_solution : tuple
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
        return (1,)

    @property
    def constraint_type(self) -> ConstraintType:
        return ConstraintType.DETERMINISTIC

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
        # return (175, 179, 143)
        return None

    @property
    def model_default_factors(self) -> dict:
        return {}

    @property
    def model_decision_factors(self) -> set[str]:
        return {"capacity"}

    @property
    def specifications(self) -> dict[str, dict]:
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
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget,
            "installation_costs": self.check_installation_costs,
            "installation_budget": self.check_installation_budget,
        }

    @property
    def dim(self) -> int:
        return self.model.factors["n_fac"]

    @property
    def lower_bounds(self) -> tuple:
        return (0,) * self.dim

    @property
    def upper_bounds(self) -> tuple:
        return (np.inf,) * self.dim

    def __init__(
        self,
        name: str = "FACSIZE-2",
        fixed_factors: dict | None = None,
        model_fixed_factors: dict | None = None,
    ) -> None:
        # Let the base class handle default arguments.
        super().__init__(
            name=name,
            fixed_factors=fixed_factors,
            model_fixed_factors=model_fixed_factors,
            model=FacilitySize,
        )

    def check_installation_costs(self) -> bool:
        if (
            len(self.factors["installation_costs"])
            != self.model.factors["n_fac"]
        ):
            return False
        elif any([elem < 0 for elem in self.factors["installation_costs"]]):
            return False
        else:
            return True

    def check_installation_budget(self) -> bool:
        return self.factors["installation_budget"] > 0

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
        factor_dict = {"capacity": vector[:]}
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
        vector = tuple(factor_dict["capacity"])
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
        objectives = (1 - response_dict["stockout_flag"],)
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
        tuple
            vector of LHSs of stochastic constraint
        """
        stoch_constraints = ()
        return stoch_constraints

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
        tuple
            vector of deterministic components of objectives
        tuple
            vector of gradients of deterministic components of objectives
        """
        det_objectives = (0,)
        det_objectives_gradients = ((0, 0, 0),)
        return det_objectives, det_objectives_gradients

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
        det_stoch_constraints_gradients = ()
        return det_stoch_constraints, det_stoch_constraints_gradients

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
        # Check budget constraint.
        budget_feasible = (
            np.dot(self.factors["installation_costs"], x)
            <= self.factors["installation_budget"]
        )
        # Check box constraints.
        box_feasible = super().check_deterministic_constraints(x)
        return budget_feasible * box_feasible

    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
        """
        Generate a random solution for starting or restarting solvers.

        Arguments
        ---------
        rand_sol_rng : mrg32k3a.mrg32k3a.MRG32k3a object
            random-number generator used to sample a new random solution

        Returns
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
