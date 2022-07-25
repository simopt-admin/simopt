"""
Summary
-------
Simulate a multi-stage revenue management system with inter-temporal dependence.
A detailed description of the model/problem can be found
`here <https://simopt.readthedocs.io/en/latest/rmitd.html>`_.

"""
import numpy as np

from base import Model, Problem


class RMITD(Model):
    """
    A model that simulates a multi-stage revenue management system with
    inter-temporal dependence.
    Returns the total revenue.

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
    def __init__(self, fixed_factors=None):
        if fixed_factors is None:
            fixed_factors = {}
        self.name = "RMITD"
        self.n_rngs = 2
        self.n_responses = 1
        self.specifications = {
            "time_horizon": {
                "description": "Time horizon.",
                "datatype": int,
                "default": 3
            },
            "prices": {
                "description": "Prices for each period.",
                "datatype": list,
                "default": [100, 300, 400]
            },
            "demand_means": {
                "description": "Mean demand for each period.",
                "datatype": list,
                "default": [50, 20, 30]
            },
            "cost": {
                "description": "Cost per unit of capacity at t = 0.",
                "datatype": float,
                "default": 80.0
            },
            "gamma_shape": {
                "description": "Shape parameter of gamma distribution.",
                "datatype": float,
                "default": 1.0
            },
            "gamma_scale": {
                "description": "Scale parameter of gamma distribution.",
                "datatype": float,
                "default": 1.0
            },
            "initial_inventory": {
                "description": "Initial inventory.",
                "datatype": int,
                "default": 100
            },
            "reservation_qtys": {
                "description": "Inventory to reserve going into periods 2, 3, ..., T.",
                "datatype": list,
                "default": [50, 30]
            }
        }
        self.check_factor_list = {
            "time_horizon": self.check_time_horizon,
            "prices": self.check_prices,
            "demand_means": self.check_demand_means,
            "cost": self.check_cost,
            "gamma_shape": self.check_gamma_shape,
            "gamma_scale": self.check_gamma_scale,
            "initial_inventory": self.check_initial_inventory,
            "reservation_qtys": self.check_reservation_qtys
        }
        # Set factors of the simulation model.
        super().__init__(fixed_factors)

    def check_time_horizon(self):
        return self.factors["time_horizon"] > 0

    def check_prices(self):
        return all(price > 0 for price in self.factors["prices"])

    def check_demand_means(self):
        return all(demand_mean > 0 for demand_mean in self.factors["demand_means"])

    def check_cost(self):
        return self.factors["cost"] > 0

    def check_gamma_shape(self):
        return self.factors["gamma_shape"] > 0

    def check_gamma_scale(self):
        return self.factors["gamma_scale"] > 0

    def check_initial_inventory(self):
        return self.factors["initial_inventory"] > 0

    def check_reservation_qtys(self):
        return all(reservation_qty > 0 for reservation_qty in self.factors["reservation_qtys"])

    def check_simulatable_factors(self):
        # Check for matching number of periods.
        if len(self.factors["prices"]) != self.factors["time_horizon"]:
            return False
        elif len(self.factors["demand_means"]) != self.factors["time_horizon"]:
            return False
        elif len(self.factors["reservation_qtys"]) != self.factors["time_horizon"] - 1:
            return False
        # Check that first reservation level is less than initial inventory.
        elif self.factors["initial_inventory"] < self.factors["reservation_qtys"][0]:
            return False
        # Check for non-increasing reservation levels.
        elif any(self.factors["reservation_qtys"][idx] < self.factors["reservation_qtys"][idx + 1] for idx in range(self.factors["time_horizon"] - 2)):
            return False
        # Check that gamma_shape*gamma_scale = 1.
        elif np.isclose(self.factors["gamma_shape"] * self.factors["gamma_scale"], 1) is False:
            return False
        else:
            return True

    def replicate(self, rng_list):
        """
        Simulate a single replication for the current model factors.

        Arguments
        ---------
        rng_list : list of rng.MRG32k3a objects
            rngs for model to use when simulating a replication

        Returns
        -------
        responses : dict
            performance measures of interest
            "revenue" = total revenue
        gradients : dict of dicts
            gradient estimates for each response
        """
        # Designate separate random number generators.
        # Outputs will be coupled when generating demand.
        X_rng = rng_list[0]
        Y_rng = rng_list[1]
        # Generate X and Y (to use for computing demand).
        # random.gammavariate takes two inputs: alpha and beta.
        #     alpha = k = gamma_shape
        #     beta = 1/theta = 1/gamma_scale
        X = X_rng.gammavariate(alpha=self.factors["gamma_shape"], beta=1./self.factors["gamma_scale"])
        Y = [Y_rng.expovariate(1) for _ in range(self.factors["time_horizon"])]
        # Track inventory over time horizon.
        remaining_inventory = self.factors["initial_inventory"]
        # Append "no reservations" for decision-making in final period.
        reservations = self.factors["reservation_qtys"]
        reservations.append(0)
        # Simulate over the time horizon and calculate the realized revenue.
        revenue = 0
        for period in range(self.factors["time_horizon"]):
            demand = self.factors["demand_means"][period]*X*Y[period]
            sell = min(max(remaining_inventory-reservations[period], 0), demand)
            remaining_inventory = remaining_inventory - sell
            revenue += sell*self.factors["prices"][period]
        revenue -= self.factors["cost"]*self.factors["initial_inventory"]
        # Compose responses and gradients.
        responses = {"revenue": revenue}
        gradients = {response_key: {factor_key: np.nan for factor_key in self.specifications} for response_key in responses}
        return responses, gradients


"""
Summary
-------
Maximize the total revenue of a multi-stage revenue management
with inter-temporal dependence problem.
"""


class RMITDMaxRevenue(Problem):
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
    rng_list : list of rng.MRG32k3a objects
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
    model_fixed_factors : dict
        subset of user-specified non-decision factors to pass through to the model

    See also
    --------
    base.Problem
    """
    def __init__(self, name="RMITD-1", fixed_factors=None, model_fixed_factors=None):
        if fixed_factors is None:
            fixed_factors = {}
        if model_fixed_factors is None:
            model_fixed_factors = {}
        self.name = name
        self.dim = 3
        self.n_objectives = 1
        self.n_stochastic_constraints = 0
        self.minmax = (1,)
        self.constraint_type = "deterministic"
        self.variable_type = "discrete"
        self.lower_bounds = (0, 0, 0)
        self.upper_bounds = (np.inf, np.inf, np.inf)
        self.gradient_available = False
        self.optimal_value = None
        self.optimal_solution = None  # (90, 50, 0)
        self.model_default_factors = {}
        self.model_decision_factors = {"initial_inventory", "reservation_qtys"}
        self.factors = fixed_factors
        self.specifications = {
            "initial_solution": {
                "description": "Initial solution from which solvers start.",
                "datatype": tuple,
                "default": (100, 50, 30)
            },
            "budget": {
                "description": "Max # of replications for a solver to take.",
                "datatype": int,
                "default": 10000
            }
        }
        self.check_factor_list = {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget
        }
        super().__init__(fixed_factors, model_fixed_factors)
        # Instantiate model with fixed factors and over-riden defaults.
        self.model = RMITD(self.model_fixed_factors)

    def vector_to_factor_dict(self, vector):
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
        factor_dict = {
            "initial_inventory": vector[0],
            "reservation_qtys": list(vector[0:])
        }
        return factor_dict

    def factor_dict_to_vector(self, factor_dict):
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
        vector = (factor_dict["initial_inventory"],) + tuple(factor_dict["reservation_qtys"])
        return vector

    def response_dict_to_objectives(self, response_dict):
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
        objectives = (response_dict["revenue"],)
        return objectives

    def check_deterministic_constraints(self, x):
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
        return all(x[idx] >= x[idx + 1] for idx in range(self.dim - 1))

    def get_random_solution(self, rand_sol_rng):
        """
        Generate a random solution for starting or restarting solvers.

        Arguments
        ---------
        rand_sol_rng : rng.MRG32k3a object
            random-number generator used to sample a new random solution

        Returns
        -------
        x : tuple
            vector of decision variables
        """
        # Generate random solution using acceptable/rejection.
        while True:
            x = tuple([200*rand_sol_rng.random() for _ in range(self.dim)])
            if self.check_deterministic_constraints(x):
                break
        return x
