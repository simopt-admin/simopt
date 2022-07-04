"""
Summary
-------
Simulate multiple periods of arrival and seating at a restaurant.
A detailed description of the model/problem can be found
`here <https://simopt.readthedocs.io/en/latest/tableallocation.html>`_.
"""
import numpy as np

from base import Model, Problem


class TableAllocation(Model):
    """
    A model that simulates a table capacity allocation problem at a restaurant
    with a homogenous Poisson arrvial process and exponential service times.
    Returns expected maximum revenue.

    Attributes
    ----------
    name : str
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

    Parameters
    ----------
    fixed_factors : dict
        fixed_factors of the simulation model

        ``n_hours``
            Number of hours to simulate (`int`)
        ``capacity``
            Maximum total capacity (`int`)
        ``table_cap``
            Capacity of each type of table (`int`)
        ``lambda``
            Average number of arrivals per hour (`flt`)
        ``service_time_means``
            Mean service time in minutes (`flt`)
        ``table_revenue``
            Per table revenue earned (`flt`)
        ``num_tables``
            Number of tables of each capacity (`int`)

    See also
    --------
    base.Model
    """
    def __init__(self, fixed_factors=None):
        if fixed_factors is None:
            fixed_factors = {}
        self.name = "TABLEALLOCATION"
        self.n_rngs = 3
        self.n_responses = 2
        self.factors = fixed_factors
        self.specifications = {
            "n_hours": {
                "description": "Number of hours to simulate.",
                "datatype": float,
                "default": 5.0
            },
            "capacity": {
                "description": "Maximum capacity of restaurant.",
                "datatype": int,
                "default": 80
            },
            "table_cap": {
                "description": "Seating capacity of each type of table.",
                "datatype": list,
                "default": [2, 4, 6, 8]
            },
            "lambda": {
                "description": "Average number of arrivals per hour.",
                "datatype": list,
                "default": [3, 6, 3, 3, 2, 4 / 3, 6 / 5, 1]
            },
            "service_time_means": {
                "description": "Mean service time (in minutes).",
                "datatype": list,
                "default": [20, 25, 30, 35, 40, 45, 50, 60]
            },
            "table_revenue": {
                "description": "Revenue earned for each group size.",
                "datatype": list,
                "default": [15, 30, 45, 60, 75, 90, 105, 120]
            },
            "num_tables": {
                "description": "Number of tables of each capacity.",
                "datatype": list,
                "default": [10, 5, 4, 2]
            }
        }
        self.check_factor_list = {
            "n_hours": self.check_n_hours,
            "capacity": self.check_capacity,
            "table_cap": self.check_table_cap,
            "lambda": self.check_lambda,
            "service_time_means": self.check_service_time_means,
            "table_revenue": self.check_table_revenue,
            "num_tables": self.check_num_tables
        }
        # Set factors of the simulation model
        super().__init__(fixed_factors)

    # Check for simulatable factors
    def check_n_hours(self):
        return self.factors["n_hours"] > 0

    def check_capacity(self):
        return self.factors["capacity"] > 0

    def check_table_cap(self):
        return self.factors["table_cap"] > [0, 0, 0, 0]

    def check_lambda(self):
        return self.factors["lambda"] >= [0] * max(self.factors["table_cap"])

    def check_service_time_means(self):
        return self.factors["service_time_means"] > [0] * max(self.factors["table_cap"])

    def check_table_revenue(self):
        return self.factors["table_revenue"] >= [0] * max(self.factors["table_cap"])

    def check_num_tables(self):
        return self.factors["num_tables"] >= [0, 0, 0, 0]

    def check_simulatable_factors(self):
        if len(self.factors["num_tables"]) != len(self.factors["table_cap"]):
            return False
        elif len(self.factors["lambda"]) != max(self.factors["table_cap"]):
            return False
        elif len(self.factors["lambda"]) != len(self.factors["service_time_means"]):
            return False
        elif len(self.factors["service_time_means"]) != len(self.factors["table_revenue"]):
            return False
        else:
            return True

    def replicate(self, rng_list):
        """
        Simulate a single replication for the current model factors.

        Arguments
        ---------
        rng_list : [list]  [rng.mrg32k3a.MRG32k3a]
            rngs for model to use when simulating a replication

        Returns
        -------
        responses : dict
            performance measures of interest

            ``total_revenue``
                Total revenue earned over the simulation period.
            ``service_rate``
                Fraction of customer arrivals that are seated.

        """
        # Designate separate random number generators.
        arrival_rng = rng_list[0]
        group_size_rng = rng_list[1]
        service_rng = rng_list[2]
        # Track total revenue.
        total_rev = 0
        # Track table availability.
        # (i,j) is the time that jth table of size i becomes available.
        table_avail = np.zeros((4, max(self.factors["num_tables"])))
        # Generate total number of arrivals in the period
        n_arrivals = arrival_rng.poissonvariate(round(self.factors["n_hours"] * sum(self.factors["lambda"])))
        # Generate arrival times in minutes
        arrival_times = 60 * np.sort([arrival_rng.uniform(0, self.factors["n_hours"]) for _ in range(n_arrivals)])
        # Track seating rate
        found = np.zeros(n_arrivals)
        # Pass through all arrivals of groups to the restaurants.
        for n in range(n_arrivals):
            # Determine group size.
            group_size = group_size_rng.choices(population=range(1, max(self.factors["table_cap"]) + 1), weights=self.factors["lambda"])[0]
            # Find smallest table size to start search.
            table_size_idx = 0
            while self.factors["table_cap"][table_size_idx] < group_size:
                table_size_idx = table_size_idx + 1
            # Find smallest available table.
            for k in range(table_size_idx, len(self.factors["num_tables"])):
                for j in range(self.factors["num_tables"][k]):
                    # Check if table is currently available.
                    if table_avail[k, j] < arrival_times[n]:
                        found[n] = 1
                        break
                if found[n] == 1:
                    break
            if found[n] == 1:
                # Sample service time.
                service_time = service_rng.expovariate(lambd=1 / self.factors["service_time_means"][group_size - 1])
                # Update table availability.
                table_avail[k, j] = table_avail[k, j] + service_time
                # Update revenue.
                total_rev = total_rev + self.factors["table_revenue"][group_size - 1]
        # Calculate responses from simulation data.
        responses = {"total_revenue": total_rev,
                     "service_rate": sum(found) / len(found)
                     }
        gradients = {response_key: {factor_key: np.nan for factor_key in self.specifications} for response_key in responses}
        return responses, gradients


"""
Summary
-------
Maximize the total expected revenue for a restaurant operation.
"""


class TableAllocationMaxRev(Problem):
    """
    Class to make table allocation simulation-optimization problems.

    Attributes
    ----------
    name : str
        name of problem
    dim : int
        number of decision variables
    n_objectives : int
        number of objectives
    n_stochastic_constraints : int
        number of stochastic constraints
    minmax : tuple of int (+/- 1)
        indicator of maximization (+1) or minimization (-1) for each objective
    constraint_type : str
        description of constraints types:
            "unconstrained", "box", "deterministic", "stochastic"
    variable_type : str
        description of variable types:
            "discrete", "continuous", "mixed"
    gradient_available : bool
        indicates if gradient of objective function is available
    optimal_value : float
        optimal objective function value
    optimal_solution : tuple
        optimal solution
    model : base.Model
        associated simulation model that generates replications
    model_default_factors : dict
        default values for overriding model-level default factors
    model_fixed_factors : dict
        combination of overriden model-level factors and defaults
    model_decision_factors : set of str
        set of keys for factors that are decision variables
    rng_list : [list]  [rng.mrg32k3a.MRG32k3a]
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
        user-specified name of problem
    fixed_factors : dict
        dictionary of user-specified problem factors
    model_fixed factors : dict
        subset of user-specified non-decision factors to pass through to the model

    See also
    --------
    base.Problem
    """
    def __init__(self, name="TABLEALLOCATION-1", fixed_factors=None, model_fixed_factors=None):
        if fixed_factors is None:
            fixed_factors = {}
        if model_fixed_factors is None:
            model_fixed_factors = {}
        self.name = name
        self.dim = 4
        self.n_objectives = 1
        self.n_stochastic_constraints = 0
        self.minmax = (1,)
        self.constraint_type = "deterministic"
        self.variable_type = "discrete"
        self.lower_bounds = ([0, 0, 0, 0])
        self.upper_bounds = ([np.inf, np.inf, np.inf, np.inf])
        self.gradient_available = False
        self.optimal_value = None
        self.optimal_solution = None
        self.model_default_factors = {}
        self.model_decision_factors = {"num_tables"}
        self.factors = fixed_factors
        self.specifications = {
            "initial_solution": {
                "description": "Initial solution from which solvers start.",
                "datatype": tuple,
                "default": (10, 5, 4, 2)
            },
            "budget": {
                "description": "Max # of replications for a solver to take.",
                "datatype": int,
                "default": 1000
            }
        }
        self.check_factor_list = {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget
        }
        super().__init__(fixed_factors, model_fixed_factors)
        # Instantiate model with fixed factors and overwritten defaults.
        self.model = TableAllocation(self.model_fixed_factors)

    def vector_to_factor_dict(self, vector):
        """
        Convert a vector of variables to a dictionary with factor keys

        Arguments
        ---------
        vector : tuple
            vector of values associated with decision variables

        Returns
        -------
        factor_dict : dict
            dictionary with factor keys and associated values
        """
        factor_dict = {
            "num_tables": vector[:]
        }
        return factor_dict

    def factor_dict_to_vector(self, factor_dict):
        """
        Convert a dictionary with factor keys to a vector
        of variables.

        Arguments
        ---------
        factor_dict : dict
            dictionary with factor keys and associated values

        Returns
        -------
        vector : tuple
            vector of values associated with decision variables
        """
        vector = (factor_dict["num_tables"],)
        return vector

    def response_dict_to_objectives(self, response_dict):
        """
        Convert a dictionary with response keys to a vector
        of objectives.

        Arguments
        ---------
        response_dict : dict
            dictionary with response keys and associated values

        Returns
        -------
        objectives : tuple
            vector of objectives
        """
        objectives = (response_dict["total_revenue"],)
        return objectives

    def response_dict_to_stoch_constraints(self, response_dict):
        """
        Convert a dictionary with response keys to a vector
        of left-hand sides of stochastic constraints: E[Y] >= 0

        Arguments
        ---------
        response_dict : dict
            dictionary with response keys and associated values

        Returns
        -------
        stoch_constraints : tuple
            vector of LHSs of stochastic constraint
        """
        stoch_constraints = None
        return stoch_constraints

    def deterministic_objectives_and_gradients(self, x):
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
        det_objectives_gradients = ((0,) * self.dim,)
        return det_objectives, det_objectives_gradients

    def deterministic_stochastic_constraints_and_gradients(self, x):
        """
        Compute deterministic components of stochastic constraints
        for a solution `x`.

        Arguments
        ---------
        x : tuple
            vector of decision variables

        Returns
        -------
        det_stoch_constraints : tuple
            vector of deterministic components of stochastic constraints
        det_stoch_constraints_gradients : tuple
            vector of gradients of deterministic components of
            stochastic constraints
        """
        det_stoch_constraints = None
        det_stoch_constraints_gradients = None
        return det_stoch_constraints, det_stoch_constraints_gradients

    def check_deterministic_constraints(self, x):
        """
        Check if a solution `x` satisfies the problem's deterministic
        constraints.

        Arguments
        ---------
        x : tuple
            vector of decision variables

        Returns
        -------
        satisfies : bool
            indicates if solution `x` satisfies the deterministic constraints.
        """
        return (np.sum(np.multiply(self.model_fixed_factors["table_cap"], x)) <= self.model_fixed_factors["capacity"])

    def get_random_solution(self, rand_sol_rng):
        """
        Generate a random solution for starting or restarting solvers.

        Arguments
        ---------
        rand_sol_rng : rng.mrg32k3a.MRG32k3a
            random-number generator used to sample a new random solution

        Returns
        -------
        x : tuple
            vector of decision variables
        """
        # Add new tables of random size to the restaurant until the capacity is reached.
        # TO DO: Replace this with call to integer_random_vector_from_simplex().
        # The different-weight case is not yet implemented.
        allocated = 0
        num_tables = [0, 0, 0, 0]
        while allocated < self.model_fixed_factors["capacity"]:
            table = rand_sol_rng.randint(0, len(self.model_fixed_factors["table_cap"]) - 1)
            if self.model_fixed_factors["table_cap"][table] <= (self.model_fixed_factors["capacity"] - allocated):
                num_tables[table] = num_tables[table] + 1
                allocated = allocated + self.model_fixed_factors["table_cap"][table]
            elif self.model_fixed_factors["table_cap"][0] > (self.model_fixed_factors["capacity"] - allocated):
                break
        return tuple(num_tables)
