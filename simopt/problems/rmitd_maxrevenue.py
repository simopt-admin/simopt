"""
Summary
-------
Maximize the total revenue of a multi-stage revenue management
with inter-temporal dependence problem.
"""
from base import Problem
from oracles.rmitd import RMITD


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
    gradient_available : bool
        indicates if gradient of objective function is available
    initial_solution : tuple
        default initial solution from which solvers start
    budget : int
        max number of replications (fn evals) for a solver to take
    optimal_bound : float
        bound on optimal objective function value
    ref_optimal_solution : tuple
        reference optimal solution
    oracle : Oracle object
        associated simulation oracle that generates replications
    rng_list : list of rng.MRG32k3a objects
        list of RNGs used to generate a random initial solution
        or a random problem instance
    factors : dict
        changeable factors of the problem
    specifications : dict
        details of each factor (for GUI, data validation, and defaults)

    Arguments
    ---------
    fixed_factors : dict
        dictionary of user-specified problem factors
    oracle_fixed_factors : dict
        subset of user-specified non-decision factors to pass through to the oracle

    See also
    --------
    base.Problem
    """
    def __init__(self, fixed_factors={}, oracle_fixed_factors={}):
        self.name = "RMITD-1"
        self.dim = 3
        self.n_objectives = 1
        self.n_stochastic_constraints = 0
        self.minmax = (1,)
        self.constraint_type = "deterministic"
        self.variable_type = "discrete"
        self.gradient_available = False
        self.budget = 10000
        self.optimal_bound = 0
        self.ref_optimal_solution = None  # (90, 50, 0)
        self.initial_solution = (100, 50, 30)
        self.oracle_default_factors = {}
        self.factors = fixed_factors
        self.specifications = {}
        super().__init__(fixed_factors, oracle_fixed_factors)
        # Instantiate oracle with fixed factors and over-riden defaults.
        self.oracle = RMITD(self.oracle_fixed_factors)

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
