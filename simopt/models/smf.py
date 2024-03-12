"""
Summary
-------
Simulate duration of a stochastic Max-Flow network (SMF).
A detailed description of the model/problem can be found
`here <https://simopt.readthedocs.io/en/latest/SMF.html>`_.
"""

import numpy as np
from ortools.graph.python import max_flow
from ..base import Model, Problem


class SMF(Model):
    """
    A model that simulates a stochastic Max-Flow problem with
    capacities deducted with multivariate distributed noise distributed durations

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
        self.name = "SMF"
        self.n_rngs = 1
        self.n_random = 1
        self.n_responses = 1
        cov_fac = np.zeros((20, 20))
        np.fill_diagonal(cov_fac, 4)
        cov_fac = cov_fac.tolist()
        self.specifications = {
            "num_nodes": {
                "description": "number of nodes, 0 being the source, highest being the sink",
                "datatype": int,
                "default": 10
            },
            "source_index": {
                "description": "source node index",
                "datatype": int,
                "default": 0
            },
            "sink_index": {
                "description": "sink node index",
                "datatype": int,
                "default": 9
            },
            "arcs": {
                "description": "list of arcs",
                "datatype": list,
                "default": [(0, 1), (0, 2), (0, 3), (1, 2), (1, 4), (2, 4), (4, 2), (3, 2), (2, 5), (4, 5), (3, 6), (3, 7), (6, 2), (6, 5), (6, 7), (5, 8), (6, 8), (6, 9), (7, 9), (8, 9)]
            },
            "assigned_capacities": {
                "description": "Assigned capacity of each arc",
                "datatype": list,
                "default": [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
            },
            "mean_noise": {
                "description": "The mean noise in reduction of arc capacities",
                "datatype": list,
                "default": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            },
            "cov_noise": {
                "description": "Covariance matrix of noise",
                "datatype": list,
                "default": cov_fac
            }

        }
        self.check_factor_list = {
            "num_nodes": self.check_num_nodes,
            "arcs": self.check_arcs,
            "assigned_capacities": self.check_capacities,
            "mean_noise": self.check_mean,
            "cov_noise": self.check_cov,
            "source_index": self.check_s,
            "sink_index": self.check_t
        }
        # Set factors of the simulation model.
        super().__init__(fixed_factors)

    def check_num_nodes(self):
        return self.factors["num_nodes"] > 0

    def dfs(self, graph, start, visited=None):
        if visited is None:
            visited = set()
        visited.add(start)
        for next in graph[start] - visited:
            self.dfs(graph, next, visited)
        return visited

    def check_arcs(self):
        if len(self.factors["arcs"]) <= 0:
            return False
        # Check source is connected to the sink.
        graph = {node: set() for node in range(0, self.factors["num_nodes"])}
        for a in self.factors["arcs"]:
            graph[a[0]].add(a[1])
        visited = self.dfs(graph, self.factors["source_index"])
        if self.factors["source_index"] in visited and self.factors["sink_index"] in visited:
            return True
        return False

    def check_capacities(self):
        positive = True
        for x in list(self.factors["assigned_capacities"]):
            positive = positive & (x > 0)
        return (len(self.factors["assigned_capacities"]) == len(self.factors["arcs"])) & positive

    def check_mean(self):
        return len(self.factors["mean_noise"]) == len(self.factors["arcs"])

    def check_cov(self):
        return np.array(self.factors["cov_noise"]).shape == (len(self.factors["arcs"]), len(self.factors["arcs"]))

    def check_s(self):
        return self.factors["source_index"] >= 0 and self.factors["source_index"] <= self.factors["num_nodes"]

    def check_t(self):
        return self.factors["sink_index"] >= 0 and self.factors["sink_index"] <= self.factors["num_nodes"]

    def replicate(self, rng_list):
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
        solver = max_flow.SimpleMaxFlow()
        exp_rng = rng_list[0]
        # From input graph generate start end end nodes.
        start_nodes = []
        end_nodes = []
        for i, j in self.factors["arcs"]:
            start_nodes.append(i)
            end_nodes.append(j)
        # Generate actual capacity.
        for i in range(len(self.factors["arcs"])):
            noise = exp_rng.mvnormalvariate(self.factors["mean_noise"], np.array(self.factors["cov_noise"]))
        capacities = []
        for i in range(len(noise)):
            capacities.append(max(1000 * (self.factors["assigned_capacities"][i] - noise[i]), 0))
        # Add arcs in bulk.
        solver.add_arcs_with_capacity(start_nodes, end_nodes, capacities)
        status = solver.solve(self.factors["source_index"], self.factors["sink_index"])
        if status != solver.OPTIMAL:
            print('There was an issue with the max flow input.')
            print(f'Status: {status}')
            exit(1)

        # Construct gradient vector (=1 if has a outflow from min-cut nodes).
        gradient = np.zeros(len(self.factors["arcs"]))
        grad_arclist = []
        min_cut_nodes = solver.get_source_side_min_cut()
        for i in min_cut_nodes:
            for j in range(self.factors['num_nodes']):
                if j not in min_cut_nodes:
                    grad_arc = (i, j)
                    if (i, j) in self.factors['arcs']:
                        grad_arclist.append(grad_arc)
        for arc in grad_arclist:
            gradient[self.factors['arcs'].index(arc)] = 1

        responses = {"Max Flow": solver.optimal_flow() / 1000}
        gradients = {response_key: {factor_key: np.nan for factor_key in self.specifications} for response_key in responses}
        gradients["Max Flow"]["assigned_capacities"] = gradient
        return responses, gradients


"""
Summary
-------
Maximize the expected max flow from the source node s to the sink node t.
"""


class SMF_Max(Problem):
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
    Ci : ndarray (or None)
        Coefficient matrix for linear inequality constraints of the form Ci@x <= di
    Ce : ndarray (or None)
        Coefficient matrix for linear equality constraints of the form Ce@x = de
    di : ndarray (or None)
        Constraint vector for linear inequality constraints of the form Ci@x <= di
    de : ndarray (or None)
        Constraint vector for linear equality constraints of the form Ce@x = de
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
    def __init__(self, name="SMF-1", fixed_factors=None, model_fixed_factors=None, random=False):
        if fixed_factors is None:
            fixed_factors = {}
        if model_fixed_factors is None:
            model_fixed_factors = {}
        self.name = name
        self.n_objectives = 1
        self.n_stochastic_constraints = 0
        self.minmax = (1, )
        self.constraint_type = "deterministic"
        self.variable_type = "continuous"
        self.gradient_available = True
        self.optimal_value = None
        self.optimal_solution = None
        self.random = random
        self.model_default_factors = {}
        self.model_decision_factors = {"assigned_capacities"}
        self.factors = fixed_factors
        self.specifications = {
            "initial_solution": {
                "description": "initial solution",
                "datatype": tuple,
                "default": (1, ) * 20
            },
            "budget": {
                "description": "max # of replications for a solver to take",
                "datatype": int,
                "default": 10000
            },
            "cap": {
                "description": "total set-capacity to be allocated to arcs.",
                "datatype": int,
                "default": 100
            }
        }
        self.check_factor_list = {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget,
            "cap": self.check_cap
        }
        super().__init__(fixed_factors, model_fixed_factors)
        # Instantiate model with fixed factors and over-riden defaults.
        self.model = SMF(self.model_fixed_factors)
        self.dim = len(self.model.factors["arcs"])
        self.lower_bounds = (0, ) * self.dim
        # self.upper_bounds = (np.inf, ) * self.dim
        self.upper_bounds = (self.factors["cap"], ) * self.dim
        self.Ci = np.ones(20)
        self.Ce = None
        self.di = np.array([self.factors["cap"]])
        self.de = None

    def check_cap(self):
        return self.factors["cap"] >= 0
    
    def attach_rngs(self, random_rng):
        self.random_rng = random_rng
        self.model.attach_rng(random_rng)
        return random_rng

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
            "assigned_capacities": vector[:]
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
        vector = tuple(factor_dict["assigned_capacities"])
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
        objectives = (response_dict["Max Flow"], )
        return objectives

    def response_dict_to_stoch_constraints(self, response_dict):
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
        stoch_constraints = None
        return stoch_constraints

    def deterministic_stochastic_constraints_and_gradients(self, x):
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
        det_stoch_constraints = None
        det_stoch_constraints_gradients = None
        return det_stoch_constraints, det_stoch_constraints_gradients

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
        det_objectives = (0, )
        det_objectives_gradients = ((0, ) * self.dim,)
        return det_objectives, det_objectives_gradients

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

        return sum(self.factors["assigned_capacities"]) <= self.factors["cap"]

    def get_random_solution(self, rand_sol_rng):
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
        x = rand_sol_rng.continuous_random_vector_from_simplex(len(self.model.factors["arcs"]), self.factors["cap"], False)
        return x
