"""
Summary
-------
"""
import numpy as np
#import cvxpy as cp
import simopt

#from base import Model, Problem
from simopt.base import Model, Problem
#from ..base import Model, Problem


class conSAN_model(Model):
    """
    A model that simulates a stochastic activity network problem with
    tasks that have exponentially distributed durations, and the selected
    means come with a cost.
    
    A model that simulates a stochastic activity network problem with
    tasks that have exponentially distributed durations, and the selected
    means come with a cost. If each arc's time is x_i, then the constraint
    is Ax <= b

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
        self.name = "SAN"
        self.n_rngs = 1
        self.n_responses = 1
        self.specifications = {
            "num_nodes": {
                "description": "number of nodes",
                "datatype": int,
                "default": 9
            },
            "arcs": {
                "description": "list of arcs",
                "datatype": list,
                "default": [(1, 2), (1, 3), (2, 3), (2, 4), (2, 6), (3, 6), (4, 5),
                            (4, 7), (5, 6), (5, 8), (6, 9), (7, 8), (8, 9)]
            },
            "arc_means": {
                "description": "mean task durations for each arc",
                "datatype": tuple,
                "default": (1,) * 13
            },
            "arc_costs": {
                "description": "Cost associated to each arc.",
                "datatype": tuple,
                "default": self.default_cost_f
                #"default": (1,) * 13
            },
            
            "cost_grad":{
                "description": "Gradient of the cost function.",
                "datatype": "function",
                "default": self.default_grad_cost_f
            }
        }
        self.check_factor_list = {
            "num_nodes": self.check_num_nodes,
            "arcs": self.check_arcs,
            "arc_means": self.check_arc_means
        }
        # Set factors of the simulation model.
        super().__init__(fixed_factors)
        self.num_arcs = len(self.factors['arcs'])
        self.cost_f = self.factors['arc_costs']
        self.grad_cost_f = self.factors['cost_grad']

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
        # Check graph is connected.
        graph = {node: set() for node in range(1, self.factors["num_nodes"] + 1)}
        for a in self.factors["arcs"]:
            graph[a[0]].add(a[1])
        visited = self.dfs(graph, 1)
        if self.factors["num_nodes"] in visited:
            return True
        return False

    def check_arc_means(self):
        positive = True
        for x in list(self.factors["arc_means"]):
            positive = positive & (x > 0)
        return (len(self.factors["arc_means"]) == len(self.factors["arcs"])) & positive
    
    def default_cost_f(self,x):
        """
        return the default cost function which is sum(1/x)
        
        x: a vector of arc means
        """
        
        return sum(1/np.array(x))
    
    def default_grad_cost_f(self,x):
        """
        return the default gradient of cost function which is sum(1/x)
        
        x: a vector of arc means
        """
        
        return tuple(-1/np.array(x)**2)
    
    def get_cost(self,x):
        
        return self.cost_f(x)
    
    def get_grad_cost(self,x):
        
        return self.grad_cost_f(x)

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
        exp_rng = rng_list[0]
        
        #print(self.check_factor_list["arc_means"]())
        #if(not self.check_factor_list["arc_means"]()):
        #    means = np.array(self.factors["arc_means"])
            
        #    if((means < -1e-5).any()):
        #        raise ValueError("Negative arc means")
        #    else:
        #        means[means < 1e-10] = 1e-8
        #        self.factors["arc_means"] = tuple(means)

        # Topological sort.
        graph_in = {node: set() for node in range(1, self.factors["num_nodes"] + 1)}
        graph_out = {node: set() for node in range(1, self.factors["num_nodes"] + 1)}
        for a in self.factors["arcs"]:
            graph_in[a[1]].add(a[0])
            graph_out[a[0]].add(a[1])
        indegrees = [len(graph_in[n]) for n in range(1, self.factors["num_nodes"] + 1)]

        queue = []
        topo_order = []
        for n in range(self.factors["num_nodes"]):
            if indegrees[n] == 0:
                queue.append(n + 1)
        while len(queue) != 0:
            u = queue.pop(0)
            topo_order.append(u)
            for n in graph_out[u]:
                indegrees[n - 1] -= 1
                if indegrees[n - 1] == 0:
                    queue.append(n)

        # Generate arc lengths.
        arc_length = {}
        for i in range(len(self.factors["arcs"])):
            arc_length[str(self.factors["arcs"][i])] = exp_rng.expovariate(1 / self.factors["arc_means"][i])
        
        #print(self.factors["arc_means"])
        #print(arc_length)
        # Calculate the length of the longest path.
        T = np.zeros(self.factors["num_nodes"])
        prev = np.zeros(self.factors["num_nodes"])
        for i in range(1, self.factors["num_nodes"]):
            vi = topo_order[i - 1]
            for j in graph_out[vi]:
                if T[j - 1] < T[vi - 1] + arc_length[str((vi, j))]:
                    T[j - 1] = T[vi - 1] + arc_length[str((vi, j))]
                    prev[j - 1] = vi
        longest_path = T[self.factors["num_nodes"] - 1]

        # Calculate the IPA gradient w.r.t. arc means.
        # If an arc is on the longest path, the component of the gradient
        # is the length of the length of that arc divided by its mean.
        # If an arc is not on the longest path, the component of the gradient is zero.
        gradient = np.zeros(len(self.factors["arcs"]))
        current = topo_order[-1]
        backtrack = int(prev[self.factors["num_nodes"] - 1])
        #print("prev: ", prev)
        while current != topo_order[0]:
            idx = self.factors["arcs"].index((backtrack, current))
            gradient[idx] = arc_length[str((backtrack, current))] / (self.factors["arc_means"][idx])
            current = backtrack
            backtrack = int(prev[backtrack - 1])
        
        #lengths = np.array([arc_length[str(self.factors["arcs"][i])] for i in range(self.num_arcs)])
        # Compose responses and gradients.
        responses = {"longest_path_length": longest_path + self.get_cost(self.factors['arc_means'])}
        gradients = {response_key: {factor_key: np.nan for factor_key in self.specifications} for response_key in responses}
        gradients["longest_path_length"]["arc_means"] = gradient + self.get_grad_cost(self.factors['arc_means'])
        return responses, gradients

    


class conSAN_problem(Problem):
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
    def __init__(self, name="conSAN", fixed_factors=None, model_fixed_factors=None):
        if fixed_factors is None:
            fixed_factors = {}
        if model_fixed_factors is None:
            model_fixed_factors = {}
        self.name = name
        self.n_objectives = 1
        self.n_stochastic_constraints = 0
        self.minmax = (-1,)
        self.constraint_type = "box"
        self.variable_type = "continuous"
        self.gradient_available = True
        self.optimal_value = None
        self.optimal_solution = None
        self.model_default_factors = {}
        self.model_decision_factors = {"arc_means"}
        self.factors = fixed_factors
        self.specifications = {
            "initial_solution": {
                "description": "initial solution",
                "datatype": tuple,
                "default": (8,) * 13
            },
            "budget": {
                "description": "max # of replications for a solver to take",
                "datatype": int,
                "default": 10000
            },
            "arc_costs": {
                "description": "Cost associated to each arc.",
                "datatype": tuple,
                "default": self.default_cost_f
                #"default": (1,) * 13
            },
            
            "cost_grad":{
                "description": "Gradient of the cost function.",
                "datatype": "function",
                "default": self.default_grad_cost_f
            },
            #"sum_lb": {
            #    "description": "Lower bound for the sum of arc means",
            #    "datatype": float,
            #    "default": 100.0
            #},
            "Ci":{
                "description": "Coefficients for inequality constraints Ci@x <= di",
                "datatype": "matrix",
                "default": -1 * np.eye(13)
            },
            "di":{
                "description": "RHS for inequality constraints Ci@x <= di",
                "datatype": "vector",
                "default": 0
            },
            "Ce":{
                "description": "Coefficients for equality constraints Ce@x == de",
                "datatype": "matrix",
                "default": None
            },
            "de":{
                "description": "RHS for equality constraints Ce@x == de",
                "datatype": "vector",
                "default": None
            } 
        }
        self.check_factor_list = {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget,
            "arc_costs": self.check_arc_costs
        }
        super().__init__(fixed_factors, model_fixed_factors)
        # Instantiate model with fixed factors and over-riden defaults.
        self.model = conSAN_model(self.model_fixed_factors)
        self.dim = len(self.model.factors["arcs"])
        self.lower_bounds = (1e-2,) * self.dim
        #self.upper_bounds = (np.inf,) * self.dim
        self.upper_bounds = (1e4,) * self.dim
        #self.Ci = -1 * np.ones(13)
        self.Ci = self.factors["Ci"]
        self.Ce = self.factors["Ce"]
        #self.di = -1 * np.array([self.factors["sum_lb"]])
        self.di = self.factors["di"]
        self.de = self.factors["de"]
        
        self.num_arcs = len(self.model.factors['arcs'])
        self.cost_f = self.model.factors['arc_costs']
        self.grad_cost_f = self.model.factors['cost_grad']

    def default_cost_f(self,x):
        """
        return the default cost function which is sum(1/x)
        
        x: a vector of arc means
        """
        
        return sum(1/x)
    
    def default_grad_cost_f(self,x):
        """
        return the default gradient of cost function which is sum(1/x)
        
        x: a vector of arc means
        """
        
        return -sum(1/x**2)
    
    def check_arc_costs(self):
        positive = True
        for x in list(self.factors["arc_costs"]):
            positive = positive & x > 0
        return (len(self.factors["arc_costs"]) != self.model.factors["num_arcs"]) & positive

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
            "arc_means": vector[:]
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
        vector = tuple(factor_dict["arc_means"])
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
        objectives = (response_dict["longest_path_length"],)
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
        det_stoch_constraints_gradients = ((0,) * self.dim,)  # tuple of tuples – of sizes self.dim by self.dim, full of zeros
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
        #det_objectives = (0,)
        det_objectives = (self.get_cost(x),)
        #det_objectives_gradients = ((0,) * self.dim,)
        det_objectives_gradients = (tuple(self.grad_cost_f(x)),)
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
        
        is_satisfy = True
        
        if((self.Ci is not None) and (self.di is not None)):
            is_satisfy = is_satisfy and (self.Ci.dot(x) <= self.di)
            
        if((self.Ce is not None) and (self.de is not None)):
            is_satisfy = is_satisfy and (self.Ce.dot(x) <= self.de)
        
        return is_satisfy
        #return self.Ci.dot(x) <= self.di
        
        #return np.all(np.array(x) >= 0)

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
        #while True:
        #    x = [rand_sol_rng.lognormalvariate(lq = 0.1, uq = 10) for _ in range(self.dim)]
        #    if np.sum(x) >= self.factors['sum_lb']:
        #        break
        #x= tuple(x)
        x = tuple([rand_sol_rng.lognormalvariate(lq = 0.1, uq = 10) for _ in range(self.dim)])
        return x
    
    def get_cost(self,x):
        """
        Generate a random solution for starting or restarting solvers.

        Arguments
        ---------
        x: current arc inputs

        Returns
        -------
        total cost: cost from purchasing the arcs
        """
        
        total_cost = self.model.factors['arc_costs'](x)
        
        return total_cost
