"""
Summary
-------
Simulate duration of a stochastic activity network (SAN).
A detailed description of the model/problem can be found
`here <https://simopt.readthedocs.io/en/latest/san.html>`_.
"""
import numpy as np
from scipy.optimize import linprog

from ..base import Model, Problem


class SAN(Model):
    """
    A model that simulates a stochastic activity network problem with
    tasks that have exponentially distributed durations, and the selected
    means come with a cost.

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
    def __init__(self, fixed_factors=None, random=False):
        if fixed_factors is None:
            fixed_factors = {}
        self.name = "SAN"
        self.n_rngs = 1
        self.n_responses = 1
        self.n_random = 2  # Number of rng used for the random instance
        self.random = random
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
            "num_arcs": {
                "description": "number of arcs to be generated",
                "datatype": int,
                "default": 13
            },
            "set_arcs": {
                "description": "list of all possible arcs",
                "datatype": list,
                "default": [(1, 2), (1, 3),(1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9),
                            (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), 
                            (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), 
                            (4, 5), (4, 6), (4, 7), (4, 8), (4, 9),
                            (5, 6), (5, 7), (5, 8), (5, 9), 
                            (6, 7), (6, 8), (6, 9),
                            (7, 8), (7, 9),
                            (8, 9)]
            }
        }
        self.check_factor_list = {
            "num_nodes": self.check_num_nodes,
            "arcs": self.check_arcs,
            "arc_means": self.check_arc_means,
            "num_arcs": self.check_num_arcs,
            "set_arcs": self.check_set_arcs
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
    
    def check_num_arcs(self):
        return self.factors["num_arcs"] > 0
    
    def check_set_arcs(self):
        return True 
    
    def allPathsStartEnd(self, graph):
        end = len(graph)
        
        def dfs(node, path, output):
            if node == end:
                output.append(path)
            
            for nx in graph[node]:
                dfs(nx, path+[nx], output)
        
        output = []
        dfs(1,[1],output)
        return output
    
    def get_arcs(self, num_nodes, num_arcs, uni_rng):
        """
        Getting a random set of valid arcs.

        Arguments
        ---------
        num_nodes: int
            number of nodes for the random graph
        num_arcs: int
            number of arcs for the random graph
        rng_list : list of mrg32k3a.mrg32k3a.MRG32k3a
            rngs for model to use when simulating a replication

        Returns
        -------
        arcs : list
            Generated random arcs to be used in the following simulation
        """
        # Calculate the total set of possible arcs in the graph
        set_arcs = []
        for n1 in range(1, num_nodes):
            for n2 in range(n1 + 1, num_nodes + 1):
                set_arcs.append((n1, n2))
        
        # Assign the arcs set with the necessary arcs    
        arcs = [(1, 2), (num_nodes - 1, num_nodes)]
        remove = []
        def get_in(arcs, num_nodes, ind, in_ind=True):
            global remove
            if len(arcs) <= 0:
                return False            
            graph = {node: set() for node in range(1, num_nodes + 1)}
            for a in arcs:
                if in_ind == True:
                    graph[a[0]].add(a[1])
                else:
                    graph[a[1]].add(a[0])
            set0 = graph[ind]
            for i in graph[ind]:
                set0 = {*set0, *graph[i]}
                for j in graph[i]:
                    set0 = {*set0, *graph[j]}
            
            if in_ind == True:      
                for j in set0 - graph[ind]:
                    if j in graph[ind]:
                        remove.append((ind, j))
            
            set0 = {*set0, ind}
            return set0
        
        # Check whether the first node can reach all other nodes
        set0 = get_in(arcs, num_nodes, 1)
        for i in range(2, num_nodes+1):
            set0 = get_in(arcs, num_nodes, 1)  # Get the set of nodes that starter node can reach
            if i not in set0:
                set1 = list(get_in(arcs, num_nodes, i, False))  # Get the set of nodes that can reach node i
                n2 = set1[uni_rng.randint(0, len(set1)-1)]  # Randomly choose one
                set2 = [i for i in set0 if i < n2]
                n1 = list(set2)[uni_rng.randint(0, len(set2)-1)]
                arc = (n1, n2)  # Connect the two nodes so that starter node can reach node i
                arcs = {*arcs, arc}
        
        # Check whether each node can reach the end node
        for i in range(2, num_nodes):
            set9 = get_in(arcs, num_nodes, i)
            if num_nodes not in set9:
                set_out = list(get_in(arcs, num_nodes, num_nodes, False))
                n1 = list(set9)[uni_rng.randint(0, len(set9)-1)]
                set2 = [i for i in set_out if i > n1]
                n2 = set2[uni_rng.randint(0, len(set2)-1)]
                arc = (n1, n2)
                arcs = {*arcs, arc}
        
        if len(arcs) < num_arcs:  # If the current arc set has less arcs than the input lower bound
            remain_num = num_arcs - len(arcs)
            remain = list(set(set_arcs) - set(arcs))
            idx = uni_rng.sample(range(0, len(remain)), remain_num)
            aa = set([remain[i] for i in idx])
            arcs = {*arcs, *aa}

        else:
            return list(arcs)
        
        return list(arcs)
    
    def attach_rng(self, random_rng):
        """
        Attach rng to random model class and generate random factors and update corresponding problem dimension.

        Arguments
        ---------
        random_rng : list of mrg32k3a.mrg32k3a.MRG32k3a
            rngs for model to use when generating random factors

        Returns
        -------
        arcs : list
            Generated random arcs to be used in the following simulation
        """
        self.random_rng = random_rng
        arcs_set = self.get_arcs(self.factors["num_nodes"], self.factors["num_arcs"], random_rng[0])
        
        arcs_set.sort(key=lambda a: a[1])
        arcs_set.sort(key=lambda a: a[0])  
        self.factors["arcs"] = arcs_set
        print('arcs: ', arcs_set)
        self.factors["num_arcs"] = len(self.factors["arcs"])
        self.factors["arc_means"] = (1,) * len(self.factors["arcs"])

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

        ## Calculate the length of the longest path.
        allpaths = self.allPathsStartEnd(graph_out)
        L = []
        for p in allpaths:
            l = 0
            for j in range(len(p)-1):
                l += arc_length[str((p[j], p[j+1]))]
            L.append(l)
        longest_path = np.max(L)
        longest_P = allpaths[np.argmax(L)]

        # Calculate the IPA gradient w.r.t. arc means.
        # If an arc is on the longest path, the component of the gradient
        # is the length of the length of that arc divided by its mean.
        # If an arc is not on the longest path, the component of the gradient is zero.
        gradient = np.zeros(len(self.factors["arcs"]))

        for i in range(len(longest_P)-1,0,-1):
            backtrack = longest_P[i-1]
            current = longest_P[i]
            idx = self.factors["arcs"].index((backtrack, current))
            gradient[idx] = arc_length[str((backtrack, current))] / (self.factors["arc_means"][idx])

        # Compose responses and gradients.
        responses = {"longest_path_length": longest_path}
        gradients = {response_key: {factor_key: np.nan for factor_key in self.specifications} for response_key in responses}
        gradients["longest_path_length"]["arc_means"] = gradient
        return responses, gradients


"""
Summary
-------
Minimize the duration of the longest path from a to i plus cost.
"""


class SANLongestPath(Problem):
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
            initial_solution : list
                default initial solution from which solvers start
            budget : int > 0
                max number of replications (fn evals) for a solver to take
    specifications : dict
        details of each factor (for GUI, data validation, and defaults)
    random: bool
        indicator of whether user want to build a random problem or a deterministic model
    n_rng: int
        Number of random number generator needed to build a random problem instance

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
    def __init__(self, name="SAN-1", fixed_factors=None, model_fixed_factors=None, random=False, random_rng=None):
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
        self.random = random  # Randomlize problem and model or not
        self.n_rngs = 3  # Number of rngs used for the random instance
        self.specifications = {
            "initial_solution": {
                "description": "initial solution",
                "datatype": tuple,
                "default": (8,) * 13
            },
            "budget": {
                "description": "max # of replications for a solver to take",
                "datatype": int,
                "default": 100000
            },
            "c": {
                "description": "cost associated to each arc",
                "datatype": tuple,
                "default": (1,) * 13
            }
        }
        self.check_factor_list = {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget,
            "c": self.check_arc_costs
        }
        super().__init__(fixed_factors, model_fixed_factors)
        # Instantiate model with fixed factors and over-riden defaults.
        self.model = SAN(self.model_fixed_factors, random)
        # If random, generate random model factors and update model class
        if random==True and random_rng != None:
            self.model.attach_rng(random_rng)
        self.dim = len(self.model.factors["arcs"])
        # Update every value and dimension according to the randomly generated case
        self.factors["initial_solution"] = (8,) * self.dim
        self.factors["c"] = (1,) * self.dim
        self.lower_bounds = (1e-2,) * self.dim
        self.upper_bounds = (np.inf,) * self.dim
        self.Ci = None
        self.Ce = None
        self.di = None
        self.de = None
    
    def check_arc_costs(self):
        """
        Check if the arc costs are positive.

        Returns
        -------
        bool
            indicates if arc costs are positive
        """
        positive = True
        for x in list(self.factors["c"]):
            positive = positive & x > 0
        return (len(self.factors["c"]) != self.dim) & positive
    
    def check_budget(self):
        """
        Check if the budget is positive.

        Returns
        -------
        bool
            indicates if the budget is positive
        """
        return self.factors["budget"] > 0

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
    
    def get_coefficient(self, exp_rng):
        """
        Generate random coefficients for each arc.

        Arguments
        ---------
        exp_rng : mrg32k3a.mrg32k3a.MRG32k3a object
            random-number generator used to sample random coefficients

        Returns
        -------
        c : list
            vector of coefficients
        """
        c = []
        for i in range(len(self.factors["c"])):
            ci = exp_rng.expovariate(1)
            c.append(ci)

        return c
    
    def random_budget(self, uni_rng):
        """
        Generate random budget for the problem, proportional to the dimension.

        Arguments
        ---------
        uni_rng : mrg32k3a.mrg32k3a.MRG32k3a object
            random-number generator used to sample random budget
        """
        l = [100, 200, 300, 400, 500]
        budget = uni_rng.choice(l) * self.dim
        return budget
                       
    def attach_rngs(self, random_rng):
        """
        Attach random-number generators to the problem.

        Arguments
        ---------
        random_rng : list of mrg32k3a.mrg32k3a.MRG32k3a objects
            list of rngs for problem to use when generating random instances
        """
        # Attach rng for problem class and generate random problem factors for random instances
        self.random_rng = random_rng
        
        if self.random == True:
            self.factors["budget"] = self.random_budget(random_rng[0])
            self.factors["c"] = self.get_coefficient(random_rng[1])
            
        print('budget: ', self.factors['budget'])
        print('c: ', self.factors["c"])
        
        return random_rng

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
        det_objectives = (np.sum(np.array(self.factors["c"]) / np.array(x))/len(x),)
        det_objectives_gradients = (-np.array(self.factors["c"]) / np.array(x) ** 2 / len(x),)
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
        return np.all(np.array(x) >= 0)

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
            vector of decision variables·
        """
        x = tuple([rand_sol_rng.lognormalvariate(lq=0.1, uq=10) for _ in range(self.dim)])
        return x


"""
Summary
-------
Minimize the duration of the longest path from a to i subject to some lower bounds in sum of arc_means.
"""

class SANLongestPathConstr(Problem):
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
            initial_solution : list
                default initial solution from which solvers start
            budget : int > 0
                max number of replications (fn evals) for a solver to take
    specifications : dict
        details of each factor (for GUI, data validation, and defaults)
    random: bool
        indicator of whether user want to build a random problem or a deterministic model
    n_rng: int
        Number of random number generator needed to build a random problem instance
    random_const: bool
        indicator of whether to generate random constraints for the random problem instance or not

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
    def __init__(self, name="SAN-2", fixed_factors=None, model_fixed_factors=None, random=False, random_rng=None):
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
        self.random = random
        self.random_const = False  # Turn on if want to generate random constraints for random problem instance
        if self.random_const:
            self.num_con = 3  # Number of random constraints to generate
        else:
            self.num_con = 1
        self.n_rngs = 3  # Number of rngs used for the random instance
        self.specifications = {
            "initial_solution": {
                "description": "initial solution",
                "datatype": tuple,
                "default": (15,) * 13
            },
            "budget": {
                "description": "max # of replications for a solver to take",
                "datatype": int,
                "default": 100000
            },
            "arc_costs": {
                "description": "cost associated to each arc",
                "datatype": tuple,
                "default": (1,) * 13
            },
            "r_const": {
                "description": "random constraint for arc rates",
                'datatype': int,
                "default": 0
            },
            "sum_lb": {
                "description": "Lower bound for the sum of arc means",
                "datatype": float,
                "default": 100.0
            },
            "lbs":{
                "description": "Lower bounds for the selected sum of arc means",
                "datatype": float,
                "default": 0.0,
            }
        }
        self.check_factor_list = {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget,
            "arc_costs": self.check_arc_costs,
            "r_const": self.check_const,
            "sum_lb": self.check_lb,
            "lbs": self.check_lbs
        }
        super().__init__(fixed_factors, model_fixed_factors)
        # Instantiate model with fixed factors and over-riden defaults.
        self.model = SAN(self.model_fixed_factors, random)
        if random==True and random_rng != None:
            self.model.attach_rng(random_rng)
        self.dim = len(self.model.factors["arcs"])
        self.factors["initial_solution"] = (15,) * self.dim
        self.factors["arc_costs"] = (1,) * self.dim 
        self.lower_bounds = (1e-2,) * self.dim
        self.upper_bounds = (np.inf,) * self.dim
        self.Ci = -1 * np.ones(self.dim)
        self.Ce = None
        self.di = -1 * np.array([self.factors["sum_lb"]])
        self.de = None
    
    def check_arc_costs(self):
        """
        Check if the arc costs are positive.

        Returns
        -------
        bool
            indicates if arc costs are positive
        """
        positive = True
        for x in list(self.factors["arc_costs"]):
            positive = positive & x > 0
        return (len(self.factors["arc_costs"]) != self.dim) & positive
    
    def check_budget(self):
        """
        Check if the budget is positive.

        Returns
        -------
        bool
            indicates if the budget is positive
        """
        return self.factors["budget"] > 0
    
    def check_const(self):
        """
        Check if the random constraint is positive.

        Returns
        -------
        bool
            indicates if the random constraint is positive
        """
        return self.factors["r_const"] >= 0
    
    def check_lb(self):
        """
        Check if the lower bound for sum of all arc rates is positive.

        Returns
        -------
        bool
            indicates if the lower bound is positive
        """
        return self.factors["sum_lb"] >= 0
    
    def check_lbs(self):
        """
        Check if other potential lower bound is positive.

        Returns
        -------
        bool
            indicates if other lower bound is positive
        """
        return self.factors["lbs"] >= 0

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
    
    def get_coefficient(self, exp_rng):
        """
        Generate random coefficients for each arc.

        Arguments
        ---------
        exp_rng : mrg32k3a.mrg32k3a.MRG32k3a object
            random-number generator used to sample random coefficients
        """
        if self.random == True:
            c = []
            for i in range(len(self.factors["arc_costs"])):
                ci = exp_rng.expovariate(1)
                c.append(ci)
            return c
        else:
            return self.factors['arc_costs']
    
    def random_budget(self, random_rng):
        """
        Generate random budget for the problem, proportional to the dimension.

        Arguments
        ---------
        uni_rng : mrg32k3a.mrg32k3a.MRG32k3a object
            random-number generator used to sample random budget
        """
        if self.random == True:
            l = [10000, 20000]
            budget = random_rng.choice(l) * self.dim
            return budget
        else:
            return self.factors['budget']
    
    def get_const(self, n, uni_rng):
        """
        Generate random constraint for the problem, proportional to the dimension.

        Arguments
        ---------
        n : int
            number of constraints want to generate
        uni_rng : mrg32k3a.mrg32k3a.MRG32k3a object
            random-number generator used to sample random budget
        """
        # Randomly choose a subset of arcs that have limited budget
        C = []
        L = []
        for i in range(n):
            if self.random_const == True:
                const = uni_rng.sample(range(0, self.dim), int(self.dim/4))
                # lb = uni_rng.uniform(0, int(self.dim/4)) * uni_rng.uniform(1, 6)
                lb = int(self.dim/4) * uni_rng.uniform(1, int(self.factors["sum_lb"]/self.dim))
                C.append(const)
                L.append(lb)
            else:
                return [[i for i in range(self.dim)]], self.factors['sum_lb']
        return C, L
                       
    def attach_rngs(self, random_rng):
        """
        Attach random-number generators to the problem.

        Arguments
        ---------
        random_rng : list of mrg32k3a.mrg32k3a.MRG32k3a objects
            list of rngs for problem to use when generating random instances
        """
        # Attach rng for problem class and generate random problem factors for random instances
        self.random_rng = random_rng
        
        if self.random:
            self.factors["budget"] = self.random_budget(random_rng[0])
            
            self.factors["arc_costs"] = self.get_coefficient(random_rng[1])
            print('c: ', self.factors["arc_costs"])
        
            # Random constraint
            if self.random_const:
                self.factors["r_const"], self.factors['lbs'] = self.get_const(self.num_con, random_rng[2])
                self.factors["lbs"].append(self.factors["sum_lb"])  # Combine the sum_lb with the partial_lb
                self.factors["r_const"].append([i for i in range(self.dim)])  # Combine the index related to sum_lb with the r_const
            else:
                self.factors["r_const"], self.factors['sum_lb'] = self.get_const(self.num_con, random_rng[2])
                self.factors["lbs"] = [self.factors["sum_lb"]]
        else:
            self.factors["r_const"] = [[i for i in range(self.dim)]]
            self.factors["lbs"] = [self.factors["sum_lb"]]
        
        self.factors["lbs"] += [0 for i in range(self.dim)]  # Require each arc means larger or equal to 0
        self.factors["r_const"] += [[i] for i in range(self.dim)]
        print('r_const: ', self.factors["r_const"])
        print('lbs: ', self.factors['lbs'])
        
        return random_rng

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

        det_objectives = (0,)
        det_objectives_gradients = ((0,) * self.dim,)
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

        return np.all(np.array(x) >= 0)
    
    def find_feasible(self):
        """
        Find an initial feasible solution (if not user-provided)
        by interior point method.

        Arguments
        ---------
        problem : Problem object
            simulation-optimization problem to solve

        Returns
        -------
        x0 : ndarray
            an initial feasible solution
        tol: float
            Floating point comparison tolerance
        """
        c = [0 for i in range(self.dim)]
        l1 = [-1 for i in range(self.dim)]
        A = [l1]
        b = [-self.factors["sum_lb"]]
        if self.random_const:
            b.extend([-plb for plb in self.factors["lbs"]])
            for idx in self.factors["r_const"]:
                l2 = [-1 if i in idx else 0 for i in range(self.dim)]
                A.append(l2)

        res = linprog(c, A_ub=A, b_ub=b, bounds=(0, None), method='interior-point')
        
        return res.x
   
    def check_feasible(self, x):
        """
        Check whether a solution is feasible or not.

        Arguments
        ---------
        x : ndarray/list
            current point

        Returns
        -------
        feasible : bool
            indicates if solution `x` satisfies the deterministic constraints.
        """
        if sum(x) >= self.factors['sum_lb']:
            for i in range(len(self.factors["lbs"])):
                if sum([x[j] for j in self.factors["r_const"][i]]) < self.factors["lbs"][i]:
                    return False
            return True
        else:
            return False
    
    def get_random_solution(self, rand_sol_rng):
        """
        Generate a random solution for starting or restarting solvers.

        Arguments
        ---------
        rand_sol_rng : mrg32k3a.mrg32k3a.MRG32k3a
            random-number generator used to sample a new random solution

        Returns
        -------
        x : tuple
            vector of decision variables
        """
        if self.check_feasible(self.factors["initial_solution"]):
            x0 = self.factors["initial_solution"]
        else:
            x0 = self.find_feasible()
        x = rand_sol_rng.hit_and_run(x0, [10 * self.factors['sum_lb']], [[i for i in range(self.dim)]], self.factors["lbs"], self.factors["r_const"], self.dim, 20)

        x = tuple(x)
        return x