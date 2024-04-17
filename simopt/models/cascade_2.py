"""
Summary
-------
Simulate a progressive cascade process in an infinite time horizon.
`here <https://simopt.readthedocs.io/en/latest/cascade.html>`_.

"""
import numpy as np
import networkx as nx
import cvxpy as cp

from ..base import Model, Problem

class Cascade(Model):
    """
    Simulate a progressive cascade process in an infinite time horizon.

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

    Arguments
    ----------
    fixed_factors : dict
        fixed_factors of the simulation model

    See also
    --------
    base.Model
    """
    def __init__(self, fixed_factors=None, random=False):
        if fixed_factors is None:
            fixed_factors = {}
        self.name = "CASCADE"
        self.n_rngs = 2
        self.n_responses = 1
        self.factors = fixed_factors
        self.G = nx.read_graphml('/Users/liulitong/Desktop/simopt-1/DAG.graphml')
        self.num_nodes = len(self.G)
        self.random = random
        self.n_random = 1
        self.specifications = {
            "num_subgraph": {
                "description": "number of subgraphs to generate",
                "datatype": int,
                "default": 10
            },
            "init_prob": {
                "description": "probability of initiating the nodes",
                "datatype": np.ndarray,
                "default": 0.1 * np.ones(self.num_nodes)
            },
            "num_nodes":{
                "description": "number of nodes in each graph",
                "datatype": int,
                "default": 30
            },
            "num_edges": {
                "description": "number of edges in each graph",
                "datatype": int,
                "default": 100
            },
            "p": {
                "description": "probability",
                "datatype": float,
                "default": 0.4
            }
        }

        self.check_factor_list = {
            "num_subgraph": self.check_num_subgraph,
            "init_prob": self.check_init_prob,
            "num_nodes": self.check_num_nodes,
            "num_edges": self.check_num_edges
        }
        # Set factors of the simulation model
        super().__init__(fixed_factors)

    # Check for simulatable factors
    def check_num_subgraph(self):
        return self.factors["num_subgraph"] > 0
    
    def check_init_prob(self):
        return np.all(self.factors["init_prob"] >= 0)
    
    def check_num_nodes(self):
        return self.factors["num_nodes"] > 0
    
    def check_num_edges(self):
        return self.factors["num_edges"] > 0

    def check_simulatable_factors(self):
        return True
    
    # def generate_random_graph(self, uni_rng):
    #     num_nodes = self.factors["num_nodes"]
    #     num_edges = self.factors["num_edges"]
    #     # Check if the number of edges is sufficient to create a weakly connected DAG
    #     if num_edges < num_nodes - 1:
    #         raise ValueError("Number of edges is too few to form a weakly connected DAG with the given number of nodes.")

    #     # Create an empty adjacency matrix
    #     adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    #     # Add edges to ensure at least one incoming and one outgoing edge for each node, ensuring weak connectivity
    #     for i in range(1, num_nodes):
    #         adj_matrix[i-1, i] = 1  # Add edge from node i-1 to i

    #     # Remaining edges to randomly add, avoiding the main diagonal and the edges already added
    #     remaining_edges = num_edges - (num_nodes - 1)
    #     upper_triangle_indices = np.triu_indices(num_nodes, k=1)
    #     # Mask out the already added edges
    #     mask = np.ones(len(upper_triangle_indices[0]), dtype=bool)
    #     for i in range(1, num_nodes):
    #         mask &= ~((upper_triangle_indices[0] == i-1) & (upper_triangle_indices[1] == i))
    #     random_indices = uni_rng.choices(np.where(mask)[0], k = remaining_edges)#, replace=False)
    #     # print(random_indices)
    #     # print(len(random_indices))

    #     # Add the additional edges
    #     for idx in random_indices:
    #         adj_matrix[upper_triangle_indices[0][idx], upper_triangle_indices[1][idx]] = 1

    #     # Create a directed graph, add nodes with costs, and add edges
    #     G = nx.DiGraph()
    #     for i in range(num_nodes):
    #         cost = uni_rng.uniform(1, 100)  # Random cost for each node
    #         G.add_node(i, cost=cost)

    #     for i in range(num_nodes):
    #         for j in range(i + 1, num_nodes):
    #             if adj_matrix[i, j] == 1:
    #                 weight = uni_rng.uniform(0, 1)
    #                 G.add_edge(i, j, weight=weight)
                    
    #     return G
    
    def generate_random_graph(self, uni_rng):
        num_nodes = self.factors["num_nodes"]
        num_edges = self.factors["num_edges"]
        p = self.factors["p"]

        # Create a directed graph, add nodes with costs, and add edges
        G = nx.DiGraph()
        cs = []
        G.add_nodes_from(range(num_nodes))
        for node in G.nodes:
            G.nodes[node]['cost'] = round(uni_rng.uniform(1, 30), 4)
            cs.append(G.nodes[node]['cost'])
        
        arcs, ws, cnt = [], [], 0
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if uni_rng.uniform(0,1) < p:
                    weight = round(uni_rng.uniform(0, 1), 4)
                    if cnt == 0:
                        if weight < 0.1:
                            weight = round(uni_rng.uniform(0.2, 0.8), 4)
                    if weight <= 0.0001:
                        weight = round(uni_rng.uniform(0.005, 0.8), 4)
                    G.add_edge(i, j, weight = weight)
                    cnt += 1
                    arcs.append((i, j))
                    ws.append(weight)
        print(cs)
        print(arcs) 
        print(ws)
        return G
    
    def attach_rng(self, random_rng):
        self.random_rng = random_rng
        self.G = self.generate_random_graph(random_rng[0])

    def replicate(self, rng_list):
        """
        Simulate a single replication for the current model factors.

        Arguments
        ---------
        rng_list : [list]  [mrg32k3a.mrg32k3a.MRG32k3a]
            rngs for model to use when simulating a replication

        Returns
        -------
        responses : dict
            performance measures of interest
            "mean_num_activated" = Mean number of activated nodes
        """
        # Designate random number generators.
        seed_rng = rng_list[0]
        activate_rng = rng_list[1]

        nodes = list(self.G.nodes)
        num_lst = []
        for _ in range(self.factors["num_subgraph"]):
            # Create seed nodes.
            seeds = [nodes[j] for j in range(self.num_nodes) if seed_rng.uniform(0, 1) < self.factors["init_prob"][j]]
            # Set all nodes as not activated.
            activated = set()
            # Add the seed nodes to the activated set.
            activated.update(set(seeds))
            # Initialize the newly activated nodes list with the seed nodes.
            newly_activated = set(seeds)
            
            # Run the model until there are no more newly activated nodes.
            while len(newly_activated) != 0:
                temp_activated = set()
                for v in newly_activated:
                    # Check for each successor if it gets activated.
                    for w in self.G.successors(v):
                        if w not in activated:
                            u = activate_rng.uniform(0, 1)
                            # print(self.G[v][w])
                            if u < self.G[v][w]["weight"]:
                                temp_activated.add(w)
                # Add newly activated nodes to the activated set.
                newly_activated = temp_activated
                activated.update(newly_activated)

            
            num_activated = len(activated)
            num_lst.append(num_activated)
    

        # Calculate responses from simulation data.
        responses = {"mean_num_activated": np.mean(num_lst)
                     }
        gradients = {response_key: {factor_key: np.nan for factor_key in self.specifications} for response_key in responses}
        return responses, gradients


"""
Summary
-------
Maximize the expected number of activated nodes.
"""

class CascadeMax(Problem):
    """
    Class to make network cascade simulation-optimization problems.

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
    rng_list : [list]  [mrg32k3a.mrg32k3a.MRG32k3a]
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
    def __init__(self, name="CASCADE-1", fixed_factors=None, model_fixed_factors=None, random=False, random_rng=None):
        if fixed_factors is None:
            fixed_factors = {}
        if model_fixed_factors is None:
            model_fixed_factors = {}
        self.name = name
        self.n_objectives = 1
        self.n_stochastic_constraints = 0
        self.minmax = (1,)
        self.constraint_type = "deterministic"
        self.variable_type = "continuous"
        self.gradient_available = False
        self.optimal_value = None
        self.optimal_solution = None
        # self.G = nx.read_graphml('/Users/liulitong/Desktop/simopt-1/DAG.graphml')
        self.model_default_factors = {}
        self.model_decision_factors = {"init_prob"}
        self.factors = fixed_factors
        self.random = random
        self.n_rngs = 2  # Number of rngs used for random instance
        self.specifications = {
            "initial_solution": {
                "description": "initial solution",
                "datatype": tuple,
                "default": tuple(0.001 * np.ones(30))  #len(self.G)
            },
            "budget": {
                "description": "max # of replications for a solver to take",
                "datatype": int,
                "default": 10000
            },
            "B": {
                "description": "budget for the activation costs",
                "datatype": int,
                "default": 200
            }
        }
        self.check_factor_list = {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget
        }
        super().__init__(fixed_factors, model_fixed_factors)
        # Instantiate model with fixed factors and overwritten defaults.
        self.model = Cascade(self.model_fixed_factors, random)
        if random == True and random_rng != None:
            self.model.attach_rng(random_rng)
        self.G = self.model.G
        self.dim = len(self.model.G)
        self.lower_bounds = (0,) * self.dim
        self.upper_bounds = (1,) * self.dim
        self.Ci = np.array([self.model.G.nodes[node]["cost"] for node in self.model.G.nodes()])
        self.Ce = None
        self.di = np.array([self.factors["B"]])
        self.de = None

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
            "init_prob": vector[:]
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
        vector = (factor_dict["init_prob"],)
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
        objectives = (response_dict["mean_num_activated"],)
        return objectives
    
    def random_budget(self, uni_rng):
        # l = [1000, 2000, 3000]
        l = 1000
        # l = [600, 700, 800, 900, 1000]
        budget = uni_rng.choice(l) * self.dim
        return budget
    
    def get_cost(self, exp_rng):
        b0 = np.dot(self.Ci, self.factors['initial_solution']) * 10
        B = max(exp_rng.expovariate(1/self.factors["B"]), self.factors["B"]/5, b0)
        return B
    
    def attach_rngs(self, random_rng):
        self.random_rng = random_rng
        
        if self.random == True:
            self.factors["budget"] = self.random_budget(random_rng[0])
            self.factors["B"] = self.get_cost(random_rng[1])
            self.di = np.array([self.factors["B"]])
            
        print("Budget: ", self.factors['budget'])
        print("B: ", self.factors["B"])

    def response_dict_to_stoch_constraints(self, response_dict):
        """
        Convert a dictionary with response keys to a vector
        of left-hand sides of stochastic constraints: E[Y] <= 0

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
        det_objectives_gradients = ((0,),)
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
        return np.dot(self.Ci, x) <= self.factors["B"]

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

        # Upper bound and lower bound.
        lower_bound = np.array(self.lower_bounds)
        upper_bound = np.array(self.upper_bounds)
        # Input inequality and equlaity constraint matrix and vector.
        # Cix <= di
        # Cex = de
        Ci = self.Ci
        di = self.di
        Ce = self.Ce
        de = self.de

        # Remove redundant upper/lower bounds.
        ub_inf_idx = np.where(~np.isinf(upper_bound))[0]
        lb_inf_idx = np.where(~np.isinf(lower_bound))[0]

        # Form a constraint coefficient matrix where all the equality constraints are put on top and
        # all the bound constraints in the bottom and a constraint coefficient vector.  
        if (Ce is not None) and (de is not None) and (Ci is not None) and (di is not None):
            C = np.vstack((Ce,  Ci))
            d = np.vstack((de.T, di.T))
        elif (Ce is not None) and (de is not None):
            C = Ce
            d = de.T
        elif (Ci is not None) and (di is not None):
            C = Ci
            d = di.T
        else:
          C = np.empty([1, self.dim])
          d = np.empty([1, 1])
        
        if len(ub_inf_idx) > 0:
            C = np.vstack((C, np.identity(upper_bound.shape[0])))
            d = np.vstack((d, upper_bound[np.newaxis].T))
        if len(lb_inf_idx) > 0:
            C = np.vstack((C, -np.identity(lower_bound.shape[0])))
            d = np.vstack((d, -lower_bound[np.newaxis].T))

        # Hit and Run
        start_pt = self.find_feasible_initial(None, C, None, d)
        tol = 1e-6

        x = start_pt
        # Generate the markov chain for sufficiently long.
        for _ in range(20):
            # Generate a random direction to travel.
            direction = np.array([rand_sol_rng.uniform(0, 1) for _ in range(self.dim)])
            direction = direction / np.linalg.norm(direction)

            dir = direction
            ra = d.flatten() - C @ x
            ra_d = C @ dir
            # Initialize maximum step size.
            s_star = np.inf
            # Perform ratio test.
            for i in range(len(ra)):
                if ra_d[i] - tol > 0:
                    s = ra[i]/ra_d[i]
                    if s < s_star:
                        s_star = s

            dir = -direction
            ra = d.flatten() - C @ x
            ra_d = C @ dir
            # Initialize maximum step size.
            s_star2 = np.inf
            # Perform ratio test.
            for i in range(len(ra)):
                if ra_d[i] - tol > 0:
                    s = ra[i]/ra_d[i]
                    if s < s_star2:
                        s_star2 = s

            # Generate random point between lambdas.
            lam = rand_sol_rng.uniform(-1 * s_star2, s_star)

            # Compute the new point.
            x += lam * direction

        x= tuple(x)
        return x

    
    def get_multiple_random_solution(self, rand_sol_rng, n_samples):
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

        # Upper bound and lower bound.
        lower_bound = np.array(self.lower_bounds)
        upper_bound = np.array(self.upper_bounds)
        # Input inequality and equlaity constraint matrix and vector.
        # Cix <= di
        # Cex = de
        Ci = self.Ci
        di = self.di
        Ce = self.Ce
        de = self.de

        # Remove redundant upper/lower bounds.
        ub_inf_idx = np.where(~np.isinf(upper_bound))[0]
        lb_inf_idx = np.where(~np.isinf(lower_bound))[0]

        # Form a constraint coefficient matrix where all the equality constraints are put on top and
        # all the bound constraints in the bottom and a constraint coefficient vector.  
        if (Ce is not None) and (de is not None) and (Ci is not None) and (di is not None):
            C = np.vstack((Ce,  Ci))
            d = np.vstack((de.T, di.T))
        elif (Ce is not None) and (de is not None):
            C = Ce
            d = de.T
        elif (Ci is not None) and (di is not None):
            C = Ci
            d = di.T
        else:
          C = np.empty([1, self.dim])
          d = np.empty([1, 1])
        
        if len(ub_inf_idx) > 0:
            C = np.vstack((C, np.identity(upper_bound.shape[0])))
            d = np.vstack((d, upper_bound[np.newaxis].T))
        if len(lb_inf_idx) > 0:
            C = np.vstack((C, -np.identity(lower_bound.shape[0])))
            d = np.vstack((d, -lower_bound[np.newaxis].T))

        # Hit and Run
        start_pt = self.find_feasible_initial(None, self.Ci, None, self.di)
        xs = []
        x = start_pt
        tol = 1e-6

        # Generate the markov chain for sufficiently long.
        for _ in range(20 + n_samples):
            # Generate a random direction to travel.
            direction = np.array([rand_sol_rng.uniform(0, 1) for _ in range(self.dim)])
            direction = direction / np.linalg.norm(direction)

            dir = direction
            ra = d.flatten() - C @ x
            ra_d = C @ dir
            # Initialize maximum step size.
            s_star = np.inf
            # Perform ratio test.
            for i in range(len(ra)):
                if ra_d[i] - tol > 0:
                    s = ra[i]/ra_d[i]
                    if s < s_star:
                        s_star = s

            dir = -direction
            ra = d.flatten() - C @ x
            ra_d = C @ dir
            # Initialize maximum step size.
            s_star2 = np.inf
            # Perform ratio test.
            for i in range(len(ra)):
                if ra_d[i] - tol > 0:
                    s = ra[i]/ra_d[i]
                    if s < s_star2:
                        s_star2 = s

            # Generate random point between lambdas.
            lam = rand_sol_rng.uniform(-1 * s_star2, s_star)

            # Compute the new point.
            x += lam * direction

            xs.append(tuple(x))

        return xs[: -n_samples]
    

    def find_feasible_initial(self, Ae, Ai, be, bi):
        '''
        Find an initial feasible solution (if not user-provided)
        by solving phase one simplex.

        Arguments
        ---------
        problem : Problem object
            simulation-optimization problem to solve
        C: ndarray
            constraint coefficient matrix
        d: ndarray
            constraint coefficient vector

        Returns
        -------
        x0 : ndarray
            an initial feasible solution
        tol: float
            Floating point comparison tolerance
        '''
        upper_bound = np.array(self.upper_bounds)
        lower_bound = np.array(self.lower_bounds)

        # Define decision variables.
        x = cp.Variable(self.dim)

        # Define constraints.
        constraints = []

        if (Ae is not None) and (be is not None):
            constraints.append(Ae @ x == be.ravel())
        if (Ai is not None) and (bi is not None):
            constraints.append(Ai @ x <= bi.ravel())

        # Removing redundant bound constraints.
        ub_inf_idx = np.where(~np.isinf(upper_bound))[0]
        if len(ub_inf_idx) > 0:
            for i in ub_inf_idx:
                constraints.append(x[i] <= upper_bound[i])
        lb_inf_idx = np.where(~np.isinf(lower_bound))
        if len(lb_inf_idx) > 0:
            for i in lb_inf_idx:
                constraints.append(x[i] >= lower_bound[i])

        # Define objective function.
        obj = cp.Minimize(0)
        
        # Create problem.
        model = cp.Problem(obj, constraints)

        # Solve problem.
        model.solve(solver = cp.SCIPY)

        # Check for optimality.
        if model.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] :
            raise ValueError("Could not find feasible x0")
        x0 = x.value

        return x0


