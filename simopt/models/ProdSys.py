"""
Summary
-------
Simulate expected revenue for a hotel.
"""
import numpy as np

from base import Model, Problem


class ProdSys(Model):
    """
    A model that simulates a production system with a normally distribute demand.

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
    def __init__(self, fixed_factors={}):
        self.name = "ProdSys"
        self.n_rngs = 3
        self.n_responses = 2
        self.specifications = {
            "num_products": {
                "description": "Number of products: (processing time, units of raw material).",
                "datatype": int,
                "default": 3
            },
            "Interarrival_Time_mean": {
                "description": "Interarrival times of orders for each product.",
                "datatype": float,
                "default": 30.0
            },
            "Interarrival_Time_StDev": {
                "description": "Interarrival times of orders for each product.",
                "datatype": float,
                "default": 5.0
            },
            "num_machines": {
                "description": "Number of machines.",
                "datatype": int,
                "default": 2
            },
            "num_edges": {
                "description": "Number of edges",
                "datatype": int,
                "default": 6
            },
            "interm_product": {
                "description": "Product quantities to be processed ahead of time; number of intermediate products presently at node ",
                "datatype": list,
                "default": [200,0,0,0,0,0]
            },
            "routing_layout": {
                "description": "Layout matrix, list of edges",
                "datatype": list,
                "default": [[1,2],
                            [1,3],
                            [2,4],
                            [2,5],
                            [3,5],
                            [3,6]]
            },
            "machine_layout": {
                "description": "List of machines, each element is the index for the machine that processes the task on each edge",
                "datatype": list,
                "default": [1,2,2,2,1,1]
            },
            "processing_time_mean": {
                "description": "Normally distributed processing times list; each element is the mean for the processing time distribution associated with the task on each edge",
                "datatype": list,
                "default": [4,3,5,4,4,3]
            },
            "processing_time_StDev": {
                "description": "Normally distributed processing times matrix; standard deviation",
                "datatype": list,
                "default": [1,1,2,1,1,1]
            },
            "product_batch_prob": {
                "description": "Batch order probabilities of product.  ",
                "datatype": list,
                "default": [.5, .35, .15]
            },
            "time_horizon": {
                "description": "Time horizon for raw material delivery. ",
                "datatype": int,
                "default": 600
            },
            "batch": {
                "description": "Batch size.",
                "datatype": int,
                "default": 10
            },
            "n_sets": {
                "description": "Set of raw material to be ordered (dependent on time horizon). ",
                "datatype": int,
                "default": 200
            },
        }
        self.check_factor_list = {
            "num_products": self.check_num_products,
            "Interarrival_Time_mean": self.check_Interarrival_Time_mean,
            "Interarrival_Time_StDev": self.check_Interarrival_Time_StDev,
            "product_batch_prob": self.check_product_batch_prob,
            "num_machines": self.check_num_machines,
            "num_edges": self.check_num_edges,
            "interm_product": self.check_interm_product,
            "n_sets": self.check_n_sets,
            "machine_layout": self.check_machine_layout,
            "batch": self.check_batch,
            "time_horizon": self.check_time_horizon


            #"routing_layout"
           # "machine_layout"
           # "processing_time_mean"
           # "processing_time_StDev"
           # "product_batch_prob"
           # "time_horizon"

        }
        # Set factors of the simulation model.
        super().__init__(fixed_factors)

    def check_num_products(self):
        return self.factors["num_products"] > 0

    def check_Interarrival_Time_mean(self):
        return len(self.factors["Interarrival_Time_mean"]) > 0

    def check_Interarrival_Time_StDev(self):
        return self.factors["Interarrival_Time_StDev"] > 0

    def check_product_batch_prob(self):
        for i in self.factors["product_batch_prob"]:
            if i <= 0:
                return False
        return len(self.factors["product_batch_prob"])== self.factors["num_products"] and sum(self.factors["product_batch_prob"]) == 1

    def check_num_machines(self):
        return self.factors["num_machines"] > 0

    def check_num_edges(self):
        return self.factors["num_edges"] > 0

    def check_interm_product(self):
        for i in self.factors["interm_products"]:
            if i <= 0:
                return False
        return sum(self.factors["interm_product"]) == self.factors["n_sets"] and len(self.factors["interm_product"]) == self.factors["num_edges"]

    def check_n_sets(self):
        return self.factors["num_sets"] >= 0

    def check_machine_layout(self):
        return len(self.factors["machine_layout"]) == "num_edges"

    def check_batch(self):
        return self.factors["batch"] > 0

    def check_time_horizon(self):
        return self.factors["time_horizon"] > 0



    def check_booking_limits(self):
        for i in list(self.factors["booking_limits"]):
            if i <= 0 or i > self.factors["num_rooms"]:
                return False
        return len(self.factors["booking_limits"]) == self.factors["num_products"]

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
            "lead_time" = time to produce each product
            "service_level" = percentage of products returned on time 
        gradients : dict of dicts
            gradient estimates for each response
        """

        m1_TimeLeft = [] # time left for each job in machine 1
        m2_TimeLeft = [] # time left for each job in machine 2
        m1_Jobs = [] # jobs left for each job in machine 1
        m2_Jobs = [] # jobs left for each job in machine 2

        node_product = self.factors["interm_product"]

        product = random.choices(np.arange(1,self.factors(["num_products"])), weights = self.factors["product_batch_prob"], k = 1)
        order_arrival_time = np.random.normal(loc=self.factors["Interarrival_Time_mean"], scale = self.factors["Interarrival_Time_StDev"])

        end_nodes = {1:4, 
                    2:5, 
                    3:6} # node 6 is product type 3's ending node

        #check_node = node_product[]
        def check_node(node_product, end_nodes, product):
            node = end_nodes[product]
            inventory = node_product[node-1]

        


    

 #       mean = self.factors["processing_time_mean"][i]
 #       std = self.factors["processing_time_Stdev"][i]
 #       processing_time.append(np.random.normal(loc=mean, scale=std))


        


        # Designate separate random number generators.
#        arr_rng = rng_list[0]
#
#        total_revenue = 0
#        b = list(self.factors["booking_limits"])
#        A = np.array(self.factors["product_incidence"])
        # Vector of next arrival time per product.
        # (Starts at time = -1*time_before, e.g., t = -168.)
#        arrival = np.zeros(self.factors["num_products"]) - self.factors["time_before"]
        # Upper bound on number of arrivals over the time period.
#        arr_bound = 10 * round(168 * np.sum(self.factors["lambda"]))
  #      arr_time = np.zeros((self.factors["num_products"], arr_bound))
  #      # Index of which arrival time to use next for each product.
  #      a = np.zeros(self.factors["num_products"], dtype=int)
  #      # Generate all interarrival times in advance.
 #       for i in range(self.factors["num_products"]):
#            arr_time[i] = np.array([arr_rng.expovariate(self.factors["lambda"][i]) for _ in range(arr_bound)])
      # Extract first arrivals.
  #      for i in range(self.factors["num_products"]):
 #           arrival[i] = arrival[i] + arr_time[i, a[i]]
 #           a[i] = 1
 #       min_time = 0  # Keeps track of minimum time of the orders not yet received.
 #       while min_time <= self.factors["runlength"]:
 #           min_time = self.factors["runlength"] + 1
    #        for i in range(self.factors["num_products"]):
    #            if ((arrival[i] < min_time) and (arrival[i] <= self.factors["time_limit"][i])):
   #                 min_time = arrival[i]
  #                  min_idx = i
  #          if min_time > self.factors["runlength"]:
  #              break
  #          if b[min_idx] > 0:
  #              if min_idx % 2 == 0:  # Rack_rate.
  #                  total_revenue += sum(self.factors["rack_rate"] * A[:, min_idx])
  #              else:  # Discount_rate.
 #                   total_revenue += sum(self.factors["discount_rate"] * A[:, min_idx])
 #               # Reduce the inventory of products sharing the same resource.
 #               for i in range(self.factors["num_products"]):
 #                   if np.dot(A[:, i].T, A[:, min_idx]) >= 1:
 #                       if b[i] != 0:
 #                           b[i] -= 1
 #           arrival[min_idx] += arr_time[min_idx, a[min_idx]]
 #           a[min_idx] = a[min_idx] + 1
        # Compose responses and gradients.
 #       responses = {"revenue": total_revenue}
 #       gradients = {response_key: {factor_key: np.nan for factor_key in self.specifications} for response_key in responses}
 #       return responses, gradients


"""
Summary
-------
Maximize the expected revenue.
"""


class HotelRevenue(Problem):
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
    def __init__(self, name="HOTEL-1", fixed_factors={}, model_fixed_factors={}):
        self.name = name
        self.n_objectives = 1
        self.n_stochastic_constraints = 0
        self.minmax = (1,)
        self.constraint_type = "box"
        self.variable_type = "discrete"
        self.gradient_available = False
        self.optimal_value = None
        self.optimal_solution = None
        self.model_default_factors = {}
        self.model_decision_factors = {"booking_limits"}
        self.factors = fixed_factors
        self.specifications = {
            "initial_solution": {
                "description": "Initial solution.",
                "datatype": tuple,
                "default": tuple([0 for _ in range(56)])
            },
            "budget": {
                "description": "Max # of replications for a solver to take.",
                "datatype": int,
                "default": 100
            }
        }
        self.check_factor_list = {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget
        }
        super().__init__(fixed_factors, model_fixed_factors)
        # Instantiate model with fixed factors and over-riden defaults.
        self.model = Hotel(self.model_fixed_factors)
        self.dim = self.model.factors["num_products"]
        self.lower_bounds = tuple(np.zeros(self.dim))
        self.upper_bounds = tuple(self.model.factors["num_rooms"] * np.ones(self.dim))

    def check_initial_solution(self):
        return len(self.factors["initial_solution"]) == self.dim

    def check_budget(self):
        return self.factors["budget"] > 0

    def check_simulatable_factors(self):
        if len(self.lower_bounds) != self.dim:
            return False
        elif len(self.upper_bounds) != self.dim:
            return False
        else:
            return True

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
            "booking_limits": vector[:]
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
        vector = tuple(factor_dict["booking_limits"])
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

    def response_dict_to_stoch_constraints(self, response_dict):
        """
        Convert a dictionary with response keys to a vector
        of left-hand sides of stochastic constraints: E[Y] >= 0

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
        return True

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
        x = tuple([rand_sol_rng.randint(0, self.model.factors["num_rooms"]) for _ in range(self.dim)])
        return x
