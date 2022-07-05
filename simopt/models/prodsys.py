"""
Summary
-------
A model that simulates a production system where orders are
demanded over time and inventory can be partially pre-processed.
"""

import numpy as np
from base import Model, Problem


class ProdSys(Model):
    """
    A model that simulates a production system where orders are
    demanded over time and inventory can be partially pre-processed.

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
        self.name = "PRODSYS"
        self.n_responses = 2
        self.specifications = {
            "num_products": {
                "description": "Number of products.",
                "datatype": int,
                "default": 3
            },
            "interarrival_time_mean": {
                "description": "Mean of interarrival times of orders for each product.",
                "datatype": float,
                "default": 30.0
            },
            "interarrival_time_stdev": {
                "description": "Standard deviation of interarrival times of orders for each product.",
                "datatype": float,
                "default": 5.0
            },
            "num_machines": {
                "description": "Number of machines.",
                "datatype": int,
                "default": 2
            },
            "num_edges": {
                "description": "Number of edges.",
                "datatype": int,
                "default": 6
            },
            "total_inventory": {
                "description": "Total inventory.",
                "datatype": int,
                "default": 200
            },
            "interm_product": {
                "description": "Product quantities to be processed ahead of time; number of intermediate products presently at each node.",
                "datatype": list,
                "default": [200, 0, 0, 0, 0, 0]
            },
            "routing_layout": {
                "description": "Layout matrix. List of edges sequences for each product type.",
                "datatype": list,
                "default": [[1, 2],
                            [1, 3],
                            [2, 4],
                            [2, 5],
                            [3, 5],
                            [3, 6]]
            },
            "machine_layout": {
                "description": "List of machines. Each element is the index for the machine that processes the task on each edge.",
                "datatype": list,
                "default": [1, 2, 2, 2, 1, 1]
            },
            "processing_time_mean": {
                "description": "Mean of normally distributed processing times. Each element is associated with a task (edge).",
                "datatype": list,
                "default": [4, 3, 5, 4, 4, 3]
            },
            "processing_time_StDev": {
                "description": "Standard deviation of normally distributed processing times. Each element is associated with a task (edge).",
                "datatype": list,
                "default": [1, 1, 2, 1, 1, 1]
            },
            "product_batch_prob": {
                "description": "Batch order probabilities of each product.",
                "datatype": list,
                "default": [0.5, 0.35, 0.15]
            },
            "time_horizon": {
                "description": "Time horizon.",
                "datatype": int,
                "default": 600
            },
            "batch": {
                "description": "Batch size.",
                "datatype": int,
                "default": 10
            }
        }
        super().__init__(fixed_factors)
        self.n_rngs = self.factors["num_machines"] + 2
        self.check_factor_list = {
            "num_products": self.check_num_products,
            "interarrival_time_mean": self.check_interarrival_time_mean,
            "interarrival_time_stdev": self.check_interarrival_time_stdev,
            "product_batch_prob": self.check_product_batch_prob,
            "num_machines": self.check_num_machines,
            "num_edges": self.check_num_edges,
            "total_inventory": self.check_total_inventory,
            "interm_product": self.check_interm_product,
            "batch": self.check_batch,
            "time_horizon": self.check_time_horizon,
            "routing_layout": self.check_routing_layout,
            "machine_layout": self.check_machine_layout,
            "processing_time_mean": self.check_processing_time_mean,
            "processing_time_stdev": self.check_processing_time_stdev
        }
        # Set factors of the simulation model.
        super().__init__(fixed_factors)

    def check_num_products(self):
        return self.factors["num_products"] > 0

    def check_interarrival_time_mean(self):
        return self.factors["interarrival_time_mean"] > 0

    def check_interarrival_time_stdev(self):
        return self.factors["interarrival_time_stdev"] > 0

    def check_product_batch_prob(self):
        all_positive = all([p > 0 for p in self.factors["product_batch_prob"]])
        sum_to_one = sum(self.factors["product_batch_prob"]) == 1
        return all_positive * sum_to_one

    def check_num_machines(self):
        return self.factors["num_machines"] > 0

    def check_num_edges(self):
        return self.factors["num_edges"] > 0

    def check_total_inventory(self):
        return self.factors["total_inventory"] > 0

    def check_interm_product(self):
        all_nonnegative = all([x >= 0 for x in self.factors["interm_product"]])

    def check_routing_layout(self):
        # Advanced logic appears in check_simulatable factors.
        return True

    def check_machine_layout(self):
        # TO DO: Add more advanced logic.
        return True

    def check_batch(self):
        return self.factors["batch"] > 0

    def check_time_horizon(self):
        return self.factors["time_horizon"] > 0

    def check_processing_time_mean(self):
        return self.factors["processing_time_mean"] > 0

    def check_processing_time_stdev(self):
        return self.factors["processing_time_stdev"] > 0

    def check_simulatable_factors(self):
        simulatable = True
        # Check lengths.
        simulatable *= len(self.factors["product_batch_prob"]) == self.factors["num_products"]
        simulatable *= len(self.factors["processing_time_mean"]) == self.factors["num_edges"]
        simulatable *= len(self.factors["processing_time_stdev"]) == self.factors["num_edges"]
        simulatable *= len(self.factors["machine_layout"]) == self.factors["num_edges"]
        simulatable *= len(self.factors["interm_product"]) == self.factors["num_edges"]
        simulatable *= len(self.factors["routing_layout"]) == self.factors["num_edges"]
        # Check sums.
        simulatable *= sum(self.factors["interm_product"]) == self.factors["total_inventory"]
        simulatable *= sum(self.factors["interm_product"]) == self.factors["total_inventory"]
        # Check that routing layout has right number of terminal nodes.
        # TO DO: Not yet implemented.
        return simulatable

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
            "avg_lead_time" = average time to fulfill each order
            "service_level" = percentage of ordered filled on time
        gradients : dict of dicts
            gradient estimates for each response
        """
        import random  # REMOVE

        def previous_node(node, check):    # Returns pre node
            """
            Helper function: return the predecessor node(s) in the network.
            """
            pre_node = 0
            i = False
            j = self.factors["num_edges"]
            t = 0
            if node == 1:
                pre_node = 0
            else:
                if check == 0:
                    while i is False:
                        if node == self.factors["routing_layout"][j - 1][1]:
                            pre_node = self.factors["routing_layout"][j - 1][0]
                            i = True
                        j -= 1
                else:
                    t = 1
                    while t == check:
                        if node == self.factors["routing_layout"][j - t][1]:
                            t += 1
                            pre_node = self.factors["routing_layout"][j - t][0]
                        j -= 1
            return(pre_node)

        def check_node(product):                    # Return inventory and corresponding node
            i = False
            node = end_nodes[product - 1]
            possible_node = []                                                                 # Product's end node from list                                      # Inventory at node from replicated list of intermediate product
            if product != 1 and product != self.factors["num_products"]:
                check = 0
                for j in range(num_nodes):
                    if self.factors["routing_layout"][j][1] == node:
                        check += 1
                for j in range(check):
                    node = end_nodes[product - 1]                                                  # Product's end node from list
                    inventory = node_product[node - 1]
                    lst_nodes = [node]
                    while inventory == 0 or i is False:
                        if previous_node(node, 0) == 1 and j != 0:
                            if node_product[node - 1] == 0:
                                lst_nodes.append(1)
                                break
                            else:
                                break
                        node = previous_node(node, j)
                        if node == 0:
                            possible_node = float('inf')
                            break
                        inventory = node_product[node - 1]
                        if inventory != 0:
                            i = True
                        lst_nodes.append(node)
                    lst_nodes.reverse()
                    if possible_node != float('inf'):
                        possible_node.append(lst_nodes)
            else:
                inventory = node_product[node - 1]
                possible_node = [node]
                while inventory == 0 and i is False:
                    node = previous_node(node, 0)
                    if node == 0:
                        possible_node = float('inf')
                        break
                    inventory = node_product[node - 1]
                    if inventory != 0:
                        i = True
                    possible_node.append(node)
                if possible_node != float('inf'):
                    possible_node.reverse()
            return(possible_node)

        def edge_route(nodes):
            edges = []
            for i in range(len(self.factors["routing_layout"])):
                for j in range(len(nodes) - 1):
                    if self.factors["routing_layout"][i][0] == nodes[j] and self.factors["routing_layout"][i][1] == nodes[j + 1]:
                        edges.append(i)
            return(edges)

        def get_sequence(prod):
            nodes = check_node(prod)
            if nodes == float('inf'):
                edges = [float('inf')]
            elif type(nodes[0]) == list:
                edges = []
                for i in range(len(nodes)):
                    edges.append(edge_route(nodes[i]))
            else:
                edges = edge_route(nodes)
            return edges

        def get_sequence_time(edges):
            order_time = []
            time = 0
            for i in edges:
                time += random.normalvariate(self.factors["processing_time_mean"][i], self.factors["processing_time_stdev"][i])
                order_time.append(time)
            return(edges, order_time)

        def get_min_seq(seq):
            current_q = [machines_q[k][-1] for k in range(len(machines_q))]
            min_seq = float('inf')
            for elem in seq:
                total_time = 0
                for i in range(len(elem)):
                    total_time += self.factors["processing_time_mean"][elem[i]]
                    mach = self.factors["machine_layout"][elem[i]]
                    if current_q[mach - 1] == float('inf'):
                        total_time += 0
                    else:
                        total_time += current_q[mach - 1]
                if total_time < min_seq:
                    min_seq = total_time
                    optimal_edges = elem
            return optimal_edges

        def update_time(prod):
            seq = get_sequence(prod)
            if seq == [float('inf')]:
                finish_time.append(float('inf'))
                lead_times.append(finish_time[-1] - arrival_time)
            else:
                if type(seq[0]) == list:
                    optimal_edges = get_min_seq(seq)
                    optimal_edges, optimal_time = get_sequence_time(optimal_edges)
                else:
                    optimal_edges, optimal_time = get_sequence_time(seq)
                machines = []
                for elem in optimal_edges:
                    machines.append(self.factors["machine_layout"][elem])
                for i in range(len(machines)):
                    for j in range(len(self.factors["machine_layout"])):
                        if self.factors["machine_layout"][j] == machines[i]:
                            edge_time[j] = optimal_time[i]
                nodes = []
                for j in optimal_edges:
                    nodes.append(self.factors["routing_layout"][j][0])
                node_product[nodes[0] - 1] -= 10
                count = 0
                new_lst2 = [machines_q[k][-1] for k in range(len(machines_q))]
                for k in new_lst2:
                    if k == float('inf'):
                        count += 1
                if count == len(new_lst2):
                    for i in machines:
                        x = clock + optimal_time[i - 2]
                        machines_q[i - 1] = [x]
                lapse_order = [machines_q[k][-1] for k in range(len(machines_q))]
                for elem in lapse_order:
                    if elem == float('inf'):
                        lapse_order.remove(elem)
                finish_time.append(max(lapse_order))
                lead_times.append(finish_time[-1] - arrival_time)
                # network_time.append(sum(optimal_time))

        # MAIN CODE
        # LIST RANDOM NUMBERS GENERATED
        # Generate/attach random machine processing times for # of machines
        for j in range(self.factors["num_machines"]):
            list_initiator = []
            for i in range(self.factors["num_edges"]):
                if self.factors["machine_layout"][i] == j + 1:
                    parameters = [self.factors["processing_time_mean"][i], self.factors["processing_time_stdev"][i]]
                    list_initiator.append(parameters)
                else:
                    list_initiator.append(0)
            rng_list[j] = list_initiator
        product_orders_rng = []
        arrival_times_rng = []

        node_product = self.factors["interm_product"]

        orders_time = 0
        num_orders = 0
        for i in range(self.factors["time_horizon"]):                        # Generate random order inter-arrival times
            order_arrival_time = random.normalvariate(self.factors["interarrival_time_mean"], self.factors["interarrival_time_stdev"])
            orders_time += order_arrival_time                                                      # Sum of arrival times
            if orders_time <= self.factors["time_horizon"]:                                                             # Attach if sum is less than time horizon
                arrival_times_rng.append(orders_time)                                                   # Track number of orders
                num_orders += 1
                product = random.choices(np.arange(1, self.factors["num_products"] + 1), weights=self.factors["product_batch_prob"], k=1)
                product_orders_rng.append(product[0])
            else:
                break
        rng_list[-2] = product_orders_rng
        rng_list[-1] = arrival_times_rng

        # CREATING END NODE LIST
        num_nodes = self.factors["routing_layout"][
            self.factors["num_edges"] - 1][1]
        end_nodes = []
        for i in range(self.factors["num_products"]):
            (end_nodes.append(num_nodes - i))
        end_nodes.reverse()
        edge_time = [0] * len(self.factors["machine_layout"])
        machines_q = [[0]] * 2
        for i in range(len(machines_q)):
            machines_q[i][0] = float('inf')
        finish_time = []
        lead_times = []
        clock = 0
        i = 0
        while len(finish_time) != len(rng_list[3]):
            new_lst = [machines_q[k][-1] for k in range(len(machines_q))]
            next_inq = min(new_lst)
            ind = new_lst.index(next_inq)
            if next_inq < rng_list[3][i] or next_inq != float("inf"):
                clock = next_inq
                machines_q[ind].remove(next_inq)
                if machines_q[ind] == []:
                    machines_q[ind].append(float("inf"))
            else:
                clock = rng_list[3][i]
                product = rng_list[2][i]
                arrival_time = clock
                update_time(product)
                i += 1

        sum_leadtime = 0
        sum_sslevel = 0
        for j in range(len(lead_times)):
            if lead_times[j] != float('inf'):
                sum_leadtime += lead_times[j]
                sum_sslevel += 1
        avg_ldtime = sum_leadtime / sum_sslevel
        avg_sslevel = sum_sslevel / len(lead_times)
        # Compose responses and gradients.
        responses = {"avg_lead_time": avg_ldtime, "service_level": avg_sslevel}
        gradients = {response_key: {factor_key: np.nan for factor_key in self.specifications} for response_key in responses}
        return responses, gradients


"""
Summary
-------
Find the optimal inventory placement that minimizes expected lead time,
while satisfying service level requirement with high probability.
"""


class ProdSysMinLeadTime(Problem):
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
    def __init__(self, name="PRODSYS-1", fixed_factors=None, model_fixed_factors=None):
        if fixed_factors is None:
            fixed_factors = {}
        if model_fixed_factors is None:
            model_fixed_factors = {}
        self.name = name
        self.n_objectives = 1
        self.n_stochastic_constraints = 1
        self.minmax = (-1,)
        self.constraint_type = "stochastic"
        self.variable_type = "discrete"
        self.gradient_available = False
        self.optimal_value = None
        self.optimal_solution = None
        self.model_default_factors = {}
        self.model_decision_factors = {"interm_product"}
        self.factors = fixed_factors
        self.specifications = {
            "initial_solution": {
                "description": "Initial solution.",
                "datatype": list,
                "default": [200, 0, 0, 0, 0, 0]
            },
            "budget": {
                "description": "Max # of replications for a solver to take.",
                "datatype": int,
                "default": 10000
            },
            "alpha": {
                "description": "Risk level parameter.",
                "datatype": float,
                "default": 0.10
            },
            "min_sslevel": {
                "description": "Minimum tolerable service level.",
                "datatype": float,
                "default": 0.5
            }
        }
        self.check_factor_list = {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget,
            "alpha": self.check_alpha,
            "min_sslevel": self.check_min_sslevel
        }
        super().__init__(fixed_factors, model_fixed_factors)
        # Instantiate model with fixed factors and over-riden defaults.
        self.model = ProdSys(self.model_fixed_factors)
        self.dim = self.model.factors["num_products"]

    def check_initial_solution(self):
        right_length = len(self.factors["initial_solution"]) == self.dim
        feasible = self.check_deterministic_constraints(x=tuple(self.factors["initial_solution"]))
        return right_length * feasible

    def check_budget(self):
        return self.factors["budget"] > 0

    def check_alpha(self):
        return self.factors["alpha"] > 0

    def check_min_sslevel(self):
        return 0 < self.factors["min_sslevel"] <= 1

    def check_simulatable_factors(self):
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
            "interm_product": vector[:]
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
        vector = tuple(factor_dict["interm_product"])
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
        objectives = (response_dict["avg_lead_time"],)
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
        stoch_constraints = (response_dict["service_level"],)
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
        det_stoch_constraints = (0,)
        det_stoch_constraints_gradients = ((0,) * self.dim,)
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
        inventory_feasible = sum(x) == self.model.factors["total_inventory"]
        box_feasible = super().check_deterministic_constraints(x)
        return inventory_feasible * box_feasible

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
        x = rand_sol_rng.integer_random_vector_from_simplex(n_elements=len(self.model.factors["routing_layout"]),
                                                            summation=self.model.factors["total_inventory"],
                                                            with_zero=False
                                                            )
        return x
