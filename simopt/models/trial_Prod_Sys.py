"""
Summary
-------
Simulate expected revenue for a hotel.
"""
import numpy as np

from base import Model, Problem


class ProdSys(Model):
    """
    A model that simulates a 
    production system with a normally distribute demand.

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
        self.n_responses = 2
        self.specifications = {
            "num_products": {
                "description": "Number of products: (processing time,"
                "units of raw material).",
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
                "default": [200, 0, 0, 0, 0, 0]
            },
            "routing_layout": {
                "description": "Layout matrix, list of edges sequences for each product type",
                "datatype": list,
                "default": [[1, 2],
                            [1, 3],
                            [2, 4],
                            [2, 5],
                            [3, 5],
                            [3, 6]]
            },
            "machine_layout": {
                "description": "List of machines, each element is the index for the machine that processes the task on each edge",
                "datatype": list,
                "default": [1, 2, 2, 2, 1, 1]
            },
            "processing_time_mean": {
                "description": "Normally distributed processing times list; each element is the mean for the processing time distribution associated with the task on each edge",
                "datatype": list,
                "default": [4, 3, 5, 4, 4, 3]
            },
            "processing_time_StDev": {
                "description": "Normally distributed processing times matrix; standard deviation",
                "datatype": list,
                "default": [1, 1, 2, 1, 1, 1]
            },
            "product_batch_prob": {
                "description": "Batch order probabilities of product.  ",
                "datatype": list,
                "default": [0.5, 0.35, 0.15]
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
        super().__init__(fixed_factors)

        self.n_rngs = self.factors["num_machines"] + 2
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
            "time_horizon": self.check_time_horizon,
            "routing_layout": self.check_routing_layout,
            "machine_layout": self.check_machine_layout,
            "processing_time_mean": self.check_processing_time_mean,
            "processing_time_StDev": self.check_processing_time_StDev,
            "product_batch_prob": self.check_product_batch_prob,
            "time_horizon": self.check_time_horizon
        }
        
        # Set factors of the simulation model.

    def check_num_products(self):
        return self.factors["num_products"] > 0

    def check_Interarrival_Time_mean(self):
        return (self.factors["Interarrival_Time_mean"] > 0)

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
        for i in self.factors["interm_product"]:
            if i < 0:
                return False
        return sum(self.factors["interm_product"]) == self.factors["n_sets"] and len(self.factors["interm_product"]) == self.factors["num_edges"]
    def check_routing_layout(self):
        if len(self.factors["routing_layout"]) != self.factors["num_edges"]:
           return False
        end_nodes = []
        num_nodes = self.factors["routing_layout"][self.factors["num_edges"]-1][1]
        for i in range(self.factors["num_products"]): (end_nodes.append(num_nodes-i))
        return len(end_nodes) != self.factors["num_products"]
        return True

    def check_n_sets(self):
        return self.factors["n_sets"] >= 0

    def check_machine_layout(self):
        return len(self.factors["machine_layout"]) == "num_edges"

    def check_batch(self):
        return self.factors["batch"] > 0

    def check_time_horizon(self):
        return self.factors["time_horizon"] > 0

    def check_processing_time_mean(self):
        return len(self.factors["processing_time_mean"]) == self.factors["num_edges"]

    def check_processing_time_StDev(self):
        return len(self.factors["processing_time_StDev"]) == self.factors["num_edges"]

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
        import random
        
        def previous_node(num_nodes, node, possible_node):    # Returns list of predecesors
            for i in range(num_nodes):       
                if node == self.factors["routing_layout"][i][1]:
                    pre_node = self.factors["routing_layout"][i][0]
                    possible_node.append(pre_node)
            if ((len(possible_node)>0) and (not(1 in possible_node))):
                for element in possible_node:
                    if pre_node == 1:
                        break
                    previous_node(num_nodes, element, possible_node)                   
            return(possible_node)
           
        def check_node(node_product, end_nodes, product):                                               # Return inventory and corresponding node  
            node = end_nodes[product-1]                                                                 # Product's end node from list                                                
            inventory = node_product[node-1]                                                            # Inventory at node from replicated list of intermediate product
            if inventory != 0:                                                                          # Updates inventory at end-node
                possible_node = node
            else:
                possible_node = []
                previous_node(num_nodes, node, possible_node)
                length= len(possible_node)
                for i in reversed(range(length)):
                    inventory = node_product[possible_node[i-1]-1]
                    if inventory == 0:
                        possible_node.remove(possible_node[i-1])
            return inventory, possible_node
        '''
        def get_sequence(product):           # Returns possible routing sequences with inventory
            end = end_nodes[product-1]
            possible_seq = []
            nodes = previous_node(num_nodes, end, possible_seq)
            nodes.reverse()
            nodes.append(end)                                               # List of predecesors
            in_route = check_node(node_product, end_nodes, rng_list[2][0])  # List of nodes with inventory
            invent_route = []                                               # Empty list for predecessors with inventory
            routes = []                                                 # Empty list for routes of inventory and end node 
            for i in in_route:
                for j in nodes:
                    if i == j:
                        invent_route.append(j)
                    invent_route.append(end)
                routes.append(invent_route)
            return routes
            print("Routing sequence for product ", product, ": ", nodes)
            if len(nodes)>num_products:
                seq = np.arange(len(nodes)/(num_products-1))
        '''
        def edge_route(nodes):
            edges = []
            for i in range(len(self.factors["routing_layout"])):
                for j in range(len(nodes)):
                    if self.factors["routing_layout"][i][0] == nodes[j] and self.factors["routing_layout"][i][1] == nodes[j+1]:
                        edges.append(i)
            return(edges)
        
        def get_sequence(product):
            end = end_nodes[product-1]
            possible_seq = []
            nodes = previous_node(num_nodes, end, possible_seq)
            nodes.reverse()
            nodes.append(end)
            if len(nodes)>1+self.factors["num_products"]:
                seq = np.arange(len(nodes)/(self.factors["num_products"]-1))
                nodes = [[1, 2, 5], [1, 3, 5]]
            if type(nodes[0]) == list:
                edges = []
                for i in range(len(nodes)):
                    edges.append(edge_route(nodes[i]))
            else:
                edges = edge_route(nodes)
            return edges

        def get_sequence_time(edges):
            total_time = 0
            order_time = []
            for i in edges:
                time = random.normalvariate(self.factors["processing_time_mean"][i], self.factors["processing_time_StDev"][i])
                print("random time ", time)
                order_time.append(time + edge_time[i])
                sum_order_time = sum(order_time)
            print(order_time, sum_order_time)
            return(edges, order_time)
        
        def get_min_time(seq):
            min_time = self.factors["time_horizon"]
            optimal_edges = []
            print("seq", seq)
            for i in range(len(seq)):
                time = []
                for j in seq[i]:
                    print("j", j, self.factors["processing_time_mean"][j])
                    time.append(self.factors["processing_time_mean"][j])
                if sum(time) < min_time:
                    min_time = sum(time)
                    optimal_edges = seq[i]
            print("optimal edgessss: ", optimal_edges)
            return optimal_edges
        
        def update_time(prod):
            invent, invent_seq = check_node(node_product, end_nodes, prod)
            seq = get_sequence(prod)
            if type(seq[0]) == list:
                optimal_edges = get_min_time(seq)
                optimal_edges, optimal_time = get_sequence_time(optimal_edges)
            else:
                optimal_edges, optimal_time = get_sequence_time(seq)
            machines = []
            print("optimal edges: ", optimal_edges)
            for elem in optimal_edges:
                machines.append(self.factors["machine_layout"][elem])
            print("time: ", optimal_time)
            for i in range(len(machines)):
                for j in range(len(self.factors["machine_layout"])):
                    if self.factors["machine_layout"][j] == machines[i]:
                        edge_time[j] =  optimal_time[i]
            print("machines ", machines)
            network_time.append(sum(optimal_time))

        # MAIN CODE

        # LIST RANDOM NUMBERS GENERATED
        for j in range(self.factors["num_machines"]):                   # Generate/attach random machine processing times for # of machines
            list_initiator = []                          
            for i in range(self.factors["num_edges"]):
                if self.factors["machine_layout"][i] == j+1:
                    parameters = [self.factors["processing_time_mean"][i], self.factors["processing_time_StDev"][i]]
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
            order_arrival_time = random.normalvariate(self.factors["Interarrival_Time_mean"], self.factors["Interarrival_Time_StDev"])
            orders_time += order_arrival_time                                                           # Sum of arrival times                                                                                                                                                                 
            if orders_time <= self.factors["time_horizon"]:                                                             # Attach if sum is less than time horizon
                arrival_times_rng.append(orders_time)                                                   # Track number of orders
                num_orders += 1
                product = random.choices(np.arange(1, self.factors["num_products"]+1), weights = self.factors["product_batch_prob"], k = 1)
                product_orders_rng.append(product[0])
            else: 
                break
        rng_list[-2] = product_orders_rng
        rng_list[-1] = arrival_times_rng
        print(rng_list)
        print("")
        # CREATING END NODE LIST
        num_nodes = self.factors["routing_layout"][self.factors["num_edges"]-1][1]
        end_nodes = []
        for i in range(self.factors["num_products"]): (end_nodes.append(num_nodes-i))
        end_nodes.reverse()
        network_time = []
        edge_time = np.zeros(len(self.factors["machine_layout"]))
        

        for i in range(3):
            product = rng_list[2][i]
            print("Product: ", product)

            update_time(product)

            print("")
            print("edges time: ",edge_time)
            print("")
        
        leadtime = []
        clock = rng_list[3][0]
        i = 0
        tot_time = 0
        product = rng_list[2][0]
        print("Product: ", product)
        update_time(product)
        print("")
        print("edges time: ",edge_time)

        while len(leadtime) <= len(rng_list[2]):
            i += 1
            print("Clock: ", clock)
            print(max(edge_time))
            print(rng_list[3][i])
            
            if max(edge_time) > rng_list[3][i]:
                clock = max(edge_time)
                print("here")
            else:
                clock = rng_list[3][i]
                product = rng_list[2][i]
                print("Product: ", product)
                update_time(product)
                print("")
                print("edges time: ",edge_time)
                print("")
            if i == 5:
                break

        print(network_time)
        rng_list = [0, 0, 0, 0]    # FIX THIS
        product = 1 

        def get_lead_time(end_nodes, product, rng_list):
            print("")



        #def get_proc_time(prod_seq, machine_layout, rng_list):                                      # Finds SINGLE path's total processing time.
        #    total_time = 0                                                                          
        #    for elem in prod_seq:                                                                   # For each element/edge in sequence
        #        total_time += machine_times_rng[(machine_layout[elem-1])-1]                         # Add the respective machine's processing time from rng list
        #        total_time += sum(time_left[machine_layout[elem-1]]                                 # **********
        #    return total_time                                                                       # Returns path's total processing time.
        #            
        #                                          
        #                                          
        #def get_best_path(prod_seq, machine_layout, rng_list)):                                     # Get optimal path from list of MULTIPLE sequences 
        #    min_time = self.factors["time_horizon"]                                                                 # Set min_time to max time 
        #    for elem in prod_seq:                                                                   # For every sequence in the list of possible paths                                                                                              
        #            sum_time = get_proc_time(prod_seq[elem], machine_layout, rng_list)              # Sums up machine processing time for current sequence                       
        #            if sum_time < min_time:
        #                min_time = sum_time                                                         # Track minimum processing time possible
        #                seq = prod_seq[elem]                                                        # Sequence with smallest processing time       
        #    return min_time, seq                                                                    # return sequence and total process time
        #                                  

        ##################################################
        ##################################################
        # For every order in rng_list[2]
        for elem in rng_list[2]:
            # Get end node 
            # Find possible path(s)
            # Identify inventory node-location
            # Match & return route with inventory
            ''' 
            invent_seq = get_sequence(elem)
            '''
            # Check fpr multiple sequences 
            # Compare machine times & return shortest lead time & path

            # Update Network
                # Inventory @ nodes
                # Machine jobs
                # Order time in system
            
            # Find Servie Level
            # Count # orders/demand met

        num_orders = len(rng_list)