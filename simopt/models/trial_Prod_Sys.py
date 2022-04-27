"""
Summary
-------
------------*
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
            "total_inventory": {
                "description": "total inventory",
                "datatype": int,
                "default": 200
            },
            "interm_product": {
                "description": "Product quantities to be processed ahead of time; number of intermediate products presently at node ",
                "datatype": list,
                "default": [100, 10, 0, 0, 0, 0]
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
            "batch": self.check_batch,
            "time_horizon": self.check_time_horizon,
            "routing_layout": self.check_routing_layout,
            "machine_layout": self.check_machine_layout,
            "processing_time_mean": self.check_processing_time_mean,
            "processing_time_StDev": self.check_processing_time_StDev,
            "total_inventory": self.check_total_inventory
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
        return len(self.factors["product_batch_prob"]) == self.factors["num_products"] and sum(self.factors["product_batch_prob"]) == 1

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
        for i in range(self.factors["num_products"]):
            (end_nodes.append(num_nodes-i))
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

    def check_total_inventory(self):
        return(sum(self.factors["interm_product"]) == self.factors["total_inventory"])

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

        def previous_node(node, check):    # Returns pre node
            pre_node = 0
            i = False
            j = self.factors["num_edges"]
            t = 0
            if check == 0:
                while i is False:
                    if node == self.factors["routing_layout"][j-1][1]:
                        pre_node = self.factors["routing_layout"][j-1][0]
                        i = True
                    j -= 1
            else:
                t = 1
                while t == check:
                    if node == self.factors["routing_layout"][j-t][1]:
                        t += 1
                        pre_node = self.factors["routing_layout"][j-t][0]
                    j -= 1
            return(pre_node)

        def check_node(product):                    # Return inventory and corresponding node
            i = False
            node = end_nodes[product-1]
            possible_node = []                                                                 # Product's end node from list
            k = 0                                       # Inventory at node from replicated list of intermediate product
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
                            if node_product[node-1] == 0:
                                lst_nodes.append(1)
                                break
                            else:
                                break
                        node = previous_node(node, j)
                        inventory = node_product[node-1]
                        if inventory != 0:
                            i = True
                        lst_nodes.append(node)
                        if k == 5:
                            break
                        k += 1
                    lst_nodes.reverse()
                    possible_node.append(lst_nodes)
            else:
                inventory = node_product[node-1]
                possible_node = [node]
                while inventory == 0 and i is False:
                    node = previous_node(node, 0)
                    inventory = node_product[node-1]
                    if inventory != 0:
                        i = True
                    possible_node.append(node)
                possible_node.reverse()
            print("Inventory: ", node_product)
            return(possible_node)

        def edge_route(nodes):
            edges = []
            for i in range(len(self.factors["routing_layout"])):
                for j in range(len(nodes)-1):
                    if self.factors["routing_layout"][i][0] == nodes[j] and self.factors["routing_layout"][i][1] == nodes[j+1]:
                        edges.append(i)
            return(edges)

        def get_sequence(prod):
            nodes = check_node(prod)
            if type(nodes[0]) == list:
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
                time += random.normalvariate(self.factors["processing_time_mean"][i], self.factors["processing_time_StDev"][i])
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
                    if current_q[mach-1] == float('inf'):
                        total_time += 0
                    else:
                        total_time += current_q[mach-1]
                if total_time < min_seq:
                    min_seq = total_time
                    optimal_edges = elem
            print("optimal edges: ", optimal_edges)
            return optimal_edges

        def update_time(prod):
            seq = get_sequence(prod)
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
            node_product[nodes[0]-1] -= 10
            count = 0
            new_lst2 = [machines_q[k][-1] for k in range(len(machines_q))]
            for k in new_lst2:
                if k == float('inf'):
                    count += 1
            if count == len(new_lst2):
                for i in machines:
                    x = clock + optimal_time[i-2]
                    machines_q[i-1] = [x]
            lapse_order = [machines_q[k][-1] for k in range(len(machines_q))]
            for elem in lapse_order:
                if elem == float('inf'):
                    lapse_order.remove(elem)
            print("Lapse order", lapse_order)
            finish_time.append(max(lapse_order))
            print("Machine Queue:", machines_q)
            # network_time.append(sum(optimal_time))

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
            orders_time += order_arrival_time                                                      # Sum of arrival times
            if orders_time <= self.factors["time_horizon"]:                                                             # Attach if sum is less than time horizon
                arrival_times_rng.append(orders_time)                                                   # Track number of orders
                num_orders += 1
                product = random.choices(np.arange(1,
                    self.factors["num_products"] + 1),
                        weights=self.factors["product_batch_prob"], k=1)
                product_orders_rng.append(product[0])
            else:
                break
        rng_list[-2] = product_orders_rng
        rng_list[-1] = arrival_times_rng
        rng_list = [[[4, 1], 0, 0, 0, [4, 1], [3, 1]], [0, [3, 1], [5, 2], [4, 1], 0, 0], [2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 1, 1, 2, 3, 1, 2, 1, 3], [29.727401676011688, 61.285048195045746, 92.6774513710674, 125.03807931458186, 154.244553001439, 180.64771027023832, 208.32458418385718, 233.3913201191581, 262.38399177585217, 293.21313863649874, 308.2088057588325, 342.68573987169464, 372.01710375272273, 401.3857848259946, 434.2295472258132, 466.42519331866515, 486.70798518555915, 516.3478106441862, 544.4080471479766, 562.9793019943288, 587.4876637499182]]
        print("")
        print(rng_list)
        print("")
        # CREATING END NODE LIST
        num_nodes = self.factors["routing_layout"][
            self.factors["num_edges"]-1][1]
        end_nodes = []
        for i in range(self.factors["num_products"]):
            (end_nodes.append(num_nodes-i))
        end_nodes.reverse()
        edge_time = [0] * len(self.factors["machine_layout"])
        machines_q = [[0]] * 2
        for i in range(len(machines_q)):
            machines_q[i][0] = float('inf')
        finish_time = []
        clock = 0
        i = 0
        for j in range(15):
            print("")
            print("Clock: ", clock)
            new_lst = [machines_q[k][-1] for k in range(len(machines_q))]
            next_inq = min(new_lst)
            print("machine queue: ", machines_q)
            ind = new_lst.index(next_inq)
            if next_inq == float('inf'):
                print("Next in queue: Arrival")
            else:
                print("Next in queue: ", next_inq)

            if next_inq < rng_list[3][i] or next_inq != float("inf"):
                clock = next_inq
                machines_q[ind].remove(next_inq)
                if machines_q[ind] == []:
                    machines_q[ind].append(float("inf"))
            else:
                clock = rng_list[3][i]
                product = rng_list[2][i]
                print("Product: ", product, "arrives at: ", clock)
                update_time(product)
                i += 1
        print("")
        print("Finish Time: ", finish_time)
        lead_times = []
        for k in range(len(finish_time)):
            lead_times.append(finish_time[k]-rng_list[3][k])
        print("")
        print("Lead Times:", lead_times)
        print("")
