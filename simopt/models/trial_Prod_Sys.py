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
        # Designate RNG for demands.       
        order_arrival_time_rng = rng_list[0]


        m1_TimeLeft = [] # time left for each job in machine 1
        m2_TimeLeft = [] # time left for each job in machine 2
        m1_Jobs = [] # jobs left for each job in machine 1
        m2_Jobs = [] # jobs left for each job in machine 2

        edge_time = {}
        for i in range(len(self.factors["routing_layout"])):
            edge_time[i] = ([self.factors["processing_time_mean"], self.factors["processing_time_StDev"]])

        node_product = self.factors["interm_product"]

        product = random.choices(np.arange(1,self.factors(["num_products"])), weights = self.factors["product_batch_prob"], k = 1)
        order_arrival_time = order_arrival_time_rng.random.normal(loc=self.factors["Interarrival_Time_mean"], scale = self.factors["Interarrival_Time_StDev"])

        end_nodes = {1:4, 
                    2:5, 
                    3:6} # node 6 is product type 3's ending node

        print(check_node(node_product, end_nodes, product,self.factors["routing_layout"]))

    def check_node(node_product, end_nodes, product, routing):
        node = end_nodes[product]
        inventory = node_product[node-1]
        #if inventory = 0:
        for i in range(len(routing)):
            if routing[i][1] == node:
                node = routing[i][0]
                inventory = node_product[node-1]
                if inventory != 0:
                    break
        return inventory, node



