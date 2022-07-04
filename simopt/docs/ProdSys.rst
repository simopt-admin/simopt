Model: Production System (ProdSys)
==========================================

Description:
------------
Let :math:`G = (V, A)` be a production network with a set of nodes :math:`V = {1, 2,..., n}`. :math:`num_products`
different final products correspond to the nodes with outdegree 0. Each node in :math:`V` represents a state of
the product while each arc denotes the process of transforming one intermediate product to another. 

Every transformation process uses one of the :math:`num_machines` machines. All processing times are random with given distributions, 
and orders for each of the :math:`num_products` final products are also random and arrive at random intervals. 
Assume that the time horizon is :math:`H`, and that the available sets of raw material equals 
the expected value of the total demand of all :math:`num_products` final products.

Sources of Randomness:
----------------------
3 sources of randomness exist: 

1. Normally distributed order inter-arrival times.

2. Product type of the order.

3. Normally distributed machine processing times.

Model Factors:
--------------
* :math:`num_products`: Number of products.

    * Default: 3

* :math:`interarrival_time_mean`: Mean of interarrival times of orders for each product.

    * Default: 30

* :math:`interarrival_time_stdev`: Standard deviation of interarrival times of orders for each product.

    * Default: 5
    
* :math:`num_machines`: Number of machines.

    * Default: 2

* :math:`num_edges`: Number of edges.

    * Default: 6

* :math:`total_inventory`: Total inventory.
    
    * Default: 200

* :math:`interm_product`: Product quantities to be processed ahead of time; number of intermediate products presently at each node.

    * Default: [200, 0, 0, 0, 0, 0]
    
* :math:`routing_layout`: Layout matrix. List of edges sequences for each product type.

    * Default: [[1, 2],
                [1, 3],
                [2, 4],
                [2, 5],
                [3, 5],
                [3, 6]]

* :math:`machine_layout`: List of machines. Each element is the index for the machine that processes the task on each edge.

    * Default: [1, 2, 2, 2, 1, 1]

* :math:`processing_time_mean`: Mean of normally distributed processing times. Each element is associated with a task (edge).
    * Default: [4, 3, 5, 4, 4, 3]

* :math:`processing_time_stdev`: Standard deviation of normally distributed processing times. Each element is associated with a task (edge).

    * Default: [1, 1, 2, 1, 1, 1]

* :math:`product_batch_prob`: Batch order probabilities of each product.

    * Default: [0.5, 0.35, 0.15]

* :math:`time_horizon`: Time horizon.
    
    * Default: 600

* :math:`batch`: Batch size.

    * Default: 10    
    
Responses:
---------
* :math:`avg_lead_time`: Average lead time.

* :math:`service_level`: Service level.


References:
===========
This model is adapted from the article Azadivar, F., Shu, J., & Ahmad, M. (1996). Simulation Optimization in Strategic Location of Semi-Finished Products in a Pull-Type Production System. 
*Proceedings of the 1996 Winter Simulation Conference (WSC)*, 1123-1128.



Optimization Problem: Minimize Lead Time (ProdSysMinLeadTime)
=============================================================
Our objective is to minimize the expected lead time, 
while satisfying a tolerable service level, :math:`b` with high probability :math:`1 − α`.

Decision Variables:
-------------------
* :math:`interm_products`

Objectives:
-----------
Minimize expected :math:`avg_lead_time`.

Constraints:
------------
interm_products must be non-negative vector of length equal to number of nodes.
1 deterministic constraint: interm_products must sum to total inventory.
1 stochastic constraint: :math:`P[{service_level} ≥ b] ≥ 1 − α`.

Problem Factors:
----------------
* :math:`alpha`: Risk level parameter.

  * Default: 0.10
  
* :math:`min_sslevel`: Minimum tolerable service level (b).

  * Default: 0.5

Fixed Model Factors:
--------------------
None

Starting Solution: 
------------------
* interm_product: [200, 0, 0, 0, 0, 0]

Random Solutions:
------------------
Generate initial inventory vectors uniformly at random from the set of vectors (of length equal to the number of nodes) whose values sum to the total inventory.


Optimal Solution:
-----------------
N/A

Optimal Objective Function Value:
---------------------------------
N/A
