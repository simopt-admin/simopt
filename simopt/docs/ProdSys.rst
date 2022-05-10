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
the expected value of the total demand, :math:`f(x1, x2, . . . , xn)` of all :math:`num_products` final products.


Sources of Randomness:
----------------------
3 sources of randomness exists: 
order inter-arrival times, which follow a normal distribution with mean, :math:`mu` and standard devaition, :math:`sigma`,
product type depending on probabilities, 
and normally distributed machine processing times with parameters :math:`mu` and standard devaition, :math:`sigma`.

Model Factors:
--------------
* :math:`num_products`: Number of different product types

    * Default: 3

* :math:`Interarrival_Time_mean`: Interarrival mean times of orders for each product

    * Default: 30

* :math:`Interarrival_Time_StDev`: Interarrival standard deviation times of orders for each product

    * Default: 5
    
* :math:`num_machines`: Number of machines

    * Default: 2

* :math:`num_edges`: Number of edges

    * Default: 6

* :math:`total_inventory`: Total inventory
    
    * Default: 200

* :math:`interm_product`: Product quantities to be processed ahead of time; number of intermediate products presently at node

    * Default: [20, 0, 0, 0, 0, 0]
    
* :math:`routing_layout`: Layout matrix, list of edges sequences for each product type

    * Default: [[1, 2],
                [1, 3],
                [2, 4],
                [2, 5],
                [3, 5],
                [3, 6]]

* :math:`machine_layout`: List of machines, each element is the index for the machine that processes the task on each edge

    * Default: [1, 2, 2, 2, 1, 1]

* :math:`processing_time_mean`: Normally distributed processing times list; each element is the mean for the processing time distribution associated with the task on each edge
    * Default: [4, 3, 5, 4, 4, 3]

* :math:`processing_time_StDev`: Normally distributed processing times matrix; standard deviation

    * Default: [1, 1, 2, 1, 1, 1]

* :math:`product_batch_prob`: Batch order probabilities of product

    * Default: [0.5, 0.35, 0.15]

* :math:`time_horizon`: Time horizon for raw material delivery
    
    * Default: 600

* :math:`batch`: Batch size

    * Default: 10
    
* :math:`n_sets`: Set of raw material to be ordered (dependent on time horizon)

    * Default: 200

    
    
Responses:
---------
* :math:`avg_ldtime`: Average lead time

* :math:`avg_sslevel`: Average service level


References:
===========
This model is adapted from the article Azadivar, F., Shu, J., & Ahmad, M. (1996). Simulation Optimization in Strategic Location of Semi-Finished Products in a Pull-Type Production System. 
Proceedings of the 1996 Winter Simulation Conference (WSC), 1123-1128.



Optimization Problem: Minimize Lead Time (ProdSys)
========================================================
Our objective is thus to minimize the expected lead time, :math:`y(x1, x2, . . . , xn)`, 
while satisfying a tolerable service level, :math:`b` with high probability :math:`1 − α`.

Decision Variables:
-------------------
* :math:`interim_products`

Objectives:
-----------
Minimize :math:`avg_leadtime`

Constraints:
------------
1 stocahstic consraint: :math:`P[g(x1, x2, . . . , xn) ≥ b] ≥ 1 − α`

Problem Factors:
----------------
* :math:`alpha`: Risk level parameter

  * Default: 0.10
  
* :math:`min_sslevel`: Minimum tolerable service leve

  * Default: 0.5

Fixed Model Factors:
--------------------
None

Starting Solution: 
------------------
* <dv1name>: <dv1initialvalue>

* <dv2name>: <dv2initialvalue>

Random Solutions: 
------------------
<description of how to generate random solutions>

Optimal Solution:
-----------------
<if known, otherwise unknown>

Optimal Objective Function Value:
---------------------------------
<if known, otherwise unknown>


Optimization Problem: <problem_name> (<problem_abbrev>)
========================================================

...
