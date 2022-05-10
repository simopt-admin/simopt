Model: Production System (ProdSys)
==========================================

Description:
------------
Orders for each of :math:`m` different products arrive according to :math:`m` independent
Poisson processes with constant arrival rates :math:`(λj : 1 ≤ j ≤ m)`. Products are made up of a collection
of items of different types. Product :math:`j` requires :math:`ajk` items of type :math:`k`, where :math:`k` ranges from :math:`1 to n`. Items
are either key items or non-key items. If any of the key items are out of stock then the product
order is lost. If all key items are in stock the order is assembled from all key items and the available
non-key items. Each item sold brings a profit :math:`pk, k = 1, 2, . . . , n`, and each item in inventory has a
holding cost per unit time of :math:`hk, k = 1, 2, . . . , n`. There are inventory capacities :math:`ck, k = 1, 2, . . . , n`
for each item, so that :math:`0 ≤ xk ≤ ck`, where math:`xk` is the inventory level of item k, :math:`k = 1, 2, . . . , n`. The
production time for each item is normally distributed with mean :math:`μk` and standard devaition :math:`σ`, :math:`k = 1, 2, . . . , n`,
truncated at 0.

The system operates under a continuous-review base stock policy under which each item has a
target base stock :math:`bk, k = 1, 2, . . . , n` and each demand for an item triggers a replenishment order for
that item. Items are produced one at a time on dedicated machines, i.e., there are :math:`n` machines, each
producing a single type of item.


Sources of Randomness:
----------------------
2 sources of randomness exists: 
Order product type depending on probabilities, 
and normally distributed processing times for items with parameters :math:`mu` and standard devaition, :math:`sigma`.

Model Factors:
--------------
* :math:`num_products`: Number of different product types

    * Default: 5

* :math:`lambda`: Constant (Poisson) arrival rates for each product type

    * Default: [3.6, 3, 2.4, 1.8, 1.2]

* :math:`num_items`: Number of different item types

    * Default: 8
    
* :math:`item_revenue`: Items' profit per item sold, per item in inventory

    * Default: (1, 2, 3, 4, 5, 6, 7, 8)

* :math:`item_holding`: Items' holding cost per unit time, per item in inventory

    * Default: (2, 2, 2, 2, 2, 2, 2, 2)

* :math:`item_cap`: Items' inventory capacity
    
    * Default: [[20]
                [20],
                [20],
                [20],
                [20],
                [20],
                [20],
                [20]]

* :math:`process_time`: Production time for each item type; normally distributed mean and standard deviation (mu, sigma)

    * Default: [[0.15, 0.0225],
                [0.40, 0.06],
                [0.25, 0.0375],
                [0.15, 0.0225],
                [0.25, 0.0375],
                [0.08, 0.012],
                [0.13, 0.0195],
                [0.40, 0.06]]
    
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
This model is adapted from the article Hong, L. J. and B . L. Nelson, 2006.
Discrete Optimization via Simulation Using COMPASS. Operations Research 54 (1), 115–129.



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
