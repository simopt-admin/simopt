Model: Production System (ProdSys)
==========================================

Description:
------------
Orders for each of :math:`m` different products arrive according to :math:`m` independent
Poisson processes with constant arrival rates :math:`(λ_j : 1 ≤ j ≤ m)`. Products are made up of a collection
of items of different types. Product :math:`j` requires :math:`a_jk` items of type :math:`k`, where :math:`k` ranges
from 1 to n`. Items are either key items or non-key items. If any of the key items are out of stock then the product
order is lost. If all key items are in stock the order is assembled from all key items and the available
non-key items. Each item sold brings a profit :math:`p_k, k = 1, 2, . . . , n`, and each item in inventory has a
holding cost per unit time of :math:`h_k, k = 1, 2, . . . , n`. There are inventory capacities
:math:`c_k, k = 1, 2, . . . , n`
for each item, so that :math:`0 ≤ x_k ≤ c_k`, where :math:`x_k` is the inventory level of item k,
:math:`k = 1, 2, . . . , n`. The
production time for each item is normally distributed with mean :math:`μ_k` and standard devaition
:math:`σ_k`, :math:`k = 1, 2, . . . , n`,
truncated at 0.

The system operates under a continuous-review base stock policy under which each item has a
target base stock :math:`b_k, k = 1, 2, . . . , n` and each demand for an item triggers a replenishment order for
that item. Items are produced one at a time on dedicated machines, i.e., there are :math:`n` machines, each
producing a single type of item.


Sources of Randomness:
----------------------
2 sources of randomness exists: 
Order product type depending on probabilities, 
and normally distributed processing times for items with parameters :math:`μ` and :math:`σ`.

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
    
* :math:`product_req`: Bill of materials; required item types/quantity for each product type

    * Default: [[1, 0, 0, 1, 0, 1, 1, 0],
                [1, 0, 0, 0, 1, 1, 1, 0],
                [0, 1, 0, 1, 0, 1, 0, 0],
                [0, 0, 1, 1, 0, 1, 0, 1],
                [0, 0, 1, 0, 1, 1, 1, 0]]

* :math:`warm_up_time`: Warm-up period in time units

    * Default: 20

* :math:`run_time`: Run-length in time units for capturing statistics

    * Default: 50

* :math:`key_items`: Number of key-items (columns)

    * Default: [6

    
    
Responses:
---------
* :math:`avg_profit`: Average profit



References:
===========
This model is adapted from the article Hong, L. J. and B . L. Nelson, 2006.
Discrete Optimization via Simulation Using COMPASS. Operations Research 54 (1), 115–129.



Optimization Problem: Minimize Lead Time (ProdSys)
========================================================
We wish to maximize the expected total profit per unit time, :math:`avg_profit` by selecting the target inventory
level vector, :math:`b`.

Decision Variables:
-------------------
* :math:`item_cap`

Objectives:
-----------
Minimize :math:`avg_leadtime`

Constraints:
------------
1 box consraint: :math:`0 ≤ xk ≤ ck`

Problem Factors:
----------------
* :math:`b`: Target inventory vector

  * Default: [20, 20, 20, 20, 20, 20, 20, 20]
  
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
