
Model: Continuous Newsvendor Problem (CNTNV)
============================================

Description:
------------

A vendor orders a fixed quantity of materials at the beginning of a day. A per-unit-material
cost :math:`c_{m}` is assessed for each type of material. As the demand is observed, the vendor 
optionally orders recourse material at a premium price :math:`c_{r}` and processes all materials 
into a variety of finished goods in a manner that maximizes profit. The finished goods are sold 
to customers at price :math:`p_{s}`. At the end of the day, any unprocessed materials can be 
salvaged at price :math:`p_{v}`.

Sources of Randomness:
----------------------

Each day's random demand for discrete product quantities follows Poisson distribution and 
is denoted by :math:`D`. The parameters of the Poisson distribution is :math:`λ`. Its 
cumulative distribution function is given by :math:`F(x; λ) = \sum_{k=0}^{x} \frac{e^{-λ}λ^k}{k!}` 
where :math:`x` and :math:`λ` are positive.

Model Factors: 
--------------

* num_material (:math:`n_{m}`): The number of material types.
 
    * Default: 4

* num_product (:math:`n_{p}`): The number of product types.
 
    * Default: 3

* mat_to_prod (:math:`A`): :math:`n_{p}` by :math:`n_{m}` matrix, mapping :math:`n_{m}` materials to :math:`n_{p}` products.

    * Default: [[1, 2, 1, 3], [1, 1, 3, 1], [2, 0, 4, 1]]

* material_cost (:math:`c_{m}`): Purchasing cost per material unit. :math:`n_{m}` sized array.

    * Default: [1, 1, 1, 1]

* recourse_cost (:math:`c_{r}`): Recourse purchasing cost per material unit. :math:`n_{m}` sized array. 

    * Default: [2, 2, 2, 2]

* process_cost (:math:`c_{p}`): Processing cost per product unit. :math:`n_{p}` sized array.

    * Default: [0.1, 0.1, 0.1]

* order_cost (:math:`c_{o}`): Fixed one-time ordering cost.

    * Default: 20

* purchase_yield (:math:`Y`): Yield rates for initially purchased materials. :math:`n_{m}` sized array. 

    * Default: [0.9, 0.9, 0.9, 0.9]

* total_budget (:math:`b`): Total budget for all materials, processing, ordering and recourse purchases.

    * Default: 600

* sales_price (:math:`p_{s}`): Sales price per product unit. :math:`n_{p}` sized array. 

    * Default: [12, 12, 12]

* salvage_price (:math:`p_{v}`): Salvage price per material unit. :math:`n_{m}` sized array. 

    * Default: [0.6, 0.6, 0.6, 0.6]

* order_quantity (:math:`x`): Initial order quantity per material. :math:`n_{m}` sized array. 

    * Default: [20, 20, 20, 20]
  
* poi_mean (:math:`λ`): Mean parameter for demand's poisson distribution. :math:`n_{p}` sized array. 

    * Default: [15, 15, 15]

Model variations:
----------------------

This newsvendor model is adaptable to various cases of the traditional model. Adjustment of
model factors is necessary if simulating a particular case. 

1) Retail vendor (traditonal): 

    The newsvendor purchases products and sells them directly to the customers. There 
    is no production involved. The decision variable in this problem is order quantity
    of *products*. This variation is subject to below factor settings and intepretations: 

    * :math:`n_{m}` = :math:`n_{p}` (there can be multiple products)
    * :math:`A` = identity matrix of size :math:`n_{p}`
    * :math:`c_{m}` = cost of each product type
    * :math:`c_{r}` = recourse cost of each product type
    * :math:`p_{v}` = salvage price of each product type
    * :math:`Y` = yield rates of each product type

2) Factory vendor:

    Initially, the newsvendor purchases materials instead of products. After demand is
    observed, materials are processed into products. This is done using an integer program
    with an objective of maximizing profit. Additionally, in this variation, materials 
    can be salvaged but products can not be salvaged. The decision variable in this problem 
    is order quantity of *materials*. This variation is subject to below factor settings 
    and intepretations:

    * :math:`n_{m}` :math:`⊥` :math:`n_{p}`. Independence between the number of materials and the number of products
    * :math:`A` = Resource requirements to produce each product. :math:`n_{p}` by :math:`n_{m}` matrix.

3) Multi-product:

    Either in a factory vendor or retail vendor setting, there can exist multiple products
    to be procured/produced and sold to customers. Previous constraints from retail or factory 
    models still hold. This variation is subject to below factor settings and intepretations:

    * :math:`n_{p}` > 1
    * len(:math:`c_{p}`) > 1
    * len(:math:`p_{s}`) > 1

4) Single-product:

    Similarly, a factory vendor or a retail vendor can sell one product only. Note that a single-
    product retail vendor presents the same senario as the traditional newsvendor problem. A single-
    product factory vendor can have multiple materials or one single material, in which case it will
    behave as the traditional newsvendor problem. This variation is subject to below factor settings 
    and intepretations:

    * :math:`n_{p}` = 1 
    * :math:`c_{p}` still exists as an array. len(:math:`c_{p}`) = 1.
    * :math:`p_{s}` still exists as an array. len(:math:`p_{s}`) = 1.
    * :math:`λ` stil exists as an array. len(:math:`λ`) = 1.

    In other words, the factors' datatype should not alter, regardless of the model being the 
    multi-product/single-product or retail/factory. 

5) Recourse:
   
    The recourse case does not change the nature of newsvendor's operations. It is an additional 
    feature can co-exist with all previous variations of the model. Recourse refers to the newsvendor's 
    decision to procure materials/products after demand is observed. Economically speaking, a vendor
    would chooses to do so only if the return is worth the premium prices paid for the recourse. 

    Recourse is disabled by default. In other words, its default input is None. 

    In the retail newsvendor setting, the recourse refers to product recourse. 

    In the factory newsvendor setting, the recourse refers to material recourse. 

    In either case, len(:math:`c_{r}`) = :math:`n_{m}`. 


6) Random yield:

    This case refers to indepedent yields of procurement. Such applies to material in the factory 
    newsvendor and products in the retail newsvendor. The randomness of yields comes from a binomial 
    distribution using a random number generator. Yield rates (in array form) are inputed as a model 
    factor. This variation is subject to below factor setting:

    * len(:math:`Y`) = :math:`n_{m}` 
    * 0 < :math:`Y_{i}` <= 1, for all :math:`Y_{i}` in :math:`Y`.

Responses:
----------

* Profit: The daily profit; can be negative if a loss is incurred.

References: 
===========

Evan L. Porteus. Stochastic inventory theory. In D. P. Heyman and M. J. Sobel, editors,
Stochastic Models, volume 2 of Handbooks in Operations Research and Management Science,
chapter 12, pages 605–652. Elsevier, New York, 1990.

Gallego, G., & Moon, I. (1993). The distribution free newsboy problem: Review and extensions.
The Journal of the Operational Research Society, 44(8), 825.
https://doi.org/10.2307/2583894

Ding, X., Puterman, M. L., & Bisi, A. (2002). The censored Newsvendor and the optimal
acquisition of Information. Operations Research, 50(3), 517–527.
https://doi.org/10.1287/opre.50.3.517.7752

Optimization Problem: Maximize Profit
=====================================

Decision Variables:
-------------------

* order_quantity (:math:`x`): Amount of raw material to be purchased at the beginning of the day. :math:`n_{m}` sized array.

Objectives: 
-----------

Maximizes the vendor's expected profit.

Constraints: 
------------

order_quantity must be an array of non-negative integers

* :math:`x_{i}` >= 0, for all :math:`x_{i}` in :math:`x`.

Problem Factors:
----------------

* Budget: Max # of replications for a solver to take.

  * Default: 3000

Fixed Model Factors:
--------------------

* N/A

Starting Solution:
------------------

* [40, 40, 100, 60]


Random Solutions: 
-----------------

If random solutions are needed, generate :math:`x` from continous_random_vector_from_simplex 
function in mrg32k3a.

Optimal Solution:
-----------------

* [82, 60, 144, 115]

Optimal Objective Function Value:
---------------------------------

For the default factors, the maximum expected profit is 343.19