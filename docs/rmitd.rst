Model: Multi-Stage Revenue Management with Inter-Temporal Dependence (RMITD)
============================================================================

Description:
------------

Consider the following inventory system: A businessperson initially purchases
:math:`b` units of a product to sell to customers over a fixed time horizon.
In the first period, a businessperson receives a certain demand for units at a
low price :math:`p_{low}`. The businessperson decision can choose how many units
to sell right now. The businessperson may wish to reserve some units for
customers arriving later who are willing to pay more for the product. The number
of units reserved for the start of period :math:`t` is denoted by :math:`r_t`. This
process of observing demand and choosing how many units to sell is repeated over
:math:`T` time periods. In order to make wise decisions about how much inventory
to reserve in each time period, the businessperson must consider future demand for
the product.

The demand in period :math:`t` is denoted by :math:`D_t = μ_tXY_t` where :math:`X`
has a gamma distribution with parameters :math:`k > 0` and :math:`θ > 0` such that
it has mean :math:`kθ = 1` and standard deviation :math:`{\sqrt{k}}θ = 1/ {\sqrt{k}}.`
:math:` Y_1, . . . , Y_T` are i.i.d. exponential with mean 1 and 
:math:`μ_t` are positive constants for all :math:`t`.

Sources of Randomness:
----------------------
Two sources of randomness are used to generate the :math:`X`s and :math:`Y`s that
form demands.

Model Factors:
--------------
* Time Horizon (T): Period of time that is considered.

    * Default: 3

* Prices: Prices for each period.

    * Default: (100, 300, 400)

* Demand Mean (μ): Mean demand for each period.

    * Default: (50, 20, 30)

* Cost (c): Cost per unit of capacity at :math:`t = 0`.

    * Default: 80

* Gamma Shape (k): Shape parameter of gamma distribution.

    * Default: 1

* Gamma Scale (θ): Scale parameter of gamma distribution.

    * Default: 1

* Initial Inventory (b): Initial inventory.

    * Default: 100

* Reservation Quantity (r): Inventory to reserve going into periods :math:`2, 3, ..., T`.

    * Default: :math:`r_2 = 50`, :math:`r_3 = 30`. 


Responses:
----------

* Revenue: Total revenue of given model

References:
===========
This example is adapted (almost verbatim) from test problem 2 by Prof. J.M. Harrison for class OIT 603
at Stanford University. (https://www.gsb.stanford.edu/faculty-research/faculty/j-michael-harrison) 


Optimization Problem: Maximize Total Revenue
============================================

Decision Variables:
-------------------

* Initial Inventory (b)

* Reservation Quantities (:math:`r_t`)

Objectives:
-----------

Maximize total revenue.

Constraints:
------------

The reserve quantities are decreasing and less than the initial capacity, i.e.,
:math:`b >= r_2 >= r_3`

Problem Factors:
----------------
  
* Budget: Max # of replications for a solver to take.

  * Default: 10000

Fixed Model Factors:
--------------------

* N/A

Starting Solution: 
------------------

* :math:`b = 100`

* :math:`r_2 = 50`

* :math:`r_3 = 30`

Random Solutions: 
-----------------

If multiple solutions are needed for Reservation Quantity (r), use :math:`r_2` ∼ Uniform(40,60) and :math:`r_3` ∼ Uniform(20,40).

Optimal Solution:
-----------------

Unknown

Optimal Objective Function Value:
---------------------------------

Unknown
