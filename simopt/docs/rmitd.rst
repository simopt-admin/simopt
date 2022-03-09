Model: Multi-Stage Revenue Management with Inter-Temporal Dependence (RMITD)
==========================================

Description:
------------

To better understand the initial capacity and selling decisions, consider the following: There is a large
initial capacity :math:`b` (units purchased). In period 1 the businessman receives a certain demand for units at low
price :math:`p_{low}`. The decision at period 1 is, essentially, choosing a number of units to be sold right now in order to
reserve some units for customers arriving later because they will pay more for the product; The opportunity
cost for each unit of capacity reserved is now :math:`p_{low}`. Thus, in order to make a wise decision at period 1, the
businessman must consider later demand.

Assume that :math:`D_t = μ_tXY_t` where :math:`X` has a gamma distribution with parameters :math:`k > 0` and :math:`θ > 0` such that
it has mean :math:`kθ = 1` and standard deviation :math:`{\sqrt{k}}θ = 1/ {\sqrt{k}}. Y_1, . . . , Y_T` are i.i.d. exponential with mean 1 and 
:math:`μ_t` are positive constants (:math:`\forall\:t`).


Model Factors:
--------------
* Time Horizon (T): Period of time that is considered

    * Default: 3

* Prices: Prices for each period

     Default: <100, 300, 400>

* Demand Mean (μ): Mean demand for each period

    * Default: <50, 20, 30>

* Cost (c): Cost per unit of capacity at :math:`t = 0`

    * Default: 80

* Gamma Shape (k): Shape parameter of gamma distribution

    * Default: :math:`1`

* Gamma Scale (θ): Scale parameter of gamma distribution

    * Default: :math:`1`

* Initial Inventory (b): Initial inventory

    * Default: 100

* Reservation Quantity (r): Inventory to reserve going into periods :math:`2, 3, ..., T`.

    * Default: :math:`r_2` = 50, :math:`r_3 = 30`. 


Respones:
---------
* Revenue: Total revenue of given model

References:
===========
This example is adapted (almost verbatim) from test problem 2 by Prof. J.M. Harrison for class OIT 603
at Stanford University. (https://ee.stanford.edu/~harris/) 


Optimization Problem: <Maimize Total Revenue> (MTR)
========================================================

Decision Variables:
-------------------

* Initial Inventory (b)

* Reservation Quantity (:math:`r_t`)

Objectives:
-----------

Our goal is to calculate how many units should purchased (:math:`b`) and how many units should be reserved for
future periods in order to maximize total revenue. In other words, we want to find b and :math:`r_t, t = 2, . . . , T` so
that, if the number of units sold in all periods before :math:`t` is less than :math:`b − r_t` (:math:`r_t` units are reserved for periods
:math:`t, t + 1, . . . , T`), revenue is maximized.

Constraints:
------------

* Gamma Shape (k) > 0

* Gamma Scale (θ) > 0 

* Initial Inventory (b) > 0

* Cost (c) > 0

* :math:`0 ≤ x_t ≤ D_t`

Problem Factors:
----------------
* Initial Solution: Initial solution from which solvers start.

  * Default: (100, 50, 30)
  
* Budget: Max # of replications for a solver to take.

  * Default: 10000

Fixed Model Factors:
--------------------
* N/a

Starting Solution: 
------------------

* :math:`b = 100`

* :math:`r_2 = 50`

* :math:`r_3 = 30`

Random Solutions: 
------------------

If multiple solutions are needed for Reservation Quantity (r), use :math:`r_2` ∼ Uniform(40,60) and :math:`r_3` ∼ Uniform(20,40).

Optimal Solution:
-----------------

Unknown

Optimal Objective Function Value:
---------------------------------

Unknown
