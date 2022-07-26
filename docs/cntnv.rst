
Model: Continuous Newsvendor Problem (CNTNV)
============================================

Description:
------------

A vendor orders a fixed quantity of liquid at the beginning of a day to be
sold to customers throughout the day. The vendor pays a per-unit order cost
:math:`c` for the initial inventory and sells it the product to customers at a per-unit price
:math:`s`. At the end of the day, any unsold liquid can be salvaged at a per-unit price, :math:`w`.

Sources of Randomness:
----------------------

Each day's random demand for liquid product follows Burr Type XII distribution and is denoted by :math:`D`.
Ther parameters of the Burr Type XII distribution are :math:`α` and :math:`β` so that its cumulative
distribution function is given by :math:`F(x) = 1 - (1-x^α)^{-β}` where :math:`x, α,` and
:math:`β` are all positive.

Model Factors: 
--------------

* Cost (:math:`c`): The price at which the newsvendor purchases one unit volume of liquid.
 
    * Default: 5

* Price (:math:`s`): The price at which the newsvendor sells one unit volume of liquid.
 
    * Default: 9 

* Salvage Price (:math:`w`): The price at which any unsold liquid is sold for salvage.

    * Default: 1

* Alpha (:math:`α`): Parameter for the demand distribution.

    * Default: 2

* Beta (:math:`β`): Parameter for the demand distribution.

    * Default: 20

* Quantity of Liquid (:math:`x`): Amount (volume) of liquid ordered at the beginning of the day.

    * Default: 0.5

Responses:
----------

* Profit: The daily profit; can be negative if a loss is incurred.

References: 
===========

Evan L. Porteus. Stochastic inventory theory. In D. P. Heyman and M. J. Sobel, editors,
Stochastic Models, volume 2 of Handbooks in Operations Research and Management Science,
chapter 12, pages 605–652. Elsevier, New York, 1990.


Optimization Problem: Maximize Profit
=====================================

Decision Variables:
-------------------

* Quantity of Liquid (:math:`x`): Amount (volume) of liquid ordered at the beginning of the day.

Objectives: 
-----------

Maximizes the vendor's expected profit.

Constraints: 
------------

Quantity of Liquid must be non-negative: :math:`x > 0`

Problem Factors:
----------------

* Budget: Max # of replications for a solver to take.

  * Default: 1000

Fixed Model Factors:
--------------------

* N/A

Starting Solution:
------------------

* :math:`x = 0`


Random Solutions: 
-----------------

If random solutions are needed, generate :math:`x` from an Exponential distribution with mean 1.

Optimal Solution:
-----------------

Global minimum at :math:`x* = (1/((1-r^)^{1/β})-1)^{1/α}`.
For the default factors, the optimal solution is :math:`x*` = 0.1878.

Optimal Objective Function Value:
---------------------------------

The maximum expected profit is 0.4635.
