Model: Newsvendor under Dynamic Consumer Substitution (DYNAMNEWS)
=================================================================

Description:
------------
A retailer sells :math:`n` substitutable products :math:`j = 1, \ldots, n`, each with price :math:`p^j` and cost :math:`c^j`.
The initial inventory levels are denoted by :math:`x = (x_1, \ldots, x_n)`.

Unlike in the classical newsvendor problem, the demand is not given by a predetermined distribution,
but depends on the initial inventory levels :math:`x` as well. The customers :math:`t = 1, \ldots, T` 
arrive in order and each can choose one product that is in-stock when he/she arrives, namely any element in
:math:`S(x_t) = \{j : x^j_t > 0\} \cup \{0\}` where :math:`0` denotes the no-purchase option.

Each customer :math:`t` assigns a utility :math:`U^j_t` to option :math:`j = 0, \ldots, n`, and thus :math:`U_t = (U^0_t, U^1_t, \ldots, U^n_t)` is his/her
vector of utilities. Note that :math:`U^j_t` is the utility of product :math:`j` net of the price :math:`p^j`, and therefore could be 
negative. Since the *no-purchase* option :math:`0` incurs neither utility nor cost, one can assume :math:`U^0_t = 0`.
Customer :math:`t` makes their choice to maximize his/her utility

.. math::
  d(x_t,U_t) = \argmax_{j\in S(x_t)} U^j_t

Sources of Randomness:
----------------------
1. Use the Multinomial Logit (MNL) model. :math:`U^0_t, U^1_t, \ldots, U^n_t` are mutually independent random variables
of the form

.. math::
  U^j_t = u^j + \epsilon^j_t

where :math:`u^j` is a constant and :math:`\epsilon^j_t`, :math:`j = 0, 1, \ldots, n` are mutually independent Gumbel random variables with
:math:`P(\epsilon^j_t \leq z) = \exp(-e^{-(z/\mu+\gamma)})` (:math:`\gamma` is Euler's constant,  :math:`\gamma \approx 0.5772`.)


Model Factors:
--------------
* num_prod: Number of Products

    * Default: 2

* num_customer: Number of Customers

    * Default: 5

* c_utility: Constant of each product's utility

    * Default: (1.0, 1.0)
  
* mu: Mu for calculating Gumbel random variable

    * Default: 1.0
  
* init_level: Initial inventory level

    * Default: (2, 3)

* price: Sell price of products

    * Default: (9, 9)
  
* cost: Cost of prodcuts

    * Default: (5, 5)

An alternative setting has 10 products, 30 customers, linearly increasing utilities
(:math:`u^j = 5 + j`) and initial inventory levels :math:`(3, 3, \ldots, 3)`.

Respones:
---------
* profit: profit in this scenario

* n_prod_stockout: number of products which are out of stock


References:
===========
This model is adapted from the article Mahajan, S., & van Ryzin, G. (2001).
Stocking Retail Assortments under Dynamic Consumer Substitution.
*Operations Research*, 49(3), 334-351.
(https://pubsonline.informs.org/doi/abs/10.1287/opre.49.3.334.11210)


Optimization Problem: Maximize Profit (<DYNAMNEWS-1)
====================================================

Decision Variables:
-------------------
* init_level

Objectives:
-----------
Let :math:`\omega = \{U_t : t = 1, \ldots, T\}`denote the sample path,
and assume that `\omega` follows the probability distribution :math:`P`.
We consider a one-period inventory model and assume :math:`P(T < +\infty) = 1`.
The retailer knows the probability measure :math:`P`.
His/her objective is to choose the initial inventory level :math:`x` that maximizes profit.

Constraints:
------------
* Initial inventory levels must be non-negative.

Problem Factors:
----------------  
* budget: Max # of replications for a solver to take.

  * Default: 1000

Fixed Model Factors:
--------------------
* N/A

Starting Solution: 
------------------
* initial_solution: (2, 3) or :math:`(3, 3, \ldots, 3)`

Random Solutions: 
-----------------
Sample uniformly from (0, 10) in the dimension of num_prod.

Optimal Solution:
-----------------
Unknown

Optimal Objective Function Value:
---------------------------------
Unknown
