Model: Parameter Estimation (PARAMESTI)
==========================================

Description:
------------
A model that simulates MLE estimators for a two-dimentinal gamma  variable.
    Returns the 2-D vector x_star that maximizes the probability of seeing
    parameters x in 2-D beta probability density function f(y, x).


Sources of Randomness:
----------------------
y is a 2-D vector that contributes randomness. Both elements of y are gamma random variables.

Model Factors:
--------------
* x_star: the unknown 2-D parameter that maximizes g(x).

    * Default: [2, 5]

* x: a 2-D variable in the probability density function.

    * Default: [1, 1]

Respones:
---------
* loglik: log likelihood of the pdf.


References:
===========
This model is designed by Raghu Pasupathy (Virginia Tech) and Shane G. Henderson (Cornell) in 2007.




Optimization Problem: Minimize Log Likelihood (ParamEstiMinLogLik)
========================================================

Decision Variables:
-------------------
* x_star

Objectives:
-----------
Minimize the log likelihood of 2-D gamma random variable.

Constraints:
------------
x is in the square [0.1, 10] × [0.1, 10].

Problem Factors:
----------------
* initial_solution: initial solution

  * Default: [1, 1]
  
* budget: Maximum number of replications for a solver to take.

  * Default: 1000

Fixed Model Factors:
--------------------
* x_star: the unknown 2-D parameter that maximizes g(x).

    * Default: [2, 5]

* x: a 2-D variable in the probability density function.

    * Default: [1, 1]

Starting Solution: 
------------------
* x_star: [2, 5]

Random Solutions: 
------------------
Generate i.i.d. uniformly in the square [0.1, 10] × [0.1, 10].

Optimal Solution:
-----------------
x_star = [2, 5]

Optimal Objective Function Value:
---------------------------------
Unknown