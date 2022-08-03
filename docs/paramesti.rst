Model: Parameter Estimation (PARAMESTI)
=======================================

Description:
------------
A model that simulates maximum likelihood estimation for the parameters of
a two-dimensional gamma distribution.

Say a simulation generates output data :math:`{Y_j}`, :math:`Y_j \in [0, \infty] \times [0, \infty]`,
that are i.i.d and known to come from a distribution with the two-dimensional density function

.. math:: f(y1, y2; x^*) = \frac{e^{-y1} y_1^{x^*_1 y_2 - 1}{\Gamma(x^*_1 y_2)} \frac{e^{-y2} y_2^{x^*_2 - 1}{\Gamma(x^*_2)}, y1, y2 > 0,
    
where :math:`x^* ≡ (x^*_1, x^*_2)`` is the unknown vector of parameters.

Noting that :math:`x_star` maximizes the function

.. math:: g(x) = E [log (f(Y ; x))] = \int_0^\infty \log (f(y; x)) f(y; x^*)dy,

and that

.. math:: G_m(x) = \frac{1}{m} \sum_{j=1}^m \log(f(Y_j ; x))

is a consistent estimator of :math:`g(x)`.
Observations are generated from the distribution specified by a given :math:`x_star`.

Sources of Randomness:
----------------------
y is a 2-D vector that contributes randomness. Both elements of y are gamma random variables.

Model Factors:
--------------
* x_star: the unknown 2-D parameter that maximizes the expected log likelihood function.

    * Default: [2, 5]

* x: a 2-D variable in the probability density function.

    * Default: [1, 1]

Respones:
---------
* loglik: log likelihood of the pdf.

References:
===========
This model is designed by Raghu Pasupathy (Virginia Tech) and Shane G. Henderson (Cornell) in 2007.


Optimization Problem: Max Log Likelihood (ParamEstiMaxLogLik)
=============================================================

Decision Variables:
-------------------
* x

Objectives:
-----------
Minimize the log likelihood of a 2-D gamma random variable.

Constraints:
------------
x is in the square (0, 10) × (0, 10).

Problem Factors:
----------------
* budget: Maximum number of replications for a solver to take.

  * Default: 1000

Fixed Model Factors:
--------------------
N/A

Starting Solution: 
------------------
* x: [1, 1]

Random Solutions: 
-----------------
Generate :math:`x` i.i.d. uniformly in the square (0, 10) × (0, 10).

Optimal Solution:
-----------------
x = [2, 5]

Optimal Objective Function Value:
---------------------------------
Known, but not evaluated.
