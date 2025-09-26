Stochastic Trust-Region Response-Surface Method (STRONG)
========================================================

See the :mod:`simopt.solvers.strong` module for API details.

Description
-----------

The solver estimates the shape of the underlying response distribution, 
through function evaluations taken within a neighborhood of the incumbent solution.
STRONG has two stages in each iteration where a sub trust region is defined: 
stage I optimizes a first-order polynomial, and stage II optimizes a second-order 
polynomial. If stage II fails to generate a good solution, an inner loop is initiated 
where value, gradient, and Hessian of the center point are further calculated.

Modifications & Implementation
------------------------------

Process within a stage:
We first find the Cauchy Point and the new solution in order to create a polynomial.
Then, the solver either shrinks trust region size, or moves the center point while the
trust region size stays constant, or moves the center point while the trust region enlarges.

Helper functions:
There are several helper functions in addition to the main algorithm:

* **cauchy_point** finds the Cauchy Point by using the gradient and Hessian matrix to find the steepest descent direction.
* **check_cons** checks the feasibility of the Cauchy point and updates the point accordingly.

Scope
-----

* objective_type: single
* constraint_type: box
* variable_type: continuous

Solver Factors
--------------

* crn_across_solns: Use CRN across solutions?
    * Default: True
* n0: Initial sample size
    * Default: 10
* n_r: Number of replications taken at each solution
    * Default: 10
* sensitivity: shrinking scale for VarBds
    * Default: 10**(-7)
* delta_threshold: maximum value of the radius
    * Default: 1.2
* delta_T: initial size of trust region
    * Default: 2
* eta_0: the constant of accepting
    * Default: 0.01
* eta_1: the constant of more confident accepting
    * Default: 0.3
* gamma_1: the constant of shrinking the trust regionthe new solution
    * Default: 0.9
* gamma_2: the constant of expanding the trust region
    * Default: 1.11
* lambda: multiplicative factor for n_r within finite difference
    * Default: 2
* lambda_2: magnifying factor for n_r in stage I and stage II
    * Default: 1.01

References
----------

This solver is adapted from the article Kuo-Hao Chang, L. Jeff Hong, Hong Wan, (2013).
Stochastic Trust-Region Response-Surface Method (STRONG) - A New Response-Surface Framework for Simulation Optimization.
*INFORMS Journal on Computing*, 25(2):230-243.
(https://doi.org/10.1287/ijoc.1120.0498)
