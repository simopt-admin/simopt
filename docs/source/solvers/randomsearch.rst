Random Search (RNDSRCH)
=======================

See the :mod:`simopt.solvers.randomsearch` module for API details.

Description
-----------

Randomly sample solutions from the feasible region and use a fixed number of replications at each solution. The sampling distribution is specified inside each problem class in **get_random_solution**.

Modifications & Implementation
------------------------------

The new random solutions maintain the type of each variable based on the sampling distributions that are discrete for integer decisions and otherwise continuous.

Scope
-----

* objective_type: single
* constraint_type: stochastic
* variable_type: mixed
* gradient_observations: not available

Solver Factors
--------------

* crn_across_solns: Use CRN across solutions?
    * Default: True
* sample_size: Sample size per solution > 1.
    * Default: 10

References
----------

This solver is adapted from the article Chia, Y.L. and Glynn, P.W., (2013). 
Limit Theorems for Simulation-Based Optimization via Random Search. 
*ACM Transactions on Modeling and Computer Simulation (TOMACS)*, 23(3), pp.1-18.
(https://dl.acm.org/doi/abs/10.1145/2499913.2499915)