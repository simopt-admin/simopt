Solver: Adaptive Sampling Trust-Region Optimization for Derivative-Free Simulations (ASTRODF)
=============================================================================================

Description:
------------
The solver progressively builds local models (quadratic with diagonal Hessian) using interpolation on a set of points on the coordinate bases of the best (incumbent) solution. Solving the local models within a trust region (closed ball around the incumbent solution) at each iteration suggests a candidate solution for the next iteration. If the candidate solution is worse than the best interpolation point, it is replaced with the latter (a.k.a. direct search). The solver then decides whether to accept the candidate solution and expand the trust-region or reject it and shrink the trust-region based on a success ratio test. The sample size at each visited point is determined adaptively and based on closeness to optimality.

Modifications & Implementation:
-------------------------------

**construct_model**: Take the model and accept it if its gradient is large relative to the trust-region radius.

**evaluate_model**: Find (proxy to) the subproblem solution.

**get_stopping_time**: Decide whether to stop at the existing sample size for a given solution.

**get_standard_basis**: Form the coordinate basis.

**get_model_coefficients**: Solve system of linear equations (interpolation) to obtain model coefficients.

**get_interpolation_points**: Choose coordinate bases.

**tune_parameters**: The initial trust-region radius is tuned before starting the search by choosing one of three choices.


Scope:
------
* objective_type: single

* constraint_type: box

* variable_type: continuous

* gradient_observations: not available

Solver Factors:
---------------
* crn_across_solns: Use CRN across solutions?

    * Default: True

* delta_max: Maximum value of the trust-region radius > 0.

    * Default: 200

* eta_1: Threshhold for a successful iteration > 0, < 1.

    * Default: 0.1

* eta_2: Threshhold for a very successful iteration > eta_1, < 1.

    * Default: 0.5

* gamma_1: Very successful step trust-region radius increase > 1.

    * Default: 1.5

* gamma_2: Unsuccessful step trust-region radius decrease < 1, > 0.

    * Default: 0.75

* w: Trust-region radius rate of shrinkage in contracation loop > 0, < 1.

    * Default: 0.85

* mu: Trust-region radius ratio upper bound in contraction loop > 0.

    * Default: 1000

* beta: Trust-region radius ratio lower bound in contraction loop < mu, > 0.

    * Default: 10

* lambda_min: Minimum sample size value, integer > 2.

    * Default: 4

* simple_solve: Subproblem solver with Cauchy point or the built-in solver? True - Cauchy point, False - built-in solver.

    * Default: True

* criticality_select: True - skip contraction loop if not near critical region, False - always run contraction loop.

    * Default: True

* criticality_threshold: Threshold on gradient norm indicating near-critical region.

    * Default: 0.1


References:
===========
This solver is adapted from the article Ha, Y., Shashaani, S., and Tran-Dinh, Q. (2021).
Improved Complexity Of Trust-Region Optimization For Zeroth-Order Stochastic Oracles with Adaptive Sampling
*Proceedings of 2021 Winter Simulation Conference*, doi: 10.1109/WSC52266.2021.9715529.
(https://ieeexplore.ieee.org/abstract/document/9715529)