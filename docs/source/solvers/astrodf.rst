Adaptive Sampling Trust-Region Optimization for Derivative-Free Simulations (ASTRODF)
=====================================================================================

See the :mod:`simopt.solvers.astrodf` module for API details.

Description
-----------

The solver progressively builds local models (quadratic with diagonal Hessian) using interpolation on a set of points on the coordinate bases of the best (incumbent) solution. Solving the local models within a trust region (closed ball around the incumbent solution) at each iteration suggests a candidate solution for the next iteration. If the candidate solution is worse than the best interpolation point, it is replaced with the latter (a.k.a. direct search). The solver then decides whether to accept the candidate solution and expand the trust-region or reject it and shrink the trust-region based on a success ratio test. The sample size at each visited point is determined adaptively and based on closeness to optimality.

Modifications & Implementation
------------------------------

**construct_model**: Construct the "qualified" local model for each iteration k with the center point x_k, reconstruct with new points in a shrunk trust-region if the model fails the criticality condition. The criticality condition keeps the model gradient norm and the trust-region size in lock-step.

**evaluate_model**: Find (proxy to) the subproblem solution.

**get_stopping_time**: Decide whether to stop at the existing sample size for a given solution.

**get_coordinate_vector**: Form the coordinate basis.

**get_rotated_basis**: Form the rotated coordinates, where the first vector comes from the visited design points.

**get_model_coefficients**: Compute the model coefficients using (2d+1) design points and their function estimates by solving a system of linear equations (interpolation).

**get_coordinate_basis_interpolation_points**: Compute the interpolation points (2d+1) using the coordinate basis.

**get_rotated_basis_interpolation_points**: Compute the interpolation points (2d+1) using the rotated coordinate basis to allow reusing one design point.

**iterate**: Run one iteration of trust-region algorithm by bulding and solving a local model and updating the current incumbent and trust-region radius, and saving the data.

Scope
-----

* objective_type: single
* constraint_type: box
* variable_type: continuous
* gradient_observations: not available

Solver Factors
--------------

* crn_across_solns: Use CRN across solutions?
    * Default: True
* eta_1: Threshhold for a successful iteration > 0, < 1.
    * Default: 0.1
* eta_2: Threshhold for a very successful iteration > eta_1, < 1.
    * Default: 0.8
* gamma_1: Trust-region radius increase rate after a very successful iteration > 1.
    * Default: 1.5
* gamma_2: Trust-region radius decrease rate after an unsuccessful iteration < 1, > 0.
    * Default: 0.5
* lambda_min: Minimum sample size value, integer > 2.
    * Default: 4
* seasy_solve: Solve the subproblem approximately with Cauchy point.
    * Default: True
* reuse_points: Reuse the previously visited points.
    * Default: True
* ps_sufficient_reduction: Use pattern search if with sufficient reduction, 0 always allows it, large value never does.
    * Default: 0.1

References
----------

This solver is adapted from the article Ha, Y., Shashaani, S., and Tran-Dinh, Q. (2021).
Improved Complexity Of Trust-Region Optimization For Zeroth-Order Stochastic Oracles with Adaptive Sampling
*Proceedings of 2021 Winter Simulation Conference*, doi: 10.1109/WSC52266.2021.9715529.
(https://ieeexplore.ieee.org/abstract/document/9715529)