Constrained Adaptive Sampling Trust-Region Optimization for Derivative-Free Simulations (C-ASTRO-DF)
==================================================================================================

See the :mod:`simopt.solvers.castrodf` module for API details.

Description
-----------

The solver progressively builds local models (quadratic with diagonal Hessian) using interpolation on a set of points around the incumbent solution. Each iteration solves the model within a trust region using linearized constraints: a normal step reduces constraint violation and a tangent step reduces the objective model within the null space of the constraint Jacobian. A penalty merit function and a success ratio test decide whether to accept the candidate and expand the trust region, or reject it and shrink it. The sample size at each visited point is determined adaptively and based on closeness to optimality.

Notes & Limitations
-------------------

* **Parameter tuning is warranted.** Default values are not necessarily optimal.
* **Direct search (pattern search) is currently disabled.**
* **Caution with inequality constraints.** Slack initialization and barrier penalty management are still a work in progress.
* **SciPy can perform poorly on the normal/tangent steps.** If results are poor, try both ``easy_solve_normal`` and ``easy_solve_tangent`` set to True/False.
* **Adaptive sampling may unnecessarily blow up the sample size.** Work in progress.
* **The problem must define** ``get_deterministic_equality_constraints``, ``get_deterministic_inequality_constraints``, ``get_deterministic_equality_constraints_gradients``, ``get_deterministic_inequality_constraints_gradients``, and ``get_deterministic_constraints_hessian`` before running.

Implementation
--------------

**construct_model**: Build the local model at the incumbent, shrinking the trust region and rebuilding if the criticality condition fails.

**get_model_coefficients**: Fit the model by interpolating (2d+1) design points.

**solve_normal_step / solve_tangent_step**: Compute the composite step. Each uses either a Cauchy point (``easy_solve_*`` True) or SciPy (False). The normal step can also use a dogleg path (``dogleg``).

**iterate**: Run one iteration: build the model, compute the step, evaluate the candidate, update the penalty parameter, incumbent, and trust-region radius.

Interpolation set management: by default the interpolation points are rebuilt each iteration (reusing one visited point if ``reuse_points``). With ``reuse_interpolation_set`` a persistent set is kept and updated on successful iterations, with Lagrange-polynomial-based point replacement always on.

Scope
-----

* objective_type: single
* constraint_type: deterministic (equality, inequality, and box)
* variable_type: continuous
* gradient_observations: not used

Solver Factors
--------------

* crn_across_solns: Use CRN across solutions?
    * Default: True
* eta_1: Threshold for a successful iteration, > 0.
    * Default: 0.1
* eta_2: Threshold for a very successful iteration, >= eta_1.
    * Default: 0.8
* gamma_1: Trust-region radius increase rate after a very successful iteration, > 1.
    * Default: 2.5
* gamma_2: Trust-region radius decrease rate after an unsuccessful iteration, > 0, < 1.
    * Default: 0.5
* lambda_min: Minimum sample size, integer > 2.
    * Default: 5
* reuse_points: Reuse previously visited points (only when reuse_interpolation_set is False).
    * Default: True
* reuse_interpolation_set: Maintain a persistent interpolation set across iterations.
    * Default: False
* dist_threshold: With a persistent set, the farthest point is replaced when its squared distance exceeds dist_threshold * delta_k^2.
    * Default: 10.0
* delta_0: Initial trust-region radius.
    * Default: 1
* delta_max: Maximum trust-region radius.
    * Default: 100
* mu: Criticality measure weight factor.
    * Default: 0.1
* a_normal: Fraction of the trust region for the normal step.
    * Default: 0.5
* a_tangent: Fraction of the trust region for the tangent step.
    * Default: 0.5
* easy_solve_normal: Compute the normal step with the Cauchy point.
    * Default: False
* easy_solve_tangent: Compute the tangent step with the Cauchy point.
    * Default: False
* dogleg: Use the dogleg method for the normal step.
    * Default: True
* nu: Ensures model improvement when the normal step worsens the objective.
    * Default: 0.001
* tau_1: Penalty parameter increase factor.
    * Default: 2.0
* tau_2: Penalty parameter increase constant.
    * Default: 1.0
* sigma_min: Initial penalty parameter.
    * Default: 10.0
* sigma_b_max: Upper bound on sigma_b (norm of the Lagrange multipliers).
    * Default: 1e8
* feas_tol: Feasibility tolerance.
    * Default: 1e-8

References
----------

This solver is adapted from the article Felice, N., Shashaani, S., & Roberts, L. (2026). Adaptive Sampling Trust Region Optimization for Derivative-free Stochastic Functions and Deterministic Equality Constraints. https://arxiv.org/abs/2608.15894
