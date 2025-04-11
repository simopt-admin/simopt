Simultaneous Perturbation Stochastic Approximation (SPSA)
=========================================================

See the :mod:`simopt.solvers.spsa` module for API details.

Description
-----------

SPSA is an iterative algorithm that approximates the gradient based on function evaluations taken at two points: one in a random direction and the other in the opposite direction.

Modifications & Implementation
------------------------------

SPSA's main feature is the gradient approximation that requires only two objective function measurements per iteration.
The gradient approximtation calculation used in this solver is the weigthed average of the two objective function measurements,
with the weights reflecting the distances between the two neighbors and the incumbent solution.

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
* alpha: Non-negative coefficient in the SPSA gain sequecence ak.
    * Default: 0.602
* gamma: Non-negative coefficient in the SPSA gain sequecence ck.
    * Default: 0.101
* step: The initial desired magnitude of change in the theta elements.
    * Default: 0.1
* gavg: Averaged SP gradients used per iteration.
    * Default: 1
* n_reps: Number of replications takes at each solution.
    * Default: 30
* n_loss: Number of loss function evaluations used in this gain calculation.
    * Default: 2
* eval_pct: Percentage of the expected number of loss evaluations per run.
    * Default: 2 / 3
* iter_pct: Percentage of the maximum expected number of iterations.
    * Default: 0.1

References
----------

This solver is adapted from the article Spall, J. C. (1998). Implementation of simultaneous perturbation algorithm for stochastic optimization. IEEE Transactions on Aerospace and Electronic Systems 34(3):817-823.
(https://ieeexplore.ieee.org/document/705889)