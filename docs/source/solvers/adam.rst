ADAM
====

See the :mod:`simopt.solvers.adam` module for API details.

Description
-----------

The solver is for efficient stochastic optimization that only requires first-order gradients
with little memory requirement. The method computes individual adaptive learning rates for
different parameters from estimates of first and second moments of the gradients.

Modifications & Implementation
------------------------------

At each timestep :math:`t`, we first evaluate the gradient w.r.t the current solution :math:`x`, either using
the IPA gradient estimates or finite difference estimates.
Then, the algorithm updates exponential moving averages of the gradient :math:`m_t` and the squared gradient
:math:`v_t` where the hyper-parameters :math:`\beta_1` and :math:`\beta_2` control the exponential decay rates of 
these moving averages. The moving averages themselves are estimates of the 1st moment (the mean) and the
2nd raw moment (the uncentered variance) of the gradient. These moving averages are
initialized as (vectors of) 0's, leading to moment estimates that are biased towards zero, especially
during the initial timesteps, and especially when the decay rates are small (i.e., the :math:`\beta` are close to 1).
The bias-corrected estimates :math:`\hat{m_t}` and :math:`\hat{v_t}`. Lastly, the new solution can be found via
current solution, the bias-corrected first and second moment estimate, and the step size.

Scope
-----

* objective_type: single
* constraint_type: box
* variable_type: continuous

Solver Factors
--------------

* crn_across_solns: Use CRN across solutions?
    * Default: True
* r: Number of replications taken at each solution.
    * Default: 30
* beta_1: Exponential decay of the rate for the first moment estimates.
    * Default: 0.9
* beta_2: Exponential decay rate for the second-moment estimates.
    * Default: 0.999
* alpha: Step size.
    * Default: 0.5
* epsilon: A small value to prevent zero-division.
    * Default: 10^(-8)
* sensitivity: shrinking scale for VarBds
    * Default: 10^(-7)

References
----------

This solver is adapted from the article Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.
