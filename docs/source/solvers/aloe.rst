Adaptive Line-search with Oracle Estimations (ALOE)
===================================================

See the :mod:`simopt.solvers.aloe` module for API details.

Description
-----------

The solver is a stochastic line search algorithm  with the gradient estimate recomputed in each iteration,
whether or not a step is accepted. The algorithm includes the relaxation of the Armijo condition by 
an additive constant :math:`2\epsilon_f`.

Modifications & Implementation
------------------------------

For each iteration, first compute the gradient approximation :math:`g_k` using either
the IPA gradient estimates or finite difference estimates.
Then, the algorithm checks for sufficient decrease. Let :math:`x_k^{+} = x_k - \alpha_k g_k`. Estimate the objective
values :math:`f(x_k^{+})` and :math:`f(x_k)`. Check the modified Arimjo condition:

.. math::
   f(x_k^{+}) \leq f(x_k) - \alpha_k \theta ||g_k||^2 + 2\epsilon_f.

If the condition holds, then set :math:`x_{k+1} \leftarrow x_{k}` and :math:`\alpha_{k+1} \leftarrow \min\{ \alpha_{max}, \gamma^{-1}\alpha_k \}`.
Otherwise, set :math:`x_{k+1} \leftarrow x_{k}` and :math:`\alpha_{k+1} \leftarrow \gamma \alpha_k`

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
* theta: Constant in the Armijo condition.
    * Default: 0.2
* gamma: Constant for shrinking the step size.
    * Default: 0.8
* alpha_max:  Maximum step size.
    * Default: 10
* alpha_0:  Initial step size.
    * Default: 1
* epsilon_f: Additive constant in the Armijo condition.
    * Default: 1
* sensitivity: Shrinking scale for variable bounds.
    * Default: 10^(-7)

References
----------

This solver is adapted from the article Jin, B., Scheinberg, K., & Xie, M. (2021). High probability complexity bounds for line search based on stochastic oracles. Advances in Neural Information Processing Systems, 34, 9193-9203.
