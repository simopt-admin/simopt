Solver: CSA_LP
============

Description:
------------
The solver is for efficient stochastic optimization with stochastic constraints 
that only requires first-order gradients of the objective function and constraints.
The method alternatively computes solutions using gradients, and improves feasibility
if the current solution is infeasible. The algorithm uses LP to solve for a direction
which improves feasibility for violated constraints. If the current solution is feasible, then it
computes the next iterate by using prox-mapping. 


Modifications & Implementation:
-------------------------------
At each timestep :math:`t`, we first evaluate the gradient w.r.t the current solution :math:`x`, either using
the IPA gradient estimates or finite difference estimates.

Then, the algorithm check the current solution's feasibility. If it is feasible, then it computes the gradient
and update the solution using a prox-mapping function with pre-determined step sizes. However, it the current
solution is infeasible, then the algorithm selects the worse constraint, or the most-violated constraint, and 
improve the feasibility using the gradients of the constraint. In these steps, the feasibility on the compact
set is maintained through the optimization in prox-mapping. 

Helper functions:
The finite_diff function uses finite difference methods to estimate the gradient of the
objective function. The function prox_fn computes the next solution using feasible set, 
gradients, and step size. The function get_constraints_dir computes a search direction 
to improve feasibility on the constraints violated. 


Scope:
------
* objective_type: single

* constraint_type: stochastic

* variable_type: continuous


Solver Factors:
---------------
* crn_across_solns: Use CRN across solutions?

    * Default: True

* r: Number of replications taken at each solution.

    * Default: 30

* h: Finite difference parameter.

    * Default: 0.1

* tolerance : A tolerance for checking feasibility

    * Default: 10^(-2)

* max_iters: maximum number of iterations

    * Default: 300


References:
===========
This solver is adapted from the article Lan G., & Zhou Z. (2020). Algorithms for stochastic optimization with functional or expectation constraints. arXiv preprint arXiv:1604.03887.
