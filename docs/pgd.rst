Solver: Projected Gradient Descent (PGD)
=============================================================================================

Description:
------------
The solver is dedicated for problems with linear constraints in the form of :math:`C_e x = de, C_i x \leq di`. 
It is based on the standard line-search gradient descent algorithm with new solutions being projected back to the feasible region.

Modifications & Implementation:
-------------------------------

**find_feasible_initial**: Find an initial feasible solution if one is not provided.

**line_search**: A back-tracking line search method.

**project_grad**: Project the vector x onto the feasible hyperplane 
by solving a quadratic projection problem below. (`x + d`) will be the projected vector.

.. math::

    \begin{align}
    \min && d^T d \\\\
    \text{s.t.} & A_e(x + d) = b_e \\
    A_i(x + d) \leq b_i \\
    x + d \geq lb \\
    x + d \leq ub
    \end{align}
        

**_feasible**:  Check whether a solution x is feasible to the problem.

**finite_diff**: Approximate objective gradient using the finite difference method.

Scope:
------
* objective_type: single

* constraint_type: box, deterministic (linear)

* variable_type: continuous

* gradient_observations: not available

Solver Factors:
---------------
* crn_across_solns: Use CRN across solutions?

    * Default: True
    
* r: number of replications taken at each solution

    * Default: 30

* alpha: Tolerance for sufficient decrease condition.

    * Default: 0.2

* beta: Step size reduction factor in line search.

    * Default: 0.9

* alpha_max: Maximum step size

    * Default: 10.0

* lambda: Magnifying factor for r inside the finite difference function

    * Default: 2

* tol: Floating point tolerance for checking tightness of constraints

    * Default: 1e-7

* finite_diff_step: Step size for finite difference

    * Default: 1e-5


References:
===========
N/A