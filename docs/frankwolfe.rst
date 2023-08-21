Solver: Frank-Wolfe (FW)
=============================================================================================

Description:
------------
The solver is dedicated for problems with linear constraints in the form of :math:`C_e x = d_e, C_i x \leq d_i`.
It employs the Frank-Wolfe algorithm with adaptive step search.

Modifications & Implementation:
-------------------------------

**find_feasible_initial**: Find an initial feasible solution if one is not provided.

**_feasible**:  Check whether a solution x is feasible to the problem.

**finite_diff**: Approximate objective gradient using the finite difference method.

**search_dir**: Compute a search direction by solving a direction-finding linear subproblem at solution x.

.. math::

    \begin{align}
    \min && (1/2)|| d ||^2+ g(x)^T d \\\\
    \text{s.t.} A_e(x + d) = b_e \\
             A_i(x + d) \leq b_i \\
             (x + d) \geq lb \\
             (x + d) \leq ub
    \end{align}

Scope:
------
* objective_type: single

* constraint_type: box,  deterministic (linear)

* variable_type: continuous

* gradient_observations: not available

Solver Factors:
---------------
* crn_across_solns: Use CRN across solutions?

    * Default: True
    
* r: number of replications taken at each solution

    * Default: 30

* theta: Constant in the Armijo condition.

    * Default: 0.2

* gamma: Constant for shrinking the step size.

    * Default: 0.8

* alpha_max: Maximum step size

    * Default: 10.0

* alpha_0: initial step size.

    * Default: 1

* epsilon_f: Additive constant in the Armijo condition.

    * Default: 1

* lambda: Magnifying factor for r inside the finite difference function

    * Default: 2

* tol: Floating point tolerance for checking tightness of constraints

    * Default: 1e-7

References:
===========
Frank, M., & Wolfe, P. (1956). An algorithm for quadratic programming. Naval research logistics quarterly, 3(1-2), 95-110.