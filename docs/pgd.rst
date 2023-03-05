Solver: Projected Gradient Descent (PGD)
=============================================================================================

Description:
------------
The solver is dedicated for problems with linear constraints in the form of Ce@x = de, Ci@x <= di. It is based on the standard gradient descent algorithm with new solutions being projected back to the feasible region.

Modifications & Implementation:
-------------------------------

**find_feasible_initial**: Find an initial feasible solution if one is not provided.

**line_search**: A back-tracking line search method.

**project_grad**: Project the vector x onto the feasible hyperplane by solving a quadratic projection problem:
        .. math::

        min d^Td
        s.t. Ae(x + d) = be
             Ai(x + d) <= bi
             (x + d) >= lb
             (x + d) <= ub
        
        (x + d) will be the projected vector.

**_feasible**:  Check whether a solution x is feasible to the problem.

**finite_diff**: Approximate objective gradient using the finite difference method.

A step-by-step description of the solver is detailed below:
1.
2.
3.

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


References:
===========
N/A