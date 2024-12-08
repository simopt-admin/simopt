Solver: Active-set Method (ACTIVE-SET)
=============================================================================================

Description:
------------
The solver is dedicated for problems with linear constraints in the form of :math:`C_e x = de, C_i x \leq di`.
It employs the active-set method mentioned in Jorge Norcedal's Numerical Optimization.

Modifications & Implementation:
-------------------------------

**find_feasible_initial**: Find an initial feasible solution if one is not provided.

**line_search**: A back-tracking line search method.

**compute_search_direction**: Compute a search direction by solving a direction-finding quadratic subproblem at solution x.
It takes the index set of the active constraints (`W`), objective gradient at solution x (`g(x)`), and the concatenated constraint
coefficient matrix (`C`) as inputs and returns the optimal search direction along with the Lagrange multipliers of the active constraints.

.. math::

    \begin{align}
    \min && (1/2)|| d ||^2+ g(x)^T d \\\\
    \text{s.t.} & C_k^T d = 0, \quad \forall k \in W \\
    \end{align}

**_feasible**:  Check whether a solution x is feasible to the problem.

**finite_diff**: Approximate objective gradient using the finite difference method.

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

* tol2: Floating point tolerance for checking closeness of dot product to zero

    * Default: 1e-7

* finite_diff_step: Step size for finite difference

    * Default: 1e-5

References:
===========
Nocedal, J., & Wright, S. J. (2006). Numerical optimization (2nd ed.). Springer.