Solver: Fully Cooperative Stochastic Approximation (FCSA)
=========================================================

See the :mod:`simopt.solvers.fcsa` module for API details.

Description
-----------

The **FCSA solver** is designed for efficient **stochastic optimization** problems with **stochastic constraints**, requiring only **first-order (gradient-based) information** of the objective and constraint functions.

The algorithm alternates between two phases:

- **Descent steps** that use stochastic gradient estimates to improve the objective value.
- **Feasibility-improvement steps** that are triggered when the current solution violates constraints.

When infeasible, the solver determines a search direction to restore feasibility by either:

1. Using the gradient of the most violated constraint (**CSA**), or
2. Solving a nonlinear program (NLP) to find the direction that forms the **maximum angle** with all violated constraints (**CSA-N**), optionally incorporating the objective gradient (**FCSA**).

When the current solution is feasible, the objective gradient alone is used to determine the search direction.
The next iterate is then computed using the **proximal mapping**.

Modifications & Implementation
------------------------------

**_direction**: Determines the search direction for infeasible iterations by finding the direction that maximizes the angle among all violated constraints, optionally including the objective constraint.

**_prox_fn**: Proximal mapping function used to compute the next solution while ensuring satisfaction of deterministic constraints.

Scope
-----

* objective_type: single
* constraint_type: stochastic and deterministic
* variable_type: continuous

Solver Factors
---------------

* **crn_across_solns**: Use Common Random Numbers (CRN) across solutions?
    *Default:* ``True``

* **r**: Number of replications performed at each solution.
    *Default:* ``30``

* **h**: Finite-difference parameter used for gradient approximation.
    *Default:* ``0.1``

* **tolerance**: Tolerance for checking feasibility.
    *Default:* ``1e-2``

* **step_type**: Type of step size.
    Options: ``"const"`` or ``"decay"``
    *Default:* ``"const"``

* **step_mult**: Step-size value for constant steps, or the multiplier of the iteration index ``k`` for decaying steps.
    *Default:* ``0.1``

* **search_direction**: Method used to determine the search direction.

    Options:

    - ``"CSA"``: use the gradient of the most violated constraint.
    - ``"CSA-N"``: solve an NLP involving all violated constraints.
    - ``"FCSA"``: solve an NLP involving all violated constraints and the objective gradient.

        *Default:* ``"FCSA"``


* **normalize_grads**: Normalize gradients used in search-direction calculations?
    *Default:* ``True``

* **feas_const**: Feasibility constant used to relax the objective constraint in the search-direction problem (FCSA only).
    *Default:* ``0.0``

* **feas_score**: Degree of feasibility score used to relax the objective constraint in the search-direction problem (FCSA only).
    *Default:* ``2``

References
----------

The solver is adapted from:

Lan, G., & Zhou, Z. (2020). *Algorithms for Stochastic Optimization with Functional or Expectation Constraints.*
*arXiv preprint arXiv:1604.03887.*

with further modifications from:

Felice, N., et al. (2025). *Diagnostic Tools for Evaluating Solvers for Stochastically Constrained Simulation Optimization Problems.*
(submitted for publication)
