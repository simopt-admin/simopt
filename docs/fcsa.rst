Solver: FCSA
============

Description:
------------
The solver is for efficient stochastic optimization with stochastic constraints 
that only requires first-order gradients of the objective function and constraints.
The method alternatively computes solutions using gradients, and improves feasibility
if the current solution is infeasible. The search direction to improve feasibility
can be found by either using the gradient of the most violated constraint or
solving an NLP for the search direction with the maximum angle to all violated constraints (with the option to included the
objective gradient as well). When the current solution is feasible, only the objective gradient is used to determine
the search direction. The next solution then is computed using prox-mapping.



Modifications & Implementation:
-------------------------------

**get_FD_grad**: get finite differencing estimate of objective gradients for problems where gradient is unavailable

**get_violated_constraints_grad**: gets gradients of all violated constraints

**get_constraints_dir**: determines the search direction for infeasible iterations by finding the direction that maximizes
the angle between all violated constraints and objective constraint (optional).

**prox_fn**: prox-mapping function used to determine the next solution while satisfying deterministic constraints


Scope:
------
* objective_type: single

* constraint_type: stochastic and deterministic

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

* step_type : type of step size used by prox to find next solution. Can be "const" or "decay".

	* Default: "const"
    
* step_mult : value of step size for constant, multiplier of k (iterration index) for decaying.
	
    * Default : 0.1

* search_direction : determines how solver finds the search direction. CSA: use gradient of most violated constraint, CSA-M: solve NLP with all violated constraints, FCSA: solve NLP with all violated constraints and objective gradient

    * Default: "FCSA"

* normalize_grads : normalize gradients used for search direction calculations?

    * Default : True

* feas_const : feasibility constant used to relax objective constraint in the search direction problem (FCSA only)

    * Default : 0.0

* feas_score : degree of feasibility score used to relax objective constraint in the search direction problem (FCSA only)

    * Default : 2

References:
===========
This solver is adapted from the article Lan G., & Zhou Z. (2020). Algorithms for stochastic optimization with functional or expectation constraints. arXiv preprint arXiv:1604.03887.
With modifications adapted from Felice, N. et. al. (2025). Diagnostic Tools for Evaluating Solvers for Stochastically Constrained Simulation Optimization Problems. (submitted for publication)