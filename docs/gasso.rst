Solver: Gradient-Based Adaptive Stochastic Search for Simulation Optimization Over Continuous Space (GASSO)
================================================================

Description:
------------
The solver iteratively generates population of candidate solutions from a sample distribution,
and uses the performance of sample distribution to update the sampling dsitribution. 
GASSO has two stages in each iteration: 
1. Stage I: Generate candidate solutions from some exponential family of distribution, and the 
2. Stage II: Evaluate candidate solutions, and update the parameter of sampling distribution via 
direct gradient search. 

Scope:
------
* objective_type: single

* constraint_type: box

* variable_type: continuous

Solver Factors:
---------------
* crn_across_solns: Use CRN across solutions?

    * Default: True

* N: Number of candidate solutions

    * Default: :math:`50 * \sqrt(dim)`

* M: Number of function evaluations per candidate

    * Default: 10

* K: Number of iterations

    * Default: Budget/(N * M)

* alpha_0: Determines the initial step size

    * Default: 50

* alpha_c: Determines the speed at which the step size decreases

    * Default: 1500

* alpha_p: Determines the rate at which step size gets smaller

    * Default: 0.6

* alpha_k: Step size

    * Default: :math:`\frac{alpha_0}{(k + \alpha_c) ^ {alpha_p}}`


References:
===========
This solver is adapted from the article Enlu Zhou, Shalabh Bhatnagar (2018).
Zhou, E., & Bhatnagar, S. (2017). Gradient-based adaptive stochastic search for simulation optimization over continuous space. 
*INFORMS Journal on Computing, 30(1), 154-167.  
(https://doi.org/10.1287/ijoc.2017.0771)
