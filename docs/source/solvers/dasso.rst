Dice and Slice Simulation Optimization (DASSO)
====

See the :mod:`simopt.solvers.dasso` module for API details.

Description
-----------

The solver is for problems involving computationally expensive simulations having high-dimensional, discrete decision-variable spaces. It decomposes the prior distribution on the objective function into a rigorously justified additive form, reducing the problem dimensionality to facilitate computationally efficient posterior updates, but without losing a full-dimensional representation. It then works iteratively between posteriors on the low-dimensional “dice” and a full-dimensional “slice” of the decision-variable space to identify the best solution to simulate on each iteration.

Modifications & Implementation
------------------------------

**_Mapping**: An object with the ability to map from solutions to coordinates and vice versa.

**_Group**: An object that contains the information of a group when the hyperparameters are given.

**_hyperparameter_estimation**: Estimate the hyperparameters for each group and the overall mean.

**_estimate_random_effect_variance**: Estimate the random-effect variance via maximum likelihood estimation.

**_estimate_theta**: Estimate hyperparameters of the precision matrix via MLE.

Scope
-----

* objective_type: single
* constraint_type: box
* variable_type: discrete

Solver Factors
--------------

* crn_across_solns: Use CRN across solutions?
    * Default: True
* r: Number of replications taken at each solution.
    * Default: 30
* n_points_for_estimation: Number of reference points to be used for parameter estimation.
    * Default: 10
* decomposition: Decomposition of dimensions; two groups of the same dimension are considered unless decomposition is specified.
    * Default: []

References
----------

This solver is adapted from the article Avci, H., et al. (2014). Dice and Slice Simulation Optimization for High-Dimensional Discrete Problems. European Journal of Operational Research.
