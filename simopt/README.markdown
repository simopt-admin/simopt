# Base classes and functions

The `base` module includes base classes for solvers, problems, oracles and solutions.

## `Oracle` class
New oracles can be added to the library by subclassing the `Oracle` class.

Any subclass <ins>**must**</ins> define the following attributes:
* `n_rng` : the number of random number generators needed to run a replication;
* `dim` : the number of decision variables;
* `n_responses` : the number of stochastic responses (performance measures);

and the following methods:
* `check_simulatable_params` : checks whether the oracle characterized by the given parameters can be simulated;
* `check_simulatable_x` : checks whether the system characterized by `x` can be simulated; and
* `replicate` : generates a single simulation replication.

The `Oracle` class also features several useful methods:
* `attach_rngs` : assigns a set of random number generators to the oracle;
* `simulate` : takes a fixed number of replications, using distinct subsubstreams for each replication.

## `Solution` class
Objects in the `Solution` class store the outputs of past replications taken a given solution.
Attributes include:
* `x` : the vector of decision variables describing the solution;
* `dim` : the length of `x`;
* `n_reps` : the number of replications taken at the solution thus far;
* `responses` : the performance measures of interest from each replication; and
* `gradients` : the gradients of the responses from each replication.

The `Solution` class includes the following methods:
* `response_mean`, `response_var`, `response_std_error`, `response_cov` : calculate summary statistics of the responses; and
* `gradient_mean`, `gradient_var`, `gradient_std_error`, `gradient_cov` : calculate summary statistics of the gradients.