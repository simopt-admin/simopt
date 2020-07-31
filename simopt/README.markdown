# Base classes and functions

The `base` module includes base classes for solvers, problems, oracles and solutions.

## `Problem` class
New problems can be added to the library by subclassing the `Problem` class.

Any subclass <ins>**must**</ins> define the following attributes:
* `minmax` : indicator of maximization (+1) or minimization (-1);
* `dim` : the number of decision variables;
* `oracle` : associated `Oracle` object for generating replications;

and the following methdos:
* `vector_to_factor_dict` : converts a vector of variables to a dictionary with factor keys;
* `list_of_factor_dict_to_vector` : converts a list of dictionaries with factor keys to a vector of variables.

The `Oracle` class also features several useful methods:
* `simulate` : takes a fixed number of replications, using distinct subsubstreams for each replication.

## `Oracle` class
New oracles can be added to the library by subclassing the `Oracle` class.

Any subclass <ins>**must**</ins> define the following attributes:
* `n_rng` : the number of random number generators needed to run a replication;
* `n_responses` : the number of stochastic responses (performance measures);
* `specifications` : dictionary describing factors and data requirements;
* `check_factor_list` : a dictionary of functions for checking that individual factors describe a simulatable model;

and the following methods:
* `check_simulatable_factors` : checks whether the oracle characterized by all of the factors can be simulated;
* `replicate` : generates a single simulation replication.

The `Oracle` class also features several useful methods:
* `attach_rngs` : assigns a set of random number generators to the oracle;
* `check_simulatable_factor` : checks whether a given factor satisfies its constraints on simulatability and data type requirements.

## `Solution` class
Objects in the `Solution` class store the outputs of past replications taken a given solution.
Attributes include:
* `x` : the vector of decision variables describing the solution;
* `dim` : the length of `x`;
* `decision_factors` : a dictionary of decision factors describing the solution;
* `n_reps` : the number of replications taken at the solution thus far;
* `responses` : the performance measures of interest from each replication; and
* `gradients` : the gradients of the responses from each replication.

The `Solution` class includes the following methods:
* `response_mean`, `response_var`, `response_std_error`, `response_cov` : calculate summary statistics of the responses; and
* `gradient_mean`, `gradient_var`, `gradient_std_error`, `gradient_cov` : calculate summary statistics of the gradients.
