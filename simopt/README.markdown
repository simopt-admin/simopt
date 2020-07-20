# Base classes and functions

The `base` module includes base classes for solvers, problems, and oracles and miscellaneous functions.

## Oracles
The `Oracle` class implements simulation oracles (models).

New oracles can be added to the library by subclassing the `Oracle` class.

Any subclass <ins>**must**</ins> define the following attributes:
* `n_rng` : the number of random number generators needed to run a replication;
* `dim` : the number of decision variables;
* `n_responses` : the number of stochastic responses (performance measures);

and the following methods:
* `check_simulatable` : checks whether the system characterized by a given vector of decision variables can be simulated;
* `simulate` : generates a single simulation replication.

The `Oracle` class also features several useful methods:
* `attach_rngs` : assigns a set of random number generators to the oracle;
* `batch` : takes a fixed number of replications, using distinct subsubstreams for each replication.

## Miscellaneous functions

The `aggregate` function takes the outputs (responses and gradient estimates) from the replications run by `oracle.batch` and calculates the sample means and covariances.
