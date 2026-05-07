Ambulance Dispatch Model
========================


Model: Ambulance Dispatch (AMBULANCE)
--------------------------------------

Description
^^^^^^^^^^^

We consider an ambulance dispatch system on a square service region.
Fixed bases and variable bases are located on the plane. Calls arrive
stochastically and are served by the **nearest available** ambulance.
If no ambulance is available at arrival, the call **waits in a FIFO queue**
until a unit becomes free.

For each call, the **response time** is
:math:`R = W + D`, where

- :math:`W` is the **waiting time** in queue (zero if served immediately),
- :math:`D` is the **travel time** from the dispatched base to the call
  location (Manhattan distance divided by speed).

Service of a call includes an outbound travel leg :math:`D`, on-scene time,
and a return leg :math:`D` back to base. The nearest-ambulance rule is used
at both immediate dispatch and when releasing a call from the queue.


Sources of Randomness
^^^^^^^^^^^^^^^^^^^^^

1. **Call arrivals** follow a Poisson process (exponential interarrival times).
2. **Call locations** :math:`(X, Y)` are drawn from Beta distributions on each axis
   and scaled to the service square, allowing spatial hot spots.
3. **On-scene service time** is exponential with a given mean.
4. Independent random streams are used for arrivals, scene times, and the two spatial axes.

Model Factors
^^^^^^^^^^^^^

* ``fixed_base_count``: Number of fixed bases (do not move).
    * Default: user-specified
* ``variable_base_count``: Number of variable bases (decision variables).
    * Default: user-specified
* ``fixed_locs``: Flattened coordinates of fixed bases ``[x0, y0, x1, y1, ...]``.
    * Default: user-specified
* ``variable_locs``: Flattened coordinates of variable bases ``[x0, y0, x1, y1, ...]``.
    * Default: user-specified
* ``call_loc_beta_x``: Beta shape parameters for the X-axis (``alpha_x, beta_x``).
    * Default: (2.0, 1.0)
* ``call_loc_beta_y``: Beta shape parameters for the Y-axis (``alpha_y, beta_y``).
    * Default: (2.0, 1.0)
* ``mean_scene_time``: Mean on-scene service time (minutes).
    * Default: ``50 / (fixed_base_count + variable_base_count)``

Responses
^^^^^^^^^

* ``avg_response_time``: Sample mean of :math:`R = W + D` over served calls.

References
^^^^^^^^^^

This model setup is based on the ambulance base location problem described
in *Biased Gradient Estimators in Simulation Optimization*  
(Eckman, D. J. and Henderson, S. G., Proceedings of the 2020 Winter Simulation Conference).  

Optimization Problem: Minimize Average Response Time (AMBULANCE-1)
------------------------------------------------------------------

Decision Variables
^^^^^^^^^^^^^^^^^^

* ``variable_locs``

Objectives
^^^^^^^^^^

Minimize :math:`E[R]`, the expected response time averaged over calls,
with :math:`R = W + D` as defined above. An IPA estimator of the gradient
with respect to ``variable_locs`` is implemented in code using a carry term
that propagates the effect of waiting across consecutive calls handled by
the same ambulance (no special symbols are used in code comments).

Constraints
^^^^^^^^^^^

* Each variable base must lie inside the service region:
  :math:`0 \le x, y \le \text{square\_width}`.

Problem Factors
^^^^^^^^^^^^^^^

* ``budget``: Max number of replications a solver may take.
    * Default: 1000
* ``rng_streams``: Separate streams for arrivals, scene times, and spatial axes.
    * Default: provided by the framework

Fixed Model Factors
^^^^^^^^^^^^^^^^^^^

* ``fixed_base_count``, ``fixed_locs``

Starting Solution
^^^^^^^^^^^^^^^^^

* ``initial_solution``: Place each variable base at coordinates ``(6, 6)``.

Random Solutions
^^^^^^^^^^^^^^^^

Sample each variable base uniformly over the square region or from the
specified Beta-shaped spatial prior used for call locations.

Optimal Solution
^^^^^^^^^^^^^^^^

Unknown

Optimal Objective Function Value
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Unknown
