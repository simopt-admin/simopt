Model: Emergency Medical Service Volunteer (VOLUNTEER)
======================================================

Description:
------------

Out of hospital cardiac arrest (OHCA) requires immediate treatment and patient survival can
be improved by combining traditional ambulance response with the dispatch of volunteers alerted via an
app. How many volunteers are needed, and from where should they be recruited?

We subdivide a large square (city) into :math:`l` squares, each of size 500 meters x 500 meters. Let 
:math:`\lambda_i` = :math:`P` (OHCA call in square :math:`i`), :math:`i = 1, 2, \ldots, l`, so that
:math:`\sum_{i = 1}^{l}\lambda_i = 1`. Within each square, assume the location of the call is uniformly
distributed. Let the mean measure of available volunteers on the mean square be :math:`\mu(\cdot) = a\upsilon(\cdot)`,
where :math:`a = E` (# available volunteers) and :math:`\upsilon(\cdot)` is a probability measure that is assumed
to be uniform in each square. Let :math:`x_i = \upsilon(square_i)` be the probability an available volunteer is
in square :math:`i`, :math:`i = 1, 2, \ldots, l` so that , :math:`\sum_{i = 1}^{l} x_i = 1`.

We want to choose the :math:`x_i`'s to either

1. Minimize :math:`P` (closest volunteer > :math:`r` meters from OHCA)

2. Maximize :math:`P` (patient survives).

For objective #2, assume we have a function :math:`s` that maps the distance of the closest volunteer to
:math:`P` (survival), which is a survival function described by De Maio et al. (2003):

.. math::

   s(t) = (1 + exp(0.679+0.262t)^{-1})

We assume OHCA events do not overlap (no queueing). Let :math:`X \in \mathbb{R}^2` be the random location
of an OHCA and let :math:`Y_1, Y_2, \ldots, Y_N` be the random locations of :math:`N \sim Poisson(a)` volunteers when
the OHCA happens. The distance of the closest volunteer is

.. math::

   D = (\min_{i = 1, \dots, N}||X - Y_i||)^{1/2},

which equals :math:`\infty` if :math:`N = 0`.


Sources of Randomness:
----------------------
There are three sources of randomness.

1. The number of volunteers available follows a Poisson distribution.

2. The locations of the volunteers follows a Poissson point process.

3. The location of the OHCA is distributed according to P_OHCA.


Model Factors:
--------------
* mean_vol: Mean number of available volunteers.

    * Default: 1600

* thre_dist: The distance within which a volunteer can reach a call within the time threshold in meters.

    * Default: 200

* num_squares: Number of squares (regions) the city is divided into.

    * Default: 400

* square_length: Length (or width) of the square in meters.

    * Default: 500

* p_OHCA: Probability of an OHCA occurs in each square.

    * Default: A 20 x 20 matrix centered at (5, 5) with probabilities decreasing gradually from the center.

* p_vol: Probability of an available volunteer is in each square.

    * Default: A 20 x 20 matrix with probabilities spreaded uniformly.

Respones:
---------
* thre_dist_flag: whether the distance of the closest volunteer exceeds the threshold distance

* p_survival: probability of survial

* OHCA_loc: location of the OHCA

* closest_loc: the closest volunteer location

* closest_dist: the distance of the closest volunteer in meters

* num_vol: total number of volunteers available


References:
===========
This model is adapted from the article van den Berg et al., "Modeling Emergency Medical Service Volunteer Response".


Optimization Problem: VolunteerDist (VOLUNTEER-1)
========================================================

Decision Variables:
-------------------
* p_vol

Objectives:
-----------
Minimize the probability of the distance of the closest volunteer exceeding the threshold distance :math:`r`.

:math:`\min && \mathcal{P}(D > r)`

Constraints:
------------
* :math:`x_i`'s should be non-negative and sum up to one.

Problem Factors:
----------------
* initial_solution: Initial solution from which solvers start.

  * Default: A 20 x 20 matrix with probabilities spreaded uniformly.
  
* budget: Max # of replications for a solver to take.

  * Default: 1000

Fixed Model Factors:
--------------------
* n/a

Starting Solution: 
------------------
* initial_solution: A 20 x 20 matrix with probabilities spreaded uniformly.

Random Solutions: 
------------------
Sample :math:`x_i` uniformly from a unit simplex.

Optimal Solution:
-----------------
Unknown

Optimal Objective Function Value:
---------------------------------
Unknown

Optimization Problem: VolunteerSurvival (VOLUNTEER-2)
========================================================

Decision Variables:
-------------------
* p_vol

Objectives:
-----------
Maximize the probability of survival of the OHCA.

:math:`\max && \mathcal{E} s(D)`

**gradient** (using likelihodd ratio estimator)

To compute the gradient estimator:

* Compute s(D)
* Let :math:`S_i =` count of volunteers in square :math:`i`.
* Estimate :math:`i^{th}` component of gradient by :math:`\frac{S_i}{x_i} s(D)` if :math:`x_i > 0` or 0 if :math:`x_i = 0`.

This has very high variance for component :math:`i` where :math:`x_i` is small.

Constraints:
------------
* :math:`x_i`'s should be non-negative and sum up to one.

Problem Factors:
----------------
* initial_solution: Initial solution from which solvers start.

  * Default: A 20 x 20 matrix with probabilities spreaded uniformly.
  
* budget: Max # of replications for a solver to take.

  * Default: 1000

Fixed Model Factors:
--------------------
* n/a

Starting Solution: 
------------------
* initial_solution: A 20 x 20 matrix with probabilities spreaded uniformly.

Random Solutions: 
------------------
Sample :math:`x_i` uniformly from a unit simplex.

Optimal Solution:
-----------------
Unknown

Optimal Objective Function Value:
---------------------------------
Unknown
...
