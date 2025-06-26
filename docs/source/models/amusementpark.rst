Amusement Park Queueing
=======================

See the :mod:`simopt.models.amusementpark` module for API details.

Model: Amusement Park Queues (AMUSEMENT)
----------------------------------------

Description
^^^^^^^^^^^

This model simulates an amusement park with 7 attractions. Visitors arrive at
each attraction according to a poisson distribution with a rate :math:`\gamma_i = 1`,
:math:`i = 1,. . . , 7`. Each attraction can only take one visitor at a time, while
others wait in a queue with capacity :math:`c_i`. If a visitor finds a queue full,
they will immediately leave the park.

After visiting each attraction, a visitor goes to another attraction (or leaves) 
according to the transition matrix:

+---+-----+-----+-----+-----+-----+-----+-----+--------+
|   |  1  |  2  |  3  |  4  |  5  |  6  |  7  | Leave  |
+===+=====+=====+=====+=====+=====+=====+=====+========+
| 1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.2 | 0.2 | 0.0 | 0.2    |
+---+-----+-----+-----+-----+-----+-----+-----+--------+
| 2 | 0.1 | 0.1 | 0.1 | 0.1 | 0.2 | 0.2 | 0.0 | 0.2    |
+---+-----+-----+-----+-----+-----+-----+-----+--------+
| 3 | 0.1 | 0.1 | 0.1 | 0.1 | 0.2 | 0.2 | 0.0 | 0.2    |
+---+-----+-----+-----+-----+-----+-----+-----+--------+
| 4 | 0.1 | 0.1 | 0.1 | 0.1 | 0.2 | 0.2 | 0.0 | 0.2    |
+---+-----+-----+-----+-----+-----+-----+-----+--------+
| 5 | 0.1 | 0.1 | 0.1 | 0.1 | 0.0 | 0.1 | 0.3 | 0.2    |
+---+-----+-----+-----+-----+-----+-----+-----+--------+
| 6 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.0 | 0.3 | 0.2    |
+---+-----+-----+-----+-----+-----+-----+-----+--------+
| 7 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.2 | 0.2    |
+---+-----+-----+-----+-----+-----+-----+-----+--------+

* **Rows** represent the ride a tourist is currently at (i.e., the ride they just completed).
* **Columns** represent the next ride the tourist chooses to go to.

The time that a visitor spends at an attraction follows an Erlang
distribution with shape parameter :math:`k = 2`` and rate :math:`\lambda = 9`.
The park opens at 9AM and closes at 5PM, and time is measured in minutes.
When the park closes, all visitors in the queue leave immediately.

Sources of Randomness
^^^^^^^^^^^^^^^^^^^^^

There are 3 sources of randomness in this model:

1. The arrival rate of visitors as a poisson distribution with rate of 1 for all :math:`i = 1, . . . , 7`.
2. The transition probabiliyt matrix that visitors follow after visiting each attraction.
3. The time spent at each attraction as an Erlang distribution with the shape parameter :math:`k = 2` and rate = 9.

The Erlang distribution is the distribution representing a sum of :math:`k` independent exponential variables with mean :math:`1/\lambda` each.
It is a special case of the gamma distribution wherein the shape of the distribution is discretized. The probability density function
of the Erlang distribution is

:math:`f(x;k,\lambda) = \frac{\lambda^{k}x^{k-1}e^{-\lambda x}}{(k-1)!} \quad for \ x, \beta >= 0`

where :math:`k` is the shape parameter, :math:`\lambda` is the rate parameter.

Alternatively, the pdf can be expressed as

:math:`f(x;k,\beta) = \frac{x^{k-1}e^{-x/\beta}}{\beta^k(k-1)!} \quad for \ x, \beta >= 0`

where :math:`\beta` is the scale parameter, which is the reciprocal of the rate parameter.

* Note: In this model, Erlang variates are generated through the gamma distribution with the scale (:math:`\beta:`) parameter set to 1/9.

Accordingly, the reciprocal of desired rate values should be used in the erlang_scale parameter.

Model Factors
^^^^^^^^^^^^^

* park_capacity: The total number of visitors waiting for attractions that can be maintained through park facilities, distributed across the attractions.
    * Default: 350
* number_attractions: The number of attractions in the park.
    * Default: 7
* time_open: The number of minutes per day the park is open.
    * Default: 480
* erlang_shape: The shape parameter of the Erlang distribution for each attraction duration.
    * Default: [2, 2, 2, 2, 2, 2, 2]
* erlang_scale: The scale parameter of the Erlang distribution for each attraction duration (reciprocal of the rate value).
    * Default: [1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9]
* depart_probabilities: The probability that a visitor will depart the park after visiting an attraction.
    * Default: [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
* queue_capacities: The capacity of the queues for the attractions based on the portion of facilities allocated.
    * Default: [50, 50, 50, 50, 50, 50, 50]
* arrival_gammas: The gamma values for the poisson distributions dictating the rates at which visitors entering the park arrive at each attraction.
    * Default: [1, 1, 1, 1, 1, 1, 1]
* transition_probabilities: The transition matrix that describes the probability of a visitor visiting each attraction after their current attraction.
    * Default:

        +---+-----+-----+-----+-----+-----+-----+-----+--------+
        |   |  1  |  2  |  3  |  4  |  5  |  6  |  7  | Leave  |
        +===+=====+=====+=====+=====+=====+=====+=====+========+
        | 1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.2 | 0.2 | 0.0 | 0.2    |
        +---+-----+-----+-----+-----+-----+-----+-----+--------+
        | 2 | 0.1 | 0.1 | 0.1 | 0.1 | 0.2 | 0.2 | 0.0 | 0.2    |
        +---+-----+-----+-----+-----+-----+-----+-----+--------+
        | 3 | 0.1 | 0.1 | 0.1 | 0.1 | 0.2 | 0.2 | 0.0 | 0.2    |
        +---+-----+-----+-----+-----+-----+-----+-----+--------+
        | 4 | 0.1 | 0.1 | 0.1 | 0.1 | 0.2 | 0.2 | 0.0 | 0.2    |
        +---+-----+-----+-----+-----+-----+-----+-----+--------+
        | 5 | 0.1 | 0.1 | 0.1 | 0.1 | 0.0 | 0.1 | 0.3 | 0.2    |
        +---+-----+-----+-----+-----+-----+-----+-----+--------+
        | 6 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.0 | 0.3 | 0.2    |
        +---+-----+-----+-----+-----+-----+-----+-----+--------+
        | 7 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.2 | 0.2    |
        +---+-----+-----+-----+-----+-----+-----+-----+--------+

    * **Rows** represent the ride a tourist is currently at (i.e., the ride they just completed).
    * **Columns** represent the next ride the tourist chooses to go to.

Responses
^^^^^^^^^

* total_departed: The total number of visitors to leave the park due to full queues.
* percent_departed: The percentage of visitors to leave the park due to full queues.
* average_number_in_system: The time average of the number of visitors in the system.
* attraction_utilization_percentages: The percent utilizations for each attraction.

References
^^^^^^^^^^

This model is adapted from the article:
Vill’en-Altamirano, J. (2009). Restart Simulation of Networks of Queues with
Erlang Service Times. *Proceedings of the 2009 Winter Simulation Conference.*

Optimization Problem: Minimize Total Departed Visitors (AMUSEMENT-1)
--------------------------------------------------------------------

Decision Variables
^^^^^^^^^^^^^^^^^^

* queue_capacities

Objectives
^^^^^^^^^^

Minimize total number of departed visitors.

Constraints
^^^^^^^^^^^

* park_capacity = 350
* :math:`\sum_{i=1}^{7}` queue_capacities = park_capacity
* queue_capacities :math:`\ge` 0

Problem Factors
^^^^^^^^^^^^^^^

* Budget: Max # of replications for a solver to take.
    * Default: 1000

Fixed Model Factors
^^^^^^^^^^^^^^^^^^^

* N/A

Starting Solution
^^^^^^^^^^^^^^^^^

* queue_capacities = [50, 50, 50, 50, 50, 50, 50]

Random Solutions
^^^^^^^^^^^^^^^^

Generate a solution uniformly from a space of vectors of length 7 that sum up to 350.

Optimal Solution
^^^^^^^^^^^^^^^^

unknown

Optimal Objective Function Value
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

unknown
