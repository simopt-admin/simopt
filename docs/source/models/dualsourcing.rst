Dual Sourcing System
====================

See the :mod:`simopt.models.dualsourcing` module for API details.

Model: Dual Sourcing System (DUALSOURCING)
------------------------------------------

Description
^^^^^^^^^^^

Consider a single-stage, incapacitated, manufacturing location facing stochastic demand. 
The manufacturer can buy the material from a “regular” supplier at cost :math:`c_r` per unit, or, 
if needed, she can get some or all of the material “expedited” at some premium cost :math:`c_e` 
per unit with :math:`c_e > c_r`. 
Regular orders arrive after :math:`l_r` periods while expedited orders arrive after :math:`l_e` periods with 
:math:`l_e < l_r`. Let the difference in lead times be :math:`l = l_r − l_e ≥ 1`. 

If there is remaining on-hand inventory at the end of period :math:`n` (after demand :math:`d_n` is satisfied), 
these items are carried over to the next period (i.e., :math:`I_{n+1} > 0`) at a holding cost per unit. 
However, if there is a stock-out (i.e., :math:`I_{n+1} < 0`), there is a penalty cost per unit 
of unsatisfied demand.

We will let the period :math:`n` expediting order be based on the on-hand inventory plus the orders that 
will arrive within :math:`l_e` periods (both regular and expedited). Regular orders that are due to arrive 
after :math:`l_e` periods are not considered in expedited ordering decisions. 
The expedited order is placed to restore the expedited inventory position :math:`IP_n^e`, 
to some target parameter level :math:`z_e`. The regular order :math:`X_n^r`, on the other hand, 
is based on the regular inventory position (sum of on-hand inventory and all outstanding orders, 
including the expedited order placed in the current period). Similarly, it tries to restore the regular 
inventory position :math:`IP_n^r` to the target parameter :math:`z_r`. Thus, under this model, we carry two inventory positions, 
one for regular orders and another for expedited orders.

Sources of Randomness
^^^^^^^^^^^^^^^^^^^^^

Demand follows a normal distribution. 

Model Factors
^^^^^^^^^^^^^

* n_days: Number of days to simulate.
    * Default: 1000
* initial_inv: Initial inventory.
    * Default: 40
* cost_reg: Regular ordering cost per unit.
    * Default: 100.00
* cost_exp: Expedited ordering cost per unit.
    * Default: 110.00
* lead_reg: Lead time for regular orders in days.
    * Default: 110.00
* lead_exp: Lead time for expedited orders in days.
    * Default: 0
* holding_cost: Holding cost per unit per period.
    * Default: 5.00
* penalty_cost: Penalty cost per unit per period for backlogging.
    * Default: 495.00
* st_dev: Standard deviation of demand distribution.
    * Default: 10.0
* mu: Mean of demand distribution.
    * Default: 30.0
* order_level_reg: Order-up-to level for regular orders.
    * Default: 80
* order_level_exp: Order-up-to level for expedited orders.
    * Default: 50

Responses
^^^^^^^^^

* average_holding_cost: The average holding cost over the time period.
* average_penalty_cost: The average penalty cost over the time period.
* average_ordering_cost: The average ordering cost over the time period.

References
^^^^^^^^^^

This model is adapted from the article `Veeraraghavan, S and Scheller-Wolf, A. Now or Later: 
A simple policy for Effective Dual Sourcing in Capacitated Systems. Operations Research (4), 850- 864. 
<https://repository.upenn.edu/oid_papers/121/>`_

Optimization Problem: Minimize total cost (DUALSOURCING-1)
----------------------------------------------------------

Decision Variables
^^^^^^^^^^^^^^^^^^

* order_level_exp
* order_level_reg

Objectives
^^^^^^^^^^

Minimize the expected total cost: sum of average_holding_cost, average_penalty_cost, average_ordering_cost.

Constraints
^^^^^^^^^^^

order_level_exp and order_level_reg are both non-negative.

Problem Factors
^^^^^^^^^^^^^^^

* budget: Max # of replications for a solver to take.
    * Default: 1000

Fixed Model Factors
^^^^^^^^^^^^^^^^^^^

N/A

Starting Solution
^^^^^^^^^^^^^^^^^

* order_level_exp: 50
* order_level_reg: 80

Random Solutions
^^^^^^^^^^^^^^^^

Draw order_level_exp from Uniform(40,60) and order_level_reg from Uniform(70,90).

Optimal Solution
^^^^^^^^^^^^^^^^

Unknown.

Optimal Objective Function Value
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Unknown.