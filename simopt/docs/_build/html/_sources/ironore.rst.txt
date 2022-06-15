Model: Newsvendor Problem with Exogenous Stochastic Price (IRONORE)
===================================================================

Description:
------------
Iron ore is traded on the spot market, facing an exogenous, stochastic price. There
is enormous demand for iron, and so for the purposes of a small or medium-sized iron ore mine, we assume
that any quantity of ore can be instantaneously sold at current market rates.

Let there be :math:`T` time periods (days), holding cost of :math:`h` per unit, production cost of :math:`c` per unit, 
maximum production per day of :math:`m` units, and maximum holding capacity of :math:`K` units. Let the iron ore market price for
the day be :math:`P_t`.

Let the decision variables be :math:`x = [x_1, x_2, x_3, x_4]`, where :math:`x_1` is the price at which to begin production, :math:`x_2`
is the inventory level at which to cease production, :math:`x_3` is the price at which to cease production, and :math:`x_4` is
the price at which to sell all current stock.

The order of operations in the simulation is as follows:

  1. Sample the market price, :math:`P_t`. Let current stock be :math:`s_t`.

  2. If production is already underway,

    (a) if :math:`P_t` ≤ :math:`x_3` or :math:`s_t` ≥ :math:`x_2`, cease production.
    
    (b) else, produce :math:`min(m, K − s_t)` at cost :math:`c` per unit.

  3. If production is not currently underway, and if :math:`P_t` ≥ :math:`x_1` and :math:`s_t` < :math:`x_2`, begin production.

  4. If :math:`P_t` ≥ :math:`x_4`, sell all stock (after production) at price :math:`P_t`.

  5. Charge a holding cost of :math:`h` per unit (after production and sales).

Sources of Randomness:
----------------------
1. Let :math:`P_t` be a meanreverting random walk, such that :math:`P_t = \mbox{trunc}(P_t - 1 + N_t (\mu,\sigma))`, 
where :math:`N_t` is a normal random variable with standard deviation :math:`\sigma = 7.5` and mean :math:`\mu_t = \mbox{sgn}(\mu_0 − P_t−1) * (| \mu_0 − P_t − 1 |)^{\frac{1}{4}}`, :math:`\mu_0 = 100`. 

:math:`\mbox{trunc}(x) = \max(\min(x, 200), 0)` (trunc bounds its argument in [0,200]). 

Model Factors:
--------------
* mean_price: Mean iron ore price per unit

    * Default: 100.0

* max_price: <short description>

    * Default: 200.0

* min_price: Minimum iron ore price per unit.

    * Default: 0.0
  
* capacity: Maximum holding capacity.

    * Default: 10000

* st_dev: Standard deviation of random walk steps for price.

    * Default: 7.5

* holding_cost: Holding cost per unit per period.

    * Default: 1.0

* prod_cost: Production cost per unit.

    * Default: 100.0

* max_prod_perday: Maximum units produced per day.

    * Default: 100

* price_prod: Price level to start production.

    * Default: 80.0

* inven_stop: Inventory level to cease production.

    * Default: 7000

* price_stop: Price level to stop production.

    * Default: 40

* price_sell: Price level to sell all stock.

    * Default: 100

* n_days: Number of days to simulate.

    * Default: 365


Respones:
---------
* total_profit: The total profit over the time period

* frac_producing: The fraction of days spent producing iron ore

* mean_stock: The average stocks over the time period


References:
===========
n/a


Optimization Problem: IronOreMaxRev (IRONORE-1)
========================================================

Decision Variables:
-------------------
* price_prod
* inven_stop
* price_stop
* price_sell

Objectives:
-----------
Maximize total_profit over the :math:`T` time periods.

Constraints:
------------
1. price_stop <= price_prod <= price_sell

Problem Factors:
----------------
* initial_solution: Initial solution from which solvers start

  * Default: (80, 7000, 40, 100)
  
* budget: Max # of replications for a solver to take

  * Default: 1000

Fixed Model Factors:
--------------------
* n/a

Starting Solution: 
------------------
* initial_solution: (80, 7000, 40, 100)

Random Solutions: 
------------------
* :math:`x_1`: Sample an integer number from (70, 90)
* :math:`x_2`: Sample an integer number from (2000, 8000)
* :math:`x_3`: Sample an integer number from (30, 50)
* :math:`x_4`: Sample an integer number from (90, 110)

Optimal Solution:
-----------------
Unknown

Optimal Objective Function Value:
---------------------------------
Unknown
