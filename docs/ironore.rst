Model: Iron Ore Production with Exogenous Stochastic Price (IRONORE)
====================================================================

Description:
------------
Iron ore is traded on the spot market, facing an exogenous, stochastic price. There
is enormous demand for iron, and so for the purposes of a small or medium-sized iron ore mine, we assume
that any quantity of ore can be instantaneously sold at current market rates.

Let there be :math:`T` time periods (days), holding cost of :math:`h` per unit, production cost of :math:`c` per unit, 
maximum production per day of :math:`m` units, and maximum holding capacity of :math:`K` units. Let the iron ore market price for
the day be :math:`P_t`.

Let :math:`x_1` be the price at which to begin production, :math:`x_2` be the inventory level at which to cease production,
:math:`x_3` be the price at which to cease production, and :math:`x_4` be the price at which to sell all current stock.

The daily order of operations in the simulation is as follows:

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
where :math:`N_t` is a normal random variable with standard deviation :math:`\sigma` and mean :math:`\mu_t = \mbox{sgn}(\mu_0 − P_t−1) * (| \mu_0 − P_t − 1 |)^{\frac{1}{4}}`.
Here :math:`\mbox{trunc}(x)` truncates the price to lie between a specified minimum and maximum price.

Model Factors:
--------------
* mean_price: Mean iron ore price per unit.

    * Default: 100.0

* max_price: Maximum iron ore price per unit.

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
N/A


Optimization Problem: Maximize Profit (IRONORE-1)
=================================================

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
All decision variables should be non-negative.
Logically, we should also have price_stop <= price_prod <= price_sell, but this is not enforced.

Problem Factors:
----------------
* budget: Max # of replications for a solver to take

  * Default: 1000

Fixed Model Factors:
--------------------
* N/A

Starting Solution: 
------------------
* initial_solution: :math:`x_1 = 80`, :math:`x_2 = 7000`, :math:`x_3 = 40`, :math:`x_4=100`

Random Solutions: 
-----------------
* :math:`x_1`: Sample an lognormal random variate with 2.5- and 97.5-percentiles of 10 and 200.
* :math:`x_2`: Sample an lognormal random variate with 2.5- and 97.5-percentiles of 1000 and 10000.
* :math:`x_3`: Sample an lognormal random variate with 2.5- and 97.5-percentiles of 10 and 200.
* :math:`x_4`: Sample an lognormal random variate with 2.5- and 97.5-percentiles of 10 and 200.

Optimal Solution:
-----------------
Unknown

Optimal Objective Function Value:
---------------------------------
Unknown
