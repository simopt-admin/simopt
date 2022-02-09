
Newsvendor Problem with Exogenous Stochastic Price
==================================================

**Problem Description**

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

The optimization problem is to maximize total revenue over the :math:`T` time periods.

**Recommended Parameter Settings:** :math:`x = [80, 7000, 40, 100]`.

Let :math:`P_t` be a meanreverting random walk, such that :math:`P_t = \mbox{trunc}(P_t - 1 + N_t (\mu,\sigma))`, 
where :math:`N_t` is a normal random variable with standard deviation :math:`\sigma = 7.5` and mean :math:`\mu_t = \mbox{sgn}(\mu_0 − P_t−1) * (| \mu_0 − P_t − 1 |)^{\frac{1}{4}}`, :math:`\mu_0 = 100`. 

:math:`\mbox{trunc}(x) = \max(\min(x, 200), 0)` (trunc bounds its argument in [0,200]). 

Let :math:`P_1 = \mu_0`.

**Measurement of Time:**  Number of simulation replications of length :math:`T`.

**Optimal Solutions:** Unknown.

**Known Structure:** Unknown.