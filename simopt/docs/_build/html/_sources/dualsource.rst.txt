
Dual Sourcing System
==================================================

**Problem Description**


This example is adapted from Veeraraghavan, S and Scheller-Wolf, A. Now or Later: 
A simple policy for Effective Dual Sourcing in Capacitated Systems. Operations Research (4), 850- 864.

Consider a single-stage, incapacitated, manufacturing location facing stochastic demand. 
The manufacturer can buy the material from a “regular” supplier at cost :math:`c_r` per unit, or, 
if needed, she can get some or all of the material “expedited” at some premium cost :math:`c_e` 
per unit with :math:`c_e > c_r`. Let the difference in cost be :math:`c = c_e − c_r`. 
Regular orders arrive after :math:`l_r` periods while expedited orders arrive after :math:`l_e` periods with 
:math:`l_e < l_r`. Let the difference in lead times be :math:`l = l_r − l_e ≥ 1`. 

If there is remaining on-hand inventory at the end of period :math:`n` (after demand dn is satisfied), 
these items are carried over to the next period (i.e., :math:`I_n+1 > 0`) at a holding cost of :math:`h` per unit. 
However, if there is a stock-out (i.e., :math:`In + 1 < 0`), there is a penalty cost :math:`p` per unit 
of unsatisfied demand. Our goal is to minimize average holding, penalty and ordering costs.

We will let the period :math:`n` expediting order be based on the on-hand inventory plus the orders that 
will arrive within :math:`l_e` periods (both regular and expedited). Regular orders that are due to arrive 
after :math:`l_e` periods are not considered in expedited ordering decisions. 
The expedited order is placed to restore the expedited inventory position :math:`IP_n^e`, 
to some target parameter level :math:`z_e`. The regular order :math:`X_n^r`, on the other hand, 
is based on the regular inventory position (sum of on-hand inventory and all outstanding orders, 
including the expedited order placed in the current period). Similarly, it tries to restore the regular 
inventory position :math:`IP_n^r` to the target parameter :math:`z_r`. Thus, under this model, we carry two inventory positions, 
one for regular orders and another for expedited orders.

**Recommended Parameter Settings:** 
One could use uniform, normal or exponential demands with mean 30 and Standard deviation 10. 
Lead times and holding, penalty and per unit costs can be adjusted as long as all conditions are satisfied. 
This simulation implements: :math:`d_n ∼ Truncated Normal(30,10), c_e = $110, c_r = $100,l_e = 0,l_r = 2,h = $5,p = $495.`


**Starting Solution(s):** :math:`z_e = 50,z_r = 80`. If multiple solutions are needed use :math:`z_e ∼ Uniform(40,60)` 
and :math:`z_r ∼ Uniform(70,90)`.

**Measurement of Time:** Number of periods simulated.

**Optimal Solutions:** Unknown.

**Known Structure:** Unknown.