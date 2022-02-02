
Metamodeling of M/M/1 call center
=======================

This example is adapted from Cheng, R and Kleijnen,J.(1999). Improved Design of Queueing Simulation Experience with Highly Heteroscedastic Responses. Operations Research, v. 47, n. 5, pp. 762-777.

Let :math:`\bar{f} (x_i)` be the average waiting time in a M/M/1 queue as determined through the simulation of :math:`t_i` arrivals when the utilization rate is :math:`x_i(x_i = \lambda / \mu)`, where :math:`\lambda`
and :math:`\mu` are the arrival and service rates, respectively). Furthermore, let :math:`x` be the vector :math:`(x_1, x_2,...x_n)` of utilization rates at which the average waiting time is estimated.
Lastly, let


.. :math:`\hat{f} (x) = (\beta_0 + \beta_1 * x + \beta_2 * x^2) / (1 - x)


The goal is to find \beta_0, \beta_1 and \beta_2 in order to approximate :math:`\bar{f} (x)` through :math:`\hat{f} (x)` as accurately as possible, i.e. 

.. min :math:`(\bar{f} (x) - \hat{f} (x)) \Gamma ^-1 (\bar{f} (x) - \hat{f} (x))`

where :math:`\Gamma` is the convenience matrix for :math:`\bar{f} (x)`. It accounts for any correction, such as the use of common random numbers, in the estimation of the average waiting times through simulation.
If the simulation at each :math:`x_i` is done independently, then :math:`\Gamma` would be a diagonal matrix.

*Recommended Parameter Settings:* Use :math:`n = 5` and :math:`x = (0.5, 0.564, 0.706, 0.859, 0.950)`. Lastly, take :math:`T = 50000(0.007, 0.024, 0.064, 0.258, 0.647)` and estimate :math:`\bar{f} (x_i)` independently for each :math:`x_i`.

*Starting Solutions:* Take :math:`\beta_1 = 1`, and :math:`\beta_1 = \beta_2 = 0`. 

If multiple solutions are needed, take :math:`\beta_0`, :math:`\beta_1`, :math:`\beta_2` uniformly distributed on [0,2].

*Measurement of Time:* Number of estimations of :math:`\bar{f} (x_i)` made.

*Optimal Solution:* :math:`\beta_0 = \beta_2 = 0` and :math:`\beta_1 = 1`.
