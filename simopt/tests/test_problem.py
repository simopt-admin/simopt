import numpy as np
import sys
import os.path as o
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))

from rng.mrg32k3a import MRG32k3a
from problems.cntnv_max_profit import CntNVMaxProfit
from problems.mm1_min_mean_sojourn_time import MM1MinMeanSojournTime
from problems.facilitysizing_totalcost import FacilitySizingTotalCost
from problems.rmitd_maxrevenue import RMITDMaxRevenue
from problems.sscont_min_cost import SSContMinCost
from base import Solution


myproblem = SSContMinCost()

x = (7, 50)
mysolution = Solution(x, myproblem)

# Create and attach rngs to solution
rng_list = [MRG32k3a(s_ss_sss_index=[0, ss, 0]) for ss in range(myproblem.oracle.n_rngs)]
# print(rng_list)
mysolution.attach_rngs(rng_list, copy=False)
# print(mysolution.rng_list)

# Test simulate()
myproblem.simulate(mysolution, m=100)
print('For 10 replications:')
#print('The individual objective estimates are {}'.format(mysolution.objectives[:10]))
print('The mean objective is {}'.format(mysolution.objectives_mean))
#print('The stochastic constraint estimates are {}'.format(mysolution.stoch_constraints[:10]))
#print('The individual gradient estimates are {}'.format(mysolution.objectives_gradients[:10]))