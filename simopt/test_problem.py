import numpy as np

from rng.mrg32k3a import MRG32k3a
from problems.cntnv_max_profit import CntNVMaxProfit
from problems.mm1_min_mean_sojourn_time import MM1MinMeanSojournTime
from base import Solution

#myproblem = CntNVMaxProfit()
myproblem = MM1MinMeanSojournTime()

x = [4]
mysolution = Solution(x, myproblem)

# Create and attach rngs to solution
rng_list = [MRG32k3a(s_ss_sss_index=[0, ss, 0]) for ss in range(myproblem.oracle.n_rngs)]
# print(rng_list)
mysolution.attach_rngs(rng_list, copy=False)
# print(mysolution.rng_list)

# Test simulate()
myproblem.simulate(mysolution, m=10)
print('For 10 replications:')
print('The individual objective estimates are {}'.format(mysolution.objectives[:10]))
print('The individual gradient estimates are {}'.format(mysolution.objectives_gradients[:10]))
