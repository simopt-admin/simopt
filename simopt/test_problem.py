import numpy as np

from rng.mrg32k3a import MRG32k3a
from problems.cntnv_max_profit import CntNVMaxProfit
from problems.mm1_min_mean_sojourn_time import MM1MinMeanSojournTime
from problems.facilitysizing_totalcost import FacilitySizingTotalCost
from problems.facilitysizing_max_service import FacilitySizingMaxService
from problems.rmitd_maxrevenue import RMITDMaxRevenue
from problems.sscont_min_cost import SSContMinCost
from base import Solution


myproblem = FacilitySizingMaxService()

x = (200, 200, 200)
mysolution = Solution(x, myproblem)

# Create and attach rngs to solution
rng_list = [MRG32k3a(s_ss_sss_index=[0, ss, 0]) for ss in range(myproblem.oracle.n_rngs)]
# print(rng_list)
mysolution.attach_rngs(rng_list, copy=False)
# print(mysolution.rng_list)

# Test simulate()
n_reps = 100
myproblem.simulate(mysolution, m=n_reps)
print('For ' + str(n_reps) + ' replications:')
#print('The individual objective estimates are {}'.format(mysolution.objectives[:n_reps]))
print('The mean objective is {}'.format(mysolution.objectives_mean))
#print('The stochastic constraint estimates are {}'.format(mysolution.stoch_constraints[:n_reps]))
#print('The individual gradient estimates are {}'.format(mysolution.objectives_gradients[:n_reps]))
