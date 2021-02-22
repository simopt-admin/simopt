import numpy as np

from rng.mrg32k3a import MRG32k3a
from problems.facilitysizing_totalcost import facilitysizingTotalCost
#from oracles.facilitysizing import facSize
from base import Solution

#fixed_factors = {
#    "mu": [10, 230, 221],
#    "cov" : [[2276, 1508, 813], [1508, 2206, 1349], [813, 1349, 1865]]
#        }

#myproblem = facilitysizingTotalCost(fixed_factors)
myproblem = facilitysizingTotalCost()
#print(myproblem.oracle.factors)

rng_list = [MRG32k3a() for _ in range(myproblem.oracle.n_rngs)]
myproblem.oracle.attach_rngs(rng_list)

decision_factors = [150, 350, 400]

mysoln = Solution(x=decision_factors, problem=myproblem)

myproblem.simulate(mysoln, m=100)
print('The mean of the objectives are {}'.format(mysoln.objectives_mean))
print('The mean of the stochastic constraints are {}'.format(mysoln.stoch_constraints_mean))