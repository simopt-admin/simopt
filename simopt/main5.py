import numpy as np

from rng.mrg32k3a import MRG32k3a
from problems.rmitd_maxrevenue import rmitdMaxRevenue
#from oracles.facilitysizing import facSize
from base import Solution

myproblem = rmitdMaxRevenue()
#print(myproblem.oracle.factors)

rng_list = [MRG32k3a() for _ in range(myproblem.oracle.n_rngs)]
myproblem.oracle.attach_rngs(rng_list)

decision_factors = [100, 50, 30]

mysoln = Solution(x=decision_factors, problem=myproblem)

myproblem.simulate(mysoln, m=5)
# stochastic constraints must be less then 0
# denote the risk of failing to satisfy demand is p(x)  
# stocahstic constraint: p(x) - epsilon < 0 
print('The mean of the objectives are {}'.format(mysoln.objectives_mean))
# ('The mean of the stochastic constraints are {}'.format(mysoln.stoch_constraints_mean))