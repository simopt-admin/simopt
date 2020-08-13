import numpy as np

from rng.mrg32k3a import MRG32k3a
from oracles.mm1queue import MM1Queue
#from oracles.ctsnews import CtsNews # names of .py file and Oracle subclass
from base import Solution

noise_factors = {
#      dictionary of non-decision variable factors
    "lambda": 1.5,
    "warmup": 20,
    "people": 50
}

myoracle = MM1Queue(noise_factors)
#myoracle = CtsNews(noise_factors)
print(myoracle.factors)

rng_list = [MRG32k3a() for _ in range(myoracle.n_rngs)]
#print(rng_list)
myoracle.attach_rngs(rng_list)
#print(myoracle.rng_list)

# Solution
mysoln_factors = {
#     dictionary of missing factors
    "mu": 3.0,
}

# Check simulatability
for key in noise_factors:
    print(key, myoracle.check_simulatable_factor(key))

myoracle.factors.update(mysoln_factors) 
for key in mysoln_factors:
    print(key, myoracle.check_simulatable_factor(key))

print(myoracle.check_simulatable_factors(mysoln_factors))

# print('For x = (1,), is_simulatable should be True and is {}'.format(myoracle.check_simulatable_factor(x=(1,))))
# print('For x = [1], is_simulatable should be True(?) and is {}'.format(myoracle.check_simulatable_factor(x=(1,))))
# print('For x = (-1,), is_simulatable should be False and is {}'.format(myoracle.check_simulatable_factor(x=(-1,))))
# print('For x = (0,), is_simulatable should be False and is {}'.format(myoracle.check_simulatable_factor(x=(0,))))
# print('For x = (1,2), is_simulatable should be False and is {}'.format(myoracle.check_simulatable_factor(x=(1,2))))
# print('For x = "hi", is_simulatable should be False and is {}'.format(myoracle.check_simulatable_factor(x='hi')))

# Test replicate()
responses, gradients = myoracle.replicate(mysoln_factors)
print('For a single replication:')
print('The responses are {}'.format(responses))
print('The gradients are {}'.format(gradients))