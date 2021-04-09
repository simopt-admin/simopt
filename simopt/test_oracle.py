import numpy as np

from rng.mrg32k3a import MRG32k3a
from oracles.mm1queue import MM1Queue
from oracles.cntnv import CntNV  # names of .py file and Oracle subclass
from base import Solution

# fixed_factors = {
#     # dictionary of non-decision variable factors
#     "purchase_price": 4.0,
#     # "sales_price": 9.0,
#     # "salvage_price": 1.0,
#     "Burr_c": 2.0,
#     "Burr_k": 20.0
# }
fixed_factors = {}

myoracle = MM1Queue(fixed_factors)
# myoracle = CntNV(fixed_factors)
print(myoracle.factors)

rng_list = [MRG32k3a() for _ in range(myoracle.n_rngs)]
# print(rng_list)
myoracle.attach_rngs(rng_list)
# print(myoracle.rng_list)

# # Solution
# mysoln_factors = {
#     # dictionary of missing factors
#     # "mu": 3.0,
#     "order_quantity": 0.2
# }
mysoln_factors = {}

# Check simulatability
for key in fixed_factors:
    print(key, myoracle.check_simulatable_factor(key))

myoracle.factors.update(mysoln_factors)
print(myoracle.factors)
for key in mysoln_factors:
    print(key, myoracle.check_simulatable_factor(key))

print(myoracle.check_simulatable_factors())

# print('For x = (1,), is_simulatable should be True and is {}'.format(myoracle.check_simulatable_factor(x=(1,))))
# print('For x = [1], is_simulatable should be True(?) and is {}'.format(myoracle.check_simulatable_factor(x=(1,))))
# print('For x = (-1,), is_simulatable should be False and is {}'.format(myoracle.check_simulatable_factor(x=(-1,))))
# print('For x = (0,), is_simulatable should be False and is {}'.format(myoracle.check_simulatable_factor(x=(0,))))
# print('For x = (1,2), is_simulatable should be False and is {}'.format(myoracle.check_simulatable_factor(x=(1,2))))
# print('For x = "hi", is_simulatable should be False and is {}'.format(myoracle.check_simulatable_factor(x='hi')))

# Test replicate()
responses, gradients = myoracle.replicate()
print('For a single replication:')
print('The responses are {}'.format(responses))
print('The gradients are {}'.format(gradients))
