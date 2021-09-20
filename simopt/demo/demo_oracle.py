import numpy as np
import sys
import os.path as o
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))


from rng.mrg32k3a import MRG32k3a
from oracles.mm1queue import MM1Queue
from oracles.cntnv import CntNV  # names of .py file and Oracle subclass
from oracles.facilitysizing import FacilitySize
from oracles.rmitd import RMITD
from oracles.contam import Contamination  # new
from oracles.sscont import SSCont
from base import Solution

# fixed_factors = {"s": 7, "S": 57}
fixed_factors = {}
# myoracle = SSCont(fixed_factors)
myoracle = Contamination(fixed_factors)
print(myoracle.factors)

# mysoln_factors = {}

# # Check simulatability
# for key in fixed_factors:
#     print(key, myoracle.check_simulatable_factor(key))

# myoracle.factors.update(mysoln_factors)
# print(myoracle.factors)
# for key in mysoln_factors:
#     print(key, myoracle.check_simulatable_factor(key))

# print(myoracle.check_simulatable_factors())

# print('For x = (1,), is_simulatable should be True and is {}'.format(myoracle.check_simulatable_factor(x=(1,))))
# print('For x = [1], is_simulatable should be True(?) and is {}'.format(myoracle.check_simulatable_factor(x=(1,))))
# print('For x = (-1,), is_simulatable should be False and is {}'.format(myoracle.check_simulatable_factor(x=(-1,))))
# print('For x = (0,), is_simulatable should be False and is {}'.format(myoracle.check_simulatable_factor(x=(0,))))
# print('For x = (1,2), is_simulatable should be False and is {}'.format(myoracle.check_simulatable_factor(x=(1,2))))
# print('For x = "hi", is_simulatable should be False and is {}'.format(myoracle.check_simulatable_factor(x='hi')))

rng_list = [MRG32k3a(s_ss_sss_index=[0, ss, 0]) for ss in range(myoracle.n_rngs)]
# print(rng_list)
# mysolution.attach_rngs(rng_list)
# print(mysolution.rng_list)


# Test replicate()
responses, gradients = myoracle.replicate(rng_list)
print('For a single replication:')
print('The responses are {}'.format(responses))
print('The gradients are {}'.format(gradients))
