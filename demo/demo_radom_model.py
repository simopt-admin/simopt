"""
This script is intended to help with debugging a random model.
It imports a model, initializes a model object with given factors,
sets up pseudorandom number generators, and runs one or more replications.
"""

"""
Instead of modifying the problem and model class, we modify the demo_model and demo_problems.
"""

import sys
import os.path as o
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))

import numpy as np
# Import random number generator.
# from mrg32k3a.mrg32k3a import MRG32k3a
from rng.mrg32k3a import MRG32k3a

# Import model.    
from simopt.models.san_2 import SAN
from simopt.models.cascade import Cascade

fixed_factors = {}
# mymodel = SAN(fixed_factors = fixed_factors, random=True)
mymodel = Cascade(fixed_factors=fixed_factors, random=True)

# from models.<filename> import <model_class_name>
# Replace <filename> with name of .py file containing model class.
# Replace <model_class_name> with name of model class.

# Fix factors of model. Specify a dictionary of factors.

# fixed_factors = {}  # Resort to all default values.
# Look at Model class definition to get names of factors.

# Initialize an instance of the specified model class.

# mymodel = <model_class_name>(fixed_factors)
# Replace <model_class_name> with name of model class.

# Working example for MM1 model.
# -----------------------------------------------
# from simopt.models.mm1queue import MM1Queue
# fixed_factors = {"lambda": 3.0, "mu": 8.0}
# mymodel = MM1Queue(fixed_factors)
# -----------------------------------------------

# The rest of this script requires no changes.

# Check that all factors describe a simulatable model.
# Check fixed factors individually.

for key, value in mymodel.factors.items():
    print(f"The factor {key} is set as {value}. Is this simulatable? {bool(mymodel.check_simulatable_factor(key))}.")
# Check all factors collectively.
print(f"Is the specified model simulatable? {bool(mymodel.check_simulatable_factors())}.")

# Create a list of RNG objects for the simulation model to use when
# running replications.
rng_list = [MRG32k3a(s_ss_sss_index=[0, ss, 0]) for ss in range(mymodel.n_rngs)]
rng_list2 = [MRG32k3a(s_ss_sss_index=[2, 4 + ss, 0]) for ss in range(mymodel.n_random)]

mymodel.attach_rng(rng_list2)
responses, gradients = mymodel.replicate(rng_list)
print("\nFor a single replication:")
print("\nResponses:")
for key, value in responses.items():
    print(f"\t {key} is {value}.")

# l = []
# for i in range(1000):
#     rng_list = [MRG32k3a(s_ss_sss_index=[0, ss + i, 0]) for ss in range(mymodel.n_rngs)]
#     rng_list2 = [MRG32k3a(s_ss_sss_index=[2, i + 4 + ss, 0]) for ss in range(mymodel.n_random)]
#     mymodel.attach_rng(rng_list2)
#     responses, gradients = mymodel.replicate(rng_list)
#     print("\nFor a single replication:")
#     print("\nResponses:")
#     for key, value in responses.items():
#         print(f"\t {key} is {value}.")
#         l.append(value)

# print('mean: ', np.mean(l))