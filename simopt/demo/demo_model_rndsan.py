"""
This script imports the random san model, initializes an model object with given factors,
sets up pseudorandom number generators, and runs one or more replications.
"""

import sys
import os.path as o
from typing import Container
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))

# Import random number generator.
from rng.mrg32k3a import MRG32k3a

# Import model.
from models.rndsan import RNDSAN

# Fix factors of model. Specify a dictionary of factors.
# Look at Model class definition to get names of factors.
# For the RNDSAN class,
#     fixed_factors = {"num_nodes": 9,
#                      "arcs": [...],
#                      "arc_means": (...)}

# Resort to all default values.
# fixed_factors = {}

# Randomly generate a network.
num_nodes = 20
fwd_arcs = 3
fwd_reach = 12
arcs = []
arc_rng = MRG32k3a(s_ss_sss_index=[1, 0, 0])
for i in range(num_nodes-1):
    indices = [arc_rng.randint(i+2, min(num_nodes, i+1+fwd_reach)) for _ in range(fwd_arcs)]
    for j in range(fwd_arcs):
        arcs.append([i+1,indices[j]])
for i in range(1,num_nodes):
    prev_index = arc_rng.randint(max(i+1-fwd_reach, 1), i)
    arcs.append([prev_index, i+1])
arcs = list(set(tuple(a) for a in arcs))
fixed_factors = {"num_nodes": num_nodes, "arcs": arcs, "arc_means": (1,)*len(arcs)}

# Initialize an instance of the specified model class.
mymodel = RNDSAN(fixed_factors)

# The rest of this script requires no changes.

# Check that all factors describe a simulatable model.
# Check fixed factors individually.
# for key, value in mymodel.factors.items():
#     print(f"The factor {key} is set as {value}. Is this simulatable? {bool(mymodel.check_simulatable_factor(key))}.")
# # Check all factors collectively.
# print(f"Is the specified model simulatable? {bool(mymodel.check_simulatable_factors())}.")

# Create a list of RNG objects for the simulation model to use when
# running replications.
rng_list = [MRG32k3a(s_ss_sss_index=[0, ss, 0]) for ss in range(mymodel.n_rngs)]

# Run a single replication of the model.
responses, gradients = mymodel.replicate(rng_list)
print("\nFor a single replication:")
print("\nResponses:")
for key, value in responses.items():
    print(f"\t {key} is {value}.")
# print("\n Gradients:")
# for outerkey in gradients:
#     print(f"\tFor the response {outerkey}:")
#     for innerkey, value in gradients[outerkey].items():
#         print(f"\t\tThe gradient w.r.t. {innerkey} is {value}.")
