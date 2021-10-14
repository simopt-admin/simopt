"""
This script is intended to help with debugging an oracle.
It imports an oracle, initializes an oracle object with given factors,
sets up pseudorandom number generators, and runs one or more replications.
"""
import numpy as np
import sys
import os.path as o
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))

# Import random number generator.
from rng.mrg32k3a import MRG32k3a

# Import oracle.
# Replace <filename> with name of .py file containing oracle class.
# Replace <oracle_class_name> with name of oracle class.
# Ex: from oracles.mm1queue import MM1Queue
from oracles.ironore import IronOre

# Fix factors of oracle. Specify a dictionary of factors.
# Look at Oracle class definition to get names of factors.
# Ex: for the MM1Queue class,
#     fixed_factors = {"lambda": 3.0,
#                      "mu": 8.0}
fixed_factors = {}  # Resort to all default values.

# Initialize an instance of the specified oracle class.
# Replace <oracle_class_name> with name of oracle class.
# Ex: myoracle = MM1Queue(fixed_factors)
myoracle = IronOre(fixed_factors)

# Working example for MM1 oracle. (Commented out)
# -----------------------------------------------
# from oracles.mm1queue import MM1Queue
# fixed_factors = {"lambda": 3.0, "mu": 8.0}
# myoracle = MM1Queue(fixed_factors)
# -----------------------------------------------

# The rest of this script requires no changes.

# Check that all factors describe a simulatable oracle.
# Check fixed factors individually.
for key, value in myoracle.factors.items():
    print(f"The factor {key} is set as {value}. Is this simulatable? {bool(myoracle.check_simulatable_factor(key))}.")
# Check all factors collectively.
print(f"Is the specified oracle simulatable? {bool(myoracle.check_simulatable_factors())}.")

# Create a list of RNG objects for the simulation oracle to use when
# running replications.
rng_list = [MRG32k3a(s_ss_sss_index=[0, ss, 0]) for ss in range(myoracle.n_rngs)]

# Run a single replication of the oracle.
responses, gradients = myoracle.replicate(rng_list)
print("\nFor a single replication:")
print("\nResponses:")
for key, value in responses.items():
    print(f"\t {key} is {value}.")
print("\n Gradients:")
for outerkey in gradients:
    print(f"\tFor the response {outerkey}:")
    for innerkey, value in gradients[outerkey].items():
        print(f"\t\tThe gradient w.r.t. {innerkey} is {value}.")

# Run multiple replications of the oracle.
REPLICATIONS = 100
total_revenue = []
frac_producing = []
mean_stock = []
for _ in range(REPLICATIONS):
    responses, gradients = myoracle.replicate(rng_list)
    total_revenue.append(responses['total_revenue'])
    frac_producing.append(responses['frac_producing'])
    mean_stock.append(responses['mean_stock'])

print("\nFor {REPLICATIONS} replications:")
print(np.mean(total_revenue))
print(np.mean(frac_producing))
print(np.mean(mean_stock))


