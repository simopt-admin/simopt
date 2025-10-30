# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: simopt
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Demo for Model Debugging.
#
# This script is intended to help with debugging a model.
#
# It imports a model, initializes a model object with given factors, sets up pseudorandom number generators, and runs one or more replications.

# %% [markdown]
# ## Append SimOpt Path
#
# Since the notebook is stored in simopt/notebooks, we need to append the parent simopt directory to the system path to import the necessary modules later on.

# %%
import sys
from pathlib import Path

# Take the current directory, find the parent, and add it to the system path
sys.path.append(str(Path.cwd().parent))

# %% [markdown]
# ## Model Configuration Parameters
#
# This section defines the core parameters for the model.
#
# To query model names, run `python scripts/list_directories.py`

# %%
# Import the model from the models directory
from simopt.models.mm1queue import MM1Queue

# Set fixed factors
# Setting to {} will resort to all default values.
fixed_factors = {"lambda": 3.0, "mu": 8.0}

# Initialize model
mymodel = MM1Queue(fixed_factors=fixed_factors)

# %%
# Create a list of RNG objects for the simulation model to use when
# running replications.

from mrg32k3a.mrg32k3a import MRG32k3a

rng_list = [MRG32k3a(s_ss_sss_index=[0, ss, 0]) for ss in range(mymodel.n_rngs)]

# %%
# Run a single replication of the model.
mymodel.before_replicate(rng_list)
responses, gradients = mymodel.replicate()
print("\nFor a single replication:")
print("\nResponses:")
for key, value in responses.items():
    print(f"\t {key} is {value}.")
print("\n Gradients:")
for outerkey in gradients:
    print(f"\tFor the response {outerkey}:")
    for innerkey, value in gradients[outerkey].items():
        print(f"\t\tThe gradient w.r.t. {innerkey} is {value}.")
