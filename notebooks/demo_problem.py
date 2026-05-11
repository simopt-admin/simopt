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
# # Demo script for the Problem class.
#
# This script is intended to help with debugging a problem.
#
# It imports a problem, initializes a problem object with given factors, sets up pseudorandom number generators, and runs multiple replications at a given solution.

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
# ## Configuration Parameters
#
# This section defines the core parameters for the demo.
#
# To query model/problem/solver names, run `python scripts/list_directories.py`

# %%
# Import problem.
# from models.<filename> import <problem_class_name>
# Replace <filename> with name of .py file containing problem class.
# Replace <problem_class_name> with name of problem class.
# Fix factors of problem. Specify a dictionary of factors.
# fixed_factors = {}  # Resort to all default values.
# Look at Problem class definition to get names of factors.
# Initialize an instance of the specified problem class.
# myproblem = <problem_class_name>(fixed_factors=fixed_factors)
# Replace <problem_class_name> with name of problem class.
# Initialize a solution x corresponding to the problem.
# x = (,)
# Look at the Problem class definition to identify the decision variables.
# x will be a tuple consisting of the decision variables.
# The following line does not need to be changed.
# mysolution = Solution(x, myproblem)
# Working example for CntNVMaxProfit problem.
# -----------------------------------------------
from simopt.models.cntnv import CntNVMaxProfit

fixed_factors = {"initial_solution": (2,), "budget": 500}
myproblem = CntNVMaxProfit(fixed_factors=fixed_factors)
x = (3,)

# -----------------------------------------------

# Another working example for FacilitySizingTotalCost problem. (Commented out)
# This example has stochastic constraints.
# -----------------------------------------------
# from models.facilitysizing import FacilitySizingTotalCost
# fixed_factors = {"epsilon": 0.1}
# myproblem = FacilitySizingTotalCost(fixed_factors=fixed_factors)
# x = (200, 200, 200)
# mysolution = Solution(x, myproblem)
# -----------------------------------------------

num_macroreps = 10

# %%
# Create and attach rngs to solution
from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.base import Solution

rng_list = [MRG32k3a(s_ss_sss_index=[0, ss, 0]) for ss in range(myproblem.model.n_rngs)]
mysolution = Solution(x, myproblem)

mysolution.attach_rngs(rng_list, copy=False)

# %%
# Simulate a fixed number of replications at the solution x.
myproblem.simulate(mysolution, num_macroreps=num_macroreps)

# %%
# Print results to console.
print(
    f"Ran {num_macroreps} replications of the {myproblem.name} problem "
    f"at solution x = {x}.\n"
)
obj_mean = round(mysolution.objectives_mean[0], 4)
obj_stderr = round(mysolution.objectives_stderr[0], 4)
print(f"The mean objective estimate was {obj_mean} with standard error {obj_stderr}.")
print("The individual observations of the objective were:")
for idx in range(num_macroreps):
    print(f"\t {round(mysolution.objectives[idx][0], 4)}")
if myproblem.gradient_available:
    print("\nThe individual observations of the gradients of the objective were:")
    for idx in range(num_macroreps):
        grads = mysolution.objectives_gradients[idx][0]
        print(f"\t {[round(g, 4) for g in grads]}")
else:
    print("\nThis problem has no known gradients.")
if myproblem.n_stochastic_constraints > 0:
    print(
        f"\nThis problem has {myproblem.n_stochastic_constraints} stochastic "
        "constraints of the form E[LHS] <= 0."
    )
    for stc_idx in range(myproblem.n_stochastic_constraints):
        stoch_const_mean = mysolution.stoch_constraints_mean[stc_idx]
        stoch_const_mean_round = round(stoch_const_mean, 4)
        stoch_const_stderr = mysolution.stoch_constraints_stderr[stc_idx]
        stoch_const_stderr_round = round(stoch_const_stderr, 4)
        print(
            f"\tFor stochastic constraint #{stc_idx + 1}, "
            f"the mean of the LHS was {stoch_const_mean_round} "
            f"with standard error {stoch_const_stderr_round}."
        )
        print("\tThe observations of the LHSs were:")
        for idx in range(num_macroreps):
            # Quick check to make sure the stochastic constraints are not None.
            if mysolution.stoch_constraints is None:
                error_msg = "Stochastic constraints should not be None."
                raise ValueError(error_msg)
            # Print out the current stochastic constraint value.
            stoch_const = mysolution.stoch_constraints[stc_idx][idx]
            stoch_const_round = round(stoch_const, 4)
            print(f"\t\t {stoch_const_round}")
else:
    print("\nThis problem has no stochastic constraints.")
