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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
# %set_env MRG32K3A_BACKEND=rust

# %%
import pickle

from simopt.experiment_base import (
    ProblemSolver,
    post_normalize,
)

# %%
from simopt.models.san import SANLongestPathStochastic
from simopt.solvers.fcsa import FCSA

# %%
# initialize problem
initial = (5,) * 13  # starting mean for each arc
constraint_nodes = [6, 8]  # nodes with corresponding stochastic constraints
max_length_to_node = [5, 5]  # max expected length to each constraint node
budget = 2000  # number of simmulation replications ran by solver
problem_factors = {
    "constraint_nodes": constraint_nodes,
    "length_to_node_constraint": max_length_to_node,
    "initial_solution": initial,
    "budget": budget,
}
problem = SANLongestPathStochastic(fixed_factors=problem_factors)

# %%
# initialize solvers
csa_factors = {
    "search_direction": "CSA",
    "normalize_grads": False,
    "report_all_solns": True,
    "crn_across_solns": False,
}
csa = FCSA(fixed_factors=csa_factors, name="CSA")
csa_n_factors = {
    "search_direction": "CSA",
    "normalize_grads": True,
    "report_all_solns": True,
    "crn_across_solns": False,
}
csa_n = FCSA(fixed_factors=csa_n_factors, name="CSA-N")
fcsa_factors = {
    "search_direction": "FCSA",
    "normalize_grads": True,
    "report_all_solns": True,
    "crn_across_solns": False,
}
fcsa = FCSA(fixed_factors=fcsa_factors, name="FCSA")
solvers = [csa, csa_n, fcsa]

# %%
e1 = ProblemSolver(solver=csa, problem=problem)
e2 = ProblemSolver(solver=csa_n, problem=problem)
e3 = ProblemSolver(solver=fcsa, problem=problem)

e1.run(n_macroreps=10)
e2.run(n_macroreps=10)
e3.run(n_macroreps=10)

e1.post_replicate(n_postreps=100)
e2.post_replicate(n_postreps=100)
e3.post_replicate(n_postreps=100)

post_normalize([e1, e2, e3], 100)

# %%
with open("test-data-2.pkl", "wb") as f:
    pickle.dump([e1, e2, e3], f)
