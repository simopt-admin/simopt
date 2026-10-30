# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 07 · Using a Solver on Your Own Model
#
# The solvers do not need the SimOpt benchmark. Describe your problem with `simopt_dsl`,
# pick a solver from `simopt_solvers`, and call `solve`.
#
# The example is a newsvendor: choose an order quantity $q$, buy at cost $c = 2$, sell
# at price $p = 5$, and face demand $D \sim \text{Uniform}(50, 150)$. Expected profit
# is maximized at $q^* = 110$.

# %%
import matplotlib.pyplot as plt
import simopt_dsl as dsl

from simopt_solvers import ADAM, solve

PRICE, COST = 5.0, 2.0

# %% [markdown]
# ## Describe the problem
#
# A simulation is a plain function `run(decisions, rngs)` that performs **one
# replication** and returns `(responses, gradients)`. Draw randomness only from the
# `rngs` it receives, so runs are reproducible.


# %%
def newsvendor(decisions, rngs):
    """Simulate one selling season for order quantity `q`."""
    q = decisions["q"]
    demand = rngs[0].uniform(50, 150)
    return {"profit": PRICE * min(q, demand) - COST * q}, {}


model = dsl.Model("newsvendor")
q = model.add_continuous_variable(lb=0, ub=300, name="q", initial=20)
season = model.add_simulation(run=newsvendor, decisions={"q": q})
model.maximize(dsl.mean(season.metric("profit")))

# %% [markdown]
# ## Solve
#
# ADAM is a gradient-based solver. The model provides no gradients, so ADAM estimates
# them by finite differences, simulating each solution with `r` replications (30 by
# default). Solver settings are passed as `fixed_factors`; the default step size
# `alpha=0.5` is too small for order quantities in the hundreds, so use `alpha=10`.
#
# `solve` runs the solver until it has used `budget` simulation replications. `seed`
# selects the random-number streams: the same seed reproduces the same run.

# %%
result = solve(model, ADAM(fixed_factors={"alpha": 10}), budget=1000, seed=0)

if result.x is None:
    raise RuntimeError("The solver returned no recommended solution.")
print(f"recommended q: {result.x[0]:.1f}")
print("replications used:", result.budget_used)

# %% [markdown]
# ## History
#
# `result.history` lists each recommended solution with the number of replications used
# when it was recommended.

# %%
result.history

# %%
q_path = [x[0] for x in result.history["solution"]]

fig, ax = plt.subplots()
ax.step(result.history["budget"], q_path, where="post", label="recommended q")
ax.axhline(110, color="black", linestyle="--", label="$q^*$")
ax.set(xlabel="replications used", ylabel="q", title="ADAM on the newsvendor")
ax.legend()
plt.show()
