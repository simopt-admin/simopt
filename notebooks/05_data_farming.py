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
# # 05 · Data Farming
#
# Data farming runs a designed experiment over the **factors** of a model, solver or
# problem to learn how they affect performance. SimOpt builds space-filling
# nearly-orthogonal Latin hypercube (NOLHS) designs.
#
# 1. Farm a **model**: how do prices and order quantity affect newsvendor profit?
# 2. Farm a **solver**: how sensitive is ASTRO-DF to its trust-region parameters?
# 3. Farm a **problem**: how does random search cope as the cost structure changes?
# 4. Load a design created elsewhere (e.g. the SimOpt GUI) from a file.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from simopt.analysis import terminal_scatterplot
from simopt.data_farming.design import DesignSpec, NumericRange, build_design
from simopt.data_farming_base import DataFarmingExperiment
from simopt.experiment.api import SimulationConfig, run
from simopt.experiment.single import ProblemSolver
from simopt.models.cntnv import CntNVMaxProfit
from simopt.models.sscont import SSContMinCost
from simopt.solvers.astrodf import ASTRODF
from simopt.solvers.randomsearch import RandomSearch

# %% [markdown]
# ## 1. Model data farming
#
# Each varied factor is given as `(lower bound, upper bound, decimals)`. Every design
# point is simulated `n_reps` times, with common random numbers across design points
# by default.

# %%
model_experiment = DataFarmingExperiment(
    model_name="CNTNEWS",
    factor_headers=["purchase_price", "sales_price", "salvage_price", "order_quantity"],
    factor_settings=[(4.0, 6.0, 1), (8.0, 12.0, 1), (0.5, 2.0, 1), (0.4, 0.6, 2)],
)
model_experiment.run(n_reps=10, crn_across_design_pts=True)

model_df = pd.DataFrame(
    [
        {
            **design_pt.model_factors,
            "mean_profit": sum(design_pt.responses["profit"]) / design_pt.n_reps,
        }
        for design_pt in model_experiment.design
    ]
)
print(f"{len(model_df)} design points")
model_df.head()

# %% [markdown]
# Raw replications can be written to a tab-separated file under the experiment
# directory with `print_to_csv`.

# %%
model_experiment.print_to_csv(csv_file_name="cntnews_data_farming", overwrite=True)

# %%
fig, axes = plt.subplots(1, 4, figsize=(14, 3), sharey=True)
for ax, factor in zip(
    axes, ["purchase_price", "sales_price", "salvage_price", "order_quantity"], strict=True
):
    ax.scatter(model_df[factor], model_df["mean_profit"])
    ax.set_xlabel(factor)
axes[0].set_ylabel("mean profit")
plt.show()

# %% [markdown]
# ## Helpers for solver and problem designs
#
# A solver or problem design is a list of factor dictionaries. For a problem design,
# each dictionary mixes problem factors and model factors, so split them before
# constructing the problem. To compare design points, summarize each experiment by the
# mean and standard deviation of its terminal normalized objective (0 = reached the
# proxy optimum, 1 = no progress from $x_0$).

# %%
simulation_config = SimulationConfig(n_mreps=3, n_preps=100, n_preps_x0_xstar=200)


def make_problem(problem_cls, factors):
    """Build a problem from a design point that mixes problem and model factors."""
    return problem_cls(
        fixed_factors={k: v for k, v in factors.items() if k in problem_cls.specifications},
        model_fixed_factors={
            k: v for k, v in factors.items() if k in problem_cls.model_class.specifications
        },
    )


def summarize(design, results, columns):
    """Join the varied design factors with each experiment's terminal performance."""
    rows = []
    for design_pt, result in zip(design, results, strict=True):
        terminal = terminal_scatterplot.analyze(result, normalize=True)
        rows.append(
            {
                **{c: design_pt[c] for c in columns},
                "terminal_mean": terminal.mean,
                "terminal_std": terminal.std,
            }
        )
    return pd.DataFrame(rows)


# %% [markdown]
# ## 2. Solver data farming
#
# `DesignSpec` describes the design: `varied` factors form the NOLHS, boolean `crossed`
# factors are fully crossed with it, and `fixed` overrides defaults. Every design point
# runs on the same problem object, so they share one normalization.

# %%
solver_design = build_design(
    DesignSpec(
        kind="solver",
        name="ASTRODF",
        varied={
            "eta_1": NumericRange(min=0.05, max=0.15, decimals=2),
            "eta_2": NumericRange(min=0.6, max=0.95, decimals=2),
        },
        crossed={"crn_across_solns": [True, False]},
    )
)
print(f"{len(solver_design)} solver design points")

problem = CntNVMaxProfit(fixed_factors={"budget": 500})
experiments = [
    ProblemSolver(solver=ASTRODF(name=f"ASTRODF_dp{i}", fixed_factors=dp), problem=problem)
    for i, dp in enumerate(solver_design)
]
solver_df = summarize(
    solver_design, run(experiments, simulation_config), ["eta_1", "eta_2", "crn_across_solns"]
)
solver_df.head()

# %%
fig, ax = plt.subplots()
for crn, group in solver_df.groupby("crn_across_solns"):
    ax.scatter(group["eta_1"], group["terminal_mean"], label=f"crn_across_solns={crn}")
ax.set_xlabel("eta_1")
ax.set_ylabel("mean terminal normalized objective")
ax.legend()
plt.show()

# %% [markdown]
# ## 3. Problem data farming
#
# Varying model factors creates one problem instance per design point. Each instance
# gets its own $x_0$/$x^*$ normalization.

# %%
problem_design = build_design(
    DesignSpec(
        kind="problem",
        name="CNTNEWS-1",
        varied={
            "purchase_price": NumericRange(min=4.0, max=6.0, decimals=1),
            "sales_price": NumericRange(min=8.0, max=12.0, decimals=1),
        },
        fixed={"budget": 500, "salvage_price": 3.0},
    )
)
print(f"{len(problem_design)} problem design points")

solver = RandomSearch(fixed_factors={"sample_size": 10})
experiments = [
    ProblemSolver(solver=solver, problem=make_problem(CntNVMaxProfit, dp)) for dp in problem_design
]
problem_df = summarize(
    problem_design, run(experiments, simulation_config), ["purchase_price", "sales_price"]
)
problem_df.head()

# %% [markdown]
# Designs can be crossed by hand: every solver design point against every problem
# design point. Keep in mind the experiment count multiplies quickly; here only a
# few points of each design are used.

# %%
crossed = [
    ProblemSolver(
        solver=ASTRODF(name=f"ASTRODF_dp{i}", fixed_factors=solver_dp),
        problem=problem,
    )
    for problem in [make_problem(CntNVMaxProfit, dp) for dp in problem_design[:3]]
    for i, solver_dp in enumerate(solver_design[:3])
]
print(f"{len(crossed)} crossed experiments")
crossed_results = run(crossed, simulation_config)

# %% [markdown]
# ## 4. Loading a design from a file
#
# Designs exported by the SimOpt GUI are tab-separated, with a header row of factor
# names and some bookkeeping columns that are not factors.

# %%
design_table = pd.read_csv(Path("data") / "load_solver_design_RNDSRCH_design.csv", sep="\t")
gui_columns = ["design_num", "name", "design_type", "num_stacks"]
loaded_design = design_table.drop(columns=gui_columns, errors="ignore").to_dict(orient="records")
print(f"{len(loaded_design)} design points, factors: {list(loaded_design[0])}")

problem = SSContMinCost(fixed_factors={"budget": 500})
experiments = [
    ProblemSolver(solver=RandomSearch(name=f"RNDSRCH_dp{i}", fixed_factors=dp), problem=problem)
    for i, dp in enumerate(loaded_design)
]
loaded_df = summarize(loaded_design, run(experiments, simulation_config), ["sample_size"])
loaded_df.sort_values("sample_size").head()
