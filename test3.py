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
import pickle

from simopt.experiment_base import (
    PlotType,
    plot_feasibility_progress,
    plot_terminal_feasibility,
    plot_terminal_progress,
)

# %%
with open("test-data-2.pkl", "rb") as f:
    e1, e2, e3 = pickle.load(f)

# %%
plot_terminal_progress([e1, e2], PlotType.VIOLIN, normalize=False)
plot_terminal_progress([e1, e2], PlotType.VIOLIN, normalize=True)
plot_terminal_progress([e1, e2], PlotType.VIOLIN, all_in_one=False, normalize=False)
plot_terminal_progress([e1, e2], PlotType.VIOLIN, all_in_one=False, normalize=True)

# %%
plot_terminal_feasibility(
    [[e1], [e2]], PlotType.FEASIBILITY_SCATTER, all_in_one=True, two_sided=True
)
plot_terminal_feasibility(
    [[e1], [e2]], PlotType.FEASIBILITY_SCATTER, all_in_one=False, two_sided=True
)
plot_terminal_feasibility(
    [[e1], [e2]], PlotType.FEASIBILITY_SCATTER, all_in_one=True, two_sided=True
)
plot_terminal_feasibility(
    [[e1], [e2]], PlotType.FEASIBILITY_SCATTER, all_in_one=False, two_sided=True
)
plot_terminal_feasibility(
    [[e1], [e2]],
    PlotType.FEASIBILITY_SCATTER,
    all_in_one=True,
    two_sided=True,
    plot_conf_ints=False,
)
plot_terminal_feasibility(
    [[e1], [e2]],
    PlotType.FEASIBILITY_SCATTER,
    all_in_one=False,
    two_sided=True,
    plot_conf_ints=False,
)

# %%
plot_terminal_feasibility(
    [[e1]], PlotType.FEASIBILITY_VIOLIN, all_in_one=True, two_sided=True
)
plot_terminal_feasibility(
    [[e1], [e2]], PlotType.FEASIBILITY_VIOLIN, all_in_one=False, two_sided=True
)

# %%
plot_feasibility_progress(
    [[e2], [e3]], PlotType.ALL_FEASIBILITY_PROGRESS, print_max_hw=False, two_sided=True
)
plot_feasibility_progress(
    [[e2], [e3]],
    PlotType.ALL_FEASIBILITY_PROGRESS,
    print_max_hw=False,
    two_sided=True,
    all_in_one=False,
)

# %%
plot_feasibility_progress(
    [[e2], [e3]],
    PlotType.MEAN_FEASIBILITY_PROGRESS,
    print_max_hw=False,
    two_sided=False,
)
plot_feasibility_progress(
    [[e2], [e3]],
    PlotType.MEAN_FEASIBILITY_PROGRESS,
    print_max_hw=False,
    two_sided=False,
    all_in_one=False,
)

# %%
plot_feasibility_progress(
    [[e2], [e3]],
    PlotType.QUANTILE_FEASIBILITY_PROGRESS,
    print_max_hw=False,
    two_sided=False,
    beta=0.9,
)
plot_feasibility_progress(
    [[e2], [e3]],
    PlotType.QUANTILE_FEASIBILITY_PROGRESS,
    print_max_hw=False,
    two_sided=False,
    beta=0.9,
    all_in_one=False,
)
