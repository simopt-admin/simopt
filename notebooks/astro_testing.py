# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python [conda env:base] *
#     language: python
#     name: conda-base-py
# ---

# %% [markdown]
# # Demo for the ProblemsSolvers class.
#
# This script is intended to help with debugging problems and solvers.
#
# It create problem-solver groups (using the directory) and runs multiple macroreplications of each problem-solver pair.

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
# Specify the names of the solver(s) and problem(s) to test.
#solver_abbr_names = [ "SQPASTRODF"]
#COBYQA testing w/ CUTEst
# solver_abbr_names = ["SQPASTRODF", "COBYQA"]
# problem_abbr_names = ["SAN-3"]
# solver_factors = [{"feas_tol": 1e-8, 
#                    "easy_solve": True, 
#                    "use_gradients":False, 
#                    "sampling_method": "fixed",
#                    "ps_sufficient_reduction": 0,
#                    "mu": .1, 
#                    "a_normal": .9,
#                    "a_tangent": .5,
#                    "eta_2": .4,
#                    "gamma_1": 3,
#                    "sigma_min": 10,
#                    #"gamma_2": .9,
#                    "reuse_interpolation_set": True,
#                    "recenter_mode": "shift",
#                    #"gamma_2": 0.85, 
#                    "theta_decrease": .9999, 
#                    "lagrange_poisedness_threshold": 20, 
#                    "delta_0":2,
#                    "dogleg": True,
#                    "kappa_scale": 100
#                   },
#                   {"sample_size": 5, 
#                    "feas_tol": 1e-8}]

#COBYQA testing w/ SAN
solver_abbr_names = ["CASTRODF", "COBYQA"]
problem_abbr_names = ["SAN-3"]
solver_factors = [{"feas_tol": 1e-5, 
                   "easy_solve_tangent": True, 
                   "use_gradients":False, 
                   "sampling_method": "adaptive",
                   "ps_sufficient_reduction": 100,
                   "mu": .1, 
                   "a_normal": .2,
                   "a_tangent": .8,
                   #"eta_2": .2,
                   #"gamma_1": 3,
                   "sigma_min": 10,
                   #"gamma_2": .9,
                   "reuse_interpolation_set": True,
                   "recenter_mode": "shift",
                   #"gamma_2": 0.85, 
                   "theta_decrease": .5, 
                   "lagrange_poisedness_threshold": 10, 
                   "delta_0":1,
                   "kappa_scale":10000,
                   "dogleg": True,
                   "lambda_min":10,
                  },
                  {"sample_size": 5, 
                   "feas_tol": 1e-5}]


# # SAN factors
problem_factors = [{"budget":5000, "total_cost": 5}]
# Quad factors
#problem_factors = [{"budget":1000, "noise_std": 0}]
# small SAN problem factors
# arcs = [
#     (1, 2),
#     (1, 3),
#     (2, 4),
#     (3, 4),
#     (4, 5),
# ]
# model_factors = {"arcs": arcs, "num_nodes":5}
# arc_means = (1.0,)*5
# initial_sol = (8.0,)*5
# arc_cost = (1.0,)*5
# problem_factors = [{"budget":30000, 
#                     "total_cost": 2,
#                     "arcs": arcs, 
#                     "num_nodes":5,
#                     "initial_solution": initial_sol,
#                     "arc_means": arc_means,
#                     "arc_costs": arc_cost
#                    }]
num_macroreps = 10
num_postreps = 50
num_postreps_init_opt = 50

# %%
# Initialize an instance of the experiment class.
from simopt.experiment_base import ProblemsSolvers

mymetaexperiment = ProblemsSolvers(
    solver_names=solver_abbr_names, 
    problem_names=problem_abbr_names, 
    solver_factors = solver_factors, 
    problem_factors = problem_factors,
    #solver_renames = solver_renames
)

# Write to log file.
mymetaexperiment.log_group_experiment_results()

# %%
# Run a fixed number of macroreplications of each solver on each problem.
mymetaexperiment.run(n_macroreps=num_macroreps)

# %%
print("Post-processing results.")
# Run a fixed number of postreplications at all recommended solutions.
mymetaexperiment.post_replicate(n_postreps=num_postreps)

# %%
print("Post-normalizing results.")

# Find an optimal solution x* for normalization.
mymetaexperiment.post_normalize(n_postreps_init_opt=num_postreps_init_opt)

# %%
print("Post-normalizing merit results.")

# Find an optimal solution x* for normalization.
mymetaexperiment.post_normalize_merit(obj_const = 1000, feas_tol_upper = 10000, feas_tol_lower = 1e-5)

# %%
#mymetaexperiment.report_group_statistics()

# %%
# Produce basic plots.

from simopt.experiment_base import (
    plot_area_scatterplots,  # noqa: F401
    plot_feasibility_progress,  # noqa: F401
    plot_progress_curves,  # noqa: F401
    plot_solvability_cdfs,  # noqa: F401
    plot_solvability_profiles,  # noqa: F401
    plot_terminal_feasibility,  # noqa: F401
    plot_terminal_progress,  # noqa: F401
    plot_terminal_scatterplots, 
    plot_det_feasibility,
    plot_det_terminal_feasibility,
    PlotType # noqa: F401
)

# %%
# Produce basic plots.

exp = mymetaexperiment.experiments[0][0]  # solver 0, problem 0
for i, curve in enumerate(exp.merit_progress_curves):
    print(f"mrep {i}: min={min(curve.y_vals):.4f}, final={curve.y_vals[-1]:.4f}")

# %%
plot_solvability_profiles(
    mymetaexperiment.experiments,
    plot_type=PlotType.CDF_SOLVABILITY,
    curve_source="merit_progress_curves",
    solve_tol=0.2,
    feas_obj_const = 1000,
    feas_tol_upper = 10000,
    feas_tol_lower = 1e-5
)


# %%
def _print_path(plot_path: list[Path]) -> None:
    print(f"Plot saved to {plot_path!s}")


_print_path(
    plot_progress_curves(
        experiments=mymetaexperiment.experiments[0] +mymetaexperiment.experiments[1]  , all_in_one = True, plot_type=PlotType.ALL, normalize=False,  save_as_pickle = True,
    )
)

print("Plotting complete!")


# %%
def _print_path(plot_path: list[Path]) -> None:
    print(f"Plot saved to {plot_path!s}")


_print_path(
    plot_terminal_progress(
        experiments=mymetaexperiment.experiments[0] + mymetaexperiment.experiments[1], normalize = False
    )
)

print("Plotting complete!")


# %%
def _print_path(plot_path: list[Path]) -> None:
    print(f"Plot saved to {plot_path!s}")


_print_path(
    plot_det_feasibility(
        experiments=mymetaexperiment.experiments,  sym_log = False,score_type = "norm", feas_tol_lower = 1e-5, save_as_pickle=True, log_scale=True,
    )
)

print("Plotting complete!")


# %%
def _print_path(plot_path: list[Path]) -> None:
    print(f"Plot saved to {plot_path!s}")


_print_path(
    plot_det_feasibility(
        experiments=mymetaexperiment.experiments, 
        log_scale = True, 
        score_type = "objective", 
        obj_const = 1e8, 
        feas_tol_upper = 1000,
        feas_tol_lower = 1e-4,
        save_as_pickle = True,
    )
)

print("Plotting complete!")


# %%
def _print_path(plot_path: list[Path]) -> None:
    print(f"Plot saved to {plot_path!s}")


_print_path(
    plot_det_terminal_feasibility(
        experiments=mymetaexperiment.experiments, all_in_one = True, plot_conf_ints = True, score_type = "norm", 
        feas_tol_lower = 1e-2,
        log_scale = True,
        sym_log= False,
        floor_feas=True,
        plot_zero = False,
        save_as_pickle=True
    )
)

print("Plotting complete!")


# %%
def _print_path(plot_path: list[Path]) -> None:
    print(f"Plot saved to {plot_path!s}")


_print_path(
    plot_det_terminal_feasibility(
        experiments=mymetaexperiment.experiments, all_in_one = True, 
        plot_conf_ints = True, sym_log = False, score_type = "norm", feas_tol = 1e-12, log_scale=False, floor_feas = True
    )
)

print("Plotting complete!")


# %%
def _print_path(plot_path: list[Path]) -> None:
    print(f"Plot saved to {plot_path!s}")


_print_path(
    plot_det_terminal_feasibility(
        experiments=mymetaexperiment.experiments, all_in_one = True, score_type = "norm", save_as_pickle = True, plot_conf_ints = True, feas_tol = 1e-4,  plot_zero = False, log_scale = False
    )
)

print("Plotting complete!")

# %%
