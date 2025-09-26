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
# # Run five solvers on 20 versions of the (s, S) inventory problem.
#
# Produces plots appearing in the INFORMS Journal on Computing submission.

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
# ## Experiment Configuration Parameters
#
# Define the options used in the experiments.

# %%
# Experiment Configuration
num_macroreps = 10
num_postreps = 100
num_postnorms = 200


# %% [markdown]
# ## Helper Functions
#
# Helper functions to streamline the experiment process.


# %%
# Classes for Problem and Solver config info
# These don't need modified
class ProblemConfig:
    """Class to hold problem config information."""

    def __init__(
        self,
        name: str,
        rename: str | None = None,
        fixed_factors: dict | None = None,
        model_fixed_factors: dict | None = None,
    ) -> None:
        """Initialize the Problem with name, rename, and problem/model fixed factors."""
        self.name = name
        self.rename = rename if rename else name
        self.fixed_factors = fixed_factors if fixed_factors else {}
        self.model_fixed_factors = model_fixed_factors if model_fixed_factors else {}


class SolverConfig:
    """Class to hold solver config information."""

    def __init__(
        self, name: str, rename: str | None = None, fixed_factors: dict | None = None
    ) -> None:
        """Initialize the Solver with name, rename, and solver fixed factors."""
        self.name = name
        self.rename = rename if rename else name
        self.fixed_factors = fixed_factors if fixed_factors else {}


# %%
from simopt.experiment_base import ProblemSolver, post_normalize


# Function to run an experiment with the given problem and solver configs.
def run_experiment(
    problems: list[ProblemConfig],
    solvers: list[SolverConfig],
) -> list[list[ProblemSolver]]:
    """Run an experiment using the provided configurations.

    Args:
        problems: List of ProblemConfig instances.
        solvers: List of SolverConfig instances.

    Returns:
        List[list[ProblemSolver]]: A list of lists containing ProblemSolver instances,
        grouped by problem.
    """
    all_experiments = []
    for problem_idx, problem in enumerate(problems):
        print(
            f"Running Problem {problem_idx + 1}/{len(problems)}: {problem.rename}...",
            end="",
            flush=True,
        )
        # Keep track of experiments on the same problem for post-processing.
        experiments_same_problem = []
        # Create each ProblemSolver and run it.
        for solver in solvers:
            new_experiment = ProblemSolver(
                solver_name=solver.name,
                solver_rename=solver.rename,
                solver_fixed_factors=solver.fixed_factors,
                problem_name=problem.name,
                problem_rename=problem.rename,
                problem_fixed_factors=problem.fixed_factors,
                model_fixed_factors=problem.model_fixed_factors,
            )
            # Run and post-replicate the experiment.
            new_experiment.run(n_macroreps=num_macroreps)
            new_experiment.post_replicate(n_postreps=num_postreps)
            experiments_same_problem.append(new_experiment)

        # Post-normalize experiments with L.
        # Provide NO proxies for f(x0), f(x*), or f(x).
        post_normalize(
            experiments=experiments_same_problem,
            n_postreps_init_opt=num_postnorms,
        )
        all_experiments.append(experiments_same_problem)
        print("Done.")
    print("All experiments completed.")
    return all_experiments


# %% [markdown]
# ## Problem/Solver Configuration Parameters
#
# Define the problems and solvers used in the experiments.

# %%
# Solvers to use in the experiment.
# Includes two versions of random search with varying sample sizes.
# The rename will be used in the plots to differentiate them.
solvers = [
    SolverConfig(
        name="RNDSRCH", rename="RNDSRCH_ss=10", fixed_factors={"sample_size": 10}
    ),
    SolverConfig(
        name="RNDSRCH", rename="RNDSRCH_ss=50", fixed_factors={"sample_size": 50}
    ),
    SolverConfig(name="ASTRODF"),
    SolverConfig(name="NELDMD"),
    SolverConfig(name="STRONG"),
]

# %%
# Configure the problem
demand_means = [25.0, 50.0, 100.0, 200.0, 400.0]
lead_means = [1.0, 3.0, 6.0, 9.0]

# Create all the problem variants.
SSCONT_problems = [
    ProblemConfig(
        name="SSCONT-1",
        rename=f"SSCONT-1_dm={dm}_lm={lm}",
        fixed_factors={"budget": 1000},
        model_fixed_factors={
            "demand_mean": dm,
            "lead_mean": lm,
        },
    )
    for dm in demand_means
    for lm in lead_means
]

# Run the experiment for SSCONT problems.
SSCONT_experiments = run_experiment(
    problems=SSCONT_problems,
    solvers=solvers,
)

# %%
# Combine the experiments into a list of lists, where the outer list
# contains all the experiments for a single solver, and the inner lists
# contain the ProblemSolver instances associated with that solver.

experiment_dict = {}
for exp_problem_list in SSCONT_experiments:
    for experiment in exp_problem_list:
        # Use the solver name as the key and append the ProblemSolver instance.
        key = experiment.solver.name
        if key not in experiment_dict:
            experiment_dict[key] = []
        experiment_dict[key].append(experiment)
# Turn the dictionary into a list of lists.
experiments = list(experiment_dict.values())

# %% [markdown]
# ## Plotting Settings
#
# Define the plotting settings for the experiments.

# %%
from simopt.experiment_base import (
    PlotType,
    plot_area_scatterplots,
    plot_progress_curves,
    plot_solvability_profiles,
    plot_terminal_progress,
    plot_terminal_scatterplots,
)

enable_confidence_intervals = True
alpha = 0.2

# %%
plot_area_scatterplots(
    experiments, all_in_one=True, plot_conf_ints=True, print_max_hw=True
)

# %%
plot_solvability_profiles(
    experiments,
    plot_type=PlotType.CDF_SOLVABILITY,
    solve_tol=0.1,
    all_in_one=True,
    plot_conf_ints=True,
    print_max_hw=True,
)

# %%
plot_solvability_profiles(
    experiments,
    plot_type=PlotType.QUANTILE_SOLVABILITY,
    solve_tol=0.1,
    beta=0.5,
    all_in_one=True,
    plot_conf_ints=True,
    print_max_hw=True,
)

# %%
plot_solvability_profiles(
    experiments=experiments,
    plot_type=PlotType.DIFFERENCE_OF_CDF_SOLVABILITY,
    solve_tol=0.1,
    ref_solver="ASTRODF",
    all_in_one=True,
    plot_conf_ints=True,
    print_max_hw=True,
)

# %%
plot_solvability_profiles(
    experiments=experiments,
    plot_type=PlotType.DIFFERENCE_OF_QUANTILE_SOLVABILITY,
    solve_tol=0.1,
    beta=0.5,
    ref_solver="ASTRODF",
    all_in_one=True,
    plot_conf_ints=True,
    print_max_hw=True,
)

# %%
plot_terminal_scatterplots(experiments, all_in_one=True)

# %%
n_problems = len(SSCONT_problems)
n_solvers = len(experiments)

for i in range(n_problems):
    plot_progress_curves(
        [experiments[solver_idx][i] for solver_idx in range(n_solvers)],
        plot_type=PlotType.MEAN,
        all_in_one=True,
        plot_conf_ints=True,
        print_max_hw=True,
        normalize=False,
    )
    plot_terminal_progress(
        [experiments[solver_idx][i] for solver_idx in range(n_solvers)],
        plot_type=PlotType.VIOLIN,
        normalize=True,
        all_in_one=True,
    )
