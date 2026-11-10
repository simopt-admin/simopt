"""Create test cases for all compatible problem-solver pairs."""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import zstandard as zstd
from colorama import Fore, init

# Append the parent directory (simopt package) to the system path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import simopt.directory as directory
from simopt.experiment import check_compatibility, from_names
from simopt.experiment.api import SimulationConfig, run
from simopt.options import CrnOptions
from tools.common import expected_results_filename

init(autoreset=True)

# Workaround for AutoAPI
problem_directory = directory.problem_directory
solver_directory = directory.solver_directory

NUM_MACROREPS = 10
NUM_POSTREPS = 100

# Setup the SimOpt directory structure
HOME_DIR = Path(__file__).resolve().parent.parent
EXPECTED_RESULTS_DIR = HOME_DIR / "tests" / "expected_results"


def _color_text(text: str, color: str | int) -> str:
    return color + text  # type: ignore


# Based off the similar function in simopt/experiment/experiment.py
def is_compatible(problem_name: str, solver_name: str) -> bool:
    """Check if a solver is compatible with a problem.

    Args:
        problem_name (str): Name of the problem.
        solver_name (str): Name of the solver.

    Returns:
        bool: Whether the solver is compatible with the problem.
    """
    experiment = from_names(solver_name, problem_name)
    # Check if the solver is compatible with the problem
    output = check_compatibility(experiment)
    return len(output) == 0


def create_test(
    problem_name: str, solver_name: str, output_dir: Path = EXPECTED_RESULTS_DIR
) -> None:
    """Create a test case for a problem and solver.

    Args:
        problem_name (str): Name of the problem.
        solver_name (str): Name of the solver.
        output_dir (Path): Directory the results file is written to.
    """
    # Run the experiment to get the expected results
    experiment = from_names(solver_name, problem_name)
    [result] = run(
        [experiment],
        SimulationConfig(
            n_mreps=NUM_MACROREPS,
            n_preps=NUM_POSTREPS,
            n_preps_x0_xstar=NUM_POSTREPS,
        ),
        # The CRN options are not stored in the results file. These are the defaults
        # of the old pipeline.
        CrnOptions(across_budget=True, across_macroreps=False, across_x0_xstar=True),
        n_jobs=1,
    )
    full_df = result.full_df
    mean_df = result.mean_df.sort_values(["mrep", "step"])
    est_df = full_df.groupby(["mrep", "step"]).agg(
        {"objective": "mean", "stochastic_constraints": "mean"}
    )

    # Curves are stored as (x_vals, y_vals) tuples to avoid packing custom classes
    # into the pickle file
    total_budget = experiment.problem.factors["budget"]
    objective_curves = []
    progress_curves = []
    for _, mrep_df in mean_df.groupby("mrep"):
        budget = mrep_df["budget"].to_numpy()
        objective_curves.append(
            (tuple(budget.astype(float).tolist()), tuple(mrep_df["objective"].tolist()))
        )
        progress_curves.append(
            (
                tuple((budget / total_budget).tolist()),
                tuple(mrep_df["normalized_objective"].tolist()),
            )
        )

    all_recommended_xs = [list(g["solution"]) for _, g in mean_df.groupby("mrep")]
    all_intermediate_budgets = [g["budget"].tolist() for _, g in mean_df.groupby("mrep")]
    all_est_objectives = [g["objective"].tolist() for _, g in est_df.groupby(level="mrep")]
    all_post_replicates = [
        [np.array(reps) for _, reps in g.groupby("step")["objective"]]
        for _, g in full_df.sort_values(["mrep", "step", "rep"]).groupby("mrep")
    ]
    all_stoch_constraints = [[] for _ in range(NUM_MACROREPS)]
    all_est_lhs = []
    if experiment.problem.n_stochastic_constraints > 0:
        all_stoch_constraints = [
            [np.array(list(reps)) for _, reps in g.groupby("step")["stochastic_constraints"]]
            for _, g in full_df.sort_values(["mrep", "step", "rep"]).groupby("mrep")
        ]
        all_est_lhs = [
            g["stochastic_constraints"].tolist() for _, g in est_df.groupby(level="mrep")
        ]

    results_dict = {
        "num_macroreps": NUM_MACROREPS,
        "num_postreps": NUM_POSTREPS,
        "problem_name": problem_name,
        "solver_name": solver_name,
        "all_recommended_xs": all_recommended_xs,
        "all_intermediate_budgets": all_intermediate_budgets,
        "all_est_objectives": all_est_objectives,
        "objective_curves": objective_curves,
        "progress_curves": progress_curves,
        "all_est_lhs": all_est_lhs,
        "all_post_replicates": all_post_replicates,
        "all_stoch_constraints": all_stoch_constraints,
        "n_postreps_init_opt": NUM_POSTREPS,
        "x0": tuple(result.x0.tolist()),
        "xstar": tuple(result.xstar.tolist()),
        "x0_postreps": result.x0_sample.tolist(),
        "xstar_postreps": result.xstar_sample.tolist(),
    }

    # Define the output file
    results_filename = expected_results_filename(problem_name, solver_name)
    results_filepath = output_dir / results_filename
    # Write the results to the file
    with zstd.open(results_filepath, "wb") as f:
        pickle.dump(results_dict, f)


def main() -> None:
    """Create test cases for all compatible problem-solver pairs."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXPECTED_RESULTS_DIR,
        help="Directory to write the results files to (default: tests/expected_results).",
    )
    parser.add_argument(
        "--pair",
        nargs=2,
        metavar=("PROBLEM", "SOLVER"),
        help="Only generate the results for this problem-solver pair.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir

    skip_problems = {"ERM-EXAMPLE-1"}
    skip_pairs = {
        ("DUALSOURCING-1", "DASSO"),
        ("HOTEL-1", "DASSO"),
    }
    if args.pair is not None:
        pair_problem, pair_solver = args.pair
        if pair_problem not in problem_directory:
            parser.error(f"Unknown problem: {pair_problem}")
        if pair_solver not in solver_directory:
            parser.error(f"Unknown solver: {pair_solver}")
        if not is_compatible(pair_problem, pair_solver):
            parser.error(f"Incompatible pair: {pair_problem} and {pair_solver}")
        compatible_pairs = [(pair_problem, pair_solver)]
    else:
        # Create a list of compatible problem-solver pairs
        compatible_pairs = [
            (problem_name, solver_name)
            for problem_name in problem_directory
            for solver_name in solver_directory
            if is_compatible(problem_name, solver_name)
        ]

    # Create the test directory if it doesn't exist
    # Create the expected directory if it doesn't exist
    Path.mkdir(output_dir, parents=True, exist_ok=True)
    existing_results = [path.name for path in output_dir.glob("*.pickle.zst")]

    # Don't generate any tests for pairs that already have tests generated
    for pair in compatible_pairs:
        problem_name = pair[0]
        solver_name = pair[1]
        # Generate the expected filenames
        results_filename = expected_results_filename(problem_name, solver_name)
        # If file exists, skip it
        if results_filename in existing_results:
            print(_color_text(f"Test for {pair} already exists", Fore.GREEN))
            continue
        if problem_name in skip_problems:
            print(_color_text(f"Skipping test for {pair}", Fore.YELLOW))
            continue
        if pair in skip_pairs:
            print(_color_text(f"Skipping test for {pair}", Fore.YELLOW))
            continue
        # If file doesn't exist, create it
        print(f"Creating test for {pair}")
        create_test(problem_name, solver_name, output_dir)
    print("All tests created!")


if __name__ == "__main__":
    main()
