"""Script to run experiments from the command line."""

# Sample usage for running a single experiment with viztracer:
# `viztracer --min_duration 0.05ms utils/run_experiment.py --problems=CHESS-1 --num_macroreps=1`

import argparse
import logging
import sys
import time
from pathlib import Path

# Append the parent directory (simopt package) to the system path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import simopt.directory as directory
from simopt.experiment import api
from simopt.experiment.experiment import Experiment, check_compatibility, from_names

# Workaround for AutoAPI
problem_directory = directory.problem_directory
solver_directory = directory.solver_directory


def load_arguments() -> tuple[list[str], list[str], str, int, int]:
    """Load command line arguments.

    Returns:
        tuple: A tuple containing the problems, solvers, log level,
               number of macroreplications, and number of postreplications.
    """
    parser = argparse.ArgumentParser(
        description="Execute problem-solver experiments with `simopt.experiment.api.run`."
    )
    parser.add_argument(
        "--problems",
        type=str,
        default="all",
        help=("Abbreviated names of problems to run (comma-separated or `all`) [default: all]"),
    )
    parser.add_argument(
        "--solvers",
        type=str,
        default="all",
        help=("Abbreviated names of solvers to run (comma-separated or `all`) [default: all]"),
    )
    parser.add_argument(
        "--log",
        type=str,
        default="INFO",
        help=("Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`) [default: INFO]"),
    )
    parser.add_argument(
        "--num_macroreps",
        type=int,
        default=10,
        help="Number of macroreplications [default: 10]",
    )
    parser.add_argument(
        "--num_postreps",
        type=int,
        default=100,
        help=(
            "Number of postreplications, also used for the x0/x* "
            "normalization replications [default: 100]"
        ),
    )

    args = parser.parse_args()
    return (
        args.problems.split(","),
        args.solvers.split(","),
        args.log.upper(),
        args.num_macroreps,
        args.num_postreps,
    )


def is_compatible(problem_name: str, solver_name: str) -> bool:
    """Check if a solver is compatible with a problem.

    Args:
        problem_name (str): Name of the problem.
        solver_name (str): Name of the solver.

    Returns:
        bool: True if the solver is compatible with the problem, False otherwise.
    """
    output = check_compatibility(from_names(solver_name, problem_name))
    return len(output) == 0


def create_experiments(problems: list[str], solvers: list[str]) -> list[Experiment]:
    """Generate valid problem-solver pairs.

    Args:
        problems (list[str]): List of problem names.
        solvers (list[str]): List of solver names.

    Returns:
        list[Experiment]: List of valid Experiment objects.
    """
    return [
        from_names(solver_name, problem_name)
        for problem_name in problem_directory
        if problems == ["all"] or problem_name in problems
        for solver_name in solver_directory
        if solvers == ["all"] or solver_name in solvers
        if is_compatible(problem_name, solver_name)
    ]


def run_experiments(
    experiments: list[Experiment], n_macroreps: int, n_postreps: int
) -> list[tuple[str, str, float]]:
    """Run each experiment with `api.run` and return the runtime of the whole call.

    Args:
        experiments (list[Experiment]): List of valid problem-solver pairs.
        n_macroreps (int): Number of macroreplications.
        n_postreps (int): Number of postreplications.

    Returns:
        list[tuple[str, str, float]]: List of tuples containing the problem name,
                                      solver name, and elapsed time.
    """
    config = api.SimulationConfig(
        n_mreps=n_macroreps, n_preps=n_postreps, n_preps_x0_xstar=n_postreps
    )
    runtimes = []
    for exp in experiments:
        problem_name = exp.problem.class_name_abbr
        solver_name = exp.solver.class_name_abbr
        logging.info(f"Experimenting with {solver_name} on {problem_name}.")

        start_time = time.time()
        api.run([exp], config, n_jobs=1)
        elapsed = time.time() - start_time
        logging.info(f"Elapsed time for {solver_name} on {problem_name}: {elapsed:.2f}s")

        runtimes.append((problem_name, solver_name, elapsed))
    return runtimes


def print_runtimes(runtimes: list[tuple[str, str, float]]) -> None:
    """Print the runtimes of the experiments.

    Args:
        runtimes (list[tuple[str, str, float]]): List of tuples containing
                                                 the problem name, solver name,
                                                 and elapsed time.
    """
    # Extra blank line to make it clear where the results start
    print()
    # Print header row
    header_fields = [
        f"{'Problem Name':<20}",
        f"{'Solver Name':<20}",
        f"{'Elapsed Time (s)':<20}",
    ]
    print(" │ ".join(header_fields))
    # Print the separator
    bars = ["─" * 20] * len(header_fields)
    print("─┼─".join(bars))
    # Print each runtime
    for problem_name, solver_name, elapsed in runtimes:
        fields = [
            f"{problem_name:<20}",
            f"{solver_name:<20}",
            f"{elapsed:.2f}",
        ]
        print(" │ ".join(fields))


# Main loop to iterate through problem/solver pairs
def main() -> None:
    """Main function to run the experiment."""
    # Load the arguments
    problems, solvers, log_level, num_macroreps, num_postreps = load_arguments()

    # Set logging level
    logging.basicConfig(
        format="%(levelname)s: %(message)s",
        level=getattr(logging, log_level, logging.INFO),
    )

    valid_pairs = create_experiments(problems, solvers)
    if len(valid_pairs) == 0:
        logging.warning("No valid problem-solver pairs found.")
        return
    runtimes = run_experiments(valid_pairs, num_macroreps, num_postreps)
    logging.info("Execution complete.")
    print_runtimes(runtimes)


if __name__ == "__main__":
    main()
