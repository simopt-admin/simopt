"""Script to run experiments from the command line."""

# Sample usage for running a single experiment with viztracer:
# viztracer --min_duration 0.05ms -m dev_tools.run_experiment --problems=CHESS-1 --num_macroreps=1

import argparse
import logging
import os
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path

from simopt.directory import problem_directory, solver_directory
from simopt.experiment_base import ProblemSolver, post_normalize

# Change the working directory to the parent directory of this script
os.chdir(Path(__file__).resolve().parent.parent)


# Function to check compatibility
def is_compatible(problem_name: str, solver_name: str) -> bool:
    myexperiment = ProblemSolver(solver_name, problem_name)
    output = myexperiment.check_compatibility()
    return len(output) == 0


def generate_valid_pairs(
    problems: list[str], solvers: list[str]
) -> list[tuple[str, str]]:
    return [
        (solver_name, problem_name)
        for problem_name in problem_directory
        if problems == ["all"] or problem_name in problems
        for solver_name in solver_directory
        if solvers == ["all"] or solver_name in solvers
        if is_compatible(problem_name, solver_name)
    ]


# Main loop to iterate through problem/solver pairs
def main(
    valid_pairs: list[tuple[str, str]],
    methods: list[str],
    num_macroreps: int,
    num_postreps: int,
) -> None:
    # Check these outside of the loop to avoid repeated checks
    method_steps = ["run", "post_replicate", "post_normalize"]
    level = (
        2
        if "all" in methods
        else max(
            (method_steps.index(m) for m in methods if m in method_steps),
            default=-1,
        )
    )

    if valid_pairs == []:
        logging.warning(
            f"No valid problem/solver pairs found for problems {problems} and solvers {solvers}."
        )
        return

    runtime_strings = []
    # For each problem/solver pair, run the experiment
    for solver_name, problem_name in valid_pairs:
        logging.info(f"Experimenting with {solver_name} on {problem_name}.")
        myexperiment = ProblemSolver(solver_name, problem_name)

        partial_list = []
        if level >= 0:
            # Append the run method
            partial_list.append(
                partial(
                    myexperiment.run,
                    n_macroreps=num_macroreps,
                )
            )
        if level >= 1:
            # Append the post_replicate method
            partial_list.append(
                partial(
                    myexperiment.post_replicate,
                    n_postreps=num_postreps,
                )
            )
        if level >= 2:
            # Append the post_normalize method
            partial_list.append(
                partial(
                    post_normalize,
                    [myexperiment],
                    n_postreps_init_opt=num_postreps,
                )
            )

        def time_function(func: Callable) -> float:
            """Decorator to print the runtime of a function."""
            start_time = time.time()
            func()
            end_time = time.time()
            elapsed_time = end_time - start_time
            return elapsed_time

        # Run the functions in the partial list
        for func_partial in partial_list:
            elapsed = time_function(func_partial)
            runtime_strings.append(
                f"{problem_name}"
                f",{solver_name}"
                f",{func_partial.func.__name__}"
                f",{elapsed:.2f}"
            )
    logging.info("Execution complete.")

    # Print the runtimes
    print()
    print("Runtimes")
    print("-" * 50)
    for runtime_string in runtime_strings:
        print(runtime_string)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Execute problem-solver experiments."
    )
    parser.add_argument(
        "--method",
        type=str,
        default="all",
        help="Methods to run (`run`, `post_replicate`, `post_normalize`, or `all`) [default: all]",
    )
    parser.add_argument(
        "--problems",
        type=str,
        default="all",
        help="Abbreviated names of problems to run (comma-separated or `all`) [default: all]",
    )
    parser.add_argument(
        "--solvers",
        type=str,
        default="all",
        help="Abbreviated names of solvers to run (comma-separated or `all`) [default: all]",
    )
    parser.add_argument(
        "--log",
        type=str,
        default="INFO",
        help="Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`) [default: INFO]",
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
        help="Number of postreplications [default: 100]",
    )

    args = parser.parse_args()

    logging.basicConfig(
        format="%(levelname)s: %(message)s",
        level=getattr(logging, args.log.upper(), logging.INFO),
    )

    methods = args.method.split(",")
    problems = args.problems.split(",")
    solvers = args.solvers.split(",")

    valid_pairs = generate_valid_pairs(problems, solvers)

    main(valid_pairs, methods, args.num_macroreps, args.num_postreps)
