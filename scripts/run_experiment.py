"""Script to run experiments from the command line."""

# Sample usage for running a single experiment with viztracer:
# `viztracer --min_duration 0.05ms utils/run_experiment.py --problems=CHESS-1 --num_macroreps=1` # noqa: E501

import argparse
import logging
import sys
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path

# Append the parent directory (simopt package) to the system path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import simopt.directory as directory
from simopt.experiment_base import ProblemSolver, post_normalize

# Workaround for AutoAPI
problem_directory = directory.problem_directory
solver_directory = directory.solver_directory


def load_arguments() -> tuple[str, list[str], list[str], str, int, int]:
    """Load command line arguments.

    Returns:
        tuple: A tuple containing the problems, solvers, method, log level,
               number of macroreplications, and number of postreplications.
    """
    parser = argparse.ArgumentParser(description="Execute problem-solver experiments.")
    parser.add_argument(
        "--method",
        type=str,
        default="all",
        help=(
            "Methods to run "
            "(`run`, `post_replicate`, `post_normalize`, or `all`) "
            "[default: all]"
        ),
    )
    parser.add_argument(
        "--problems",
        type=str,
        default="all",
        help=(
            "Abbreviated names of problems to run "
            "(comma-separated or `all`) "
            "[default: all]"
        ),
    )
    parser.add_argument(
        "--solvers",
        type=str,
        default="all",
        help=(
            "Abbreviated names of solvers to run "
            "(comma-separated or `all`) "
            "[default: all]"
        ),
    )
    parser.add_argument(
        "--log",
        type=str,
        default="INFO",
        help=(
            "Logging level "
            "(`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`) "
            "[default: INFO]"
        ),
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
    return (
        args.method,
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
    myexperiment = ProblemSolver(solver_name, problem_name)
    output = myexperiment.check_compatibility()
    return len(output) == 0


def gen_partial_funcs(
    experiments: list[ProblemSolver], method: str, n_macroreps: int, n_postreps: int
) -> list[tuple[str, str, partial]]:
    """Generate partial functions for the experiment.

    Args:
        experiments (list[ProblemSolver]): List of valid problem-solver pairs.
        method (str): Method to run.
        n_macroreps (int): Number of macroreplications.
        n_postreps (int): Number of postreplications.

    Returns:
        list[tuple[str, str, partial]]: List of tuples containing the solver name,
                                            problem name, and the partial function.
    """
    # Each append needs to call the function that runs before it, otherwise it won't be
    # properly configured. This could be avoided in the future if loading from a file
    # (similar to how the tests work).
    # TODO: Refactor this to avoid the need for a chain of partials.

    def append(part: partial, exp: ProblemSolver) -> None:
        """Append the partial function to the list with problem/solver names.

        Args:
            part (partial): The partial function to append.
            exp (ProblemSolver): The experiment object.
        """
        prob_name = exp.problem.class_name_abbr
        solv_name = exp.solver.class_name_abbr
        exp_partials.append((prob_name, solv_name, part))

    exp_partials = []

    methods = ["run", "post_replicate", "post_normalize", "all"]
    try:
        method_idx = methods.index(method)
    except ValueError as e:
        raise ValueError(f"Invalid method '{method}'. Choose from {methods}.") from e

    for exp in experiments:
        # We always need to run the exp
        append(partial(exp.run, n_macroreps=n_macroreps), exp)
        # Only replicate if the method is post_rep or later
        if method_idx >= 1:
            append(partial(exp.post_replicate, n_postreps=n_postreps), exp)
        # Only normalize if the method is post_norm or later
        if method_idx >= 2:
            append(partial(post_normalize, [exp], n_postreps_init_opt=n_postreps), exp)

    return exp_partials


def create_experiments(problems: list[str], solvers: list[str]) -> list[ProblemSolver]:
    """Generate valid problem-solver pairs.

    Args:
        problems (list[str]): List of problem names.
        solvers (list[str]): List of solver names.

    Returns:
        list[ProblemSolver]: List of valid ProblemSolver objects.
    """
    return [
        ProblemSolver(solver_name, problem_name)
        for problem_name in problem_directory
        if problems == ["all"] or problem_name in problems
        for solver_name in solver_directory
        if solvers == ["all"] or solver_name in solvers
        if is_compatible(problem_name, solver_name)
    ]


def time_function(func: Callable) -> float:
    """Decorator to print the runtime of a function.

    Args:
        func (Callable): The function to time.
    """
    start_time = time.time()
    func()
    end_time = time.time()
    return end_time - start_time


def run_partials(
    partial_funcs: list[tuple[str, str, partial]],
) -> list[tuple[str, str, str, float]]:
    """Run the partial functions and return their runtimes.

    Args:
        partial_funcs (list[tuple[str, str, partial]]): List of tuples containing
                                                         the solver name,
                                                         problem name, and the
                                                         partial function.

    Returns:
        list[tuple[str, str, str, float]]: List of tuples containing the
                                                  problem name, solver name,
                                                  function name, and elapsed time.
    """
    runtimes = []
    for problem_name, solver_name, part_func in partial_funcs:
        logging.info(f"Experimenting with {solver_name} on {problem_name}.")

        elapsed = time_function(part_func)
        logging.info(
            f"Elapsed time for {solver_name} on {problem_name}: {elapsed:.2f}s"
        )

        runtime_tuple = (
            problem_name,
            solver_name,
            part_func.func.__name__,
            elapsed,
        )
        runtimes.append(runtime_tuple)
    return runtimes


def print_runtimes(runtimes: list[tuple[str, str, str, float]]) -> None:
    """Print the runtimes of the experiments.

    Args:
        runtimes (list[tuple[str, str, str, float]]): List of tuples containing
                                                            the problem name,
                                                            solver name, function
                                                            name, and elapsed time.
    """
    # Extra blank line to make it clear where the results start
    print()
    # Print header row
    header_fields = [
        f"{'Problem Name':<20}",
        f"{'Solver Name':<20}",
        f"{'Function Name':<20}",
        f"{'Elapsed Time (s)':<20}",
    ]
    print(" │ ".join(header_fields))
    # Print the separator
    bars = ["─" * 20] * len(header_fields)
    print("─┼─".join(bars))
    # Print each runtime
    for problem_name, solver_name, func_name, elapsed in runtimes:
        fields = [
            f"{problem_name:<20}",
            f"{solver_name:<20}",
            f"{func_name:<20}",
            f"{elapsed:.2f}",
        ]
        print(" │ ".join(fields))


# Main loop to iterate through problem/solver pairs
def main() -> None:
    """Main function to run the experiment.

    Args:
        valid_pairs (list[tuple[str, str]]): List of valid problem-solver pairs.
        methods (list[str]): List of methods to run.
        num_macroreps (int): Number of macroreplications.
        num_postreps (int): Number of postreplications.
    """
    # Load the arguments
    method, problems, solvers, log_level, num_macroreps, num_postreps = load_arguments()

    # Set logging level
    logging.basicConfig(
        format="%(levelname)s: %(message)s",
        level=getattr(logging, log_level, logging.INFO),
    )

    valid_pairs = create_experiments(problems, solvers)
    if len(valid_pairs) == 0:
        logging.warning("No valid problem-solver pairs found.")
        return
    partial_funcs = gen_partial_funcs(valid_pairs, method, num_macroreps, num_postreps)
    runtimes = run_partials(partial_funcs)
    logging.info("Execution complete.")
    print_runtimes(runtimes)


if __name__ == "__main__":
    main()
