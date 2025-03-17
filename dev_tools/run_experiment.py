import logging
import os
import argparse

from simopt.directory import problem_directory, solver_directory
from simopt.experiment_base import ProblemSolver, post_normalize

# If running from /dev_tools, change to root directory
if os.path.basename(os.getcwd()) == "dev_tools":
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))


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
    should_run = "run" in methods or "all" in methods
    should_post_replicate = "post_replicate" in methods or "all" in methods
    should_post_normalize = "post_normalize" in methods or "all" in methods

    # For each problem/solver pair, run the experiment
    for solver_name, problem_name in valid_pairs:
        logging.info(f"Experimenting with {solver_name} on {problem_name}.")
        myexperiment = ProblemSolver(solver_name, problem_name)
        if should_run:
            logging.info("Executing `run` method.")
            myexperiment.run(n_macroreps=num_macroreps)
        if should_post_replicate:
            logging.info("Executing `post_replicate` method.")
            myexperiment.post_replicate(n_postreps=num_postreps)
        if should_post_normalize:
            logging.info("Executing `post_normalize` method.")
            post_normalize([myexperiment], n_postreps_init_opt=num_postreps)

    logging.info("Execution complete.")


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
