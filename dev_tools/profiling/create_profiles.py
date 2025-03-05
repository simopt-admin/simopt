import cProfile
import csv
import datetime
import logging
import os
import pstats
from contextlib import redirect_stdout

from simopt.directory import problem_directory, solver_directory
from simopt.experiment_base import ProblemSolver, post_normalize

os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))

# Constants for number of macro and post reps
NUM_MACROREPS: int = 20
NUM_POSTREPS: int = 100


# Function to check compatibility
def is_compatible(problem_name: str, solver_name: str) -> bool:
    myexperiment = ProblemSolver(solver_name, problem_name)
    output = myexperiment.check_compatibility()
    return len(output) == 0


# Function to profile a specific method and save results
def profile_method(
    experiment_dir: str, method_name: str, method: callable, filename: str
) -> None:
    pr = cProfile.Profile()
    pr.enable()
    method()
    pr.disable()

    ps = pstats.Stats(pr)
    ps.strip_dirs().sort_stats(pstats.SortKey.CUMULATIVE)

    # Save results to CSV
    file_path = os.path.join(experiment_dir, filename)
    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Function",
                "Calls",
                "Total Time",
                "Per Call",
                "Cumulative Time",
                "Per Call Cumulative",
            ]
        )
        for func, stats in ps.stats.items():
            writer.writerow(
                [
                    func,
                    stats[0],
                    stats[2],
                    stats[2] / stats[0] if stats[0] else 0,
                    stats[3],
                    stats[3] / stats[0] if stats[0] else 0,
                ]
            )

    print(f"Profiling complete for {method_name} and saved to {file_path}")


# Main loop to iterate through problem/solver pairs
def main() -> None:
    cwd = os.getcwd()
    experiment_dir = os.path.join(
        cwd,
        "experiments",
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "profiling",
    )
    created_directory = False

    with open(os.devnull, "w") as fnull, redirect_stdout(fnull):
        for problem_name in problem_directory:
            for solver_name in solver_directory:
                if is_compatible(problem_name, solver_name):
                    if not created_directory:
                        os.makedirs(experiment_dir, exist_ok=True)
                        created_directory = True

                    # Create and run experiment
                    myexperiment = ProblemSolver(solver_name, problem_name)

                    # Profile run
                    profile_method(
                        experiment_dir,
                        "run",
                        lambda exp=myexperiment: exp.run(
                            n_macroreps=NUM_MACROREPS
                        ),
                        f"profiling_{solver_name}_{problem_name}_run.csv",
                    )

                    # Profile post_replicate
                    profile_method(
                        experiment_dir,
                        "post_replicate",
                        lambda exp=myexperiment: exp.post_replicate(
                            n_postreps=NUM_POSTREPS
                        ),
                        f"profiling_{solver_name}_{problem_name}_post_replicate.csv",
                    )

                    # Profile post_normalize
                    profile_method(
                        experiment_dir,
                        "post_normalize",
                        lambda exp=myexperiment: post_normalize(
                            [exp], n_postreps_init_opt=NUM_POSTREPS
                        ),
                        f"profiling_{solver_name}_{problem_name}_post_normalize.csv",
                    )

    if created_directory:
        print(f"Profiling complete. Results saved in {experiment_dir}")
    else:
        print(
            "No compatible problem/solver pairs found. No profiling performed."
        )


if __name__ == "__main__":
    logging.basicConfig(format="%(levelname)s: %(message)s")
    logging.getLogger().setLevel(logging.INFO)
    main()
