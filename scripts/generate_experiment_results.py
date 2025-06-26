"""Create test cases for all compatible problem-solver pairs."""

import sys
from pathlib import Path

import yaml

# Append the parent directory (simopt package) to the system path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import simopt.directory as directory
from simopt.experiment_base import ProblemSolver, post_normalize

# Workaround for AutoAPI
problem_directory = directory.problem_directory
solver_directory = directory.solver_directory

NUM_MACROREPS = 10
NUM_POSTREPS = 100

# Setup the SimOpt directory structure
HOME_DIR = Path(__file__).resolve().parent.parent
EXPECTED_RESULTS_DIR = HOME_DIR / "test" / "expected_results"


# Based off the similar function in simopt/experiment_base.py
def is_compatible(problem_name: str, solver_name: str) -> bool:
    """Check if a solver is compatible with a problem.

    Args:
        problem_name (str): Name of the problem.
        solver_name (str): Name of the solver.

    Returns:
        bool: Whether the solver is compatible with the problem.
    """
    # Create a ProblemSolver object
    myexperiment = ProblemSolver(solver_name, problem_name)
    # Check if the solver is compatible with the problem
    output = myexperiment.check_compatibility()
    return len(output) == 0


def create_test(problem_name: str, solver_name: str) -> None:
    """Create a test case for a problem and solver.

    Args:
        problem_name (str): Name of the problem.
        solver_name (str): Name of the solver.
    """
    # Run the experiment to get the expected results
    myexperiment = ProblemSolver(solver_name, problem_name)
    myexperiment.run(n_macroreps=NUM_MACROREPS)
    myexperiment.post_replicate(n_postreps=NUM_POSTREPS)
    post_normalize([myexperiment], n_postreps_init_opt=NUM_POSTREPS)

    # Loop through each curve object and convert it into a tuple
    # This is done to avoid packing custom classes into the YAML file
    for i in range(len(myexperiment.objective_curves)):
        myexperiment.objective_curves[i] = (  # type: ignore
            myexperiment.objective_curves[i].x_vals,
            myexperiment.objective_curves[i].y_vals,
        )
    for i in range(len(myexperiment.progress_curves)):
        myexperiment.progress_curves[i] = (  # type: ignore
            myexperiment.progress_curves[i].x_vals,
            myexperiment.progress_curves[i].y_vals,
        )

    results_dict = {
        "num_macroreps": NUM_MACROREPS,
        "num_postreps": NUM_POSTREPS,
        "problem_name": problem_name,
        "solver_name": solver_name,
        "all_recommended_xs": myexperiment.all_recommended_xs,
        "all_intermediate_budgets": myexperiment.all_intermediate_budgets,
        "all_est_objectives": myexperiment.all_est_objectives,
        "objective_curves": myexperiment.objective_curves,
        "progress_curves": myexperiment.progress_curves,
    }

    # Define the directory and output file
    file_problem_name = "".join(e for e in problem_name if e.isalnum())
    file_solver_name = "".join(e for e in solver_name if e.isalnum())
    results_filename = f"{file_problem_name}_{file_solver_name}.yaml"
    results_filepath = EXPECTED_RESULTS_DIR / results_filename
    # Write the results to the file
    with Path.open(results_filepath, "w") as f:
        yaml.dump(results_dict, f)


def main() -> None:
    """Create test cases for all compatible problem-solver pairs."""
    # Create a list of compatible problem-solver pairs
    compatible_pairs = [
        (problem_name, solver_name)
        for problem_name in problem_directory
        for solver_name in solver_directory
        if is_compatible(problem_name, solver_name)
    ]

    # Create the test directory if it doesn't exist
    # Create the expected directory if it doesn't exist
    Path.mkdir(EXPECTED_RESULTS_DIR, parents=True, exist_ok=True)
    existing_results = [path.name for path in EXPECTED_RESULTS_DIR.glob("*.yaml")]

    # Don't generate any tests for pairs that already have tests generated
    for pair in compatible_pairs:
        problem_name = pair[0]
        solver_name = pair[1]
        # Generate the expected filenames
        file_problem_name = "".join(e for e in problem_name if e.isalnum())
        file_solver_name = "".join(e for e in solver_name if e.isalnum())
        results_filename = f"{file_problem_name}_{file_solver_name}.yaml"
        # If file exists, skip it
        if results_filename in existing_results:
            print(f"Test for {pair} already exists")
            continue
        # If file doesn't exist, create it
        print(f"Creating test for {pair}")
        create_test(problem_name, solver_name)
    print("All tests created!")


if __name__ == "__main__":
    main()
