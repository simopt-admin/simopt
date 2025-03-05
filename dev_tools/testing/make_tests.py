"""Create test cases for all compatible problem-solver pairs."""
# TO RUN FROM TOP DIRECTORY:
# python -m test.make_tests

import yaml
import os

from simopt.directory import problem_directory, solver_directory
from simopt.experiment_base import ProblemSolver, post_normalize

NUM_MACROREPS = 10
NUM_POSTREPS = 100

# Constants for the template file
TEMPLATE_NAME = "testing_template.py"
TEMPLATE_DIR = os.path.join(os.getcwd(), "dev_tools", "testing")
TEMPLATE_FILEPATH = os.path.join(TEMPLATE_DIR, TEMPLATE_NAME)

# Constants for the test directory
TEST_DIR = os.path.join(os.getcwd(), "test")
EXPECTED_RESULTS_DIR = os.path.join(TEST_DIR, "expected_results")


# Check compatibility of a solver with a problem
# Based off the similar function in simopt/experiment_base.py
def is_compatible(problem_name: str, solver_name: str) -> bool:
    """Check if a solver is compatible with a problem.

    Parameters
    ----------
    problem_name : str
        Name of the problem.
    solver_name : str
        Name of the solver.

    Returns
    -------
    bool
        True if the solver is compatible with the problem, False otherwise.

    """
    # Create a ProblemSolver object
    myexperiment = ProblemSolver(solver_name, problem_name)
    # Check if the solver is compatible with the problem
    output = myexperiment.check_compatibility()
    return len(output) == 0


# Create a test case for a problem and solver
def create_test(problem_name: str, solver_name: str) -> None:
    """Create a test case for a problem and solver.

    Parameters
    ----------
    problem_name : str
        Name of the problem.
    solver_name : str
        Name of the solver.

    """
    # Setup the names
    file_problem_name = "".join(e for e in problem_name if e.isalnum())
    file_solver_name = "".join(e for e in solver_name if e.isalnum())

    filename_core = file_problem_name + "_" + file_solver_name

    # Run the experiment to get the expected results
    myexperiment = ProblemSolver(solver_name, problem_name)
    myexperiment.run(n_macroreps=NUM_MACROREPS)
    myexperiment.post_replicate(n_postreps=NUM_POSTREPS)
    post_normalize([myexperiment], n_postreps_init_opt=NUM_POSTREPS)

    # Loop through each curve object and convert it into a tuple
    # This is done to avoid pickling issues
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

    # Strip any non-alphanumeric characters from the problem and solver names
    filename = "test_" + filename_core.lower() + ".py"

    # Open the file template and read it
    with open(TEMPLATE_FILEPATH, "rb") as f:
        template = f.read()

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
    results_filename = filename_core + ".yaml"
    results_filepath = os.path.join(EXPECTED_RESULTS_DIR, results_filename)
    # Write the results to the file
    with open(results_filepath, "w") as f:
        yaml.dump(results_dict, f)

    # Replace the placeholders in the template with the actual values
    class_name = "Test" + file_problem_name.title() + file_solver_name.title()
    # Replace the class name
    template = template.replace(
        b"TestProblemSolver",
        class_name.encode(),
    )
    # Replace the filename for the results file
    template = template.replace(
        b"{{FILE}}",
        results_filename.encode(),
    )

    # Write the new test into the new file
    with open(os.path.join(TEST_DIR, filename), "wb") as f:
        f.write(template)


def main() -> None:
    """Create test cases for all compatible problem-solver pairs."""
    # Create a list of compatible problem-solver pairs
    compatible_pairs = []
    for problem_name in problem_directory:
        for solver_name in solver_directory:
            if is_compatible(problem_name, solver_name):
                pair = (problem_name, solver_name)
                compatible_pairs.append(pair)

    # Create the test directory if it doesn't exist
    os.makedirs(TEST_DIR, exist_ok=True)
    existing_test_files = os.listdir(TEST_DIR)
    # Create the expected directory if it doesn't exist
    os.makedirs(EXPECTED_RESULTS_DIR, exist_ok=True)
    existing_result_files = os.listdir(EXPECTED_RESULTS_DIR)

    # Don't generate any tests for pairs that already have tests generated
    for pair in compatible_pairs:
        problem_name = pair[0]
        solver_name = pair[1]
        # Generate the expected filenames
        file_problem_name = "".join(e for e in problem_name if e.isalnum())
        file_solver_name = "".join(e for e in solver_name if e.isalnum())
        filename_core = file_problem_name + "_" + file_solver_name
        test_filename = "test_" + filename_core.lower() + ".py"
        results_filename = filename_core + ".yaml"
        # Check if the files exist or if the test needs created
        test_exists = test_filename in existing_test_files
        results_exist = results_filename in existing_result_files
        if test_exists and results_exist:
            print("Test already exists for", pair)
        else:
            print("Creating test for", pair)
            create_test(problem_name, solver_name)
        # These files exist, so we don't need to delete them later
        if test_exists:
            existing_test_files.remove(test_filename)
        if results_exist:
            existing_result_files.remove(results_filename)

    # Remove any tests that are no longer needed
    for test_file in existing_test_files:
        if test_file.startswith("test_") and test_file.endswith(".py"):
            path = os.path.join(TEST_DIR, test_file)
            print(f"Removing unneeded test file: {path}")
            os.remove(path)
    for result_file in existing_result_files:
        if result_file.endswith(".yaml"):
            path = os.path.join(EXPECTED_RESULTS_DIR, result_file)
            print(f"Removing unneeded result file: {path}")
            os.remove(path)


if __name__ == "__main__":
    main()
