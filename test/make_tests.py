"""Create test cases for all compatible problem-solver pairs."""
# TO RUN FROM TOP DIRECTORY:
# python -m test.make_tests

import os

from types import SimpleNamespace
from simopt.directory import problem_directory, solver_directory
from simopt.experiment_base import ProblemSolver, post_normalize

NUM_MACROREPS = 24
NUM_POSTREPS = 200


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
    file_problem_name = "".join(e for e in problem_name if e.isalnum())
    file_solver_name = "".join(e for e in solver_name if e.isalnum())
    filename = "test_" + file_problem_name + "_" + file_solver_name + ".py"

    # Open the file template and read it
    cwd = os.getcwd()  # Get the current working directory
    test_dir = os.path.join(cwd, "test")
    template_filepath = os.path.join(test_dir, "template.py")
    with open(template_filepath, "rb") as f:
        template = f.read()

    to_pickle = SimpleNamespace()
    to_pickle.num_macroreps = NUM_MACROREPS
    to_pickle.num_postreps = NUM_POSTREPS
    to_pickle.problem_name = problem_name
    to_pickle.solver_name = solver_name
    to_pickle.all_recommended_xs = myexperiment.all_recommended_xs
    to_pickle.all_intermediate_budgets = myexperiment.all_intermediate_budgets
    to_pickle.all_est_objectives = myexperiment.all_est_objectives
    to_pickle.objective_curves = myexperiment.objective_curves
    to_pickle.progress_curves = myexperiment.progress_curves

    # Dump the expected results into a pickle file
    expected_dir = os.path.join(test_dir, "expected_results")
    pickle_filename = (
        "expected_" + file_problem_name + "_" + file_solver_name + ".pickle"
    )
    pickle_filepath = os.path.join(expected_dir, pickle_filename)
    with open(pickle_filepath, "wb") as f:
        import pickle

        pickle.dump(to_pickle, f)

    # Replace the placeholders in the template with the actual values
    problem_name_title = file_problem_name.title()
    solver_name_title = file_solver_name.title()
    class_name = "Test" + problem_name_title + solver_name_title
    # Replace the class name
    template = template.replace(
        b"TestProblemSolver",
        class_name.encode(),
    ) 
    # Replace the filename for the pickle file
    template = template.replace(
        b"{{FILE}}",
        pickle_filename.encode(),
    )

    # Write the new test into the new file
    with open(os.path.join(test_dir, filename), "wb") as f:
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

    # Setup the directory structure for the test cases
    cwd = os.getcwd()
    # Create the test directory if it doesn't exist
    test_directory = os.path.join(cwd, "test")
    os.makedirs(test_directory, exist_ok=True)
    test_files = os.listdir(test_directory)
    # Create the expected directory if it doesn't exist
    pickle_directory = os.path.join(test_directory, "expected_results")
    os.makedirs(pickle_directory, exist_ok=True)
    pickle_files = os.listdir(pickle_directory)

    # Don't generate any tests for pairs that already have tests generated
    for pair in compatible_pairs:
        problem_name = pair[0]
        solver_name = pair[1]
        file_problem_name = "".join(e for e in problem_name if e.isalnum())
        file_solver_name = "".join(e for e in solver_name if e.isalnum())

        filename_core = file_problem_name + "_" + file_solver_name
        test_filename = "test_" + filename_core + ".py"
        pickle_filename = "expected_" + filename_core + ".pickle"
        cwd = os.getcwd()
        if test_filename in os.listdir(
            test_directory
        ) and pickle_filename in os.listdir(pickle_directory):
            # Remove pairs and files that already have tests generated
            compatible_pairs.remove(pair)
            test_files.remove(test_filename)
            pickle_files.remove(pickle_filename)

    # Delete any files that have tests generated but no pickle files
    for test_filename in test_files:
        filepath = os.path.join(test_directory, test_filename)
        if os.path.isfile(filepath) and test_filename.startswith("test_"):
            os.remove(filepath)
    # Delete any pickle files that have no tests generated
    for pickle_filename in pickle_files:
        filepath = os.path.join(pickle_directory, pickle_filename)
        os.remove(filepath)

    # Create all the test cases that don't already exist
    for pair in compatible_pairs:
        problem_name = pair[0]
        solver_name = pair[1]
        create_test(problem_name, solver_name)


if __name__ == "__main__":
    main()
