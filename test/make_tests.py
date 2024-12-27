"""Create test cases for all compatible problem-solver pairs."""
# TO RUN FROM TOP DIRECTORY:
# python -m test.make_tests

import os

from simopt.directory import problem_directory, solver_directory
from simopt.experiment_base import ProblemSolver, post_normalize


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
    myexperiment.run(n_macroreps=24)
    myexperiment.post_replicate(n_postreps=200)
    post_normalize([myexperiment], n_postreps_init_opt=200)

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
    filepath = cwd + "/test/" + filename
    with open(cwd + "/test/template.py", "rb") as f:
        template = f.read()

    # Loop through the template and replace the placeholders with the actual values
    template = template.replace(b"{{PROBLEM_NAME}}", problem_name.encode())
    template = template.replace(b"{{SOLVER_NAME}}", solver_name.encode())
    template = template.replace(
        b"{{ALL_RECOMMENDED_XS}}", str(myexperiment.all_recommended_xs).encode()
    )
    template = template.replace(
        b"{{ALL_INTERMEDIATE_BUDGETS}}",
        str(myexperiment.all_intermediate_budgets).encode(),
    )
    template = template.replace(
        b"{{ALL_EST_OBJECTIVES}}", str(myexperiment.all_est_objectives).encode()
    )
    template = template.replace(
        b"{{OBJECTIVE_CURVES}}", str(myexperiment.objective_curves).encode()
    )
    template = template.replace(
        b"{{PROGRESS_CURVES}}", str(myexperiment.progress_curves).encode()
    )
    problem_name_title = problem_name.title()
    solver_name_title = solver_name.title()
    template = template.replace(
        b"TestProblemSolver",
        ("Test" + problem_name_title + solver_name_title).encode(),
    )  # Replace the class name

    # Write the new test into the new file
    with open(filepath, "xb") as f:
        f.write(template)


def main() -> None:
    """Create test cases for all compatible problem-solver pairs."""
    # Delete all files beginning with "test_" in the test directory
    cwd = os.getcwd()  # Get the current working directory
    test_directory = cwd + "/test"
    for filename in os.listdir(test_directory):
        if filename.startswith("test_"):
            os.remove(test_directory + "/" + filename)

    # Loop through all the problems and solvers
    for problem_name in problem_directory:
        for solver_name in solver_directory:
            # Check if the solver is compatible with the problem
            if is_compatible(problem_name, solver_name):
                # Create the test case
                create_test(problem_name, solver_name)
                print("Created test for", problem_name, solver_name)
            else:
                print(
                    "Skipping",
                    problem_name,
                    solver_name,
                    "as it is not compatible",
                )


if __name__ == "__main__":
    main()
