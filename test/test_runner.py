# TO RUN FROM TOP DIRECTORY:
# python -m test.test_runner

# SHA512 is used to hash the files and avoid retesting unchanged files
# Changes to core files (base.py, directory.py, experiment_base.py) will cause
# all files to be retested
# Otherwise, only files that have changed will be retested

import hashlib
import inspect
import os
import pickle
import sys
import unittest
from test import run_template

from simopt.directory import problem_directory, solver_directory
from simopt.experiment_base import ProblemSolver, post_normalize

file_list = [
    "base.py",
    "directory.py",
    "experiment_base.py",
    "test/test_runner.py",
    "test/run_template.py",
]

# Check compatibility of a solver with a problem
# Based off the similar function in simopt/experiment_base.py
def is_compatible(problem_name: str, solver_name: str) -> bool:
    # Get the problem and solver
    problem = problem_directory[problem_name]()
    solver = solver_directory[solver_name]()

    # Check number of objectives.
    if solver.objective_type == "single" and problem.n_objectives > 1:
        return False
    if solver.objective_type == "multi" and problem.n_objectives == 1:
        return False
    # Check constraint types.
    constraint_types = ["unconstrained", "box", "deterministic", "stochastic"]
    if constraint_types.index(solver.constraint_type) < constraint_types.index(
        problem.constraint_type
    ):
        return False
    # Check variable types.
    if solver.variable_type == "discrete" and problem.variable_type != "discrete":
        return False
    if solver.variable_type == "continuous" and problem.variable_type != "continuous":
        return False
    # Check for existence of gradient estimates.
    if solver.gradient_needed and not problem.gradient_available:
        return False
    return True


# Create a test case for a problem and solver
def create_test(filename: str, problem_name: str, solver_name: str) -> None:
    # Run the experiment to get the expected results
    myexperiment = ProblemSolver(solver_name, problem_name)
    myexperiment.run(n_macroreps=10)
    myexperiment.post_replicate(n_postreps=200)
    post_normalize([myexperiment], n_postreps_init_opt=200)

    # Check if the file exists
    if os.path.isfile(filename):
        os.remove(filename)

    # Loop through each curve object and convert it into a tuple
    # This is done to avoid pickling issues
    for i in range(len(myexperiment.objective_curves)):
        myexperiment.objective_curves[i] = (myexperiment.objective_curves[i].x_vals, myexperiment.objective_curves[i].y_vals)
    for i in range(len(myexperiment.progress_curves)):
        myexperiment.progress_curves[i] = (myexperiment.progress_curves[i].x_vals, myexperiment.progress_curves[i].y_vals)

    # Put everything we want to write into a dictionary
    results: dict[str, ] = {"problem_name": problem_name,
        "solver_name": solver_name,
        "all_recommended_xs": myexperiment.all_recommended_xs,
        "all_intermediate_budgets": myexperiment.all_intermediate_budgets,
        "all_est_objectives": myexperiment.all_est_objectives,
        "objective_curves": myexperiment.objective_curves,
        "progress_curves": myexperiment.progress_curves
        }

    # Write the results to the file via pickle
    with open(filename, "xb") as f:
        pickle.dump(results, f, protocol=4)


# Create a test suite
# The suite will contain all of the tests for the problems and solvers
# Missing tests are automatically created
def suite(run_all: bool = False) -> unittest.TestSuite:
    # Create the sample test suite
    suite = unittest.TestSuite()

    # Get the list of problems and solvers to skip
    unchanged_files = getUnchangedClasses()
    if (len(unchanged_files) == 0):
        print("Retesting all files...")

    num_incompatible = 0
    num_unchanged = 0

    # Loop through all of the cases
    for problem_name in problem_directory:
        # Strip non-alphanumeric characters for the filename
        problem_filename = "".join(e for e in problem_name if e.isalnum())
        for solver_name in solver_directory:
            # If they aren't compatible, skip
            if not is_compatible(problem_name, solver_name):
                num_incompatible += 1
                continue

            # Lookup the problem class and solver class
            problem_class = problem_directory[problem_name]
            problem_class_name = problem_class.__name__
            solver_class = solver_directory[solver_name]
            solver_class_name = solver_class.__name__
            # Strip non-alphanumeric characters for the filename
            solver_filename = "".join(e for e in solver_name if e.isalnum())
            # Get current working directory
            cwd = os.getcwd()
            # Check to see if the test exists in the tests directory
            filename = (
                cwd
                + r"\test\expected_data\results_"
                + problem_filename
                + "_"
                + solver_filename
            )

            # If the test doesn't exist, create it
            if not os.path.isfile(filename):
                print("Creating test for " + problem_name + " and " + solver_name)
                create_test(filename, problem_name, solver_name)

            # Check to see if the problem and solver are in the unchanged files
            if (
                run_all is False
                and problem_class_name in unchanged_files
                and solver_class_name in unchanged_files
            ):
                num_unchanged += 1
                continue

            # Use the run_template to create the test
            test = run_template.run_template(solver_name, problem_name)
            # print("Adding test for " + problem_name + " and " + solver_name)
            suite.addTest(test)

    print("Number of skipped (incompatible) combos: ", num_incompatible)
    print("Number of skipped (unchanged) combos: ", num_unchanged)
    print("Number of tests: ", suite.countTestCases())
    return suite


def getHashes():
    # Get the current working directory
    cwd = os.getcwd()
    # Create a new hash dictionary
    hash_dict = {}
    # Get the hashes for any files in the file list
    for file in file_list:
        if not os.path.isfile(cwd + r"\simopt\\" + file):
            continue
        with open(cwd + r"\simopt\\" + file, "rb") as f:
            hash_dict[file] = hashlib.sha512(f.read()).hexdigest()
    # Get the list of files in the simopt/models directory
    files = os.listdir(cwd + r"\simopt\models")
    # Loop through the files
    for file in files:
        # If it's a folder, not a .py file, or the __init__ file, skip
        if (
            os.path.isdir(cwd + r"\simopt\models\\" + file)
            or not file.endswith(".py")
            or file == "__init__.py"
        ):
            continue
        # Get the hash of the file
        with open(cwd + r"\simopt\models\\" + file, "rb") as f:
            hash_dict["models/" + file] = hashlib.sha512(f.read()).hexdigest()
    # Get the list of files in the simopt/solvers directory
    files = os.listdir(cwd + r"\simopt\solvers")
    # Loop through the files
    for file in files:
        # If it's a folder, not a .py file, or the __init__ file, skip
        if (
            os.path.isdir(cwd + r"\simopt\solvers\\" + file)
            or not file.endswith(".py")
            or file == "__init__.py"
        ):
            continue
        # Get the hash of the file
        with open(cwd + r"\simopt\solvers\\" + file, "rb") as f:
            hash_dict["solvers/" + file] = hashlib.sha512(f.read()).hexdigest()
    # Return the hash list
    return hash_dict


def getUnchangedClasses() -> list:
    # Get the current working directory
    cwd = os.getcwd()
    # If the hash dict doesn't exist, return an empty hash dict
    if not (os.path.isfile(cwd + r"\test\hash_dict")):
        hash_dict = {}
        return hash_dict
    # Load the hash list
    with open(cwd + r"\test\hash_dict", "rb") as f:
        expected_hashes = pickle.load(f)

    # Get the current hash list
    hash_dict: dict[str, str] = getHashes()

    return_empty: bool = False

    # If any of the base files have changed, return an empty hash dict
    # This means everything needs retested
    for file in file_list:
        if file not in hash_dict:
            print("Missing hash for file: ", file)
            return_empty = True
        elif hash_dict[file] != expected_hashes[file]:
            print(file, " updated, retesting all files")
            return_empty = True
        if file in hash_dict:
            del hash_dict[file]
        if file in expected_hashes:
            del expected_hashes[file]

    # Loop through what's on the system
    for file in hash_dict:
        # If the file isn't in the expected hashes, skip
        # This just means that the file is new and hasn't been tested
        if file not in expected_hashes:
            print("Added file: ", file)
            continue
        # If the hash is different, remove it from the expected_hashes
        if hash_dict[file] != expected_hashes[file]:
            print("Changed File: ", file)
            del expected_hashes[file]
    # Remove any files that are in the expected_hashes but not in the system
    for file in list(expected_hashes.keys()):
        if file not in hash_dict:
            print("Removed file: ", file)
            del expected_hashes[file]
        # else:
        #     print("Unchanged file: ", file)
    # At this point, the only files left are ones that were in the expected
    # hashes as well as the system hashes and have the same hash

    if (return_empty):
        return []

    # Convert each to classes
    unchanged_files: list = []
    for file in expected_hashes:
        # Get the class name
        class_name = file.split("/")[-1].split(".")[0]
        # Get the class
        try:
            for name, cls in inspect.getmembers(
                sys.modules["simopt.models." + class_name], inspect.isclass
            ):
                unchanged_files.append(cls.__name__)
        except KeyError:
            for name, cls in inspect.getmembers(
                sys.modules["simopt.solvers." + class_name], inspect.isclass
            ):
                unchanged_files.append(cls.__name__)

    return unchanged_files


def saveHashes() -> None:
    # Get the current working directory
    cwd = os.getcwd()
    # Delete the hash list if it exists
    if os.path.isfile(cwd + r"\test\hash_dict"):
        os.remove(cwd + r"\test\hash_dict")

    # Dump to the hash list
    hash_dict = getHashes()
    pickle.dump(hash_dict, open(cwd + r"\test\hash_dict", "wb"), protocol=4)


def inputCheck() -> unittest.TestSuite:
    # Check if the user wants help
    if (
        len(sys.argv) > 1
        and sys.argv[1] == "help"
        or len(sys.argv) > 1
        and sys.argv[1] == "-h"
        or len(sys.argv) > 1
        and sys.argv[1] == "--help"
    ):
        print("To reset the expected values, run:")
        print("\tpython -m test.test_runner reset_expected")
        print("")
        print("To run all tests, run:")
        print("\tpython -m test.test_runner run_all")
        print("")
        print("To only run tests that have changed, run:")
        print("\tpython -m test.test_runner")
        print("")
        sys.exit()

    # Check if the user wants to recreate the tests
    # This resets the expected values
    if len(sys.argv) > 1 and sys.argv[1] == "reset_expected":
        # Delete everything in the expected_data directory
        cwd = os.getcwd()
        files = os.listdir(cwd + r"\test\expected_data")
        for file in files:
            os.remove(cwd + r"\test\expected_data\\" + file)

        return suite(run_all=True)
    # Check if the user wants to run all the tests
    # This does not change the expected values
    elif len(sys.argv) > 1 and sys.argv[1] == "run_all":
        return suite(run_all=True)
    elif len(sys.argv) > 1:
        print("Invalid command")
        sys.exit()
    else:
        return suite()


if __name__ == "__main__":
    # Check to see if the user put in a help command
    test_suite = inputCheck()

    # Run the test suite
    runner = unittest.TextTestRunner()
    if runner.run(test_suite).wasSuccessful():
        saveHashes()
