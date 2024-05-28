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
import types
import unittest
from test import run_template

from simopt.directory import problem_directory, solver_directory
from simopt.experiment_base import ProblemSolver, post_normalize


# Check compatibility of a solver with a problem
# Based off the similar function in simopt/experiment_base.py
def is_compatible(problem_name, solver_name):
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
def create_test(filename, problem_name, solver_name):
    # Create a results object
    results = types.SimpleNamespace()
    # Set the problem and solver name
    results.solver_name = solver_name
    results.problem_name = problem_name

    # Run the experiment to get the expected results
    myexperiment = ProblemSolver(solver_name, problem_name)
    myexperiment.run(n_macroreps=10)

    # Set the solutions and budgets
    results.all_recommended_xs = myexperiment.all_recommended_xs
    results.all_intermediate_budgets = myexperiment.all_intermediate_budgets

    # Check actual post-replication results against expected
    myexperiment.post_replicate(n_postreps=200)
    results.all_est_objectives = myexperiment.all_est_objectives

    # Check actual post-normalization results against expected
    post_normalize([myexperiment], n_postreps_init_opt=200)
    results.objective_curves = myexperiment.objective_curves
    results.progress_curves = myexperiment.progress_curves

    # Write the results to the file via pickle
    with open(filename, "xb") as f:
        pickle.dump(results, f, protocol=3)


# Create a test suite
# The suite will contain all of the tests for the problems and solvers
# Missing tests are automatically created
def suite():
    # Create the sample test suite
    suite = unittest.TestSuite()

    # Get the list of problems and solvers to skip
    unchanged_files = getUnchangedClasses()

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
            # Check to see if the problem and solver are in the unchanged files
            if (
                problem_class_name in unchanged_files
                and solver_class_name in unchanged_files
            ):
                num_unchanged += 1
                continue

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
    # Get the hashes for the base, directory, and experiment_base files
    with open(cwd + r"\simopt\base.py", "rb") as f:
        hash_dict["base.py"] = hashlib.sha512(f.read()).hexdigest()
    with open(cwd + r"\simopt\directory.py", "rb") as f:
        hash_dict["directory.py"] = hashlib.sha512(f.read()).hexdigest()
    with open(cwd + r"\simopt\experiment_base.py", "rb") as f:
        hash_dict["experiment_base.py"] = hashlib.sha512(f.read()).hexdigest()
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


def getUnchangedClasses():
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
    hash_dict = getHashes()

    # If any of the base files have changed, return an empty hash dict
    # This means everything needs retested
    if hash_dict["base.py"] != expected_hashes["base.py"]:
        print("base.py updated, retesting all files")
        hash_dict = {}
        return hash_dict
    del hash_dict["base.py"]
    del expected_hashes["base.py"]

    if hash_dict["directory.py"] != expected_hashes["directory.py"]:
        print("directory.py updated, retesting all files")
        hash_dict = {}
        return hash_dict
    del hash_dict["directory.py"]
    del expected_hashes["directory.py"]

    if hash_dict["experiment_base.py"] != expected_hashes["experiment_base.py"]:
        print("experiment_base.py updated, retesting all files")
        hash_dict = {}
        return hash_dict
    del hash_dict["experiment_base.py"]
    del expected_hashes["experiment_base.py"]

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

    # Convert each to classes
    unchanged_files = []
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


def setUnchangedFiles():
    # Get the current working directory
    cwd = os.getcwd()
    # Delete the hash list if it exists
    if os.path.isfile(cwd + r"\test\hash_dict"):
        os.remove(cwd + r"\test\hash_dict")

    # Dump to the hash list
    hash_dict = getHashes()
    pickle.dump(hash_dict, open(cwd + r"\test\hash_dict", "wb"), protocol=3)


if __name__ == "__main__":
    # Run the test suite
    runner = unittest.TextTestRunner()
    if runner.run(suite()).wasSuccessful():
        setUnchangedFiles()
