Testing
=======

Preface
-------

The SimOpt testbed does not use testing in the traditional sense of verifying correctness of code.
Instead, testing is used to check experiment outputs against known results to ensure that changes to the code base do not inadvertently change results.
This means that even though the tests may fail, it does not necessarily indicate a problem with the code.
The tests are primarily intended for developers making changes to the code base.

Testing Framework
-----------------

The tests are written using the `unittest <https://docs.python.org/3/library/unittest.html>`_ framework, which is included in the Python standard library.

The tests are located in the `test` directory at the root of the repository.
Inside the `test` directory, there is a `expected_results` directory that contains the expected results for the tests.
Additionally, there are two test scripts of note: `experiment_test_core.py` and `test_experiments.py`.

- `test_experiments.py` is the main test script that is run to execute all tests.
  It dynamically creates test classes for each YAML file in the `expected_results` directory.
  Each test class inherits from a base class in `experiment_test_core.py` that contains the actual test methods.
  This design allows for easy addition of new test cases by simply adding a new YAML file to the `expected_results` directory.
- `experiment_test_core.py` contains the base class and mixin class for the tests.

  - The `ExperimentTest` class is responsible for the common setup logic needed to run an experiment.
  - The `ExperimentTestMixin` class contains the actual test methods that check the results. It's called a mixin because its sole purpose is to be "mixed in" with the base class to provide specific, reusable testing functionality without being a complete class on its own.

Running Tests
-------------

Setup
^^^^^

To run the tests, you will need to have the SimOpt development environment set up.
Please refer to the `Environment Setup <environment_setup.html>`_ guide for instructions on how to set up the development environment.

Execution
^^^^^^^^^

Executing the tests can be done in several ways:

1. **Using the command line**:

    - Open a terminal and navigate to the root directory of the SimOpt repository.
    - Activate the SimOpt conda environment if it is not already activated:

        - Windows (cmd): ``conda activate simopt``
        - Windows (PowerShell): ``conda activate simopt``
        - MacOS/Linux: ``conda activate simopt``

    - Run the tests using the following command: ``python -m unittest discover -s test -p "test_*.py"``

        - This command will discover and run all test files in the `test` directory that match the pattern `test_*.py`.
    
2. **Using Visual Studio Code**:

   Since VS Code's testing interface automatically discovers tests via the `unittest` framework, you can run the tests directly from the VS Code interface.

    - Open the SimOpt repository in VS Code.
    - Ensure that the SimOpt conda environment is selected as the interpreter.
    - Open the Testing sidebar by clicking on the beaker icon on the left sidebar.
    - Click on the "Run Tests" button (the double triangle icon) to execute all tests.

Generating Expected Results
---------------------------

The `generate_experiment_results.py` script in the `scripts` directory is used to generate the expected results for the tests.
It automatically checks for any missing `.yaml` files in the `expected_results` directory and generates them by running the corresponding experiments, skipping any experiments that are not compatible.

To run the script, use the following command:
``python scripts/generate_experiment_results.py``

Dealing with Test Failures
--------------------------

If a test fails, it indicates that the output of the experiment does not match the expected results.
While this could be due to a bug in the code, it could also be due to legitimate changes/improvements in the code that change the end results.

To determine the cause of the failure, you should:

    1. Review the changes you made to the code to see if they could have affected the results.
    2. If you don't believe the changes should have affected the results, roll back SimOpt to the last known good state (e.g., the last commit where all tests passed) and re-run the tests.
       If the tests do not pass, it is likely that a dependencies update or other external factor has caused the failure.
       If the tests do pass, you can incrementally re-apply your changes until you identify the specific change that caused the failure.
    3. If you believe the changes you made should have affected the results, or if you have determined that the failure is due to a dependencies update or other external factor, you can update the expected results by deleting the relevant `.yaml` files in the `expected_results` directory and re-running the `generate_experiment_results.py` script.
       Please ensure that you carefully review the changes in the expected results to confirm that they are indeed correct and expected.
