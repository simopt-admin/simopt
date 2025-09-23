Testing
=======

Preface
-------

The SimOpt testbed does not use testing in the traditional sense of verifying correctness of code.
Instead, testing is used to check experiment outputs against known results to ensure that changes to the code base do not inadvertently change results.
This means that even though the tests may fail, it does not necessarily indicate a problem with the code.
The tests are primarily intended for developers making changes to the code base.
Non-developers of SimOpt should not need to run the tests, but they are welcome to do so if they wish.

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
  The base class sets up the experiment, while the mixin class contains the test methods that check the results of the experiment against the expected results.

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
        - MacOS/Linux: ``source activate simopt`` or ``conda activate simopt``

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

Coming Soon