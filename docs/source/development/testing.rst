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

The tests are written with `pytest <https://docs.pytest.org/>`_ and are located in the ``tests`` directory at the root of the repository.
Inside it, the ``expected_results`` directory contains one ``*.pickle.zst`` file of expected results for each model and solver pair.

The main test is ``test_experiment`` in ``tests/test_experiments.py``.
It is parametrized over the pickle files, so every pair is its own test case, named after the file (for example ``CNTNEWS1_ADAM``).
Adding a new pickle to ``expected_results`` adds a new test case without any code changes.

Each test case runs the pair once with ``simopt.experiment.api.run``, using the number of macroreplications and postreplications stored in the pickle, and compares the result with the pickle stage by stage:

1. **Solver run**: the recommended solutions and the intermediate budgets.
2. **Post-replication**: the estimated objectives and, for problems with stochastic constraints, the estimated constraint values.
3. **Normalization**: ``x0``, the ``x0`` and ``xstar`` samples, the objective curves, the progress curves, and finally ``xstar``.

Failure messages name the stage, the problem, the solver and the macroreplication.
Because the pair is run end to end, an early difference usually makes the later stages fail as well, so look at the first failing stage.

Some solvers give different results on different machines.
Their pickles end in ``_darwin_arm64`` or ``_linux_x64`` and are only used on the matching platform.
If the pickle for the current platform is missing, the test case fails with a message saying so.

To run a single case, select it with ``-k``:

.. code-block:: bash

    python -m pytest tests/test_experiments.py -n 4 -k CNTNEWS1_ADAM

To create expected results for new pairs, run ``scripts/generate_experiment_results.py``.
It runs every compatible pair that does not have a pickle yet and writes the pickles to ``tests/expected_results``.
Existing pickles are never overwritten.
It accepts two options: ``--pair PROBLEM SOLVER`` generates only that pair and fails if the pair is unknown or incompatible, and ``--output-dir DIR`` writes the pickles to another directory.

Running Tests
-------------

Setup
^^^^^

To run the tests, you will need to have the SimOpt development environment set up.
Please refer to the `Environment Setup <environment_setup.html>`_ guide for instructions on how to set up the development environment.

Execution
^^^^^^^^^

Run the tests from the root directory of the SimOpt repository with ``uv``:

.. code-block:: bash

    MRG32K3A_BACKEND=rust uv run python -m pytest -n 4 tests/test_experiments.py

``MRG32K3A_BACKEND=rust`` selects the random-number generator backend for test runs.
To run a single case, add ``-k`` followed by the case name, for example ``-k CNTNEWS1_ADAM``.

Generating Expected Results
---------------------------

Expected results are generated with ``scripts/generate_experiment_results.py``, as described in `Testing Framework`_.
Run it from the repository root:

.. code-block:: bash

    uv run python scripts/generate_experiment_results.py
    uv run python scripts/generate_experiment_results.py --pair PROBLEM SOLVER

Dealing with Test Failures
--------------------------

If a test fails, it indicates that the output of the experiment does not match the expected results.
While this could be due to a bug in the code, it could also be due to legitimate changes/improvements in the code that change the end results.

To determine the cause of the failure, you should:

    1. Review the changes you made to the code to see if they could have affected the results.
    2. If you don't believe the changes should have affected the results, roll back SimOpt to the last known good state (e.g., the last commit where all tests passed) and re-run the tests.
       If the tests do not pass, it is likely that a dependencies update or other external factor has caused the failure.
       If the tests do pass, you can incrementally re-apply your changes until you identify the specific change that caused the failure.
    3. If you believe the changes you made should have affected the results, or if you have determined that the failure is due to a dependencies update or other external factor, you can update the expected results by deleting the relevant ``.pickle.zst`` files in the ``expected_results`` directory and re-running the ``generate_experiment_results.py`` script.
       Please ensure that you carefully review the changes in the expected results to confirm that they are indeed correct and expected.
