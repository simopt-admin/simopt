"""Experiment test module.

This file is split up into two sections:
1. The ExperimentTest class acts as a base class for each test case.
   It inherits from unittest.TestCase and sets up the experiment.
2. The ExperimentTestMixin class contains the test methods that check
   the results of the experiment against the expected results.

The reason these are separate is to prevent the test cases from being
executed without first being setup. When the test cases were in the same
class as the setup, they would be executed whenever they were discovered
by unittest, even if the setup had not been run. This caused problems as
the setup was not run and the test cases would fail.
"""

import math
import unittest
from abc import abstractmethod
from pathlib import Path
from typing import Any

import yaml

from simopt.experiment_base import ProblemSolver, post_normalize


class ExperimentTest(unittest.TestCase):
    """Base class for experiment tests.

    This class sets up the experiment and checks that the solver and
    problem names match the expected values. It also loads the expected
    results from a YAML file in the test/expected_results directory
    (specified by the file attribute in the test class).
    """

    @property
    @abstractmethod
    def filepath(self) -> Path:
        """Return the name of the experiment file."""
        error_msg = "The file name must be set in the test class."
        raise NotImplementedError(error_msg)

    def setUp(self) -> None:
        # Set the name of the experiment file
        with open(self.filepath, "rb") as f:
            expected_results = yaml.load(f, Loader=yaml.Loader)

        self.num_macroreps = expected_results["num_macroreps"]
        self.num_postreps = expected_results["num_postreps"]
        self.expected_problem_name = expected_results["problem_name"]
        self.expected_solver_name = expected_results["solver_name"]
        self.expected_all_recommended_xs = expected_results[
            "all_recommended_xs"
        ]
        self.expected_all_intermediate_budgets = expected_results[
            "all_intermediate_budgets"
        ]
        self.expected_all_est_objectives = expected_results[
            "all_est_objectives"
        ]
        self.expected_objective_curves = expected_results["objective_curves"]
        self.expected_progress_curves = expected_results["progress_curves"]

        # Get rid of it to save memory
        del expected_results

        # Setup the solver and experiment
        self.myexperiment = ProblemSolver(
            self.expected_solver_name, self.expected_problem_name
        )
        self.assertEqual(
            self.myexperiment.solver.name,
            self.expected_solver_name,
            "Solver name does not match (expected: "
            + self.expected_solver_name
            + ", actual: "
            + self.myexperiment.solver.name
            + ")",
        )
        self.assertEqual(
            self.myexperiment.problem.name,
            self.expected_problem_name,
            "Problem name does not match (expected: "
            + self.expected_problem_name
            + ", actual: "
            + self.myexperiment.problem.name
            + ")",
        )


class ExperimentTestMixin:
    """Mixin class for experiment tests.

    This class contains the test methods that check the results of the
    experiment against the expected results. It expects that any classes
    that inherit from it will have already set up the experiment in the
    setUp method of the ExperimentTest class.
    """

    def test_run(self: Any) -> None:
        # Check actual run results against expected
        self.myexperiment.run(n_macroreps=self.num_macroreps)
        self.assertEqual(
            self.myexperiment.n_macroreps,
            self.num_macroreps,
            "Number of macro-replications for problem "
            + self.expected_problem_name
            + " and solver "
            + self.expected_solver_name
            + " does not match.",
        )
        # For each macroreplication
        for mrep in range(self.num_macroreps):
            # Check to make sure the list lengths are the same
            self.assertEqual(
                len(self.myexperiment.all_recommended_xs[mrep]),
                len(self.expected_all_recommended_xs[mrep]),
                "Length of recommended solutions for problem "
                + self.expected_problem_name
                + " and solver "
                + self.expected_solver_name
                + " do not match.",
            )
            # For each list of recommended solutions
            for sol_list_idx in range(
                len(self.myexperiment.all_recommended_xs[mrep])
            ):
                # Check to make sure the tuples are the same length
                self.assertEqual(
                    len(
                        self.myexperiment.all_recommended_xs[mrep][sol_list_idx]
                    ),
                    len(self.expected_all_recommended_xs[mrep][sol_list_idx]),
                    "Recommended solutions for problem "
                    + self.expected_problem_name
                    + " and solver "
                    + self.expected_solver_name
                    + " do not match at mrep "
                    + str(mrep)
                    + " and index "
                    + str(sol_list_idx)
                    + ".",
                )
                # For each tuple of recommended solutions
                for sol_idx in range(
                    len(
                        self.myexperiment.all_recommended_xs[mrep][sol_list_idx]
                    )
                ):
                    self.assertAlmostEqual(
                        self.myexperiment.all_recommended_xs[mrep][
                            sol_list_idx
                        ][sol_idx],
                        self.expected_all_recommended_xs[mrep][sol_list_idx][
                            sol_idx
                        ],
                        5,
                        "Recommended solutions for problem "
                        + self.expected_problem_name
                        + " and solver "
                        + self.expected_solver_name
                        + " do not match at mrep "
                        + str(mrep)
                        + " and index "
                        + str(sol_list_idx)
                        + " and tuple "
                        + str(sol_idx)
                        + ".",
                    )
            # Check to make sure the list lengths are the same
            self.assertEqual(
                len(self.myexperiment.all_intermediate_budgets[mrep]),
                len(self.expected_all_intermediate_budgets[mrep]),
                "Length of intermediate budgets for problem "
                + self.expected_problem_name
                + " and solver "
                + self.expected_solver_name
                + " do not match.",
            )
            # For each list of intermediate budgets
            for sol_list_idx in range(
                len(self.myexperiment.all_intermediate_budgets[mrep])
            ):
                # Check the values in the list
                self.assertAlmostEqual(
                    self.myexperiment.all_intermediate_budgets[mrep][
                        sol_list_idx
                    ],
                    self.expected_all_intermediate_budgets[mrep][sol_list_idx],
                    5,
                    "Intermediate budgets for problem "
                    + self.expected_problem_name
                    + " and solver "
                    + self.expected_solver_name
                    + " do not match at mrep "
                    + str(mrep)
                    + " and index "
                    + str(sol_list_idx)
                    + ".",
                )

    def test_post_replicate(self: Any) -> None:
        # Simulate results from the run method
        self.myexperiment = ProblemSolver(
            self.expected_solver_name, self.expected_problem_name
        )
        self.myexperiment.n_macroreps = self.num_macroreps
        self.myexperiment.all_recommended_xs = self.expected_all_recommended_xs
        self.myexperiment.all_intermediate_budgets = (
            self.expected_all_intermediate_budgets
        )

        # Check actual post-replication results against expected
        self.myexperiment.post_replicate(n_postreps=self.num_postreps)
        self.assertEqual(
            self.myexperiment.n_postreps,
            self.num_postreps,
            "Number of post-replications for problem "
            + self.expected_problem_name
            + " and solver "
            + self.expected_solver_name
            + " does not match.",
        )
        # For each macroreplication
        for mrep in range(self.num_macroreps):
            # Check to make sure the list lengths are the same
            self.assertEqual(
                len(self.myexperiment.all_est_objectives[mrep]),
                len(self.expected_all_est_objectives[mrep]),
                "Estimated objectives for problem "
                + self.expected_problem_name
                + " and solver "
                + self.expected_solver_name
                + " do not match.",
            )
            # For each list in the estimated objectives
            for objective_idx in range(
                len(self.myexperiment.all_est_objectives[mrep])
            ):
                # Check the values in the list
                self.assertAlmostEqual(
                    self.myexperiment.all_est_objectives[mrep][objective_idx],
                    self.expected_all_est_objectives[mrep][objective_idx],
                    5,
                    "Estimated objectives for problem "
                    + self.expected_problem_name
                    + " and solver "
                    + self.expected_solver_name
                    + " do not match at mrep "
                    + str(mrep)
                    + " and index "
                    + str(objective_idx)
                    + ".",
                )

    def test_post_normalize(self: Any) -> None:
        # Simulate results from the post_replicate method
        self.myexperiment = ProblemSolver(
            self.expected_solver_name, self.expected_problem_name
        )
        self.myexperiment.n_macroreps = self.num_macroreps
        self.myexperiment.n_postreps = self.num_postreps
        self.myexperiment.all_recommended_xs = self.expected_all_recommended_xs
        self.myexperiment.all_intermediate_budgets = (
            self.expected_all_intermediate_budgets
        )
        self.myexperiment.all_est_objectives = self.expected_all_est_objectives
        self.myexperiment.has_run = True
        self.myexperiment.has_postreplicated = True
        # Check actual post-normalization results against expected
        post_normalize(
            [self.myexperiment], n_postreps_init_opt=self.num_postreps
        )

        objective_curves = []
        progress_curves = []

        # Loop through each curve object and convert it into a tuple
        for i in range(len(self.myexperiment.objective_curves)):
            objective_curves.append(
                [
                    self.myexperiment.objective_curves[i].x_vals,
                    self.myexperiment.objective_curves[i].y_vals,
                ]
            )
        for i in range(len(self.myexperiment.progress_curves)):
            progress_curves.append(
                [
                    self.myexperiment.progress_curves[i].x_vals,
                    self.myexperiment.progress_curves[i].y_vals,
                ]
            )

        for mrep in range(self.num_macroreps):
            # Check to make sure the same number of objective curves are present
            # This should probably always be 2 (x and y)
            self.assertEqual(
                len(objective_curves[mrep]),
                len(self.expected_objective_curves[mrep]),
                "Number of objective curves for problem "
                + self.expected_problem_name
                + " and solver "
                + self.expected_solver_name
                + " does not match.",
            )
            # Make sure that curves are only checked if they exist
            if len(objective_curves[mrep]) > 0:
                # Make sure the lengths of the X and Y values are the same
                self.assertEqual(
                    len(objective_curves[mrep][0]),
                    len(self.expected_objective_curves[mrep][0]),
                    "Length of X values for problem "
                    + self.expected_problem_name
                    + " and solver "
                    + self.expected_solver_name
                    + " do not match.",
                )
                self.assertEqual(
                    len(objective_curves[mrep][1]),
                    len(self.expected_objective_curves[mrep][1]),
                    "Length of Y values for problem "
                    + self.expected_problem_name
                    + " and solver "
                    + self.expected_solver_name
                    + " do not match.",
                )
                # Check X (0) and Y (1) values
                for x_index in range(len(objective_curves[mrep][0])):
                    # If the value is NaN, make sure we're expecting NaN
                    if math.isnan(objective_curves[mrep][0][x_index]):
                        self.assertTrue(
                            math.isnan(
                                self.expected_objective_curves[mrep][0][x_index]
                            ),
                            "X values for problem "
                            + self.expected_problem_name
                            + " and solver "
                            + self.expected_solver_name
                            + " do not match at mrep "
                            + str(mrep)
                            + " and index "
                            + str(x_index)
                            + ".",
                        )
                    # Otherwise, check the value normally
                    else:
                        self.assertAlmostEqual(
                            objective_curves[mrep][0][x_index],
                            self.expected_objective_curves[mrep][0][x_index],
                            5,
                            "X values for problem "
                            + self.expected_problem_name
                            + " and solver "
                            + self.expected_solver_name
                            + " do not match at mrep "
                            + str(mrep)
                            + " and index "
                            + str(x_index)
                            + ".",
                        )
                for y_index in range(len(objective_curves[mrep][1])):
                    # If the value is NaN, make sure we're expecting NaN
                    if math.isnan(objective_curves[mrep][1][y_index]):
                        self.assertTrue(
                            math.isnan(
                                self.expected_objective_curves[mrep][1][y_index]
                            ),
                            "Y values for problem "
                            + self.expected_problem_name
                            + " and solver "
                            + self.expected_solver_name
                            + " do not match at mrep "
                            + str(mrep)
                            + " and index "
                            + str(y_index)
                            + ".",
                        )
                    # Otherwise, check the value normally
                    else:
                        self.assertAlmostEqual(
                            objective_curves[mrep][1][y_index],
                            self.expected_objective_curves[mrep][1][y_index],
                            5,
                            "Y values for problem "
                            + self.expected_problem_name
                            + " and solver "
                            + self.expected_solver_name
                            + " do not match at mrep "
                            + str(mrep)
                            + " and index "
                            + str(y_index)
                            + ".",
                        )

            # Check to make sure the same number of progress curves are present
            # This should probably always be 2 (x and y)
            self.assertEqual(
                len(progress_curves[mrep]),
                len(self.expected_progress_curves[mrep]),
                "Number of progress curves for problem "
                + self.expected_problem_name
                + " and solver "
                + self.expected_solver_name
                + " does not match.",
            )
            # Make sure that curves are only checked if they exist
            if len(progress_curves[mrep]) > 0:
                # Make sure the lengths of the X and Y values are the same
                self.assertEqual(
                    len(progress_curves[mrep][0]),
                    len(self.expected_progress_curves[mrep][0]),
                    "Length of X values for problem "
                    + self.expected_problem_name
                    + " and solver "
                    + self.expected_solver_name
                    + " do not match.",
                )
                self.assertEqual(
                    len(progress_curves[mrep][1]),
                    len(self.expected_progress_curves[mrep][1]),
                    "Length of Y values for problem "
                    + self.expected_problem_name
                    + " and solver "
                    + self.expected_solver_name
                    + " do not match.",
                )
                # Check X (0) and Y (1) values
                for x_index in range(len(progress_curves[mrep][0])):
                    # If the value is NaN, make sure we're expecting NaN
                    if math.isnan(progress_curves[mrep][0][x_index]):
                        self.assertTrue(
                            math.isnan(
                                self.expected_progress_curves[mrep][0][x_index]
                            ),
                            "X values for problem "
                            + self.expected_problem_name
                            + " and solver "
                            + self.expected_solver_name
                            + " do not match at mrep "
                            + str(mrep)
                            + " and index "
                            + str(x_index)
                            + ".",
                        )
                    # Otherwise, check the value normally
                    else:
                        self.assertAlmostEqual(
                            progress_curves[mrep][0][x_index],
                            self.expected_progress_curves[mrep][0][x_index],
                            5,
                            "X values for problem "
                            + self.expected_problem_name
                            + " and solver "
                            + self.expected_solver_name
                            + " do not match at mrep "
                            + str(mrep)
                            + " and index "
                            + str(x_index)
                            + ".",
                        )
                for y_index in range(len(progress_curves[mrep][1])):
                    # If the value is NaN, make sure we're expecting NaN
                    if math.isnan(progress_curves[mrep][1][y_index]):
                        self.assertTrue(
                            math.isnan(
                                self.expected_progress_curves[mrep][1][y_index]
                            ),
                            "Y values for problem "
                            + self.expected_problem_name
                            + " and solver "
                            + self.expected_solver_name
                            + " do not match at mrep "
                            + str(mrep)
                            + " and index "
                            + str(y_index)
                            + ".",
                        )
                    # Otherwise, check the value normally
                    else:
                        self.assertAlmostEqual(
                            progress_curves[mrep][1][y_index],
                            self.expected_progress_curves[mrep][1][y_index],
                            5,
                            "Y values for problem "
                            + self.expected_problem_name
                            + " and solver "
                            + self.expected_solver_name
                            + " do not match at mrep "
                            + str(mrep)
                            + " and index "
                            + str(y_index)
                            + ".",
                        )
