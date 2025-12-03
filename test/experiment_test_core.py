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
import pickle
import unittest
from abc import abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import zstandard as zstd

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
        """Set up the experiment and load expected results."""
        # Set the name of the experiment file
        with zstd.open(self.filepath, "rb") as f:
            expected_results = pickle.load(f)

        self.num_macroreps = expected_results["num_macroreps"]
        self.num_postreps = expected_results["num_postreps"]
        self.expected_problem_name = expected_results["problem_name"]
        self.expected_solver_name = expected_results["solver_name"]
        self.expected_all_recommended_xs = expected_results["all_recommended_xs"]
        self.expected_all_intermediate_budgets = expected_results[
            "all_intermediate_budgets"
        ]
        self.expected_all_est_objectives = expected_results["all_est_objectives"]
        self.expected_all_est_lhs = expected_results.get("all_est_lhs", [])
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
        )
        self.assertEqual(
            self.myexperiment.problem.name,
            self.expected_problem_name,
        )


class ExperimentTestMixin:
    """Mixin class for experiment tests.

    This class contains the test methods that check the results of the
    experiment against the expected results. It expects that any classes
    that inherit from it will have already set up the experiment in the
    setUp method of the ExperimentTest class.
    """

    def test_run(self: Any) -> None:
        """Test the run method of the experiment."""
        ps_names = f"{self.expected_problem_name} | {self.expected_solver_name}"
        # Check actual run results against expected
        self.myexperiment.run(n_macroreps=self.num_macroreps)
        self.assertEqual(
            self.myexperiment.n_macroreps,
            self.num_macroreps,
            f"[{ps_names}] Number of macro-replications does not match",
        )
        # For each macroreplication
        for mrep in range(self.num_macroreps):
            rec_xs = self.myexperiment.all_recommended_xs[mrep]
            expected_rec_xs = self.expected_all_recommended_xs[mrep]
            # Check to make sure the list lengths are the same
            self.assertEqual(
                len(rec_xs),
                len(expected_rec_xs),
                f"[{ps_names} | {mrep}] Length of recommended solutions do not match",
            )
            # For each list of recommended solutions
            for sol_list_idx in range(len(rec_xs)):
                rec_xs_list = rec_xs[sol_list_idx]
                expected_rec_xs_list = expected_rec_xs[sol_list_idx]
                # Check to make sure the tuples are the same length
                self.assertEqual(
                    len(rec_xs_list),
                    len(expected_rec_xs_list),
                    f"[{ps_names} | {mrep} | {sol_list_idx}] "
                    f"Length of recommended solutions do not match",
                )
                # For each tuple of recommended solutions
                for sol_idx in range(len(rec_xs_list)):
                    rec_xs_tup = rec_xs_list[sol_idx]
                    expected_rec_xs_tup = expected_rec_xs_list[sol_idx]
                    self.assertAlmostEqual(
                        rec_xs_tup,
                        expected_rec_xs_tup,
                        5,
                        f"[{ps_names} | {mrep} | {sol_list_idx} | {sol_idx}] "
                        f"Recommended solutions do not match",
                    )
            # Check to make sure the list lengths are the same
            int_budg = self.myexperiment.all_intermediate_budgets[mrep]
            expected_int_budg = self.expected_all_intermediate_budgets[mrep]
            self.assertEqual(
                len(int_budg),
                len(expected_int_budg),
                f"[{ps_names} | {mrep}] Length of intermediate budgets do not match",
            )
            # For each list of intermediate budgets
            for sol_list_idx in range(len(int_budg)):
                int_budg_list = int_budg[sol_list_idx]
                expected_int_budg_list = expected_int_budg[sol_list_idx]
                # Check the values in the list
                self.assertAlmostEqual(
                    int_budg_list,
                    expected_int_budg_list,
                    5,
                    f"[{ps_names} | {mrep} | {sol_list_idx}] "
                    f"Intermediate budgets do not match",
                )

    def test_post_replicate(self: Any) -> None:
        """Test the post_replicate method of the experiment."""
        ps_names = f"{self.expected_problem_name} | {self.expected_solver_name}"
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
            f"[{ps_names}] Number of post-replications does not match",
        )
        # For each macroreplication
        for mrep in range(self.num_macroreps):
            est_obj = self.myexperiment.all_est_objectives[mrep]
            expected_est_obj = self.expected_all_est_objectives[mrep]
            # Check to make sure the list lengths are the same
            self.assertEqual(
                len(est_obj),
                len(expected_est_obj),
                f"[{ps_names} | {mrep}] Length of estimated objectives do not match",
            )
            # For each list in the estimated objectives
            for objective_idx in range(len(est_obj)):
                est_obj_list = est_obj[objective_idx]
                expected_est_obj_list = expected_est_obj[objective_idx]
                # Check the values in the list
                self.assertAlmostEqual(
                    est_obj_list,
                    expected_est_obj_list,
                    5,
                    f"[{ps_names} | {mrep} | {objective_idx}] "
                    f"Estimated objectives do not match",
                )

            self.assertEqual(
                len(self.myexperiment.all_est_lhs),
                len(self.expected_all_est_lhs),
                f"[{ps_names} | {mrep}] Length of `all_est_lhs` do not match",
            )

            if len(self.myexperiment.all_est_lhs) == 0:
                continue

            est_lhs = self.myexperiment.all_est_lhs[mrep]
            expected_est_lhs = self.expected_all_est_lhs[mrep]
            self.assertEqual(
                len(est_lhs),
                len(expected_est_lhs),
                f"[{ps_names} | {mrep}] Length of `est_lhs` do not match",
            )
            assert np.allclose(est_lhs, expected_est_lhs)

    def test_post_normalize(self: Any) -> None:
        """Test the post_normalize method of the experiment."""
        ps_names = f"{self.expected_problem_name} | {self.expected_solver_name}"
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
        self.myexperiment.all_est_lhs = self.expected_all_est_lhs
        self.myexperiment.has_run = True
        self.myexperiment.has_postreplicated = True
        # Check actual post-normalization results against expected
        post_normalize([self.myexperiment], n_postreps_init_opt=self.num_postreps)

        objective_curves = [
            [curve.x_vals, curve.y_vals] for curve in self.myexperiment.objective_curves
        ]

        progress_curves = [
            [curve.x_vals, curve.y_vals] for curve in self.myexperiment.progress_curves
        ]

        for mrep in range(self.num_macroreps):
            obj_curves = objective_curves[mrep]
            expected_obj_curves = self.expected_objective_curves[mrep]
            # Check to make sure the same number of objective curves are present
            # This should probably always be 2 (x and y)
            self.assertEqual(
                len(obj_curves),
                len(expected_obj_curves),
                f"[{ps_names} | {mrep}] Number of objective curves do not match",
            )
            # Make sure that curves are only checked if they exist
            # TODO: check if this is necessary
            if len(obj_curves) > 0:
                obj_curve_x = obj_curves[0]
                expected_obj_curve_x = expected_obj_curves[0]
                # Make sure the lengths of the X and Y values are the same
                self.assertEqual(
                    len(obj_curve_x),
                    len(expected_obj_curve_x),
                    f"[{ps_names} | {mrep}] Length of X values do not match",
                )
                obj_curve_y = obj_curves[1]
                expected_obj_curve_y = expected_obj_curves[1]
                self.assertEqual(
                    len(obj_curve_y),
                    len(expected_obj_curve_y),
                    f"[{ps_names} | {mrep}] Length of Y values do not match",
                )
                # Check X (0) and Y (1) values
                for x_index in range(len(obj_curve_x)):
                    obj_curve_x_val = obj_curve_x[x_index]
                    expected_obj_curve_x_val = expected_obj_curve_x[x_index]
                    # If the value is NaN, make sure we're expecting NaN
                    if math.isnan(obj_curve_x_val):
                        self.assertTrue(
                            math.isnan(expected_obj_curve_x_val),
                            f"[{ps_names} | {mrep} | {x_index}] "
                            f"Unexpected NaN value in X values",
                        )
                    # Otherwise, check the value normally
                    else:
                        self.assertAlmostEqual(
                            obj_curve_x_val,
                            expected_obj_curve_x_val,
                            5,
                            f"[{ps_names} | {mrep} | {x_index}] X values do not match",
                        )
                for y_index in range(len(obj_curve_y)):
                    obj_curve_y_val = obj_curve_y[y_index]
                    expected_obj_curve_y_val = expected_obj_curve_y[y_index]
                    # If the value is NaN, make sure we're expecting NaN
                    if math.isnan(obj_curve_y_val):
                        self.assertTrue(
                            math.isnan(expected_obj_curve_y_val),
                            f"[{ps_names} | {mrep} | {y_index}] "
                            f"Unexpected NaN value in Y values",
                        )
                    # Otherwise, check the value normally
                    else:
                        self.assertAlmostEqual(
                            obj_curve_y_val,
                            expected_obj_curve_y_val,
                            5,
                            f"[{ps_names} | {mrep} | {y_index}] Y values do not match",
                        )

            # Check to make sure the same number of progress curves are present
            # This should probably always be 2 (x and y)
            prog_curves = progress_curves[mrep]
            expected_prog_curves = self.expected_progress_curves[mrep]
            self.assertEqual(
                len(prog_curves),
                len(expected_prog_curves),
                f"[{ps_names} | {mrep}] Number of progress curves do not match",
            )
            # Make sure that curves are only checked if they exist
            # TODO: check if this is necessary
            if len(prog_curves) > 0:
                prog_curve_x = prog_curves[0]
                expected_prog_curve_x = expected_prog_curves[0]
                # Make sure the lengths of the X and Y values are the same
                self.assertEqual(
                    len(prog_curve_x),
                    len(expected_prog_curve_x),
                    f"[{ps_names} | {mrep}] Length of X values do not match",
                )
                prog_curve_y = prog_curves[1]
                expected_prog_curve_y = expected_prog_curves[1]
                self.assertEqual(
                    len(prog_curve_y),
                    len(expected_prog_curve_y),
                    f"[{ps_names} | {mrep}] Length of Y values do not match",
                )
                # Check X (0) and Y (1) values
                for x_index in range(len(prog_curve_x)):
                    prog_curve_x_val = prog_curve_x[x_index]
                    expected_prog_curve_x_val = expected_prog_curve_x[x_index]
                    # If the value is NaN, make sure we're expecting NaN
                    if math.isnan(prog_curve_x_val):
                        self.assertTrue(
                            math.isnan(expected_prog_curve_x_val),
                            f"[{ps_names} | {mrep} | {x_index}] "
                            f"Unexpected NaN value in X values",
                        )
                    # Otherwise, check the value normally
                    else:
                        self.assertTrue(
                            math.isclose(
                                prog_curve_x_val,
                                expected_prog_curve_x_val,
                                rel_tol=1e-8,
                                abs_tol=1e-8,
                            ),
                            f"[{ps_names} | {mrep} | {x_index}] X values do not match",
                        )
                for y_index in range(len(prog_curve_y)):
                    prog_curve_y_val = prog_curve_y[y_index]
                    expected_prog_curve_y_val = expected_prog_curve_y[y_index]
                    # If the value is NaN, make sure we're expecting NaN
                    if math.isnan(prog_curve_y_val):
                        self.assertTrue(
                            math.isnan(expected_prog_curve_y_val),
                            f"[{ps_names} | {mrep} | {y_index}] "
                            f"Unexpected NaN value in Y values",
                        )
                    # Otherwise, check the value normally
                    else:
                        self.assertTrue(
                            math.isclose(
                                prog_curve_y_val,
                                expected_prog_curve_y_val,
                                rel_tol=1e-8,
                                abs_tol=1e-8,
                            ),
                            f"[{ps_names} | {mrep} | {y_index}] Y values do not match",
                        )
