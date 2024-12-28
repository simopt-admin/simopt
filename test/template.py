import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.


class TestProblemSolver(unittest.TestCase):
    def setUp(self) -> None:
        # Expected values
        self.expected_problem_name = "{{PROBLEM_NAME}}"
        self.expected_solver_name = "{{SOLVER_NAME}}"
        self.expected_all_recommended_xs = "{{ALL_RECOMMENDED_XS}}"
        self.expected_all_intermediate_budgets = "{{ALL_INTERMEDIATE_BUDGETS}}"
        self.expected_all_est_objectives = "{{ALL_EST_OBJECTIVES}}"
        self.expected_objective_curves = "{{OBJECTIVE_CURVES}}"
        self.expected_progress_curves = "{{PROGRESS_CURVES}}"

        # Convert the expected values from string to their actual types
        self.expected_all_recommended_xs = eval(
            self.expected_all_recommended_xs,
            {"nan": float("nan"), "inf": float("inf")},
        )
        self.expected_all_intermediate_budgets = eval(
            self.expected_all_intermediate_budgets,
            {"nan": float("nan"), "inf": float("inf")},
        )
        self.expected_all_est_objectives = eval(
            self.expected_all_est_objectives,
            {"nan": float("nan"), "inf": float("inf")},
        )
        self.expected_objective_curves = eval(
            self.expected_objective_curves,
            {"nan": float("nan"), "inf": float("inf")},
        )
        self.expected_progress_curves = eval(
            self.expected_progress_curves,
            {"nan": float("nan"), "inf": float("inf")},
        )

        # Number of macro-replications and post-replications
        self.num_macroreps = 24
        self.num_postreps = 200

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

    def test_run(self) -> None:
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

    def test_post_replicate(self) -> None:
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

    def test_post_normalize(self) -> None:
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

        # Loop through each curve object and convert it into a tuple
        for i in range(len(self.myexperiment.objective_curves)):
            self.myexperiment.objective_curves[i] = (
                self.myexperiment.objective_curves[i].x_vals,
                self.myexperiment.objective_curves[i].y_vals,
            )
        for i in range(len(self.myexperiment.progress_curves)):
            self.myexperiment.progress_curves[i] = (
                self.myexperiment.progress_curves[i].x_vals,
                self.myexperiment.progress_curves[i].y_vals,
            )

        for mrep in range(self.num_macroreps):
            # Check to make sure the same number of objective curves are present
            # This should probably always be 2 (x and y)
            self.assertEqual(
                len(self.myexperiment.objective_curves[mrep]),
                len(self.expected_objective_curves[mrep]),
                "Number of objective curves for problem "
                + self.expected_problem_name
                + " and solver "
                + self.expected_solver_name
                + " does not match.",
            )
            # Make sure that curves are only checked if they exist
            if len(self.myexperiment.objective_curves[mrep]) > 0:
                # Make sure the lengths of the X and Y values are the same
                self.assertEqual(
                    len(self.myexperiment.objective_curves[mrep][0]),
                    len(self.expected_objective_curves[mrep][0]),
                    "Length of X values for problem "
                    + self.expected_problem_name
                    + " and solver "
                    + self.expected_solver_name
                    + " do not match.",
                )
                self.assertEqual(
                    len(self.myexperiment.objective_curves[mrep][1]),
                    len(self.expected_objective_curves[mrep][1]),
                    "Length of Y values for problem "
                    + self.expected_problem_name
                    + " and solver "
                    + self.expected_solver_name
                    + " do not match.",
                )
                # Check X (0) and Y (1) values
                for x_index in range(
                    len(self.myexperiment.objective_curves[mrep][0])
                ):
                    # If the value is NaN, make sure we're expecting NaN
                    if math.isnan(
                        self.myexperiment.objective_curves[mrep][0][x_index]
                    ):
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
                            self.myexperiment.objective_curves[mrep][0][
                                x_index
                            ],
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
                for y_index in range(
                    len(self.myexperiment.objective_curves[mrep][1])
                ):
                    # If the value is NaN, make sure we're expecting NaN
                    if math.isnan(
                        self.myexperiment.objective_curves[mrep][1][y_index]
                    ):
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
                            self.myexperiment.objective_curves[mrep][1][
                                y_index
                            ],
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
                len(self.myexperiment.progress_curves[mrep]),
                len(self.expected_progress_curves[mrep]),
                "Number of progress curves for problem "
                + self.expected_problem_name
                + " and solver "
                + self.expected_solver_name
                + " does not match.",
            )
            # Make sure that curves are only checked if they exist
            if len(self.myexperiment.progress_curves[mrep]) > 0:
                # Make sure the lengths of the X and Y values are the same
                self.assertEqual(
                    len(self.myexperiment.progress_curves[mrep][0]),
                    len(self.expected_progress_curves[mrep][0]),
                    "Length of X values for problem "
                    + self.expected_problem_name
                    + " and solver "
                    + self.expected_solver_name
                    + " do not match.",
                )
                self.assertEqual(
                    len(self.myexperiment.progress_curves[mrep][1]),
                    len(self.expected_progress_curves[mrep][1]),
                    "Length of Y values for problem "
                    + self.expected_problem_name
                    + " and solver "
                    + self.expected_solver_name
                    + " do not match.",
                )
                # Check X (0) and Y (1) values
                for x_index in range(
                    len(self.myexperiment.progress_curves[mrep][0])
                ):
                    # If the value is NaN, make sure we're expecting NaN
                    if math.isnan(
                        self.myexperiment.progress_curves[mrep][0][x_index]
                    ):
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
                            self.myexperiment.progress_curves[mrep][0][x_index],
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
                for y_index in range(
                    len(self.myexperiment.progress_curves[mrep][1])
                ):
                    # If the value is NaN, make sure we're expecting NaN
                    if math.isnan(
                        self.myexperiment.progress_curves[mrep][1][y_index]
                    ):
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
                            self.myexperiment.progress_curves[mrep][1][y_index],
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
