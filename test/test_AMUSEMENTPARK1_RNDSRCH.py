import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class test_AMUSEMENTPARK1_RNDSRCH(unittest.TestCase):
    def setUp(self):
        # Expected values
        self.expected_problem_name = "AMUSEMENTPARK-1"
        self.expected_solver_name = "RNDSRCH"
        self.expected_all_recommended_xs = "[[(344, 1, 1, 1, 1, 1, 1), (138, 4, 2, 32, 27, 12, 135), (16, 5, 67, 38, 166, 46, 12), (16, 5, 67, 38, 166, 46, 12)], [(344, 1, 1, 1, 1, 1, 1), (103, 56, 23, 63, 23, 69, 13), (37, 106, 5, 21, 112, 20, 49), (40, 75, 17, 8, 41, 39, 130), (65, 28, 33, 37, 114, 54, 19), (65, 28, 33, 37, 114, 54, 19)], [(344, 1, 1, 1, 1, 1, 1), (51, 47, 44, 74, 70, 17, 47), (100, 29, 14, 5, 39, 134, 29), (100, 29, 14, 5, 39, 134, 29)], [(344, 1, 1, 1, 1, 1, 1), (13, 67, 122, 96, 18, 17, 17), (48, 24, 17, 30, 183, 24, 24), (48, 24, 17, 30, 183, 24, 24)], [(344, 1, 1, 1, 1, 1, 1), (2, 107, 28, 69, 34, 86, 24), (74, 35, 6, 47, 99, 51, 38), (65, 44, 14, 23, 124, 32, 48), (65, 44, 14, 23, 124, 32, 48)], [(344, 1, 1, 1, 1, 1, 1), (3, 17, 98, 26, 142, 1, 63), (73, 73, 9, 16, 34, 59, 86), (5, 36, 68, 45, 56, 75, 65)], [(344, 1, 1, 1, 1, 1, 1), (39, 3, 80, 27, 52, 21, 128), (40, 10, 18, 25, 62, 109, 86), (40, 10, 18, 25, 62, 109, 86)], [(344, 1, 1, 1, 1, 1, 1), (41, 23, 24, 168, 28, 38, 28), (16, 55, 46, 12, 103, 112, 6), (29, 11, 115, 20, 55, 51, 69), (29, 11, 115, 20, 55, 51, 69)], [(344, 1, 1, 1, 1, 1, 1), (32, 64, 29, 37, 32, 40, 116), (51, 10, 24, 62, 41, 79, 83), (33, 45, 49, 9, 177, 18, 19), (33, 45, 49, 9, 177, 18, 19)], [(344, 1, 1, 1, 1, 1, 1), (11, 3, 62, 16, 57, 50, 151), (12, 83, 13, 42, 50, 86, 64), (46, 39, 28, 36, 49, 108, 44)]]"
        self.expected_all_intermediate_budgets = "[[0, 20, 30, 100], [0, 20, 40, 60, 70, 100], [0, 20, 30, 100], [0, 20, 40, 100], [0, 20, 30, 50, 100], [0, 20, 30, 100], [0, 20, 30, 100], [0, 20, 50, 80, 100], [0, 20, 50, 90, 100], [0, 20, 30, 100]]"
        self.expected_all_est_objectives = "[[1696.43, 660.17, 437.83, 437.83], [1692.15, 503.78, 462.48, 457.25, 411.28, 411.28], [1693.035, 459.575, 433.76, 433.76], [1695.61, 565.765, 375.95, 375.95], [1689.125, 559.7, 435.43, 395.31, 395.31], [1695.015, 870.48, 453.305, 448.115], [1696.11, 518.37, 373.12, 373.12], [1693.415, 500.37, 428.155, 433.745, 433.745], [1696.255, 442.125, 419.305, 410.145, 410.145], [1694.79, 491.105, 414.635, 395.73]]"
        self.expected_objective_curves = "[([0, 20, 30, 100], [1697.795, 660.17, 437.83, 437.83]), ([0, 20, 40, 60, 70, 100], [1697.795, 503.78, 462.48, 457.25, 411.28, 411.28]), ([0, 20, 30, 100], [1697.795, 459.575, 433.76, 433.76]), ([0, 20, 40, 100], [1697.795, 565.765, 375.95, 375.95]), ([0, 20, 30, 50, 100], [1697.795, 559.7, 435.43, 395.31, 395.31]), ([0, 20, 30, 100], [1697.795, 870.48, 453.305, 448.115]), ([0, 20, 30, 100], [1697.795, 518.37, 371.33, 371.33]), ([0, 20, 50, 80, 100], [1697.795, 500.37, 428.155, 433.745, 433.745]), ([0, 20, 50, 90, 100], [1697.795, 442.125, 419.305, 410.145, 410.145]), ([0, 20, 30, 100], [1697.795, 491.105, 414.635, 395.73])]"
        self.expected_progress_curves = "[([0.0, 0.2, 0.3, 1.0], [1.0, 0.21775169341068173, 0.050133248898387814, 0.050133248898387814]), ([0.0, 0.2, 0.4, 0.6, 0.7, 1.0], [1.0, 0.09985186190363106, 0.06871647574568497, 0.06477366534360122, 0.03011764351113673, 0.03011764351113673]), ([0.0, 0.2, 0.3, 1.0], [1.0, 0.06652644434643959, 0.04706494328911807, 0.04706494328911807]), ([0.0, 0.2, 0.4, 1.0], [1.0, 0.1465813270610231, 0.003482941502414315, 0.003482941502414315]), ([0.0, 0.2, 0.3, 0.5, 1.0], [1.0, 0.14200902398480175, 0.04832392863739338, 0.018078124941102867, 0.018078124941102867]), ([0.0, 0.2, 0.3, 1.0], [1.0, 0.37630092011474103, 0.06179959516459161, 0.057886940100191124]), ([0.0, 0.2, 0.3, 1.0], [1.0, 0.11085102132359316, 0.0, 0.0]), ([0.0, 0.2, 0.5, 0.8, 1.0], [1.0, 0.0972811193661348, 0.04283942659625394, 0.047053635037486864, 0.047053635037486864]), ([0.0, 0.2, 0.5, 0.9, 1.0], [1.0, 0.05337117828212581, 0.036167558133836944, 0.02926198580437478, 0.02926198580437478]), ([0.0, 0.2, 0.3, 1.0], [1.0, 0.09029638927525417, 0.032646922459318564, 0.01839475598677691])]"

        # Convert the expected values from string to their actual types
        self.expected_all_recommended_xs = eval(self.expected_all_recommended_xs, {'nan': float('nan'), 'inf': float('inf')})
        self.expected_all_intermediate_budgets = eval(self.expected_all_intermediate_budgets, {'nan': float('nan'), 'inf': float('inf')})
        self.expected_all_est_objectives = eval(self.expected_all_est_objectives, {'nan': float('nan'), 'inf': float('inf')})
        self.expected_objective_curves = eval(self.expected_objective_curves, {'nan': float('nan'), 'inf': float('inf')})
        self.expected_progress_curves = eval(self.expected_progress_curves, {'nan': float('nan'), 'inf': float('inf')})
        
        # Number of macro-replications and post-replications
        self.num_macroreps = 10
        self.num_postreps = 200

        # Setup the solver and experiment
        self.myexperiment = ProblemSolver(self.expected_solver_name, self.expected_problem_name)
        self.assertEqual(self.myexperiment.solver.name, self.expected_solver_name, "Solver name does not match (expected: " + self.expected_solver_name + ", actual: " + self.myexperiment.solver.name + ")")
        self.assertEqual(self.myexperiment.problem.name, self.expected_problem_name, "Problem name does not match (expected: " + self.expected_problem_name + ", actual: " + self.myexperiment.problem.name + ")")

    def test_run(self):
        # Check actual run results against expected
        self.myexperiment.run(n_macroreps=self.num_macroreps)
        self.assertEqual(self.myexperiment.n_macroreps, self.num_macroreps, "Number of macro-replications for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " does not match.")
        # For each macroreplication
        for mrep in range(self.num_macroreps):
            # Check to make sure the list lengths are the same
            self.assertEqual(len(self.myexperiment.all_recommended_xs[mrep]), len(self.expected_all_recommended_xs[mrep]), "Length of recommended solutions for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match.")
            # For each list of recommended solutions
            for list in range(len(self.myexperiment.all_recommended_xs[mrep])):
                # Check to make sure the tuples are the same length
                self.assertEqual(len(self.myexperiment.all_recommended_xs[mrep][list]), len(self.expected_all_recommended_xs[mrep][list]), "Recommended solutions for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match at mrep " + str(mrep) + " and index " + str(list) + ".")
                # For each tuple of recommended solutions
                for tuple in range(len(self.myexperiment.all_recommended_xs[mrep][list])):
                    self.assertAlmostEqual(self.myexperiment.all_recommended_xs[mrep][list][tuple], self.expected_all_recommended_xs[mrep][list][tuple], 5, "Recommended solutions for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match at mrep " + str(mrep) + " and index " + str(list) + " and tuple " + str(tuple) + ".")
            # Check to make sure the list lengths are the same
            self.assertEqual(len(self.myexperiment.all_intermediate_budgets[mrep]), len(self.expected_all_intermediate_budgets[mrep]), "Length of intermediate budgets for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match.")
            # For each list of intermediate budgets
            for list in range(len(self.myexperiment.all_intermediate_budgets[mrep])):
                # Check the values in the list
                self.assertAlmostEqual(self.myexperiment.all_intermediate_budgets[mrep][list], self.expected_all_intermediate_budgets[mrep][list], 5, "Intermediate budgets for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match at mrep " + str(mrep) + " and index " + str(list) + ".")
            
    def test_post_replicate(self):
        # Simulate results from the run method
        self.myexperiment = ProblemSolver(self.expected_solver_name, self.expected_problem_name)
        self.myexperiment.n_macroreps = self.num_macroreps
        self.myexperiment.all_recommended_xs = self.expected_all_recommended_xs
        self.myexperiment.all_intermediate_budgets = self.expected_all_intermediate_budgets

        # Check actual post-replication results against expected
        self.myexperiment.post_replicate(n_postreps=self.num_postreps)
        self.assertEqual(self.myexperiment.n_postreps, self.num_postreps, "Number of post-replications for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " does not match.")
        # For each macroreplication
        for mrep in range(self.num_macroreps):
            # Check to make sure the list lengths are the same
            self.assertEqual(len(self.myexperiment.all_est_objectives[mrep]), len(self.expected_all_est_objectives[mrep]), "Estimated objectives for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match.")
            # For each list in the estimated objectives
            for list in range(len(self.myexperiment.all_est_objectives[mrep])):
                # Check the values in the list
                self.assertAlmostEqual(self.myexperiment.all_est_objectives[mrep][list], self.expected_all_est_objectives[mrep][list], 5, "Estimated objectives for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match at mrep " + str(mrep) + " and index " + str(list) + ".")

    def test_post_normalize(self):
        # Simulate results from the post_replicate method
        self.myexperiment = ProblemSolver(self.expected_solver_name, self.expected_problem_name)
        self.myexperiment.n_macroreps = self.num_macroreps
        self.myexperiment.n_postreps = self.num_postreps
        self.myexperiment.all_recommended_xs = self.expected_all_recommended_xs
        self.myexperiment.all_intermediate_budgets = self.expected_all_intermediate_budgets
        self.myexperiment.all_est_objectives = self.expected_all_est_objectives

        # Check actual post-normalization results against expected
        post_normalize([self.myexperiment], n_postreps_init_opt=self.num_postreps)

        # Loop through each curve object and convert it into a tuple
        for i in range(len(self.myexperiment.objective_curves)):
            self.myexperiment.objective_curves[i] = (self.myexperiment.objective_curves[i].x_vals, self.myexperiment.objective_curves[i].y_vals)
        for i in range(len(self.myexperiment.progress_curves)):
            self.myexperiment.progress_curves[i] = (self.myexperiment.progress_curves[i].x_vals, self.myexperiment.progress_curves[i].y_vals)

        for mrep in range(self.num_macroreps):
            # Check to make sure the same number of objective curves are present
            # This should probably always be 2 (x and y)
            self.assertEqual(len(self.myexperiment.objective_curves[mrep]), len(self.expected_objective_curves[mrep]), "Number of objective curves for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " does not match.")
            # Make sure that curves are only checked if they exist
            if (len(self.myexperiment.objective_curves[mrep]) > 0):
                # Make sure the lengths of the X and Y values are the same
                self.assertEqual(len(self.myexperiment.objective_curves[mrep][0]), len(self.expected_objective_curves[mrep][0]), "Length of X values for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match.")
                self.assertEqual(len(self.myexperiment.objective_curves[mrep][1]), len(self.expected_objective_curves[mrep][1]), "Length of Y values for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match.")
                # Check X (0) and Y (1) values
                for x_index in range(len(self.myexperiment.objective_curves[mrep][0])):
                    # If the value is NaN, make sure we're expecting NaN
                    if (math.isnan(self.myexperiment.objective_curves[mrep][0][x_index])):
                        self.assertTrue(math.isnan(self.expected_objective_curves[mrep][0][x_index]), "X values for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match at mrep " + str(mrep) + " and index " + str(x_index) + ".")
                    # Otherwise, check the value normally
                    else:
                        self.assertAlmostEqual(self.myexperiment.objective_curves[mrep][0][x_index], self.expected_objective_curves[mrep][0][x_index], 5, "X values for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match at mrep " + str(mrep) + " and index " + str(x_index) + ".")
                for y_index in range(len(self.myexperiment.objective_curves[mrep][1])):
                    # If the value is NaN, make sure we're expecting NaN
                    if (math.isnan(self.myexperiment.objective_curves[mrep][1][y_index])):
                        self.assertTrue(math.isnan(self.expected_objective_curves[mrep][1][y_index]), "Y values for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match at mrep " + str(mrep) + " and index " + str(y_index) + ".")
                    # Otherwise, check the value normally
                    else:
                        self.assertAlmostEqual(self.myexperiment.objective_curves[mrep][1][y_index], self.expected_objective_curves[mrep][1][y_index], 5, "Y values for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match at mrep " + str(mrep) + " and index " + str(y_index) + ".")
            
            # Check to make sure the same number of progress curves are present
            # This should probably always be 2 (x and y)
            self.assertEqual(len(self.myexperiment.progress_curves[mrep]), len(self.expected_progress_curves[mrep]), "Number of progress curves for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " does not match.")
            # Make sure that curves are only checked if they exist
            if (len(self.myexperiment.progress_curves[mrep]) > 0):
                # Make sure the lengths of the X and Y values are the same
                self.assertEqual(len(self.myexperiment.progress_curves[mrep][0]), len(self.expected_progress_curves[mrep][0]), "Length of X values for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match.")
                self.assertEqual(len(self.myexperiment.progress_curves[mrep][1]), len(self.expected_progress_curves[mrep][1]), "Length of Y values for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match.")
                # Check X (0) and Y (1) values
                for x_index in range(len(self.myexperiment.progress_curves[mrep][0])):
                    # If the value is NaN, make sure we're expecting NaN
                    if (math.isnan(self.myexperiment.progress_curves[mrep][0][x_index])):
                        self.assertTrue(math.isnan(self.expected_progress_curves[mrep][0][x_index]), "X values for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match at mrep " + str(mrep) + " and index " + str(x_index) + ".")
                    # Otherwise, check the value normally
                    else:
                        self.assertAlmostEqual(self.myexperiment.progress_curves[mrep][0][x_index], self.expected_progress_curves[mrep][0][x_index], 5, "X values for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match at mrep " + str(mrep) + " and index " + str(x_index) + ".")
                for y_index in range(len(self.myexperiment.progress_curves[mrep][1])):
                    # If the value is NaN, make sure we're expecting NaN
                    if (math.isnan(self.myexperiment.progress_curves[mrep][1][y_index])):
                        self.assertTrue(math.isnan(self.expected_progress_curves[mrep][1][y_index]), "Y values for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match at mrep " + str(mrep) + " and index " + str(y_index) + ".")
                    # Otherwise, check the value normally
                    else:
                        self.assertAlmostEqual(self.myexperiment.progress_curves[mrep][1][y_index], self.expected_progress_curves[mrep][1][y_index], 5, "Y values for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match at mrep " + str(mrep) + " and index " + str(y_index) + ".")      
