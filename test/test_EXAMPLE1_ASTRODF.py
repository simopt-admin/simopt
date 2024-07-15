import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class test_EXAMPLE1_ASTRODF(unittest.TestCase):
    def setUp(self):
        # Expected values
        self.expected_problem_name = "EXAMPLE-1"
        self.expected_solver_name = "ASTRODF"
        self.expected_all_recommended_xs = "[[(2.0, 2.0), (-0.2360679774997898, -0.23606797749978936), (0.18319476828142112, 0.18319476828142078), (0.18319476828142112, 0.18319476828142078)], [(2.0, 2.0), (-0.23606797749979025, -0.23606797749978936), (0.18319476828142112, 0.18319476828142028), (-0.131252291054487, -0.13125229105448755)], [(2.0, 2.0), (-0.23606797749979025, -0.23606797749978936), (0.23606797749978758, 0.2360679774997868), (-0.2360679774997851, -0.23606797749978536), (0.08170719434686402, 0.09099523864654202), (0.08170719434686402, 0.09099523864654202)], [(2.0, 2.0), (-0.23606797749979025, -0.23606797749978936), (0.23606797749978908, 0.23606797749978847), (-0.0783790818361188, -0.07837908183611941)], [(2.0, 2.0), (-0.23606797749979025, -0.23606797749978936), (0.2360679774997882, 0.23606797749978725), (-0.22525570372117154, -0.22525570372117065), (0.22525570372117143, 0.22525570372117049), (-0.11619624388108618, -0.1161962438810964), (-0.11619624388108618, -0.1161962438810964)], [(2.0, 2.0), (-0.2360679774997898, -0.23606797749978936), (0.18319476828142123, 0.18319476828142078), (0.18319476828142123, 0.18319476828142078)], [(2.0, 2.0), (-0.23606797749979025, -0.23606797749978936), (0.18319476828142128, 0.18319476828142034), (0.18319476828142128, 0.18319476828142034)], [(2.0, 2.0), (-0.23606797749979025, -0.23606797749978936), (0.18319476828142128, 0.18319476828142023), (-0.13125229105448777, -0.1312522910544866)], [(2.0, 2.0), (-0.23606797749979025, -0.23606797749978936), (0.18319476828142112, 0.18319476828142034), (0.18319476828142112, 0.18319476828142034)], [(2.0, 2.0), (-0.23606797749979025, -0.23606797749978936), (0.23606797749978753, 0.23606797749978675), (0.23606797749978753, 0.23606797749978675)]]"
        self.expected_all_intermediate_budgets = "[[4, 24, 475, 1000], [4, 24, 374, 1000], [4, 24, 40, 56, 871, 1000], [4, 24, 72, 1000], [4, 24, 40, 170, 344, 918, 1000], [4, 24, 434, 1000], [4, 24, 433, 1000], [4, 24, 320, 1000], [4, 24, 450, 1000], [4, 24, 40, 1000]]"
        self.expected_all_est_objectives = "[[7.984539704940337, 0.0959958849420176, 0.05166035119170231, 0.05166035119170231], [8.081590387702734, 0.19304656770441356, 0.14871103395409793, 0.11604471551683475], [7.9253347189439385, 0.03679089894561915, 0.036790898945616685, 0.03679089894561484, -0.059709081991685994, -0.059709081991685994], [8.073099810658121, 0.18455599065980263, 0.1845559906598017, 0.08538637159706623], [7.880122723414122, -0.008421096584198224, -0.008421096584200163, -0.018397012468040694, -0.018397012468040843, -0.09287414240173267, -0.09287414240173267], [8.025785950362149, 0.13724213036382893, 0.09290659661351369, 0.09290659661351369], [8.015084462897443, 0.12654064289912453, 0.08220510914880892, 0.08220510914880892], [7.994852045957048, 0.10630822595872841, 0.06197269220841279, 0.02930637377114952], [7.910902809206077, 0.02235898920775803, -0.021976544542557642, -0.021976544542557642], [7.943417039435916, 0.05487321943759722, 0.05487321943759472, 0.05487321943759472]]"
        self.expected_objective_curves = "[([4, 24, 475, 1000], [8.090508544469758, 0.0959958849420176, 0.05166035119170231, 0.05166035119170231]), ([4, 24, 374, 1000], [8.090508544469758, 0.19304656770441356, 0.14871103395409793, 0.11604471551683475]), ([4, 24, 40, 56, 871, 1000], [8.090508544469758, 0.03679089894561915, 0.036790898945616685, 0.03679089894561484, -0.059709081991685994, -0.059709081991685994]), ([4, 24, 72, 1000], [8.090508544469758, 0.18455599065980263, 0.1845559906598017, 0.08538637159706623]), ([4, 24, 40, 170, 344, 918, 1000], [8.090508544469758, -0.008421096584198224, -0.008421096584200163, -0.018397012468040694, -0.018397012468040843, -0.09287414240173267, -0.09287414240173267]), ([4, 24, 434, 1000], [8.090508544469758, 0.13724213036382893, 0.09290659661351369, 0.09290659661351369]), ([4, 24, 433, 1000], [8.090508544469758, 0.12654064289912453, 0.08220510914880892, 0.08220510914880892]), ([4, 24, 320, 1000], [8.090508544469758, 0.10630822595872841, 0.06197269220841279, 0.02930637377114952]), ([4, 24, 450, 1000], [8.090508544469758, 0.02235898920775803, -0.021976544542557642, -0.021976544542557642]), ([4, 24, 40, 1000], [8.090508544469758, 0.05487321943759722, 0.05487321943759472, 0.05487321943759472])]"
        self.expected_progress_curves = "[([0.004, 0.024, 0.475, 1.0], [1.0, 0.011865247334499795, 0.006385303335105502, 0.006385303335105502]), ([0.004, 0.024, 0.374, 1.0], [1.0, 0.023860869393231152, 0.01838092539383682, 0.014343315365034347]), ([0.004, 0.024, 0.04, 0.056, 0.871, 1.0], [1.0, 0.004547414880461062, 0.004547414880460758, 0.00454741488046053, -0.007380139538014573, -0.007380139538014573]), ([0.004, 0.024, 0.072, 1.0], [1.0, 0.022811420276659285, 0.022811420276659167, 0.010553894248765341]), ([0.004, 0.024, 0.04, 0.17, 0.344, 0.918, 1.0], [1.0, -0.0010408612187863566, -0.0010408612187865962, -0.002273900628980352, -0.00227390062898037, -0.011479394884913199, -0.011479394884913199]), ([0.004, 0.024, 0.434, 1.0], [1.0, 0.016963350277608983, 0.0114834062782147, 0.0114834062782147]), ([0.004, 0.024, 0.433, 1.0], [1.0, 0.015640629041251183, 0.010160685041856852, 0.010160685041856852]), ([0.004, 0.024, 0.32, 1.0], [1.0, 0.013139869437675221, 0.00765992543828089, 0.0036223154094784066]), ([0.004, 0.024, 0.45, 1.0], [1.0, 0.0027636073906678518, -0.002716336608726486, -0.002716336608726486]), ([0.004, 0.024, 0.04, 1.0], [1.0, 0.006782419069949026, 0.0067824190699487166, 0.0067824190699487166])]"

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
