import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class test_IRONORECONT1_SPSA(unittest.TestCase):
    def setUp(self):
        # Expected values
        self.expected_problem_name = "IRONORECONT-1"
        self.expected_solver_name = "SPSA"
        self.expected_all_recommended_xs = "[[(80, 40, 100), (79.9, 40.1, 99.9), (79.840173975191, 40.15982622854404, 99.840173975191), (79.84017403301645, 40.1598262863695, 99.84017403301645), (79.84017408493288, 40.15982633828593, 99.84017408493288), (79.84017408493288, 40.15982633828593, 99.84017408493288)], [(80, 40, 100), (80.06129025754778, 39.93870974245221, 99.78429614545477), (80.09129718066005, 39.90870281933994, 99.64861717997532), (80.09129718066005, 39.90870281933994, 99.64861717997532)], [(80, 40, 100), (80.33334426584184, 40.01185870179294, 99.66665573415816), (80.40503045448851, 39.94017251314627, 99.59496954551149), (80.4692621253788, 39.87594084225598, 99.5307378746212), (80.55866495203203, 39.84851023696141, 99.44133504796797), (80.58005459226503, 39.82712059672841, 99.36304204235721), (80.59859120535364, 39.80858398363979, 99.29394302000642), (80.61504505119117, 39.79213013780226, 99.23180828458527), (80.61504505119117, 39.79213013780226, 99.23180828458527)], [(80, 40, 100), (79.88944512188709, 40.11055487811291, 99.88944512188709), (79.80790898027158, 40.192091019728416, 99.80790898027158), (79.80790898027158, 40.192091019728416, 99.80790898027158)], [(80, 40, 100), (0.0, 120.0, 20.0), (0.0, 120.0, 20.0), (20.0, 100.0, 0.0), (120.0, 0.0, 100.0), (120.0, 0.0, 100.0)], [(80, 40, 100), (80.13705796125292, 39.862942038747086, 99.86294203874708), (80.23814058672366, 39.76185941327634, 99.76185941327634), (80.36987752490744, 39.729603412142836, 99.63012247509256), (80.58861475418223, 39.58210760329671, 99.41138607558304), (80.58861475418223, 39.58210760329671, 99.41138607558304)], [(80, 40, 100), (80.14163977030795, 39.858360229692046, 99.85836022969205), (80.30557029889549, 39.81336717992019, 99.69442970110451), (80.37767127777165, 39.74126620104403, 99.62232872222835), (80.37767127777165, 39.74126620104403, 99.62232872222835)], [(80, 40, 100), (120.0, 0.0, 60.0), (120.0, 0.0, 60.0)], [(80, 40, 100), (80, 40, 100)], [(80, 40, 100), (80, 40, 100)]]"
        self.expected_all_intermediate_budgets = "[[0, 210, 330, 390, 450, 1000], [0, 270, 390, 1000], [0, 390, 450, 510, 630, 750, 870, 990, 1000], [0, 210, 270, 1000], [0, 210, 270, 570, 690, 1000], [0, 210, 270, 390, 750, 1000], [0, 210, 330, 390, 1000], [0, 210, 1000], [0, 1000], [0, 1000]]"
        self.expected_all_est_objectives = "[[149895.8055235529, 149490.01490668004, 149055.9839961229, 149055.9839961229, 149055.9839961229, 149055.9839961229], [122793.09736468189, 123777.93057210676, 123146.51073289948, 123146.51073289948], [99852.80349485856, 102406.22573755184, 102070.64177815235, 101588.43733685819, 103428.73599471145, 103411.91603451176, 103702.52248035539, 103749.21753314632, 103749.21753314632], [126011.12695446546, 125345.33865181894, 124448.63770230618, 124448.63770230618], [136147.71179130895, 65822.07046203454, 65822.07046203454, 133596.77111131456, 130997.4391892812, 130997.4391892812], [132850.26196652921, 133244.21392665227, 133023.7769571096, 132848.328269142, 131372.08599689155, 131372.08599689155], [134982.68434045353, 134233.36659889162, 133557.24319635433, 133138.57078580675, 133138.57078580675], [161256.2908821113, 48055.83014443732, 48055.83014443732], [146337.47315675917, 146337.47315675917], [134867.2205665852, 134867.2205665852]]"
        self.expected_objective_curves = "[([0, 210, 330, 390, 450, 1000], [121270.73497283501, 149490.01490668004, 149055.9839961229, 149055.9839961229, 149055.9839961229, 149055.9839961229]), ([0, 270, 390, 1000], [121270.73497283501, 123777.93057210676, 123146.51073289948, 123146.51073289948]), ([0, 390, 450, 510, 630, 750, 870, 990, 1000], [121270.73497283501, 102406.22573755184, 102070.64177815235, 101588.43733685819, 103428.73599471145, 103411.91603451176, 103702.52248035539, 103749.21753314632, 103749.21753314632]), ([0, 210, 270, 1000], [121270.73497283501, 125345.33865181894, 124448.63770230618, 124448.63770230618]), ([0, 210, 270, 570, 690, 1000], [121270.73497283501, 65822.07046203454, 65822.07046203454, 133596.77111131456, 130997.4391892812, 130997.4391892812]), ([0, 210, 270, 390, 750, 1000], [121270.73497283501, 133244.21392665227, 133023.7769571096, 132848.328269142, 131372.08599689155, 131372.08599689155]), ([0, 210, 330, 390, 1000], [121270.73497283501, 134233.36659889162, 133557.24319635433, 133138.57078580675, 133138.57078580675]), ([0, 210, 1000], [121270.73497283501, 48055.83014443732, 48055.83014443732]), ([0, 1000], [121270.73497283501, 121270.73497283501]), ([0, 1000], [121270.73497283501, 121270.73497283501])]"
        self.expected_progress_curves = "[([0.0, 0.21, 0.33, 0.39, 0.45, 1.0], [nan, inf, inf, inf, inf, inf]), ([0.0, 0.27, 0.39, 1.0], [nan, inf, inf, inf]), ([0.0, 0.39, 0.45, 0.51, 0.63, 0.75, 0.87, 0.99, 1.0], [nan, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf]), ([0.0, 0.21, 0.27, 1.0], [nan, inf, inf, inf]), ([0.0, 0.21, 0.27, 0.57, 0.69, 1.0], [nan, -inf, -inf, inf, inf, inf]), ([0.0, 0.21, 0.27, 0.39, 0.75, 1.0], [nan, inf, inf, inf, inf, inf]), ([0.0, 0.21, 0.33, 0.39, 1.0], [nan, inf, inf, inf, inf]), ([0.0, 0.21, 1.0], [nan, -inf, -inf]), ([0.0, 1.0], [nan, nan]), ([0.0, 1.0], [nan, nan])]"

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
