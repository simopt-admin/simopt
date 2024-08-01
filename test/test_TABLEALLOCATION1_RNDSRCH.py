import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class test_TABLEALLOCATION1_RNDSRCH(unittest.TestCase):
    def setUp(self):
        # Expected values
        self.expected_problem_name = "TABLEALLOCATION-1"
        self.expected_solver_name = "RNDSRCH"
        self.expected_all_recommended_xs = "[[(10, 5, 4, 2), (4, 3, 2, 6), (4, 3, 2, 6)], [(10, 5, 4, 2), (4, 2, 4, 5), (4, 2, 4, 5)], [(10, 5, 4, 2), (4, 3, 6, 3), (5, 6, 1, 5), (4, 5, 2, 5), (4, 5, 2, 5)], [(10, 5, 4, 2), (5, 4, 5, 3), (4, 4, 4, 4), (4, 4, 4, 4)], [(10, 5, 4, 2), (2, 4, 6, 3), (9, 3, 3, 4), (9, 3, 3, 4)], [(10, 5, 4, 2), (5, 4, 1, 6), (5, 4, 1, 6)], [(10, 5, 4, 2), (6, 5, 0, 6), (6, 2, 6, 3), (6, 2, 6, 3)], [(10, 5, 4, 2), (4, 3, 6, 3), (5, 5, 3, 4), (2, 3, 4, 5), (5, 3, 3, 5), (7, 0, 3, 6), (4, 3, 2, 6), (4, 3, 2, 6)], [(10, 5, 4, 2), (3, 6, 3, 4), (3, 6, 3, 4)], [(10, 5, 4, 2), (8, 4, 4, 3), (5, 5, 3, 4), (5, 5, 3, 4)], [(10, 5, 4, 2), (6, 8, 2, 3), (6, 8, 2, 3)], [(10, 5, 4, 2), (10, 0, 6, 3), (2, 1, 4, 6), (2, 3, 4, 5), (4, 2, 4, 5), (4, 2, 4, 5)], [(10, 5, 4, 2), (3, 5, 5, 3), (5, 3, 3, 5), (5, 3, 3, 5)], [(10, 5, 4, 2), (2, 6, 2, 5), (1, 7, 3, 4), (5, 3, 3, 5), (2, 5, 4, 4), (3, 2, 3, 6), (3, 2, 3, 6)], [(10, 5, 4, 2), (3, 1, 5, 5), (2, 5, 4, 4), (2, 5, 4, 4)], [(10, 5, 4, 2), (3, 3, 5, 4), (3, 3, 5, 4)], [(10, 5, 4, 2), (8, 6, 4, 2), (4, 4, 4, 4), (4, 4, 4, 4)], [(10, 5, 4, 2), (3, 5, 1, 6), (3, 5, 1, 6)], [(10, 5, 4, 2), (6, 5, 4, 3), (7, 1, 5, 4), (3, 4, 3, 5), (3, 4, 3, 5)], [(10, 5, 4, 2), (9, 2, 5, 3), (2, 4, 2, 6), (2, 4, 2, 6)], [(10, 5, 4, 2), (6, 2, 6, 3), (4, 5, 2, 5), (4, 5, 2, 5)], [(10, 5, 4, 2), (0, 5, 6, 3), (8, 2, 4, 4), (8, 2, 4, 4)], [(10, 5, 4, 2), (4, 2, 4, 5), (4, 2, 4, 5)], [(10, 5, 4, 2), (6, 2, 6, 3), (5, 3, 3, 5), (5, 3, 3, 5)]]"
        self.expected_all_intermediate_budgets = "[[0, 20, 1000], [0, 20, 1000], [0, 20, 50, 60, 1000], [0, 30, 40, 1000], [0, 20, 30, 1000], [0, 30, 1000], [0, 20, 30, 1000], [0, 20, 30, 70, 200, 230, 250, 1000], [0, 20, 1000], [0, 20, 30, 1000], [0, 30, 1000], [0, 20, 40, 70, 250, 1000], [0, 30, 40, 1000], [0, 20, 50, 70, 120, 230, 1000], [0, 20, 30, 1000], [0, 20, 1000], [0, 20, 30, 1000], [0, 20, 1000], [0, 30, 50, 80, 1000], [0, 40, 50, 1000], [0, 20, 40, 1000], [0, 20, 30, 1000], [0, 20, 1000], [0, 30, 60, 1000]]"
        self.expected_all_est_objectives = "[[5024.1, 5222.325, 5222.325], [5101.275, 5279.475, 5279.475], [5011.95, 5168.625, 5195.775, 5213.25, 5213.25], [5078.175, 5209.35, 5229.45, 5229.45], [5085.45, 5258.25, 5271.6, 5271.6], [5033.55, 5205.6, 5205.6], [5109.45, 5284.2, 5274.975, 5274.975], [5107.5, 5291.175, 5314.65, 5313.9, 5316.375, 5285.4, 5318.025, 5318.025], [5132.325, 5345.925, 5345.925], [5119.125, 5265.6, 5307.975, 5307.975], [5108.325, 5202.6, 5202.6], [5082.0, 5206.35, 5261.85, 5289.075, 5292.15, 5292.15], [5046.45, 5217.375, 5257.125, 5257.125], [5134.725, 5318.4, 5318.475, 5323.05, 5320.575, 5323.35, 5323.35], [5067.3, 5303.4, 5299.125, 5299.125], [5154.525, 5340.3, 5340.3], [5120.85, 5123.625, 5341.125, 5341.125], [5096.925, 5293.2, 5293.2], [5139.6, 5313.375, 5344.95, 5361.9, 5361.9], [5123.025, 5275.275, 5313.225, 5313.225], [5080.5, 5228.1, 5267.775, 5267.775], [5081.175, 5230.65, 5247.825, 5247.825], [5132.475, 5324.025, 5324.025], [5056.8, 5225.475, 5262.225, 5262.225]]"
        self.expected_objective_curves = "[([0, 20, 1000], [5118.675, 5222.325, 5222.325]), ([0, 20, 1000], [5118.675, 5279.475, 5279.475]), ([0, 20, 50, 60, 1000], [5118.675, 5168.625, 5195.775, 5213.25, 5213.25]), ([0, 30, 40, 1000], [5118.675, 5209.35, 5229.45, 5229.45]), ([0, 20, 30, 1000], [5118.675, 5258.25, 5271.6, 5271.6]), ([0, 30, 1000], [5118.675, 5205.6, 5205.6]), ([0, 20, 30, 1000], [5118.675, 5284.2, 5274.975, 5274.975]), ([0, 20, 30, 70, 200, 230, 250, 1000], [5118.675, 5291.175, 5314.65, 5313.9, 5316.375, 5285.4, 5318.025, 5318.025]), ([0, 20, 1000], [5118.675, 5345.925, 5345.925]), ([0, 20, 30, 1000], [5118.675, 5265.6, 5307.975, 5307.975]), ([0, 30, 1000], [5118.675, 5202.6, 5202.6]), ([0, 20, 40, 70, 250, 1000], [5118.675, 5206.35, 5261.85, 5289.075, 5292.15, 5292.15]), ([0, 30, 40, 1000], [5118.675, 5217.375, 5257.125, 5257.125]), ([0, 20, 50, 70, 120, 230, 1000], [5118.675, 5318.4, 5318.475, 5323.05, 5320.575, 5323.35, 5323.35]), ([0, 20, 30, 1000], [5118.675, 5303.4, 5299.125, 5299.125]), ([0, 20, 1000], [5118.675, 5340.3, 5340.3]), ([0, 20, 30, 1000], [5118.675, 5123.625, 5341.125, 5341.125]), ([0, 20, 1000], [5118.675, 5293.2, 5293.2]), ([0, 30, 50, 80, 1000], [5118.675, 5313.375, 5344.95, 5329.65, 5329.65]), ([0, 40, 50, 1000], [5118.675, 5275.275, 5313.225, 5313.225]), ([0, 20, 40, 1000], [5118.675, 5228.1, 5267.775, 5267.775]), ([0, 20, 30, 1000], [5118.675, 5230.65, 5247.825, 5247.825]), ([0, 20, 1000], [5118.675, 5324.025, 5324.025]), ([0, 30, 60, 1000], [5118.675, 5225.475, 5262.225, 5262.225])]"
        self.expected_progress_curves = "[([0.0, 0.02, 1.0], [1.0, 0.5087095627444015, 0.5087095627444015]), ([0.0, 0.02, 1.0], [1.0, 0.2378243867756815, 0.2378243867756815]), ([0.0, 0.02, 0.05, 0.06, 1.0], [1.0, 0.763242090295059, 0.6345538570920741, 0.5517241379310341, 0.5517241379310341]), ([0.0, 0.03, 0.04, 1.0], [1.0, 0.5702097404905775, 0.47493778883754034, 0.47493778883754034]), ([0.0, 0.02, 0.03, 1.0], [1.0, 0.3384287237824378, 0.27515108425168583, 0.27515108425168583]), ([0.0, 0.03, 1.0], [1.0, 0.5879843583362938, 0.5879843583362938]), ([0.0, 0.02, 0.03, 1.0], [1.0, 0.21542836829008147, 0.2591539281905411, 0.2591539281905411]), ([0.0, 0.02, 0.03, 0.07, 0.2, 0.23, 0.25, 1.0], [1.0, 0.1823675790970473, 0.07109847138286546, 0.07465339495200872, 0.0629221471738342, 0.20974049057945307, 0.055101315321720724, 0.055101315321720724]), ([0.0, 0.02, 1.0], [1.0, -0.0771418414504116, -0.0771418414504116]), ([0.0, 0.02, 0.03, 1.0], [1.0, 0.30359047280483203, 0.10273729114823713, 0.10273729114823713]), ([0.0, 0.03, 1.0], [1.0, 0.6022040526128669, 0.6022040526128669]), ([0.0, 0.02, 0.04, 0.07, 0.25, 1.0], [1.0, 0.5844294347671506, 0.3213650906505484, 0.19232136509065018, 0.17774617845716362, 0.17774617845716362]), ([0.0, 0.03, 0.04, 1.0], [1.0, 0.5321720583007462, 0.34376110913615276, 0.34376110913615276]), ([0.0, 0.02, 0.05, 0.07, 0.12, 0.23, 1.0], [1.0, 0.05332385353714909, 0.052968361180231315, 0.03128332740845821, 0.043014575186632736, 0.02986135798080004, 0.02986135798080004]), ([0.0, 0.02, 0.03, 1.0], [1.0, 0.12442232492001454, 0.14468538926412947, 0.14468538926412947]), ([0.0, 0.02, 1.0], [1.0, -0.05047991468183706, -0.05047991468183706]), ([0.0, 0.02, 0.03, 1.0], [1.0, 0.9765375044436553, -0.0543903306078938, -0.0543903306078938]), ([0.0, 0.02, 1.0], [1.0, 0.1727692854603622, 0.1727692854603622]), ([0.0, 0.03, 0.05, 0.08, 1.0], [1.0, 0.0771418414504073, -0.07252044081052363, -0.0, -0.0]), ([0.0, 0.04, 0.05, 1.0], [1.0, 0.2577319587628873, 0.07785282616423422, 0.07785282616423422]), ([0.0, 0.02, 0.04, 1.0], [1.0, 0.48133665126199565, 0.29328119445432, 0.29328119445432]), ([0.0, 0.02, 0.03, 1.0], [1.0, 0.469249911126912, 0.38784216139353017, 0.38784216139353017]), ([0.0, 0.02, 1.0], [1.0, 0.026661926768574543, 0.026661926768574543]), ([0.0, 0.03, 0.06, 1.0], [1.0, 0.49377888375399714, 0.31958762886597675, 0.31958762886597675])]"

        # Convert the expected values from string to their actual types
        self.expected_all_recommended_xs = eval(self.expected_all_recommended_xs, {'nan': float('nan'), 'inf': float('inf')})
        self.expected_all_intermediate_budgets = eval(self.expected_all_intermediate_budgets, {'nan': float('nan'), 'inf': float('inf')})
        self.expected_all_est_objectives = eval(self.expected_all_est_objectives, {'nan': float('nan'), 'inf': float('inf')})
        self.expected_objective_curves = eval(self.expected_objective_curves, {'nan': float('nan'), 'inf': float('inf')})
        self.expected_progress_curves = eval(self.expected_progress_curves, {'nan': float('nan'), 'inf': float('inf')})
        
        # Number of macro-replications and post-replications
        self.num_macroreps = 24
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
