import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class test_IRONORECONT1_STRONG(unittest.TestCase):
    def setUp(self):
        # Expected values
        self.expected_problem_name = "IRONORECONT-1"
        self.expected_solver_name = "STRONG"
        self.expected_all_recommended_xs = "[[(80, 40, 100), (80.01930400897865, 40.07446358899551, 101.99852008475052), (80.14767474106527, 40.269184088079335, 103.98487475660046), (80.15666435688847, 40.37311093294335, 101.98759700983514), (80.15666435688847, 40.37311093294335, 101.98759700983514)], [(80, 40, 100), (80.06393142612873, 40.0, 98.00102206796744), (80.06393142612873, 40.0, 98.00102206796744)], [(80, 40, 100), (80.0, 40.055414221101834, 101.99923216863367), (80.0, 40.39479867850451, 100.02823795502636), (80.01353109797068, 40.70761923101493, 102.00357595428491), (80.01353109797068, 40.70761923101493, 102.00357595428491)], [(80, 40, 100), (80, 40, 100)], [(80, 40, 100), (79.99888053118211, 40.13989122284082, 101.9951012988221), (79.99888053118211, 40.13989122284082, 101.9951012988221)], [(80, 40, 100), (80.00436602023422, 40.04296730554915, 101.99953363275569), (80.02053417576117, 40.16107893403187, 100.00308973573178), (80.02552329457308, 40.19167835244931, 102.00284941736504), (80.02552329457308, 40.19167835244931, 102.00284941736504)], [(80, 40, 100), (80, 40, 100)], [(80, 40, 100), (80.0, 40.05108013805388, 101.99934759846715), (80.0, 40.05407218808596, 99.99934983655925), (80.0, 40.074015964360434, 101.99925039553409), (80.0, 40.07745417300037, 99.99925335085594), (80.0, 40.4452531368485, 101.96514346024462), (80.0, 40.55662898298251, 99.96824701303011), (80.0, 40.55662898298251, 99.96824701303011)], [(80, 40, 100), (80, 40, 100)], [(80, 40, 100), (79.99636768431262, 40.35706723857862, 98.03213572789774), (80.04898217424699, 42.261261984477635, 98.6414583724298), (80.04898217424699, 42.261261984477635, 98.6414583724298)]]"
        self.expected_all_intermediate_budgets = "[[10, 80, 430, 535, 1000], [10, 157, 1000], [10, 80, 332, 430, 1000], [10, 1000], [10, 80, 1000], [10, 80, 241, 332, 1000], [10, 1000], [10, 80, 157, 241, 332, 535, 892, 1000], [10, 1000], [10, 80, 332, 1000]]"
        self.expected_all_est_objectives = "[[149895.8055235529, 148477.7274778582, 141044.12952090046, 148492.96399261276, 148492.96399261276], [122793.09736468189, 120197.12182066315, 120197.12182066315], [99852.80349485856, 84568.07900562166, 99958.3701362542, 84698.53949921897, 84698.53949921897], [126011.12695446546, 126011.12695446546], [136147.71179130895, 133702.03874348794, 133702.03874348794], [132850.26196652921, 131969.56278031142, 132947.82175185136, 132016.88135926626, 132016.88135926626], [134982.68434045353, 134982.68434045353], [161256.2908821113, 158053.61037127997, 161256.2908821113, 158053.61037127997, 161257.80298595421, 157786.45114511155, 161189.10153840613, 161189.10153840613], [146337.47315675917, 146337.47315675917], [134867.2205665852, 123320.73717829157, 129089.95839731517, 129089.95839731517]]"
        self.expected_objective_curves = "[([10, 80, 430, 535, 1000], [121270.73497283501, 148477.7274778582, 141044.12952090046, 148492.96399261276, 148492.96399261276]), ([10, 157, 1000], [121270.73497283501, 120197.12182066315, 120197.12182066315]), ([10, 80, 332, 430, 1000], [121270.73497283501, 84568.07900562166, 99958.3701362542, 84698.53949921897, 84698.53949921897]), ([10, 1000], [121270.73497283501, 121270.73497283501]), ([10, 80, 1000], [121270.73497283501, 133702.03874348794, 133702.03874348794]), ([10, 80, 241, 332, 1000], [121270.73497283501, 131969.56278031142, 132947.82175185136, 132016.88135926626, 132016.88135926626]), ([10, 1000], [121270.73497283501, 121270.73497283501]), ([10, 80, 157, 241, 332, 535, 892, 1000], [121270.73497283501, 158053.61037127997, 161256.2908821113, 158053.61037127997, 121310.03521131905, 157786.45114511155, 161189.10153840613, 161189.10153840613]), ([10, 1000], [121270.73497283501, 121270.73497283501]), ([10, 80, 332, 1000], [121270.73497283501, 123320.73717829157, 129089.95839731517, 129089.95839731517])]"
        self.expected_progress_curves = "[([0.01, 0.08, 0.43, 0.535, 1.0], [1.0, -691.2856846294831, -502.13675720052595, -691.6733798532856, -691.6733798532856]), ([0.01, 0.157, 1.0], [1.0, 28.318235043483448, 28.318235043483448]), ([0.01, 0.08, 0.332, 0.43, 1.0], [1.0, 934.9041538417559, 543.296068896318, 931.5845685509364, 931.5845685509364]), ([0.01, 1.0], [1.0, 1.0]), ([0.01, 0.08, 1.0], [1.0, -315.3162426025082, -315.3162426025082]), ([0.01, 0.08, 0.241, 0.332, 1.0], [1.0, -271.2331522701114, -296.12508700830136, -272.4371800516239, -272.4371800516239]), ([0.01, 1.0], [1.0, 1.0]), ([0.01, 0.08, 0.157, 0.241, 0.332, 0.535, 0.892, 1.0], [1.0, -934.945348356751, -1016.4379966044768, -934.945348356751, -0.0, -928.1474449222227, -1014.7283544675098, -1014.7283544675098]), ([0.01, 1.0], [1.0, 1.0]), ([0.01, 0.08, 0.332, 1.0], [1.0, -51.16258945322328, -197.96122074822213, -197.96122074822213])]"

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
