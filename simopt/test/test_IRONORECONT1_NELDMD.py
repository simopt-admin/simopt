import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class test_IRONORECONT1_NELDMD(unittest.TestCase):
    def setUp(self):
        # Expected values
        self.expected_problem_name = "IRONORECONT-1"
        self.expected_solver_name = "NELDMD"
        self.expected_all_recommended_xs = "[[(80, 40, 100), (71.84677291324282, 59.313560684071284, 103.72927038686149), (56.13379149117459, 80.51840568135151, 101.48396323850142), (50.924717227140476, 88.038075807242, 102.50401828489335), (40.179801595873954, 99.84577986522817, 99.31842913731958), (43.12847863441446, 99.109866009233, 101.36142144891875), (43.12847863441446, 99.109866009233, 101.36142144891875)], [(80, 40, 100), (81.75200674676572, 40.59763606068053, 100.07218623080225), (81.75200674676572, 40.59763606068053, 100.07218623080225)], [(80, 40, 100), (72.99402336126965, 61.01444415352947, 100.27025191408055), (77.78692868226355, 62.860690909278254, 101.01622523655048), (67.02403415536372, 97.2866127479025, 98.17365677049997), (67.02403415536372, 97.2866127479025, 98.17365677049997)], [(80, 40, 100), (77.05992216766845, 102.0282154661873, 99.37299152796103), (79.28183844548062, 89.42792181444659, 99.98720277010761), (88.6060134858426, 97.82621392212657, 96.13571418562641), (88.6060134858426, 97.82621392212657, 96.13571418562641)], [(80, 40, 100), (80.94647133838481, 41.5986778194078, 102.09635916368613), (71.40647090729274, 43.5469072828665, 101.68315577157618), (96.25182809429387, 42.141913172792854, 101.95920876321753), (98.014269732472, 41.10309619221593, 101.24898242315336), (98.014269732472, 41.10309619221593, 101.24898242315336)], [(80, 40, 100), (100.37163734231885, 94.01676050768417, 100.33881897742998), (100.37163734231885, 94.01676050768417, 100.33881897742998)], [(80, 40, 100), (81.56825765245154, 39.40202263934076, 100.85877312834089), (81.56825765245154, 39.40202263934076, 100.85877312834089)], [(80, 40, 100), (80.75142988425276, 99.76906168314724, 91.24710742446698), (80.75142988425276, 99.76906168314724, 91.24710742446698)], [(80, 40, 100), (99.94506698217431, 106.1208741871661, 95.75645937152836), (101.7018336226699, 103.17509215814889, 95.30266657128709), (101.7018336226699, 103.17509215814889, 95.30266657128709)], [(80, 40, 100), (86.63835328843174, 38.52866321064976, 97.49557513910626), (84.74948605720766, 39.1933811584683, 97.47698081469643), (84.74948605720766, 39.1933811584683, 97.47698081469643)]]"
        self.expected_all_intermediate_budgets = "[[0, 480, 630, 720, 840, 900, 1000], [0, 810, 1000], [0, 540, 720, 900, 1000], [0, 840, 900, 990, 1000], [0, 690, 750, 810, 870, 1000], [0, 870, 1000], [0, 990, 1000], [0, 780, 1000], [0, 660, 960, 1000], [0, 600, 900, 1000]]"
        self.expected_all_est_objectives = "[[149895.8055235529, 153706.6747016864, 183253.04288145647, 191502.65160461538, 200315.3982262673, 202507.49597840346, 202507.49597840346], [122793.09736468189, 123208.33385017878, 123208.33385017878], [99852.80349485856, 120481.85206985116, 127101.83947312846, 184780.73634381968, 184780.73634381968], [126011.12695446546, 207115.8568076835, 208585.30770905715, 229588.6733113368, 229588.6733113368], [136147.71179130895, 134563.3775276705, 134519.62742134838, 135591.0723866743, 137173.95510059112, 137173.95510059112], [132850.26196652921, 250128.8744846016, 250128.8744846016], [134982.68434045353, 134859.37231653478, 134859.37231653478], [161256.2908821113, 219326.3821098657, 219326.3821098657], [146337.47315675917, 254054.42514026337, 259610.41605255895, 259610.41605255895], [134867.2205665852, 118851.84356596734, 118639.63598780682, 118639.63598780682]]"
        self.expected_objective_curves = "[([0, 480, 630, 720, 840, 900, 1000], [121270.73497283501, 153706.6747016864, 183253.04288145647, 191502.65160461538, 200315.3982262673, 202507.49597840346, 202507.49597840346]), ([0, 810, 1000], [121270.73497283501, 123208.33385017878, 123208.33385017878]), ([0, 540, 720, 900, 1000], [121270.73497283501, 120481.85206985116, 127101.83947312846, 184780.73634381968, 184780.73634381968]), ([0, 840, 900, 990, 1000], [121270.73497283501, 207115.8568076835, 208585.30770905715, 229588.6733113368, 229588.6733113368]), ([0, 690, 750, 810, 870, 1000], [121270.73497283501, 134563.3775276705, 134519.62742134838, 135591.0723866743, 137173.95510059112, 137173.95510059112]), ([0, 870, 1000], [121270.73497283501, 250128.8744846016, 250128.8744846016]), ([0, 990, 1000], [121270.73497283501, 134859.37231653478, 134859.37231653478]), ([0, 780, 1000], [121270.73497283501, 219326.3821098657, 219326.3821098657]), ([0, 660, 960, 1000], [121270.73497283501, 254054.42514026337, 239610.50678627988, 239610.50678627988]), ([0, 600, 900, 1000], [121270.73497283501, 118851.84356596734, 118639.63598780682, 118639.63598780682])]"
        self.expected_progress_curves = "[([0.0, 0.48, 0.63, 0.72, 0.84, 0.9, 1.0], [1.0, 0.7259083803204844, 0.47623434658693925, 0.4065231362580576, 0.3320532730277595, 0.3135295111635584, 0.3135295111635584]), ([0.0, 0.81, 1.0], [1.0, 0.9836268158400857, 0.9836268158400857]), ([0.0, 0.54, 0.72, 0.9, 1.0], [1.0, 1.006666253372767, 0.9507257415580805, 0.4633249633850558, 0.4633249633850558]), ([0.0, 0.84, 0.9, 0.99, 1.0], [1.0, 0.2745877356415907, 0.26217051631747257, 0.08468694270208539, 0.08468694270208539]), ([0.0, 0.69, 0.75, 0.81, 0.87, 1.0], [1.0, 0.8876739210230141, 0.8880436201161566, 0.8789896482442573, 0.8656139023757243, 0.8656139023757243]), ([0.0, 0.87, 1.0], [1.0, -0.08888277826750646, -0.08888277826750646]), ([0.0, 0.99, 1.0], [1.0, 0.885172692701137, 0.885172692701137]), ([0.0, 0.78, 1.0], [1.0, 0.17140581197326304, 0.17140581197326304]), ([0.0, 0.66, 0.96, 1.0], [1.0, -0.12205464090934207, -0.0, -0.0]), ([0.0, 0.6, 0.9, 1.0], [1.0, 1.020440223686428, 1.0222334295960613, 1.0222334295960613])]"

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
