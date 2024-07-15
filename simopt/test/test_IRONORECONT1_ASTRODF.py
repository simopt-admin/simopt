import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class test_IRONORECONT1_ASTRODF(unittest.TestCase):
    def setUp(self):
        # Expected values
        self.expected_problem_name = "IRONORECONT-1"
        self.expected_solver_name = "ASTRODF"
        self.expected_all_recommended_xs = "[[(80, 40, 100), (80.0, 61.54434690031883, 100.0), (102.85123068417624, 84.39557758449507, 100.0), (90.7325355527469, 84.39557758449507, 100.0), (90.7325355527469, 93.48459893306708, 100.0), (103.81640422318958, 89.65242252364862, 100.0), (88.70653721981503, 103.43308320561509, 100.0), (93.99872666755927, 89.03730278910193, 100.0), (101.30280199440337, 97.92415967328105, 100.0), (100.85454352514225, 97.82511829685517, 82.75116891248267), (100.85454352514225, 97.82511829685517, 82.75116891248267)], [(80, 40, 100), (80.0, 61.54434690031883, 100.0), (102.85123068417624, 84.39557758449507, 100.0), (102.85123068417624, 108.63296784735377, 100.0), (101.51656070762795, 105.9877273428278, 63.76484712432305), (100.3433755509445, 103.15781782271317, 64.7649178229156), (100.3433755509445, 103.15781782271317, 64.7649178229156)], [(80, 40, 100), (80.0, 61.54434690031883, 100.0), (102.85123068417624, 84.39557758449507, 100.0), (102.85123068417624, 96.51427271592442, 100.0), (93.76220933560424, 96.51427271592442, 100.0), (80.67834066516156, 92.68209630650597, 100.0), (90.60019761418162, 90.2101684948632, 100.0), (104.68571053045387, 95.09673799160517, 103.60076761126263), (104.68571053045387, 95.09673799160517, 103.60076761126263)], [(80, 40, 100), (80.0, 61.54434690031883, 100.0), (102.85123068417624, 84.39557758449507, 100.0), (102.85123068417624, 96.51427271592442, 100.0), (98.3398259887333, 96.30674773485862, 82.39189398727507), (102.94634355437549, 96.81347750522232, 89.89282543426249), (102.94634355437549, 96.81347750522232, 89.89282543426249)], [(80, 40, 100), (101.54434690031883, 40.0, 100.0), (101.54434690031883, 40.0, 100.0)], [(80, 40, 100), (80.0, 61.54434690031883, 100.0), (102.85123068417624, 84.39557758449507, 100.0), (102.85123068417624, 108.63296784735377, 100.0), (100.75923529322884, 106.7836957730319, 63.75129390893861), (98.66723990228144, 104.93442369871002, 27.502587817877213), (98.66723990228144, 104.93442369871002, 27.502587817877213)], [(80, 40, 100), (80.0, 61.54434690031883, 100.0), (102.85123068417624, 84.39557758449507, 100.0), (102.85123068417624, 96.51427271592442, 100.0), (98.30672000989024, 96.51427271592442, 100.0), (98.30672000989024, 96.51427271592442, 100.0)], [(80, 40, 100), (80, 40, 100)], [(80, 40, 100), (80.0, 61.54434690031883, 100.0), (102.85123068417624, 84.39557758449507, 100.0), (102.85123068417624, 108.63296784735377, 100.0), (100.08371729481047, 104.88446598685137, 63.943730799786636), (99.64599126304472, 99.41172343027145, 59.9032130501551), (105.91906455769842, 100.82371640585382, 67.85359038079794), (101.34368894274294, 92.29798642692896, 79.75416302499504), (99.41269642548383, 102.42008913471471, 84.8668967684458), (99.41269642548383, 102.42008913471471, 84.8668967684458)], [(80, 40, 100), (80.0, 61.54434690031883, 100.0), (102.85123068417624, 84.39557758449507, 100.0), (102.85123068417624, 108.63296784735377, 100.0), (100.6006573610449, 105.37895790218238, 63.85983834858097), (100.22918626129629, 99.78082109302638, 59.987994815766406)]]"
        self.expected_all_intermediate_budgets = "[[4, 50, 74, 180, 395, 441, 465, 544, 662, 700, 1000], [4, 116, 144, 231, 261, 333, 1000], [4, 33, 57, 144, 275, 304, 433, 499, 1000], [4, 34, 58, 170, 197, 221, 1000], [4, 52, 1000], [7, 169, 222, 347, 378, 402, 1000], [4, 32, 56, 138, 494, 1000], [12, 1000], [4, 58, 86, 144, 170, 733, 804, 834, 930, 1000], [4, 66, 90, 145, 170, 1000]]"
        self.expected_all_est_objectives = "[[149895.8055235529, 168672.8854247067, 234246.2104416266, 226579.34140707366, 246287.55924321295, 244971.2836550668, 240244.5629438077, 241624.1881675321, 255301.35525856088, 253641.1034161115, 253641.1034161115], [122793.09736468189, 149208.78165929313, 225923.64699964703, 234812.961762674, 243226.71537637338, 248954.8476191005, 248954.8476191005], [99852.80349485856, 128851.6769256802, 222643.254239316, 242235.31187345437, 239957.03647690467, 207672.48537593015, 224247.45667839752, 235849.50264935606, 235849.50264935606], [126011.12695446546, 149497.58590082, 224796.7933466262, 244299.21507732666, 241764.20246457632, 242055.05987657208, 242055.05987657208], [136147.71179130895, 138760.07226255184, 138760.07226255184], [132850.26196652921, 155746.52281976608, 230967.4195246391, 235391.84738181814, 242216.9947892452, 245931.82229351025, 245931.82229351025], [134982.68434045353, 156666.3695691511, 228131.5277213762, 246350.84901681027, 248510.77482358748, 248510.77482358748], [161256.2908821113, 161256.2908821113], [146337.47315675917, 172018.86646661864, 243106.80654491155, 245561.12025966725, 257146.2154042707, 262492.0966056358, 253994.89799126453, 249447.24155473537, 261047.8595872894, 261047.8595872894], [134867.2205665852, 158491.93299492102, 233135.8221024051, 235313.86318976388, 245548.44919351285, 252059.317496046]]"
        self.expected_objective_curves = "[([4, 50, 74, 180, 395, 441, 465, 544, 662, 700, 1000], [121270.73497283501, 168672.8854247067, 234246.2104416266, 226579.34140707366, 246287.55924321295, 244971.2836550668, 240244.5629438077, 241624.1881675321, 255301.35525856088, 253641.1034161115, 253641.1034161115]), ([4, 116, 144, 231, 261, 333, 1000], [121270.73497283501, 149208.78165929313, 225923.64699964703, 234812.961762674, 243226.71537637338, 248954.8476191005, 248954.8476191005]), ([4, 33, 57, 144, 275, 304, 433, 499, 1000], [121270.73497283501, 128851.6769256802, 222643.254239316, 242235.31187345437, 239957.03647690467, 207672.48537593015, 224247.45667839752, 235849.50264935606, 235849.50264935606]), ([4, 34, 58, 170, 197, 221, 1000], [121270.73497283501, 149497.58590082, 224796.7933466262, 244299.21507732666, 241764.20246457632, 242055.05987657208, 242055.05987657208]), ([4, 52, 1000], [121270.73497283501, 138760.07226255184, 138760.07226255184]), ([7, 169, 222, 347, 378, 402, 1000], [121270.73497283501, 155746.52281976608, 230967.4195246391, 235391.84738181814, 242216.9947892452, 245931.82229351025, 245931.82229351025]), ([4, 32, 56, 138, 494, 1000], [121270.73497283501, 156666.3695691511, 228131.5277213762, 246350.84901681027, 248510.77482358748, 248510.77482358748]), ([12, 1000], [121270.73497283501, 121270.73497283501]), ([4, 58, 86, 144, 170, 733, 804, 834, 930, 1000], [121270.73497283501, 172018.86646661864, 243106.80654491155, 245561.12025966725, 257146.2154042707, 242424.29953872776, 253994.89799126453, 249447.24155473537, 261047.8595872894, 261047.8595872894]), ([4, 66, 90, 145, 170, 1000], [121270.73497283501, 158491.93299492102, 233135.8221024051, 235313.86318976388, 245548.44919351285, 252059.317496046])]"
        self.expected_progress_curves = "[([0.004, 0.05, 0.074, 0.18, 0.395, 0.441, 0.465, 0.544, 0.662, 0.7, 1.0], [1.0, 0.608743245634422, 0.06750184467460114, 0.1307840853748581, -0.03188729707068622, -0.021022774901136226, 0.017991518472694753, 0.006604109206877759, -0.10628705615037497, -0.0925833582988238, -0.0925833582988238]), ([0.004, 0.116, 0.144, 0.231, 0.261, 0.333, 1.0], [1.0, 0.7693997136067487, 0.13619617877694712, 0.06282388639018636, -0.006623130244006991, -0.05390306181887797, -0.05390306181887797]), ([0.004, 0.033, 0.057, 0.144, 0.275, 0.304, 0.433, 0.499, 1.0], [1.0, 0.9374270003527457, 0.16327249941253918, 0.0015599018151100988, 0.020364758318614725, 0.28684103754864654, 0.15003143263229585, 0.05426829093250341, 0.05426829093250341]), ([0.004, 0.034, 0.058, 0.17, 0.197, 0.221, 1.0], [1.0, 0.7670159270251349, 0.1454972146734846, -0.01547552930297131, 0.005448432957929474, 0.003047699533057175, 0.003047699533057175]), ([0.004, 0.052, 1.0], [1.0, 0.8556432297111262, 0.8556432297111262]), ([0.007, 0.169, 0.222, 0.347, 0.378, 0.402, 1.0], [1.0, 0.7154372801950829, 0.09456494371535829, 0.0580457717617118, 0.0017110907980740818, -0.028951048756595408, -0.028951048756595408]), ([0.004, 0.032, 0.056, 0.138, 0.494, 1.0], [1.0, 0.7078448766807418, 0.11797235903511573, -0.03240969006691459, -0.050237690543140506, -0.050237690543140506]), ([0.012, 1.0], [1.0, 1.0]), ([0.004, 0.058, 0.086, 0.144, 0.17, 0.733, 0.804, 0.834, 0.93, 1.0], [1.0, 0.5811255601465788, -0.005633404255411652, -0.025891278826001375, -0.12151450861799468, -0.0, -0.09550357427778176, -0.05796727517817264, -0.15371863069231187, -0.15371863069231187]), ([0.004, 0.066, 0.09, 0.145, 0.17, 1.0], [1.0, 0.6927766990971015, 0.07666697607787563, 0.05868945230329289, -0.025786692005136405, -0.079527317184943])]"

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
