import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class test_SSCONT1_RNDSRCH(unittest.TestCase):
    def setUp(self):
        # Expected values
        self.expected_problem_name = "SSCONT-1"
        self.expected_solver_name = "RNDSRCH"
        self.expected_all_recommended_xs = "[[(600, 600), (359.64365512847405, 465.33501295716064), (448.4333218676209, 310.8963458265232), (448.4333218676209, 310.8963458265232)], [(600, 600), (379.4984318811467, 885.7474017443209), (325.34753297971815, 484.396706029854), (349.5306522240082, 410.27378677181), (370.70555006408216, 436.0331898637314), (589.5536241931459, 238.46041135411085), (455.5779816688252, 273.4468864216472), (455.5779816688252, 273.4468864216472)], [(600, 600), (201.9742725596509, 780.8933400714283), (392.20484473409556, 456.927538294216), (236.91888662053393, 425.03253740381695), (362.6919082172776, 354.3557218892089), (362.6919082172776, 354.3557218892089)], [(600, 600), (270.51550828210134, 330.22973611572723), (563.1986444042213, 251.7030137502581), (626.4364857411043, 175.81234105838976), (592.2418396925617, 148.6990089807615), (592.2418396925617, 148.6990089807615)], [(600, 600), (290.4813924950597, 261.4662596466684), (483.61266152187284, 508.8728086787097), (460.58375448622513, 226.7074912071731), (326.54107511059124, 455.5743039431546), (450.52768085160676, 312.2453259596492), (450.52768085160676, 312.2453259596492)], [(600, 600), (409.74224692763124, 535.1203903690143), (481.3923763982421, 312.5808672336022), (481.3923763982421, 312.5808672336022)], [(600, 600), (477.3830349535785, 647.41569100937), (370.95683999105216, 181.27446475403576), (360.07380838009834, 284.1229158649434), (617.7140354330088, 305.547888556553), (561.5913293809684, 280.0377568572764), (530.1524959071461, 206.53998668477863), (530.1524959071461, 206.53998668477863)], [(600, 600), (238.95919282377542, 384.82460068594133), (288.4513939553728, 390.67891574809835), (481.2727419601838, 388.8134007448537), (383.97289195347247, 391.48484260076674), (383.97289195347247, 391.48484260076674)], [(600, 600), (476.5545786705003, 356.2619054680448), (339.7929445338787, 241.71203036349), (615.6162331057836, 193.5195467569682), (511.5095718637328, 212.24733099847668), (511.5095718637328, 212.24733099847668)], [(600, 600), (503.25052162181316, 446.79327484259204), (419.1449837747667, 403.8396019725111), (357.96744157512427, 324.1947641853848), (388.4854466145964, 330.62760034162926), (467.74485172048503, 276.0076074487548), (467.74485172048503, 276.0076074487548)]]"
        self.expected_all_intermediate_budgets = "[[0, 30, 80, 1000], [0, 30, 40, 250, 300, 560, 900, 1000], [0, 20, 30, 80, 100, 1000], [0, 20, 40, 560, 740, 1000], [0, 20, 120, 130, 800, 940, 1000], [0, 30, 260, 1000], [0, 30, 40, 60, 80, 110, 300, 1000], [0, 20, 40, 140, 680, 1000], [0, 30, 90, 130, 270, 1000], [0, 30, 90, 110, 170, 470, 1000]]"
        self.expected_all_est_objectives = "[[618.5809976153716, 533.3809189890569, 524.6956810502536, 524.6956810502536], [619.371245290233, 617.0013871604638, 535.1566693494801, 531.8047516351805, 532.5540902393947, 529.4790860906369, 524.6017259605693, 524.6017259605693], [620.2040298994102, 558.2160445991162, 537.1368353845496, 545.4253221207432, 533.5709689924389, 533.5709689924389], [620.3929887875448, 546.078807843856, 527.145481326377, 522.60774171131, 517.620364115135, 517.620364115135], [617.140803174291, 543.2060261863802, 553.1628117457508, 515.8191104014048, 529.5690253522938, 519.9401355735813, 519.9401355735813], [617.6250759903628, 542.7328033712926, 523.4597411567702, 523.4597411567702], [622.8299886318688, 585.0447275257981, 534.5729153987116, 526.7188411663461, 543.3303346926017, 526.2093936498089, 516.5363149993058, 516.5363149993058], [617.1638109984892, 550.0044627557086, 542.3876126622054, 538.2641591809906, 534.7083064386326, 534.7083064386326], [625.4509909440814, 533.830456186644, 543.9036167612861, 527.6161568132001, 524.745025896469, 524.745025896469], [616.3517529689802, 542.8864270263207, 530.1130514049365, 529.5912733771099, 525.9884721979162, 525.9454774712535, 525.9454774712535]]"
        self.expected_objective_curves = "[([0, 30, 80, 1000], [624.4131899421741, 533.3809189890569, 524.6956810502536, 524.6956810502536]), ([0, 30, 40, 250, 300, 560, 900, 1000], [624.4131899421741, 617.0013871604638, 535.1566693494801, 531.8047516351805, 532.5540902393947, 529.4790860906369, 524.6017259605693, 524.6017259605693]), ([0, 20, 30, 80, 100, 1000], [624.4131899421741, 558.2160445991162, 537.1368353845496, 545.4253221207432, 533.5709689924389, 533.5709689924389]), ([0, 20, 40, 560, 740, 1000], [624.4131899421741, 546.078807843856, 527.145481326377, 522.60774171131, 517.620364115135, 517.620364115135]), ([0, 20, 120, 130, 800, 940, 1000], [624.4131899421741, 543.2060261863802, 553.1628117457508, 526.7574017452295, 529.5690253522938, 519.9401355735813, 519.9401355735813]), ([0, 30, 260, 1000], [624.4131899421741, 542.7328033712926, 523.4597411567702, 523.4597411567702]), ([0, 30, 40, 60, 80, 110, 300, 1000], [624.4131899421741, 585.0447275257981, 534.5729153987116, 526.7188411663461, 543.3303346926017, 526.2093936498089, 516.5363149993058, 516.5363149993058]), ([0, 20, 40, 140, 680, 1000], [624.4131899421741, 550.0044627557086, 542.3876126622054, 538.2641591809906, 534.7083064386326, 534.7083064386326]), ([0, 30, 90, 130, 270, 1000], [624.4131899421741, 533.830456186644, 543.9036167612861, 527.6161568132001, 524.745025896469, 524.745025896469]), ([0, 30, 90, 110, 170, 470, 1000], [624.4131899421741, 542.8864270263207, 530.1130514049365, 529.5912733771099, 525.9884721979162, 525.9454774712535, 525.9454774712535])]"
        self.expected_progress_curves = "[([0.0, 0.03, 0.08, 1.0], [1.0, 0.06782513731259486, -0.02111211975288083, -0.02111211975288083]), ([0.0, 0.03, 0.04, 0.25, 0.3, 0.56, 0.9, 1.0], [1.0, 0.9241027806077119, 0.08600890699188947, 0.05168510728490537, 0.05935837087787257, 0.027870179491240633, -0.022074224420909505, -0.022074224420909505]), ([0.0, 0.02, 0.03, 0.08, 0.1, 1.0], [1.0, 0.32213802617048587, 0.10628590307814338, 0.191160408616699, 0.0697712585501678, 0.0697712585501678]), ([0.0, 0.02, 0.04, 0.56, 0.74, 1.0], [1.0, 0.1978521340656288, 0.003973953703234255, -0.04249271968959694, -0.09356370778215023, -0.09356370778215023]), ([0.0, 0.02, 0.12, 0.13, 0.8, 0.94, 1.0], [1.0, 0.16843471078210367, 0.27039267705534203, 0.0, 0.028791161885807357, -0.06980913571553611, -0.06980913571553611]), ([0.0, 0.03, 0.26, 1.0], [1.0, 0.1635888862403645, -0.03376820411104372, -0.03376820411104372]), ([0.0, 0.03, 0.04, 0.06, 0.08, 0.11, 0.3, 1.0], [1.0, 0.5968650384861903, 0.08003123826844204, -0.000394862195014683, 0.16970763590530066, -0.005611629433734614, -0.10466442322200696, -0.10466442322200696]), ([0.0, 0.02, 0.04, 0.14, 0.68, 1.0], [1.0, 0.2380510304580845, 0.16005411666386943, 0.11782975334298926, 0.08141764907337955, 0.08141764907337955]), ([0.0, 0.03, 0.09, 0.13, 0.27, 1.0], [1.0, 0.0724284199841814, 0.1755780720491191, 0.008793693480193257, -0.02060682613817095, -0.02060682613817095]), ([0.0, 0.03, 0.09, 0.11, 0.17, 0.47, 1.0], [1.0, 0.16516199990689198, 0.03436201500867163, 0.029018982737258104, -0.007873875798970299, -0.00831414388196332, -0.00831414388196332])]"

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
