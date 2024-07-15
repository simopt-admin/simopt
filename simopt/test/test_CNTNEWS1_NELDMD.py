import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class test_CNTNEWS1_NELDMD(unittest.TestCase):
    def setUp(self):
        # Expected values
        self.expected_problem_name = "CNTNEWS-1"
        self.expected_solver_name = "NELDMD"
        self.expected_all_recommended_xs = "[[(0,), (0.2591296074337594,), (0.17275307162250628,), (0.21594133952813283,), (0.19434720557531954,), (0.19434720557531954,)], [(0,), (0.1336908966858528,), (0.16711362085731601,), (0.1712914613787489,), (0.1712914613787489,)], [(0,), (0.2410077344101471,), (0.22092375654263485,), (0.23096574547639098,), (0.23096574547639098,)], [(0,), (0.21450798830012197,), (0.14300532553341463,), (0.1787566569167683,), (0.16088099122509147,), (0.15976376211936166,), (0.15976376211936166,)], [(0,), (0.12036491544512594,), (0.1504561443064074,), (0.14669474069874722,), (0.14857544250257732,), (0.14834035477709856,), (0.14834035477709856,)], [(0,), (0.14667615039338538,), (0.1444537844783341,), (0.1444537844783341,)], [(0,), (0.13570434398518863,), (0.18659347297963438,), (0.18871385335440294,), (0.18871385335440294,)], [(0,), (0.19332816548160742,), (0.24166020685200926,), (0.21749418616680832,), (0.22957719650940878,), (0.22655644392375868,), (0.22806682021658373,), (0.22806682021658373,)], [(0,), (0.23109768282550885,), (0.23880093891969248,), (0.23880093891969248,)], [(0,), (0.11334357260860685,), (0.1558474123368344,), (0.1558474123368344,)]]"
        self.expected_all_intermediate_budgets = "[[0, 120, 180, 240, 300, 1000], [0, 150, 240, 420, 1000], [0, 120, 240, 300, 1000], [0, 120, 180, 240, 300, 600, 1000], [0, 210, 300, 480, 540, 720, 1000], [0, 360, 420, 1000], [0, 210, 360, 540, 1000], [0, 180, 270, 330, 390, 510, 570, 1000], [0, 360, 420, 1000], [0, 180, 330, 1000]]"
        self.expected_all_est_objectives = "[[0.0, 0.37576344253154204, 0.447496245826092, 0.4292835494593317, 0.44442091867189343, 0.44442091867189343], [0.0, 0.44776173256248425, 0.4952379274549976, 0.4982748554627797, 0.4982748554627797], [0.0, 0.39263560106052237, 0.42046578871462864, 0.40732629247790597, 0.40732629247790597], [0.0, 0.49279438759414335, 0.45952166948816253, 0.49680205193391586, 0.4826426505476509, 0.4814132709661594, 0.4814132709661594], [0.0, 0.3751008886366243, 0.4101673956500514, 0.40785311040869887, 0.40908080342849407, 0.40893034728418765, 0.40893034728418765], [0.0, 0.4526584724149607, 0.44968631833442196, 0.44968631833442196], [0.0, 0.4348815867197554, 0.4756846361605565, 0.47567363950421077, 0.47567363950421077], [0.0, 0.44902713797570704, 0.395712722259235, 0.42642476363566095, 0.41275720564514384, 0.41627503494787654, 0.41455647456682215, 0.41455647456682215], [0.0, 0.40512701785430333, 0.3931908125920347, 0.3931908125920347], [0.0, 0.35652152726201053, 0.4191837067727468, 0.4191837067727468]]"
        self.expected_objective_curves = "[([0, 120, 180, 240, 300, 1000], [0.0, 0.37576344253154204, 0.447496245826092, 0.4292835494593317, 0.44442091867189343, 0.44442091867189343]), ([0, 150, 240, 420, 1000], [0.0, 0.44776173256248425, 0.4952379274549976, 0.5037389673405301, 0.5037389673405301]), ([0, 120, 240, 300, 1000], [0.0, 0.39263560106052237, 0.42046578871462864, 0.40732629247790597, 0.40732629247790597]), ([0, 120, 180, 240, 300, 600, 1000], [0.0, 0.49279438759414335, 0.45952166948816253, 0.49680205193391586, 0.4826426505476509, 0.4814132709661594, 0.4814132709661594]), ([0, 210, 300, 480, 540, 720, 1000], [0.0, 0.3751008886366243, 0.4101673956500514, 0.40785311040869887, 0.40908080342849407, 0.40893034728418765, 0.40893034728418765]), ([0, 360, 420, 1000], [0.0, 0.4526584724149607, 0.44968631833442196, 0.44968631833442196]), ([0, 210, 360, 540, 1000], [0.0, 0.4348815867197554, 0.4756846361605565, 0.47567363950421077, 0.47567363950421077]), ([0, 180, 270, 330, 390, 510, 570, 1000], [0.0, 0.44902713797570704, 0.395712722259235, 0.42642476363566095, 0.41275720564514384, 0.41627503494787654, 0.41455647456682215, 0.41455647456682215]), ([0, 360, 420, 1000], [0.0, 0.40512701785430333, 0.3931908125920347, 0.3931908125920347]), ([0, 180, 330, 1000], [0.0, 0.35652152726201053, 0.4191837067727468, 0.4191837067727468])]"
        self.expected_progress_curves = "[([0.0, 0.12, 0.18, 0.24, 0.3, 1.0], [1.0, 0.25405127080922446, 0.11165052767581068, 0.14780555547307136, 0.11775552918171885, 0.11775552918171885]), ([0.0, 0.15, 0.24, 0.42, 1.0], [1.0, 0.11112349531658326, 0.016875883020155115, -0.0, -0.0]), ([0.0, 0.12, 0.24, 0.3, 1.0], [1.0, 0.22055741859037348, 0.1653101785346068, 0.1913941170198348, 0.1913941170198348]), ([0.0, 0.12, 0.18, 0.24, 0.3, 0.6, 1.0], [1.0, 0.02172668873358809, 0.0877781960879681, 0.013770853271958402, 0.041879461706638324, 0.0443199709012753, 0.0443199709012753]), ([0.0, 0.21, 0.3, 0.48, 0.54, 0.72, 1.0], [1.0, 0.25536654307893925, 0.18575408645570174, 0.1903483016969222, 0.1879111405889048, 0.18820981937704925, 0.18820981937704925]), ([0.0, 0.36, 0.42, 1.0], [1.0, 0.1014027070314748, 0.10730289397994558, 0.10730289397994558]), ([0.0, 0.21, 0.36, 0.54, 1.0], [1.0, 0.13669258303423407, 0.05569219972813571, 0.05571402979699815, 0.05571402979699815]), ([0.0, 0.18, 0.27, 0.33, 0.39, 0.51, 0.57, 1.0], [1.0, 0.10861146925692891, 0.21444885562777766, 0.1534806888437606, 0.1806129118335254, 0.1736294749131995, 0.17704108388625048, 0.17704108388625048]), ([0.0, 0.36, 0.42, 1.0], [1.0, 0.19576001834212803, 0.21945523756506277, 0.21945523756506277]), ([0.0, 0.18, 0.33, 1.0], [1.0, 0.29224945780102785, 0.16785531009083826, 0.16785531009083826])]"

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
