import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class test_PARAMESTI1_ASTRODF(unittest.TestCase):
    def setUp(self):
        # Expected values
        self.expected_problem_name = "PARAMESTI-1"
        self.expected_solver_name = "ASTRODF"
        self.expected_all_recommended_xs = "[[(1, 1), (1.5149285897956326, 4.120071881769887), (2.513063822133061, 4.430739007802012), (2.513063822133061, 4.430739007802012)], [(1, 1), (1.0, 4.16227766016838), (1.1034868749749513, 5.343607616915401), (1.4185931120402107, 6.1753068921120144), (2.0823865428708404, 6.109545427342727), (1.3408250070660603, 5.437819561378433), (2.08711054629488, 5.359123684531698), (2.08711054629488, 5.359123684531698)], [(1, 1), (2.988348405084185, 3.4589572220758904), (1.836484603166295, 3.7408433370784984), (2.183380222906105, 4.559793307470667), (2.3896338255504723, 5.877839060715303), (2.3586844131042524, 6.544163623471122), (1.6847672970322431, 5.804592889155728), (1.6847672970322431, 5.804592889155728)], [(1, 1), (2.827479762870058, 3.5807591356615207), (1.6882502677498659, 3.910010039150322), (1.9943299880287926, 4.74507340790384), (1.9943299880287926, 4.74507340790384)], [(1, 1), (3.5889705285536357, 2.815828076190229), (1.281015745990938, 6.475085612424144), (2.9511955867168447, 5.863071070413266), (2.291520534447577, 5.764201891818322), (2.1203659444727943, 5.294107865129809), (2.1203659444727943, 5.294107865129809)], [(1, 1), (2.564533422142073, 3.748133033715875), (1.6855383569167222, 4.029418827623008), (1.6855383569167222, 4.029418827623008)], [(1, 1), (3.5041045641201816, 2.9311810717673454), (2.7390078646882223, 5.176092437616412), (2.2829700315291777, 4.504726722333899), (2.2829700315291777, 4.504726722333899)], [(1, 1), (3.157720222638415, 3.3117619775438016), (1.4536762757777157, 4.438459893547172), (2.6711911979027776, 6.128695360062492), (2.024491949444858, 5.829380927701434), (2.024491949444858, 5.829380927701434)], [(1, 1), (3.4314935812327083, 3.0218404893621407), (1.5892925862415654, 7.3929153564919226), (3.2104334546814286, 3.212411850623414), (1.8631737585377213, 4.373856559228505), (1.8631737585377213, 3.7068136152867375), (2.3301157311394154, 4.591739795605449), (2.3301157311394154, 4.591739795605449)], [(1, 1), (1.0415628563574333, 4.162004511219333), (1.7688663872221673, 3.2253724735483535), (1.7688663872221673, 4.114763065470711), (1.7688663872221673, 4.114763065470711)]]"
        self.expected_all_intermediate_budgets = "[[4, 27, 184, 1000], [4, 24, 128, 277, 610, 663, 942, 1000], [4, 24, 189, 344, 368, 587, 627, 1000], [4, 24, 82, 202, 1000], [4, 25, 41, 103, 359, 731, 1000], [4, 25, 103, 1000], [4, 24, 64, 80, 1000], [4, 24, 40, 72, 104, 1000], [4, 24, 40, 56, 105, 447, 499, 1000], [4, 24, 296, 686, 1000]]"
        self.expected_all_est_objectives = "[[-9.16944227603262, -5.087901713959044, -5.038434341062556, -5.038434341062556], [-8.940090362495347, -6.105529335023807, -5.672060760205896, -5.065096221668084, -4.696451969437038, -5.099026054774974, -4.602907076090707, -4.602907076090707], [-9.121210005202611, -6.1327216667434845, -4.834807231469014, -4.674149226964931, -4.856235025198865, -4.96295717979425, -4.736035238913933, -4.736035238913933], [-8.779886386724968, -5.633801495862974, -4.753957603975482, -4.5377413626394985, -4.5377413626394985], [-8.99288952739613, -7.333600595755891, -5.7523757908576885, -5.4806204040842665, -4.704794901253494, -4.580481029440518, -4.580481029440518], [-8.87740808504234, -5.054919789182981, -4.767843414834182, -4.767843414834182], [-9.024638576352391, -7.6104184714524274, -5.290415040394047, -4.760361694139695, -4.760361694139695], [-8.921050660074993, -6.596419499573125, -4.986486714056616, -5.349440327820792, -4.671136822330359, -4.671136822330359], [-8.550164686658025, -6.655817932151406, -5.5509882159508335, -6.064705878619034, -4.517347115958269, -4.60968959603932, -4.560859802350674, -4.560859802350674], [-8.983830735669818, -6.0981523912146915, -5.03604768870009, -4.730473300524734, -4.730473300524734]]"
        self.expected_objective_curves = "[([4, 27, 184, 1000], [-9.265122221743944, -5.087901713959044, -5.038434341062556, -5.038434341062556]), ([4, 24, 128, 277, 610, 663, 942, 1000], [-9.265122221743944, -6.105529335023807, -5.672060760205896, -5.065096221668084, -4.696451969437038, -5.099026054774974, -4.602907076090707, -4.602907076090707]), ([4, 24, 189, 344, 368, 587, 627, 1000], [-9.265122221743944, -6.1327216667434845, -4.834807231469014, -4.674149226964931, -4.856235025198865, -4.96295717979425, -4.736035238913933, -4.736035238913933]), ([4, 24, 82, 202, 1000], [-9.265122221743944, -5.633801495862974, -4.753957603975482, -4.5377413626394985, -4.5377413626394985]), ([4, 25, 41, 103, 359, 731, 1000], [-9.265122221743944, -7.333600595755891, -5.7523757908576885, -5.4806204040842665, -4.704794901253494, -4.580481029440518, -4.580481029440518]), ([4, 25, 103, 1000], [-9.265122221743944, -5.054919789182981, -4.767843414834182, -4.767843414834182]), ([4, 24, 64, 80, 1000], [-9.265122221743944, -7.6104184714524274, -5.290415040394047, -4.760361694139695, -4.760361694139695]), ([4, 24, 40, 72, 104, 1000], [-9.265122221743944, -6.596419499573125, -4.986486714056616, -5.349440327820792, -4.671136822330359, -4.671136822330359]), ([4, 24, 40, 56, 105, 447, 499, 1000], [-9.265122221743944, -6.655817932151406, -5.5509882159508335, -6.064705878619034, -4.517347115958269, -4.60968959603932, -4.560859802350674, -4.560859802350674]), ([4, 24, 296, 686, 1000], [-9.265122221743944, -6.0981523912146915, -5.03604768870009, -4.730473300524734, -4.730473300524734])]"
        self.expected_progress_curves = "[([0.004, 0.027, 0.184, 1.0], [1.0, 0.1032120166568272, 0.09259209714373186, 0.09259209714373186]), ([0.004, 0.024, 0.128, 0.277, 0.61, 0.663, 0.942, 1.0], [1.0, 0.32168174321021686, 0.22862239709019858, 0.09831601190853982, 0.019173497187754653, 0.1056002494895799, -0.0009092195612812687, -0.0009092195612812687]), ([0.004, 0.024, 0.189, 0.344, 0.368, 0.587, 0.627, 1.0], [1.0, 0.32751953805007317, 0.04887634294160907, 0.014385425452659974, 0.05347657601301961, 0.07638825750535845, 0.02767144464878974, 0.02767144464878974]), ([0.004, 0.024, 0.082, 0.202, 1.0], [1.0, 0.22040869411465794, 0.031519113592847765, -0.014899342572531648, -0.014899342572531648]), ([0.004, 0.025, 0.041, 0.103, 0.359, 0.731, 1.0], [1.0, 0.5853306330069499, 0.2458648563369957, 0.18752296013759823, 0.020964602298175677, -0.0057237628951325855, -0.0057237628951325855]), ([0.004, 0.025, 0.103, 1.0], [1.0, 0.0961312810931721, 0.03450019354633626, 0.03450019354633626]), ([0.004, 0.024, 0.064, 0.08, 1.0], [1.0, 0.6447593713358437, 0.14668865794863883, 0.03289397783398828, 0.03289397783398828]), ([0.004, 0.024, 0.04, 0.072, 0.104, 1.0], [1.0, 0.4270686625477895, 0.08143970344670298, 0.1593605215428717, 0.013738706355064604, 0.013738706355064604]), ([0.004, 0.024, 0.04, 0.056, 0.105, 0.447, 0.499, 1.0], [1.0, 0.43982063493381723, 0.20262989738893183, 0.3129175458033275, -0.019277688249753663, 0.0005468880048300608, -0.009936152602553377, -0.009936152602553377]), ([0.004, 0.024, 0.296, 0.686, 1.0], [1.0, 0.3200980215585868, 0.09207971788218985, 0.026477378043439748, 0.026477378043439748])]"

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
