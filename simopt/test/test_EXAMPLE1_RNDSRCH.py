import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class test_EXAMPLE1_RNDSRCH(unittest.TestCase):
    def setUp(self):
        # Expected values
        self.expected_problem_name = "EXAMPLE-1"
        self.expected_solver_name = "RNDSRCH"
        self.expected_all_recommended_xs = "[[(2.0, 2.0), (-1.3868011021291058, 0.49063838865551596), (0.2115038131722952, -0.08431540722394781), (0.2115038131722952, -0.08431540722394781)], [(2.0, 2.0), (-0.2167740101241827, -0.5538476178659764), (0.12687336202193622, 0.052303295948191206), (-0.09430454247719171, 0.08651966838113309), (-0.09430454247719171, 0.08651966838113309)], [(2.0, 2.0), (-1.0431970971246272, -1.3162835942382718), (1.0112373369677947, 0.7853278705520694), (-0.19476147601830907, -0.4887022831093096), (0.04691925587326419, -0.11812494545691514), (0.06770843725647267, 0.1031303030460592), (0.00468157077450062, -0.09667757601710734), (0.00468157077450062, -0.09667757601710734)], [(2.0, 2.0), (-1.4832334273540664, 1.8264167120807142), (0.010656711702209958, 0.5282211357902047), (0.18919944896434113, -0.07267000594574827), (0.07681549011763708, -0.13713304270118076), (0.055050796066149786, -0.012043261964488862), (0.055050796066149786, -0.012043261964488862)], [(2.0, 2.0), (1.0548215642825145, -0.8187946674699075), (0.2359883900016888, -0.04708505389818333), (-0.17212562250056127, -0.12933033822625667), (-0.04112143724769097, -0.011072643049349488), (-0.04112143724769097, -0.011072643049349488)], [(2.0, 2.0), (-1.1142801791955343, -0.38199824445827724), (0.6458443828015042, -0.12646858482813178), (0.14117915836210093, -0.6354823054367144), (-0.04280188271227483, -0.5390524467483465), (-0.5124452038133245, 0.038637385647904206), (0.19406244612126436, 0.09489264208815747), (0.05574214650338575, 0.048311443091672554), (-0.012688852133630664, 0.028498450045270945), (-0.012688852133630664, 0.028498450045270945)], [(2.0, 2.0), (1.2053569856330164, -0.1582720784943877), (0.5537596915151112, 0.2213151328202144), (-0.06758837896000809, 0.15022735628207726), (-0.04634735860929966, 0.08266767400099762), (-0.04634735860929966, 0.08266767400099762)], [(2.0, 2.0), (0.796174007590585, 2.471569257261906), (0.052129225868331214, -0.2011908405992195), (0.04759936485487095, -0.04800553494589603), (0.04759936485487095, -0.04800553494589603)], [(2.0, 2.0), (0.3237604145032038, 0.9263520893948654), (-0.49585556617049353, 0.4382678306206567), (-0.04626283224666211, 0.11500177618792813), (0.04751655305887319, 0.050578584908096635), (0.04751655305887319, 0.050578584908096635)], [(2.0, 2.0), (0.24343264403622802, -1.0766276147143954), (-0.1903203606151546, -0.6403790559679711), (0.2880881151820605, -0.0991174259220615), (0.12927626566841813, -0.07924480142294815), (-0.0630666873707622, -0.1267795400375171), (-0.0630666873707622, -0.1267795400375171)]]"
        self.expected_all_intermediate_budgets = "[[0, 20, 30, 1000], [0, 20, 110, 200, 1000], [0, 20, 40, 50, 230, 600, 680, 1000], [0, 20, 30, 80, 570, 610, 1000], [0, 20, 50, 470, 780, 1000], [0, 20, 60, 220, 300, 410, 430, 450, 710, 1000], [0, 20, 30, 80, 420, 1000], [0, 20, 30, 450, 1000], [0, 20, 80, 220, 320, 1000], [0, 20, 30, 90, 100, 250, 1000]]"
        self.expected_all_est_objectives = "[[7.984539704940337, 2.1484830302293187, 0.03638265582209661, 0.03638265582209661], [8.081590387702734, 0.435328542983867, 0.10042287246052442, 0.09796938745134476, 0.09796938745134476], [7.9253347189439385, 2.7461974028540084, 1.5646755348875019, 0.20209667300101886, -0.05851036174516336, -0.05944498917397769, -0.06529681024660318, -0.06529681024660318], [8.073099810658121, 5.608879216846336, 0.352230944457918, 0.11417717191068556, 0.09780590158061683, 0.07627544096438224, 0.07627544096438224], [7.880122723414122, 1.6631959633666866, -0.06196975406969674, -0.0735237102789571, -0.11806370056046667, -0.11806370056046667], [8.025785950362149, 1.4133289268793845, 0.45889522010683287, 0.4495552656411394, 0.3181954918712409, 0.2898788848431312, 0.0724507968791864, 0.031227132792551425, 0.0267591189855986, 0.0267591189855986], [8.015084462897443, 1.4930199765426875, 0.37071464685958216, 0.04222091044338588, 0.02406648487223625, 0.02406648487223625], [7.994852045957048, 6.737399689762069, 0.03804725648769793, -0.0005777231229253443, -0.0005777231229253443], [7.910902809206077, 0.8738518087315944, 0.3488542430652729, -0.07373153262006339, -0.08428117472802223, -0.08428117472802223], [7.943417039435916, 1.1618035123839925, 0.389724214423027, 0.036236065666482194, -0.03359086914635138, -0.03653250173603942, -0.03653250173603942]]"
        self.expected_objective_curves = "[([0, 20, 30, 1000], [8.090508544469758, 2.1484830302293187, 0.03638265582209661, 0.03638265582209661]), ([0, 20, 110, 200, 1000], [8.090508544469758, 0.435328542983867, 0.10042287246052442, 0.09796938745134476, 0.09796938745134476]), ([0, 20, 40, 50, 230, 600, 680, 1000], [8.090508544469758, 2.7461974028540084, 1.5646755348875019, 0.20209667300101886, -0.05851036174516336, -0.05944498917397769, -0.06529681024660318, -0.06529681024660318]), ([0, 20, 30, 80, 570, 610, 1000], [8.090508544469758, 5.608879216846336, 0.352230944457918, 0.11417717191068556, 0.09780590158061683, 0.07627544096438224, 0.07627544096438224]), ([0, 20, 50, 470, 780, 1000], [8.090508544469758, 1.6631959633666866, -0.06196975406969674, -0.0735237102789571, -0.11806370056046667, -0.11806370056046667]), ([0, 20, 60, 220, 300, 410, 430, 450, 710, 1000], [8.090508544469758, 1.4133289268793845, 0.45889522010683287, 0.4495552656411394, 0.3181954918712409, 0.2898788848431312, 0.0724507968791864, 0.031227132792551425, 0.0267591189855986, 0.0267591189855986]), ([0, 20, 30, 80, 420, 1000], [8.090508544469758, 1.4930199765426875, 0.37071464685958216, 0.04222091044338588, 0.02406648487223625, 0.02406648487223625]), ([0, 20, 30, 450, 1000], [8.090508544469758, 6.737399689762069, 0.03804725648769793, -0.0005777231229253443, -0.0005777231229253443]), ([0, 20, 80, 220, 320, 1000], [8.090508544469758, 0.8738518087315944, 0.3488542430652729, -0.07373153262006339, -0.08428117472802223, -0.08428117472802223]), ([0, 20, 30, 90, 100, 250, 1000], [8.090508544469758, 1.1618035123839925, 0.389724214423027, 0.036236065666482194, -0.03359086914635138, -0.03653250173603942, -0.03653250173603942])]"
        self.expected_progress_curves = "[([0.0, 0.02, 0.03, 1.0], [1.0, 0.2655559929786994, 0.004496955367158701, 0.004496955367158701]), ([0.0, 0.02, 0.11, 0.2, 1.0], [1.0, 0.053807315150965936, 0.012412430183905827, 0.012109175450820261, 0.012109175450820261]), ([0.0, 0.02, 0.04, 0.05, 0.23, 0.6, 0.68, 1.0], [1.0, 0.3394344604865615, 0.19339643809622215, 0.024979477110763502, -0.007231975768094075, -0.007347497236698567, -0.008070791828189417, -0.008070791828189417]), ([0.0, 0.02, 0.03, 0.08, 0.57, 0.61, 1.0], [1.0, 0.6932665834313059, 0.043536316972149346, 0.014112483941288341, 0.012088968331597864, 0.009427768420876347, 0.009427768420876347]), ([0.0, 0.02, 0.05, 0.47, 0.78, 1.0], [1.0, 0.205573723113309, -0.007659562279562263, -0.00908765003767458, -0.014592865196486164, -0.014592865196486164]), ([0.0, 0.02, 0.06, 0.22, 0.3, 0.41, 0.43, 0.45, 0.71, 1.0], [1.0, 0.17468975146753427, 0.05672019472997272, 0.05556576118424985, 0.03932947973817325, 0.035829501106117365, 0.00895503619839941, 0.00385972434500383, 0.0033074705796942412, 0.0033074705796942412]), ([0.0, 0.02, 0.03, 0.08, 0.42, 1.0], [1.0, 0.18453969467262188, 0.045820932617762704, 0.005218573123224232, 0.0029746566288081882, 0.0029746566288081882]), ([0.0, 0.02, 0.03, 0.45, 1.0], [1.0, 0.8327535472868881, 0.0047027027137503015, -7.14075165670822e-05, -7.14075165670822e-05]), ([0.0, 0.02, 0.08, 0.22, 0.32, 1.0], [1.0, 0.10800950322571665, 0.043118951194202886, -0.009113337216665119, -0.010417290120240013, -0.010417290120240013]), ([0.0, 0.02, 0.03, 0.09, 0.1, 0.25, 1.0], [1.0, 0.14360080160574573, 0.04817054605169681, 0.004478836585773245, -0.004151885998478095, -0.004515476565562878, -0.004515476565562878])]"

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
