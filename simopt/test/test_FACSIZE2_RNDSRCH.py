import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class test_FACSIZE2_RNDSRCH(unittest.TestCase):
    def setUp(self):
        # Expected values
        self.expected_problem_name = "FACSIZE-2"
        self.expected_solver_name = "RNDSRCH"
        self.expected_all_recommended_xs = "[[(100, 100, 100), [115.6462517228025, 67.32898403527885, 192.94242761843483], [139.59332113978707, 132.72747399921406, 187.89481955629827], [139.59332113978707, 132.72747399921406, 187.89481955629827]], [(100, 100, 100), [85.51334282075412, 108.05785690341946, 159.85981322611704], [111.94696947582293, 221.24355205303496, 146.38163509019188], [193.07904193653746, 144.79805380524957, 103.75268316840702], [174.29099191271848, 177.04697100579972, 125.28524002044689], [202.61429556267652, 147.1833935040389, 129.27141166023293], [202.61429556267652, 147.1833935040389, 129.27141166023293]], [(100, 100, 100), [130.23611195131934, 127.5900994750533, 123.57708618604435], [143.38104154990444, 123.02091801267837, 209.77163574018053], [202.9846610317029, 125.0475615984511, 154.51891374307075], [149.98298922936942, 207.51450400869754, 101.79967309216315], [154.79572808777718, 217.56350057041462, 119.11991536080424], [183.20262236663734, 156.23222256458885, 154.11687899760685], [183.20262236663734, 156.23222256458885, 154.11687899760685]], [(100, 100, 100), [134.88010283444572, 140.23529311384567, 149.33646518317622], [129.7767278723324, 188.73960018573257, 179.88256817114868], [129.7767278723324, 188.73960018573257, 179.88256817114868]], [(100, 100, 100), [100.1269501462592, 131.88544228490724, 242.38508670034304], [126.8167221867196, 141.80684147771055, 207.59284826445213], [152.90481767249344, 171.68763773772605, 161.89283707032675], [152.90481767249344, 171.68763773772605, 161.89283707032675]], [(100, 100, 100), [167.0290317251437, 123.85734281556849, 84.39079789754139], [146.86641785041778, 166.3188830936159, 176.07253455163146], [146.86641785041778, 166.3188830936159, 176.07253455163146]], [(100, 100, 100), [135.62609423190068, 112.43768348056781, 206.21176690143704], [138.79616194628218, 193.19949664303462, 163.71978017355175], [138.79616194628218, 193.19949664303462, 163.71978017355175]], [(100, 100, 100), [81.39324547485334, 150.4271587563793, 241.51037836777013], [97.40375326014605, 223.38757767915172, 148.56049206587073], [151.54239268061187, 223.8076726328572, 115.71030005992911], [178.62339565848615, 172.09138960927007, 109.26936432440476], [153.0967413783358, 127.57417362542547, 139.45692719589937], [201.80272007709507, 147.0601377283476, 145.22338593529167], [201.80272007709507, 147.0601377283476, 145.22338593529167]], [(100, 100, 100), [195.09946102758124, 123.75768349543176, 115.28776946474234], [160.50667871846568, 136.86294748156638, 154.29282306994008], [141.35691451426555, 126.7568940216289, 221.52372917089042], [149.116179327519, 161.74594537428501, 165.8252220348562], [149.116179327519, 161.74594537428501, 165.8252220348562]], [(100, 100, 100), [141.85285929250406, 85.81791069594341, 201.91756449613095], [197.68606338619747, 110.78972172556963, 153.87472510476192], [197.68606338619747, 110.78972172556963, 153.87472510476192]]]"
        self.expected_all_intermediate_budgets = "[[0, 50, 170, 10000], [0, 110, 120, 280, 350, 1300, 10000], [0, 50, 620, 1130, 1840, 3710, 8680, 10000], [0, 30, 730, 10000], [0, 60, 4830, 6610, 10000], [0, 220, 690, 10000], [0, 30, 140, 10000], [0, 40, 70, 190, 240, 660, 4120, 10000], [0, 70, 160, 690, 4160, 10000], [0, 20, 540, 10000]]"
        self.expected_all_est_objectives = "[[0.275, 0.215, 0.685, 0.685], [0.24, 0.255, 0.535, 0.5, 0.645, 0.65, 0.65], [0.21, 0.525, 0.705, 0.7, 0.42, 0.595, 0.835, 0.835], [0.185, 0.62, 0.71, 0.71], [0.29, 0.545, 0.72, 0.85, 0.85], [0.25, 0.345, 0.81, 0.81], [0.235, 0.58, 0.72, 0.72], [0.3, 0.34, 0.495, 0.56, 0.56, 0.62, 0.76, 0.76], [0.28, 0.525, 0.74, 0.72, 0.83, 0.83], [0.2, 0.345, 0.525, 0.525]]"
        self.expected_objective_curves = "[([0, 50, 170, 10000], [0.25, 0.215, 0.685, 0.685]), ([0, 110, 120, 280, 350, 1300, 10000], [0.25, 0.255, 0.535, 0.5, 0.645, 0.65, 0.65]), ([0, 50, 620, 1130, 1840, 3710, 8680, 10000], [0.25, 0.525, 0.705, 0.7, 0.42, 0.595, 0.835, 0.835]), ([0, 30, 730, 10000], [0.25, 0.62, 0.71, 0.71]), ([0, 60, 4830, 6610, 10000], [0.25, 0.545, 0.72, 0.835, 0.835]), ([0, 220, 690, 10000], [0.25, 0.345, 0.81, 0.81]), ([0, 30, 140, 10000], [0.25, 0.58, 0.72, 0.72]), ([0, 40, 70, 190, 240, 660, 4120, 10000], [0.25, 0.34, 0.495, 0.56, 0.56, 0.62, 0.76, 0.76]), ([0, 70, 160, 690, 4160, 10000], [0.25, 0.525, 0.74, 0.72, 0.83, 0.83]), ([0, 20, 540, 10000], [0.25, 0.345, 0.525, 0.525])]"
        self.expected_progress_curves = "[([0.0, 0.005, 0.017, 1.0], [1.0, 1.0598290598290598, 0.2564102564102563, 0.2564102564102563]), ([0.0, 0.011, 0.012, 0.028, 0.035, 0.13, 1.0], [1.0, 0.9914529914529915, 0.5128205128205128, 0.5726495726495726, 0.32478632478632474, 0.31623931623931617, 0.31623931623931617]), ([0.0, 0.005, 0.062, 0.113, 0.184, 0.371, 0.868, 1.0], [1.0, 0.5299145299145298, 0.22222222222222224, 0.2307692307692308, 0.7094017094017094, 0.41025641025641024, -0.0, -0.0]), ([0.0, 0.003, 0.073, 1.0], [1.0, 0.3675213675213675, 0.2136752136752137, 0.2136752136752137]), ([0.0, 0.006, 0.483, 0.661, 1.0], [1.0, 0.49572649572649563, 0.19658119658119658, -0.0, -0.0]), ([0.0, 0.022, 0.069, 1.0], [1.0, 0.8376068376068376, 0.04273504273504258, 0.04273504273504258]), ([0.0, 0.003, 0.014, 1.0], [1.0, 0.43589743589743596, 0.19658119658119658, 0.19658119658119658]), ([0.0, 0.004, 0.007, 0.019, 0.024, 0.066, 0.412, 1.0], [1.0, 0.8461538461538461, 0.5811965811965811, 0.47008547008546997, 0.47008547008546997, 0.3675213675213675, 0.12820512820512814, 0.12820512820512814]), ([0.0, 0.007, 0.016, 0.069, 0.416, 1.0], [1.0, 0.5299145299145298, 0.16239316239316237, 0.19658119658119658, 0.008547008547008555, 0.008547008547008555]), ([0.0, 0.002, 0.054, 1.0], [1.0, 0.8376068376068376, 0.5299145299145298, 0.5299145299145298])]"

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
