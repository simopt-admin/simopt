import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class test_CHESS1_RNDSRCH(unittest.TestCase):
    def setUp(self):
        # Expected values
        self.expected_problem_name = "CHESS-1"
        self.expected_solver_name = "RNDSRCH"
        self.expected_all_recommended_xs = "[[(150,), (143.9959354336992,), (113.6269068626152,), (105.08828691067168,), (98.32992367837494,), (93.19677707732481,), (93.19677707732481,)], [(150,), (119.27048503018092,), (103.10156276095827,), (102.53925144519908,), (101.07902970687569,), (101.07902970687569,)], [(150,), (122.72928023719989,), (114.51330244302284,), (112.07126704757705,), (109.70428450521894,), (104.1369744076302,), (104.1369744076302,)], [(150,), (104.66583726274527,), (102.56893879857294,), (96.3539568789001,), (96.3539568789001,)], [(150,), (146.40270006499802,), (131.76472018707562,), (115.58445281553676,), (115.12938585410117,), (110.31212977724317,), (103.8139402247456,), (97.81325466504936,), (97.81325466504936,)], [(150,), (127.32339425713342,), (123.0554345330125,), (113.56452280248605,), (108.12599267323786,), (97.4876059250656,), (97.4876059250656,)], [(150,), (143.3698978636649,), (116.87967131671496,), (113.7524445719657,), (96.50746113024549,), (96.50746113024549,)], [(150,), (120.73420496683445,), (104.51921544298852,), (100.42258968518165,), (97.20843481870226,), (97.20843481870226,)], [(150,), (101.05580426232791,), (98.8905279005489,), (98.8905279005489,)], [(150,), (129.56816424339834,), (120.18826149831811,), (115.11223524428766,), (113.13627700861365,), (102.72878653310667,), (99.39214831242,), (98.35893958209186,), (98.35893958209186,)]]"
        self.expected_all_intermediate_budgets = "[[0, 20, 40, 120, 150, 370, 1000], [0, 40, 60, 850, 890, 1000], [0, 40, 180, 190, 290, 480, 1000], [0, 30, 250, 270, 1000], [0, 40, 60, 120, 260, 270, 560, 760, 1000], [0, 40, 70, 220, 450, 460, 1000], [0, 40, 60, 100, 150, 1000], [0, 30, 100, 370, 470, 1000], [0, 20, 800, 1000], [0, 20, 70, 180, 200, 210, 530, 720, 1000]]"
        self.expected_all_est_objectives = "[[72.4911062284239, 69.57034983565977, 55.01381796625596, 50.78505218148534, 47.60439421153544, 45.12652173261736, 45.12652173261736], [72.64117621579616, 57.80677970323161, 49.94824303532387, 49.63654910064693, 48.91841750415076, 48.91841750415076], [72.72774151770889, 59.52506493995998, 55.52388854241122, 54.399243765169494, 53.298820525804075, 50.64659913321464, 50.64659913321464], [72.50861580800658, 50.59981601207795, 49.58102619153274, 46.556667424060485, 46.556667424060485], [72.54634030480123, 70.82067379040747, 63.7243991504278, 55.807699350008114, 55.64499322523014, 53.44118985591325, 50.28403778001126, 47.464369973424255, 47.464369973424255], [72.59953486291201, 61.49289196191016, 59.46794435318925, 54.95796084036429, 52.3830749849172, 47.23272463374439, 47.23272463374439], [72.42808293921915, 69.3284406551505, 56.60606152637405, 55.01629576741032, 46.60491022097211, 46.60491022097211], [72.64581320272977, 58.40002311404187, 50.64106652788252, 48.63992806740216, 47.03614893289617, 47.03614893289617], [72.53636458538912, 48.94828258981628, 47.85997353070289, 47.85997353070289], [72.5003673807578, 62.627246645457326, 58.100956747449096, 55.736507771453255, 54.725401586995865, 49.54212521241625, 47.95268303308411, 47.41682786429558, 47.41682786429558]]"
        self.expected_objective_curves = "[([0, 20, 40, 120, 150, 370, 1000], [72.52547100786995, 69.57034983565977, 55.01381796625596, 50.78505218148534, 47.60439421153544, 45.1104722136117, 45.1104722136117]), ([0, 40, 60, 850, 890, 1000], [72.52547100786995, 57.80677970323161, 49.94824303532387, 49.63654910064693, 48.91841750415076, 48.91841750415076]), ([0, 40, 180, 190, 290, 480, 1000], [72.52547100786995, 59.52506493995998, 55.52388854241122, 54.399243765169494, 53.298820525804075, 50.64659913321464, 50.64659913321464]), ([0, 30, 250, 270, 1000], [72.52547100786995, 50.59981601207795, 49.58102619153274, 46.556667424060485, 46.556667424060485]), ([0, 40, 60, 120, 260, 270, 560, 760, 1000], [72.52547100786995, 70.82067379040747, 63.7243991504278, 55.807699350008114, 55.64499322523014, 53.44118985591325, 50.28403778001126, 47.464369973424255, 47.464369973424255]), ([0, 40, 70, 220, 450, 460, 1000], [72.52547100786995, 61.49289196191016, 59.46794435318925, 54.95796084036429, 52.3830749849172, 47.23272463374439, 47.23272463374439]), ([0, 40, 60, 100, 150, 1000], [72.52547100786995, 69.3284406551505, 56.60606152637405, 55.01629576741032, 46.60491022097211, 46.60491022097211]), ([0, 30, 100, 370, 470, 1000], [72.52547100786995, 58.40002311404187, 50.64106652788252, 48.63992806740216, 47.03614893289617, 47.03614893289617]), ([0, 20, 800, 1000], [72.52547100786995, 48.94828258981628, 47.85997353070289, 47.85997353070289]), ([0, 20, 70, 180, 200, 210, 530, 720, 1000], [72.52547100786995, 62.627246645457326, 58.100956747449096, 55.736507771453255, 54.725401586995865, 49.54212521241625, 47.95268303308411, 47.41682786429558, 47.41682786429558])]"
        self.expected_progress_curves = "[([0.0, 0.02, 0.04, 0.12, 0.15, 0.37, 1.0], [1.0, 0.8922078678759928, 0.361238234112868, 0.20698815310771101, 0.09096925433555243, 0.0, 0.0]), ([0.0, 0.04, 0.06, 0.85, 0.89, 1.0], [1.0, 0.46311537654632334, 0.1764643820712253, 0.16509491468528403, 0.1389000714213633, 0.1389000714213633]), ([0.0, 0.04, 0.18, 0.19, 0.29, 0.48, 1.0], [1.0, 0.5257922071974429, 0.3798437638808318, 0.3388207900816402, 0.2986813303784397, 0.2019378866710882, 0.2019378866710882]), ([0.0, 0.03, 0.25, 0.27, 1.0], [1.0, 0.20023140761968333, 0.1630696397789865, 0.05275197060200759, 0.05275197060200759]), ([0.0, 0.04, 0.06, 0.12, 0.26, 0.27, 0.56, 0.76, 1.0], [1.0, 0.93781516350752, 0.6789687308217051, 0.39019615564005866, 0.38426122469225743, 0.30387444861191537, 0.18871295983726624, 0.08586167657631086, 0.08586167657631086]), ([0.0, 0.04, 0.07, 0.22, 0.45, 0.46, 1.0], [1.0, 0.5975714196175549, 0.5237086547887996, 0.3592007681873408, 0.26527824516368975, 0.07741209241187966, 0.07741209241187966]), ([0.0, 0.04, 0.06, 0.1, 0.15, 1.0], [1.0, 0.8833838959209062, 0.41931752027543273, 0.36132861533713717, 0.054511693346249585, 0.054511693346249585]), ([0.0, 0.03, 0.1, 0.37, 0.47, 1.0], [1.0, 0.4847547504986034, 0.20173607723919137, 0.12874178402406722, 0.07024172183030619, 0.07024172183030619]), ([0.0, 0.02, 0.8, 1.0], [1.0, 0.1399894417288234, 0.10029186350601037, 0.10029186350601037]), ([0.0, 0.02, 0.07, 0.18, 0.2, 0.21, 0.53, 0.72, 1.0], [1.0, 0.6389485756794674, 0.4738458911243177, 0.387599344343836, 0.3507178477570423, 0.16165067275993128, 0.10367357083625602, 0.08412751238810616, 0.08412751238810616])]"

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
