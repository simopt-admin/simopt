import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class test_CNTNEWS1_ADAM(unittest.TestCase):
    def setUp(self):
        # Expected values
        self.expected_problem_name = "CNTNEWS-1"
        self.expected_solver_name = "ADAM"
        self.expected_all_recommended_xs = "[[(0,), (0.3050233636669336,), (0.2185317126338197,), (0.2115705602775329,), (0.2034744970188075,), (0.20535476709242967,), (0.20535476709242967,)], [(0,), (0.2725771987747624,), (0.2034732984953683,), (0.1795409828609915,), (0.16317885111803027,), (0.16317885111803027,)], [(0,), (0.2806815845546738,), (0.27640546292420576,), (0.2588670862825194,), (0.21583603286051675,), (0.22102581271128185,), (0.22102581271128185,)], [(0,), (0.3050233636669336,), (0.05516878405414677,), (0.0632553264439476,), (0.20685036996348966,), (0.158589812817196,), (0.1643735906708466,), (0.1643735906708466,)], [(0,), (0.3050233636669336,), (0.04805048595026229,), (0.06867616073995278,), (0.2248357885029174,), (0.22158598672334645,), (0.10862099391658879,), (0.1777928651002421,), (0.12582058057141132,), (0.16886924231685357,), (0.15915350778089685,), (0.14217170001851342,), (0.1446770220893471,), (0.1446770220893471,)], [(0,), (0.3050233636669336,), (0.04805048595026229,), (0.16569355019956494,), (0.13887261336221146,), (0.1434989459659648,), (0.1434989459659648,)], [(0,), (0.2725771987747624,), (0.22306227804941628,), (0.15230777951704738,), (0.20387067040674817,), (0.18093045387729906,), (0.18093045387729906,)], [(0,), (0.3050233636669336,), (0.22468852268986378,), (0.22900671490707683,), (0.22900671490707683,)], [(0,), (0.2725771987747624,), (0.21759199014126873,), (0.26097962117772194,), (0.22192926958495923,), (0.22192926958495923,)], [(0,), (0.2725771987747624,), (0.088601251460549,), (0.24293162791944506,), (0.20857393715254818,), (0.1838878244881581,), (0.1675013793965652,), (0.16547008519472642,), (0.16547008519472642,)]]"
        self.expected_all_intermediate_budgets = "[[0, 120, 240, 360, 480, 990, 1000], [0, 120, 240, 360, 870, 1000], [0, 120, 240, 570, 690, 810, 1000], [0, 120, 150, 210, 240, 360, 690, 1000], [0, 120, 150, 210, 240, 330, 360, 450, 570, 660, 750, 840, 960, 1000], [0, 120, 150, 240, 360, 450, 1000], [0, 120, 240, 450, 570, 810, 1000], [0, 120, 240, 480, 1000], [0, 120, 240, 510, 630, 1000], [0, 120, 210, 240, 360, 480, 630, 990, 1000]]"
        self.expected_all_est_objectives = "[[0.0, 0.2866944995117209, 0.4270040211263273, 0.432962800065976, 0.4389898785373815, 0.43770004290136166, 0.43770004290136166], [0.0, 0.41288144547221917, 0.5028310521215722, 0.5030272191109962, 0.4919194521060183, 0.4919194521060183], [0.0, 0.3084932225679554, 0.3194851879256968, 0.3599298190036173, 0.42641588145254405, 0.4203351568187604, 0.4203351568187604], [0.0, 0.33367362050453614, 0.21375997803437585, 0.24361252286227952, 0.4976984022484743, 0.48009844774773397, 0.48615874613394794, 0.48615874613394794], [0.0, 0.20452682706858255, 0.1863253400472016, 0.25396435174168486, 0.37485167137452247, 0.37979137007947017, 0.35366879069931345, 0.4127187348063703, 0.38441236963740516, 0.4142319251871544, 0.4131756712983314, 0.40418969723550285, 0.4062792513744203, 0.4062792513744203], [0.0, 0.31589897626071384, 0.18979268131943974, 0.4727909006761207, 0.44132901687544773, 0.448336501970893, 0.448336501970893], [0.0, 0.37611932004794185, 0.45736623665394505, 0.45779123160934304, 0.4730057444140837, 0.47505162977475124, 0.47505162977475124], [0.0, 0.26516847939808513, 0.41836710672983884, 0.41344178356794214, 0.41344178356794214], [0.0, 0.3259754261383471, 0.42296778632489923, 0.3511455433674098, 0.417807809989936, 0.417807809989936], [0.0, 0.32811930952150914, 0.3056856268769611, 0.3819612974960285, 0.42230304506457644, 0.43107548310011623, 0.4268483052570336, 0.42595453580822457, 0.42595453580822457]]"
        self.expected_objective_curves = "[([0, 120, 240, 360, 480, 990, 1000], [0.0, 0.2866944995117209, 0.4270040211263273, 0.432962800065976, 0.4389898785373815, 0.43770004290136166, 0.43770004290136166]), ([0, 120, 240, 360, 870, 1000], [0.0, 0.41288144547221917, 0.5028310521215722, 0.5078709170979363, 0.4919194521060183, 0.4919194521060183]), ([0, 120, 240, 570, 690, 810, 1000], [0.0, 0.3084932225679554, 0.3194851879256968, 0.3599298190036173, 0.42641588145254405, 0.4203351568187604, 0.4203351568187604]), ([0, 120, 150, 210, 240, 360, 690, 1000], [0.0, 0.33367362050453614, 0.21375997803437585, 0.24361252286227952, 0.4976984022484743, 0.48009844774773397, 0.48615874613394794, 0.48615874613394794]), ([0, 120, 150, 210, 240, 330, 360, 450, 570, 660, 750, 840, 960, 1000], [0.0, 0.20452682706858255, 0.1863253400472016, 0.25396435174168486, 0.37485167137452247, 0.37979137007947017, 0.35366879069931345, 0.4127187348063703, 0.38441236963740516, 0.4142319251871544, 0.4131756712983314, 0.40418969723550285, 0.4062792513744203, 0.4062792513744203]), ([0, 120, 150, 240, 360, 450, 1000], [0.0, 0.31589897626071384, 0.18979268131943974, 0.4727909006761207, 0.44132901687544773, 0.448336501970893, 0.448336501970893]), ([0, 120, 240, 450, 570, 810, 1000], [0.0, 0.37611932004794185, 0.45736623665394505, 0.45779123160934304, 0.4730057444140837, 0.47505162977475124, 0.47505162977475124]), ([0, 120, 240, 480, 1000], [0.0, 0.26516847939808513, 0.41836710672983884, 0.41344178356794214, 0.41344178356794214]), ([0, 120, 240, 510, 630, 1000], [0.0, 0.3259754261383471, 0.42296778632489923, 0.3511455433674098, 0.417807809989936, 0.417807809989936]), ([0, 120, 210, 240, 360, 480, 630, 990, 1000], [0.0, 0.32811930952150914, 0.3056856268769611, 0.3819612974960285, 0.42230304506457644, 0.43107548310011623, 0.4268483052570336, 0.42595453580822457, 0.42595453580822457])]"
        self.expected_progress_curves = "[([0.0, 0.12, 0.24, 0.36, 0.48, 0.99, 1.0], [1.0, 0.4354973087454118, 0.15922726277317997, 0.14749440164835284, 0.13562705845444586, 0.13816675031825676, 0.13816675031825676]), ([0.0, 0.12, 0.24, 0.36, 0.87, 1.0], [1.0, 0.18703467441786917, 0.00992351561527244, -0.0, 0.03140850254444081, 0.03140850254444081]), ([0.0, 0.12, 0.24, 0.57, 0.69, 0.81, 1.0], [1.0, 0.3925755301549064, 0.3709323035245032, 0.2912966525818813, 0.16038531229714953, 0.17235828501338615, 0.17235828501338615]), ([0.0, 0.12, 0.15, 0.21, 0.24, 0.36, 0.69, 1.0], [1.0, 0.34299521931437643, 0.5791056923364741, 0.5203259043571066, 0.02002972508760604, 0.05468411049976853, 0.04275135715204077, 0.04275135715204077]), ([0.0, 0.12, 0.15, 0.21, 0.24, 0.33, 0.36, 0.45, 0.57, 0.66, 0.75, 0.84, 0.96, 1.0], [1.0, 0.597285805934144, 0.6331246114428104, 0.49994310918041573, 0.26191546167579194, 0.25218917387578554, 0.3036246439937158, 0.1873550524122986, 0.2430904060543474, 0.18437557410424593, 0.1864553425124443, 0.20414876373486035, 0.20003442273093458, 0.20003442273093458]), ([0.0, 0.12, 0.15, 0.24, 0.36, 0.45, 1.0], [1.0, 0.37799356957508784, 0.626297401702093, 0.06907270182405602, 0.13102128509882133, 0.11722351708428874, 0.11722351708428874]), ([0.0, 0.12, 0.24, 0.45, 0.57, 0.81, 1.0], [1.0, 0.2594194560358885, 0.09944393101417164, 0.09860711413592531, 0.06864967358847462, 0.06462131659501245, 0.06462131659501245]), ([0.0, 0.12, 0.24, 0.48, 1.0], [1.0, 0.47788213407984764, 0.17623338402509434, 0.18593136631957355, 0.18593136631957355]), ([0.0, 0.12, 0.24, 0.51, 0.63, 1.0], [1.0, 0.3581529968263826, 0.1671746262971475, 0.3085929287427657, 0.17733464168934251, 0.17733464168934251]), ([0.0, 0.12, 0.21, 0.24, 0.36, 0.48, 0.63, 0.99, 1.0], [1.0, 0.3539316813090213, 0.39810369803472406, 0.24791657754568333, 0.16848350467144235, 0.15121053679671734, 0.15953386798338473, 0.1612937038367866, 0.1612937038367866])]"

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
