import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class test_MM11_RNDSRCH(unittest.TestCase):
    def setUp(self):
        # Expected values
        self.expected_problem_name = "MM1-1"
        self.expected_solver_name = "RNDSRCH"
        self.expected_all_recommended_xs = "[[(5,), (4.296923990261382,), (2.377685478570012,), (2.9477762989744125,), (2.8725872351049095,), (2.772194393803471,), (2.772194393803471,)], [(5,), (3.985053300589926,), (2.5692082889849917,), (2.678966873705628,), (2.7655068232601367,), (2.7655068232601367,)], [(5,), (3.8484877158342767,), (3.0893992471426497,), (3.0474371736837815,), (2.699569241853814,), (2.8627075639581703,), (2.8627075639581703,)], [(5,), (3.773796218317722,), (2.638230326582391,), (2.9681054377092044,), (2.9600716575770796,), (2.9600716575770796,)], [(5,), (1.9121296882013312,), (2.865358591560582,), (2.865358591560582,)], [(5,), (3.4301202689575265,), (2.3471789635658644,), (2.971703077078345,), (2.7473614222541527,), (2.7473614222541527,)], [(5,), (3.061329038300266,), (2.6213500819625137,), (2.75793594287965,), (2.781142773961063,), (2.781142773961063,)], [(5,), (3.4557418828007354,), (2.9876013780902837,), (2.891632856275529,), (2.754896253073898,), (2.754896253073898,)], [(5,), (3.0560414166146552,), (2.798595332742974,), (2.798595332742974,)], [(5,), (2.0277986812387487,), (2.46722995472664,), (3.0783373963808143,), (2.8431466204775697,), (2.833368087907808,), (2.833368087907808,)]]"
        self.expected_all_intermediate_budgets = "[[0, 60, 130, 230, 620, 800, 1000], [0, 20, 240, 370, 990, 1000], [0, 80, 200, 370, 420, 690, 1000], [0, 40, 60, 120, 130, 1000], [0, 40, 70, 1000], [0, 20, 50, 150, 370, 1000], [0, 20, 200, 230, 420, 1000], [0, 80, 170, 320, 410, 1000], [0, 120, 340, 1000], [0, 40, 100, 240, 540, 890, 1000]]"
        self.expected_all_est_objectives = "[[2.7852321578182706, 2.2031388012791577, 1.6933515906330974, 1.5532659579862353, 1.5466636819482629, 1.5466569750020915, 1.5466569750020915], [2.7857037031168543, 1.9900956095535671, 1.5875356719512517, 1.559610708067732, 1.549585693229223, 1.549585693229223], [2.7866293625352507, 1.9102060479378624, 1.5927288694029482, 1.5844274565222343, 1.5767432376944044, 1.5651124409141386, 1.5651124409141386], [2.7889080044387127, 1.8699426487564068, 1.5888074513529902, 1.572898711985816, 1.5719529323204375, 1.5719529323204375], [2.7833651638972787, 2.716122200172547, 1.5430157510654368, 1.5430157510654368], [2.787955763524055, 1.7005668937245308, 1.7731170121688715, 1.5731787324099513, 1.5737336931676196, 1.5737336931676196], [2.7843462630059106, 1.5730671354449515, 1.567980237243296, 1.5471789891695702, 1.5460352551469974, 1.5460352551469974], [2.7907221687784363, 1.720987076480683, 1.587321657640274, 1.579083169733054, 1.5840464809440606, 1.5840464809440606], [2.789502875694011, 1.5850413713046327, 1.56332141744088, 1.56332141744088], [2.7891645344327056, 2.3911416272269825, 1.677984111037892, 1.5955139999950716, 1.5721201794950088, 1.572282353214191, 1.572282353214191]]"
        self.expected_objective_curves = "[([0, 60, 130, 230, 620, 800, 1000], [2.7854035060729516, 2.2031388012791577, 1.6933515906330974, 1.5532659579862353, 1.5466636819482629, 1.5466569750020915, 1.5466569750020915]), ([0, 20, 240, 370, 990, 1000], [2.7854035060729516, 1.9900956095535671, 1.5875356719512517, 1.559610708067732, 1.549585693229223, 1.549585693229223]), ([0, 80, 200, 370, 420, 690, 1000], [2.7854035060729516, 1.9102060479378624, 1.5927288694029482, 1.5844274565222343, 1.5767432376944044, 1.5651124409141386, 1.5651124409141386]), ([0, 40, 60, 120, 130, 1000], [2.7854035060729516, 1.8699426487564068, 1.5888074513529902, 1.572898711985816, 1.5719529323204375, 1.5719529323204375]), ([0, 40, 70, 1000], [2.7854035060729516, 2.716122200172547, 1.5532395188735884, 1.5532395188735884]), ([0, 20, 50, 150, 370, 1000], [2.7854035060729516, 1.7005668937245308, 1.7731170121688715, 1.5731787324099513, 1.5737336931676196, 1.5737336931676196]), ([0, 20, 200, 230, 420, 1000], [2.7854035060729516, 1.5730671354449515, 1.567980237243296, 1.5471789891695702, 1.5460352551469974, 1.5460352551469974]), ([0, 80, 170, 320, 410, 1000], [2.7854035060729516, 1.720987076480683, 1.587321657640274, 1.579083169733054, 1.5840464809440606, 1.5840464809440606]), ([0, 120, 340, 1000], [2.7854035060729516, 1.5850413713046327, 1.56332141744088, 1.56332141744088]), ([0, 40, 100, 240, 540, 890, 1000], [2.7854035060729516, 2.3911416272269825, 1.677984111037892, 1.5955139999950716, 1.5721201794950088, 1.572282353214191, 1.572282353214191])]"
        self.expected_progress_curves = "[([0.0, 0.06, 0.13, 0.23, 0.62, 0.8, 1.0], [1.0, 0.5274454448898093, 0.11371219514212193, 2.1457462579381675e-05, -0.0053368196065135595, -0.005342262831799323, -0.005342262831799323]), ([0.0, 0.02, 0.24, 0.37, 0.99, 1.0], [1.0, 0.35454379061420804, 0.02783408169201282, 0.005170731542499412, -0.0029653728580969012, -0.0029653728580969012]), ([0.0, 0.08, 0.2, 0.37, 0.42, 0.69, 1.0], [1.0, 0.2897069974229957, 0.0320487783603519, 0.025311515328032167, 0.01907515482110349, 0.009635829454435441, 0.009635829454435441]), ([0.0, 0.04, 0.06, 0.12, 0.13, 1.0], [1.0, 0.25703001643690804, 0.02886623278143817, 0.015955013550519224, 0.015187437420065876, 0.015187437420065876]), ([0.0, 0.04, 0.07, 1.0], [1.0, 0.943772658006442, 0.0, 0.0]), ([0.0, 0.02, 0.05, 0.15, 0.37, 1.0], [1.0, 0.1195679928820262, 0.178448238691874, 0.016182272606167978, 0.016632667816086136, 0.016632667816086136]), ([0.0, 0.02, 0.2, 0.23, 0.42, 1.0], [1.0, 0.016091702709499025, 0.011963276416812413, -0.004918606424939742, -0.005846838409038288, -0.005846838409038288]), ([0.0, 0.08, 0.17, 0.32, 0.41, 1.0], [1.0, 0.1361406106247067, 0.027660391896497764, 0.02097419753210514, 0.025002323059688438, 0.025002323059688438]), ([0.0, 0.12, 0.34, 1.0], [1.0, 0.025809756462147653, 0.00818227011341831, 0.00818227011341831]), ([0.0, 0.04, 0.1, 0.24, 0.54, 0.89, 1.0], [1.0, 0.6800248319689141, 0.1012402516712412, 0.034309135440300176, 0.01532317192968373, 0.01545478892293051, 0.01545478892293051])]"

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
