import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class test_CNTNEWS1_RNDSRCH(unittest.TestCase):
    def setUp(self):
        # Expected values
        self.expected_problem_name = "CNTNEWS-1"
        self.expected_solver_name = "RNDSRCH"
        self.expected_all_recommended_xs = "[[(0,), (0.08637653581125314,), (0.16201092394781372,), (0.220031799278404,), (0.18069759039261452,), (0.18069759039261452,)], [(0,), (0.3422671900811551,), (0.25411599712618166,), (0.19694301248184054,), (0.24037175375360612,), (0.23334625077738846,)], [(0,), (0.16067182294009807,), (0.27075075921979186,), (0.21701449798421404,), (0.21701449798421404,)], [(0,), (0.07150266276670732,), (0.16380950044050657,), (0.17407003882377992,), (0.1922372233210708,), (0.1922372233210708,)], [(0,), (0.23124098139956537,), (0.12808012265783883,), (0.15379949965266126,), (0.14781215577002094,), (0.14781215577002094,)], [(0,), (0.1422314185632828,), (0.14837586376590187,), (0.3022077181845406,), (0.20663001343178547,), (0.2576797005148086,), (0.2561104719278581,), (0.2561104719278581,)], [(0,), (0.09590415802387352,), (0.1800190952064003,), (0.1800190952064003,)], [(0,), (0.041719078922958744,), (0.054813625384861384,), (0.24350561539939186,), (0.22592668331808533,), (0.22592668331808533,)], [(0,), (0.05706434183349318,), (0.22615291294721201,), (0.2623222684966682,), (0.2623222684966682,)], [(0,), (0.15178070996172968,), (0.16594898609478428,), (0.16594898609478428,)]]"
        self.expected_all_intermediate_budgets = "[[0, 20, 80, 520, 940, 1000], [0, 30, 40, 370, 700, 1000], [0, 20, 120, 230, 1000], [0, 20, 60, 130, 290, 1000], [0, 30, 60, 620, 640, 1000], [0, 20, 50, 70, 160, 510, 940, 1000], [0, 60, 120, 1000], [0, 70, 80, 90, 130, 1000], [0, 50, 70, 840, 1000], [0, 30, 340, 1000]]"
        self.expected_all_est_objectives = "[[0.0, 0.3152152013202121, 0.4439426039369449, 0.42568320017429656, 0.44830230926972503, 0.44830230926972503], [0.0, 0.23176170058690984, 0.4460923882748864, 0.5055759457041912, 0.4655738923120514, 0.4742936269626158], [0.0, 0.43898245555866006, 0.33349582987942683, 0.42509600051400326, 0.42509600051400326], [0.0, 0.27316144251245833, 0.4856172195128214, 0.49446071587216717, 0.5000004738655908, 0.5000004738655908], [0.0, 0.3650021020471091, 0.38761217452342295, 0.41162594195133495, 0.40859229991965795, 0.40859229991965795], [0.0, 0.44649448526660557, 0.45489213464277095, 0.32355753197282283, 0.4837049698664353, 0.4280386926410918, 0.4308645786107479, 0.4308645786107479], [0.0, 0.34673483336643507, 0.4748257480816718, 0.4748257480816718], [0.0, 0.1631811912751526, 0.21181775198412514, 0.3929077012672134, 0.4169803668262306, 0.4169803668262306], [0.0, 0.21515166984373782, 0.4121027610273431, 0.34835283694400165, 0.34835283694400165], [0.0, 0.4149320032019911, 0.42616525220425006, 0.42616525220425006]]"
        self.expected_objective_curves = "[([0, 20, 80, 520, 940, 1000], [0.0, 0.3152152013202121, 0.4439426039369449, 0.42568320017429656, 0.44830230926972503, 0.44830230926972503]), ([0, 30, 40, 370, 700, 1000], [0.0, 0.23176170058690984, 0.4460923882748864, 0.5095842940352483, 0.4655738923120514, 0.4742936269626158]), ([0, 20, 120, 230, 1000], [0.0, 0.43898245555866006, 0.33349582987942683, 0.42509600051400326, 0.42509600051400326]), ([0, 20, 60, 130, 290, 1000], [0.0, 0.27316144251245833, 0.4856172195128214, 0.49446071587216717, 0.5000004738655908, 0.5000004738655908]), ([0, 30, 60, 620, 640, 1000], [0.0, 0.3650021020471091, 0.38761217452342295, 0.41162594195133495, 0.40859229991965795, 0.40859229991965795]), ([0, 20, 50, 70, 160, 510, 940, 1000], [0.0, 0.44649448526660557, 0.45489213464277095, 0.32355753197282283, 0.4837049698664353, 0.4280386926410918, 0.4308645786107479, 0.4308645786107479]), ([0, 60, 120, 1000], [0.0, 0.34673483336643507, 0.4748257480816718, 0.4748257480816718]), ([0, 70, 80, 90, 130, 1000], [0.0, 0.1631811912751526, 0.21181775198412514, 0.3929077012672134, 0.4169803668262306, 0.4169803668262306]), ([0, 50, 70, 840, 1000], [0.0, 0.21515166984373782, 0.4121027610273431, 0.34835283694400165, 0.34835283694400165]), ([0, 30, 340, 1000], [0.0, 0.4149320032019911, 0.42616525220425006, 0.42616525220425006])]"
        self.expected_progress_curves = "[([0.0, 0.02, 0.08, 0.52, 0.94, 1.0], [1.0, 0.38142677274427844, 0.12881419397467322, 0.16464615342942307, 0.12025877854328915, 0.12025877854328915]), ([0.0, 0.03, 0.04, 0.37, 0.7, 1.0], [1.0, 0.5451945766388187, 0.12459549186178434, -0.0, 0.08636530253845047, 0.0692538358927354]), ([0.0, 0.02, 0.12, 0.23, 1.0], [1.0, 0.13854790915456405, 0.34555316209106185, 0.1657984645723813, 0.1657984645723813]), ([0.0, 0.02, 0.06, 0.13, 0.29, 1.0], [1.0, 0.463952390782351, 0.04703260049998537, 0.029678265872996122, 0.018807134132345554, 0.018807134132345554]), ([0.0, 0.03, 0.06, 0.62, 0.64, 1.0], [1.0, 0.2837257617247881, 0.2393561201542614, 0.19223189024961884, 0.19818506044576928, 0.19818506044576928]), ([0.0, 0.02, 0.05, 0.07, 0.16, 0.51, 0.94, 1.0], [1.0, 0.12380642321028593, 0.10732701151243537, 0.36505591761734685, 0.05078516836514375, 0.16002377300214818, 0.15447829995140178, 0.15447829995140178]), ([0.0, 0.06, 0.12, 1.0], [1.0, 0.31957315516782553, 0.06820960999079036, 0.06820960999079036]), ([0.0, 0.07, 0.08, 0.09, 0.13, 1.0], [1.0, 0.6797758620404708, 0.5843322597193831, 0.22896426387891045, 0.18172445323170075, 0.18172445323170075]), ([0.0, 0.05, 0.07, 0.84, 1.0], [1.0, 0.5777898330813633, 0.19129618818503524, 0.31639801104249526, 0.31639801104249526]), ([0.0, 0.03, 0.34, 1.0], [1.0, 0.18574412897174192, 0.16370018230041464, 0.16370018230041464])]"

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
