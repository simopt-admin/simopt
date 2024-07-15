import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class test_DYNAMNEWS1_NELDMD(unittest.TestCase):
    def setUp(self):
        # Expected values
        self.expected_problem_name = "DYNAMNEWS-1"
        self.expected_solver_name = "NELDMD"
        self.expected_all_recommended_xs = "[[(3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0), (3.788628550587098, 2.9690270305893973, 3.6618743849995004, 1.3708918586738332, 1.8337725612056026, 5.153673502821542, 1.6739925431977731, 2.5108215209525153, 1.0000000005838672e-07, 2.971872975456893), (3.788628550587098, 2.9690270305893973, 3.6618743849995004, 1.3708918586738332, 1.8337725612056026, 5.153673502821542, 1.6739925431977731, 2.5108215209525153, 1.0000000005838672e-07, 2.971872975456893)], [(3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0), (2.7596227692266817, 2.2539689940688, 3.7574242021156183, 1.8315381898254686, 0.8166826851810021, 2.9622061821551453, 3.07201957059754, 1.0000000005838672e-07, 4.884884709493207, 2.4370248289431746), (2.7596227692266817, 2.2539689940688, 3.7574242021156183, 1.8315381898254686, 0.8166826851810021, 2.9622061821551453, 3.07201957059754, 1.0000000005838672e-07, 4.884884709493207, 2.4370248289431746)], [(3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0), (3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0)], [(3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0), (0, 2.52050093909559, 5.155234798695307, 4.737329032448124, 5.447280848946661, 1.216542196206054, 1.2877324686884093, 5.164962716896884, 0.9069141085212036, 0.11753358904933564), (0, 2.52050093909559, 5.155234798695307, 4.737329032448124, 5.447280848946661, 1.216542196206054, 1.2877324686884093, 5.164962716896884, 0.9069141085212036, 0.11753358904933564)], [(3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0), (1.4670575882704087, 1.3319794877658824, 4.692800272497435, 0.45053594686274, 1.0000000000287557e-07, 3.319077232124695, 3.0256766422664474, 1.8007354005986675, 2.788753652105876, 3.8654810852166546), (1.4670575882704087, 1.3319794877658824, 4.692800272497435, 0.45053594686274, 1.0000000000287557e-07, 3.319077232124695, 3.0256766422664474, 1.8007354005986675, 2.788753652105876, 3.8654810852166546)], [(3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0), (3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0)], [(3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0), (4.053782719829777, 3.6994943782067593, 2.698288209269849, 0.3506753755802086, 4.696737048265246, 0.913611555858698, 3.291140292113798, 4.204993628948301, 1.0000000000287557e-07, 3.6796593676571114), (4.053782719829777, 3.6994943782067593, 2.698288209269849, 0.3506753755802086, 4.696737048265246, 0.913611555858698, 3.291140292113798, 4.204993628948301, 1.0000000000287557e-07, 3.6796593676571114)], [(3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0), (3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0)], [(3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0), (2.6216171987786967, 4.2153282206462865, 5.307706078205278, 2.542785255336476, 4.731713533416262, 1.0000000000287557e-07, 4.0796439387078856, 0.938396798989215, 1.4908806665590884, 1.0790692460209108), (2.6216171987786967, 4.2153282206462865, 5.307706078205278, 2.542785255336476, 4.731713533416262, 1.0000000000287557e-07, 4.0796439387078856, 0.938396798989215, 1.4908806665590884, 1.0790692460209108)], [(3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0), (3.782046916200016, 1.0000000000287557e-07, 5.519650989287856, 1.0407439804824272, 4.357126866728561, 4.188607403380507, 0.580472079356594, 0.6110250987218866, 6.3758161364705925, 1.2728806756792095), (3.782046916200016, 1.0000000000287557e-07, 5.519650989287856, 1.0407439804824272, 4.357126866728561, 4.188607403380507, 0.580472079356594, 0.6110250987218866, 6.3758161364705925, 1.2728806756792095)]]"
        self.expected_all_intermediate_budgets = "[[0, 990, 1000], [0, 690, 1000], [0, 1000], [0, 720, 1000], [0, 690, 1000], [0, 1000], [0, 750, 1000], [0, 1000], [0, 690, 1000], [0, 810, 1000]]"
        self.expected_all_est_objectives = "[[120.0, 140.32722485757924, 140.32722485757924], [120.0, 137.12313884196683, 137.12313884196683], [120.0, 120.0], [120.0, 137.22984650726218, 137.22984650726218], [120.0, 138.28951296145598, 138.28951296145598], [120.0, 120.0], [120.0, 132.0580866213513, 132.0580866213513], [120.0, 120.0], [120.0, 134.96429481669952, 134.96429481669952], [120.0, 131.35814876846172, 131.35814876846172]]"
        self.expected_objective_curves = "[([0, 990, 1000], [120.0, 140.32722485757924, 140.32722485757924]), ([0, 690, 1000], [120.0, 137.12313884196683, 137.12313884196683]), ([0, 1000], [120.0, 120.0]), ([0, 720, 1000], [120.0, 137.22984650726218, 137.22984650726218]), ([0, 690, 1000], [120.0, 138.28951296145598, 138.28951296145598]), ([0, 1000], [120.0, 120.0]), ([0, 750, 1000], [120.0, 132.0580866213513, 132.0580866213513]), ([0, 1000], [120.0, 120.0]), ([0, 690, 1000], [120.0, 134.96429481669952, 134.96429481669952]), ([0, 810, 1000], [120.0, 131.35814876846172, 131.35814876846172])]"
        self.expected_progress_curves = "[([0.0, 0.99, 1.0], [1.0, -0.0, -0.0]), ([0.0, 0.69, 1.0], [1.0, 0.1576253540786569, 0.1576253540786569]), ([0.0, 1.0], [1.0, 1.0]), ([0.0, 0.72, 1.0], [1.0, 0.15237585907661, 0.15237585907661]), ([0.0, 0.69, 1.0], [1.0, 0.10024545457632753, 0.10024545457632753]), ([0.0, 1.0], [1.0, 1.0]), ([0.0, 0.75, 1.0], [1.0, 0.4068011395635594, 0.4068011395635594]), ([0.0, 1.0], [1.0, 1.0]), ([0.0, 0.69, 1.0], [1.0, 0.26382991669815115, 0.26382991669815115]), ([0.0, 0.81, 1.0], [1.0, 0.4412346570650198, 0.4412346570650198])]"

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
