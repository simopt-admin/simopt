import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class test_CNTNEWS1_ASTRODF(unittest.TestCase):
    def setUp(self):
        # Expected values
        self.expected_problem_name = "CNTNEWS-1"
        self.expected_solver_name = "ASTRODF"
        self.expected_all_recommended_xs = "[[(0,), (0.1839393916800366,), (0.1839393916800366,)], [(0,), (0.3276936701871986,), (0.3276936701871986,)], [(0,), (0.2186816038882345,), (0.2186816038882345,)], [(0,), (0.1815451365366126,), (0.1815451365366126,)], [(0,), (0.2295649318636742,), (0.2295649318636742,)], [(0,), (0.21460552673782216,), (0.21460552673782216,)], [(0,), (0.16040227435398843,), (0.16040227435398843,)], [(0,), (0.262806097725263,), (0.262806097725263,)], [(0,), (0.13526240989754468,), (0.13526240989754468,)], [(0,), (0.2062607994280207,), (0.2062607994280207,)], [(0,), (0.25884442898145643,), (0.25884442898145643,)], [(0,), (0.2572709520577886,), (0.2572709520577886,)], [(0,), (0.2188195525709549,), (0.2188195525709549,)], [(0,), (0.1426064204577237,), (0.1426064204577237,)], [(0,), (0.19113748154111243,), (0.19113748154111243,)], [(0,), (0.1606965664173894,), (0.1606965664173894,)], [(0,), (0.19040247791908904,), (0.19040247791908904,)], [(0,), (0.2589188815763174,), (0.2589188815763174,)], [(0,), (0.3002432873798844,), (0.3002432873798844,)], [(0,), (0.2099852631233528,), (0.2099852631233528,)], [(0,), (0.17749355708643055,), (0.17749355708643055,)], [(0,), (0.20084818544361496,), (0.20084818544361496,)], [(0,), (0.1712826184611227,), (0.1712826184611227,)], [(0,), (0.2509023715632027,), (0.2509023715632027,)]]"
        self.expected_all_intermediate_budgets = "[[4, 40, 1000], [4, 32, 1000], [4, 16, 1000], [4, 40, 1000], [4, 16, 1000], [4, 40, 1000], [4, 16, 1000], [4, 32, 1000], [4, 16, 1000], [4, 16, 1000], [4, 16, 1000], [4, 32, 1000], [4, 16, 1000], [4, 40, 1000], [4, 16, 1000], [4, 40, 1000], [4, 40, 1000], [4, 16, 1000], [4, 16, 1000], [4, 16, 1000], [4, 16, 1000], [4, 16, 1000], [4, 40, 1000], [4, 16, 1000]]"
        self.expected_all_est_objectives = "[[0.0, 0.4480283071454626, 0.4480283071454626], [0.0, 0.27678994295105985, 0.27678994295105985], [0.0, 0.42320897082332876, 0.42320897082332876], [0.0, 0.4978219390578274, 0.4978219390578274], [0.0, 0.3676167393230994, 0.3676167393230994], [0.0, 0.48059316259077356, 0.48059316259077356], [0.0, 0.46571047244921915, 0.46571047244921915], [0.0, 0.3609680118634468, 0.3609680118634468], [0.0, 0.4085048038485643, 0.4085048038485643], [0.0, 0.42402664462770145, 0.42402664462770145], [0.0, 0.44518563758033963, 0.44518563758033963], [0.0, 0.35133550490068055, 0.35133550490068055], [0.0, 0.46483837922754156, 0.46483837922754156], [0.0, 0.43814190567347366, 0.43814190567347366], [0.0, 0.4405707246589916, 0.4405707246589916], [0.0, 0.442081763999216, 0.442081763999216], [0.0, 0.4329350934718565, 0.4329350934718565], [0.0, 0.34455117729680973, 0.34455117729680973], [0.0, 0.3513720053257927, 0.3513720053257927], [0.0, 0.4922939506310095, 0.4922939506310095], [0.0, 0.4490218509239012, 0.4490218509239012], [0.0, 0.4699547418333453, 0.4699547418333453], [0.0, 0.45957940145382636, 0.45957940145382636], [0.0, 0.40030465919770875, 0.40030465919770875]]"
        self.expected_objective_curves = "[([4, 40, 1000], [0.0, 0.4480283071454626, 0.4480283071454626]), ([4, 32, 1000], [0.0, 0.27678994295105985, 0.27678994295105985]), ([4, 16, 1000], [0.0, 0.42320897082332876, 0.42320897082332876]), ([4, 40, 1000], [0.0, 0.5087018446197824, 0.5087018446197824]), ([4, 16, 1000], [0.0, 0.3676167393230994, 0.3676167393230994]), ([4, 40, 1000], [0.0, 0.48059316259077356, 0.48059316259077356]), ([4, 16, 1000], [0.0, 0.46571047244921915, 0.46571047244921915]), ([4, 32, 1000], [0.0, 0.3609680118634468, 0.3609680118634468]), ([4, 16, 1000], [0.0, 0.4085048038485643, 0.4085048038485643]), ([4, 16, 1000], [0.0, 0.42402664462770145, 0.42402664462770145]), ([4, 16, 1000], [0.0, 0.44518563758033963, 0.44518563758033963]), ([4, 32, 1000], [0.0, 0.35133550490068055, 0.35133550490068055]), ([4, 16, 1000], [0.0, 0.46483837922754156, 0.46483837922754156]), ([4, 40, 1000], [0.0, 0.43814190567347366, 0.43814190567347366]), ([4, 16, 1000], [0.0, 0.4405707246589916, 0.4405707246589916]), ([4, 40, 1000], [0.0, 0.442081763999216, 0.442081763999216]), ([4, 40, 1000], [0.0, 0.4329350934718565, 0.4329350934718565]), ([4, 16, 1000], [0.0, 0.34455117729680973, 0.34455117729680973]), ([4, 16, 1000], [0.0, 0.3513720053257927, 0.3513720053257927]), ([4, 16, 1000], [0.0, 0.4922939506310095, 0.4922939506310095]), ([4, 16, 1000], [0.0, 0.4490218509239012, 0.4490218509239012]), ([4, 16, 1000], [0.0, 0.4699547418333453, 0.4699547418333453]), ([4, 40, 1000], [0.0, 0.45957940145382636, 0.45957940145382636]), ([4, 16, 1000], [0.0, 0.40030465919770875, 0.40030465919770875])]"
        self.expected_progress_curves = "[([0.004, 0.04, 1.0], [1.0, 0.1192713140634842, 0.1192713140634842]), ([0.004, 0.032, 1.0], [1.0, 0.4558896416860053, 0.4558896416860053]), ([0.004, 0.016, 1.0], [1.0, 0.1680608684648143, 0.1680608684648143]), ([0.004, 0.04, 1.0], [1.0, -0.0, -0.0]), ([0.004, 0.016, 1.0], [1.0, 0.27734341203762264, 0.27734341203762264]), ([0.004, 0.04, 1.0], [1.0, 0.05525571083788391, 0.05525571083788391]), ([0.004, 0.016, 1.0], [1.0, 0.08451192506033899, 0.08451192506033899]), ([0.004, 0.032, 1.0], [1.0, 0.2904134009318482, 0.2904134009318482]), ([0.004, 0.016, 1.0], [1.0, 0.19696614398186052, 0.19696614398186052]), ([0.004, 0.016, 1.0], [1.0, 0.1664534950828997, 0.1664534950828997]), ([0.004, 0.016, 1.0], [1.0, 0.12485939988465447, 0.12485939988465447]), ([0.004, 0.032, 1.0], [1.0, 0.30934886787509436, 0.30934886787509436]), ([0.004, 0.016, 1.0], [1.0, 0.08622627548171284, 0.08622627548171284]), ([0.004, 0.04, 1.0], [1.0, 0.13870588379534413, 0.13870588379534413]), ([0.004, 0.016, 1.0], [1.0, 0.13393134049221672, 0.13393134049221672]), ([0.004, 0.04, 1.0], [1.0, 0.13096095743541103, 0.13096095743541103]), ([0.004, 0.04, 1.0], [1.0, 0.14894137292652446, 0.14894137292652446]), ([0.004, 0.016, 1.0], [1.0, 0.32268541791049205, 0.32268541791049205]), ([0.004, 0.016, 1.0], [1.0, 0.3092771157761032, 0.3092771157761032]), ([0.004, 0.016, 1.0], [1.0, 0.03225444169764435, 0.03225444169764435]), ([0.004, 0.016, 1.0], [1.0, 0.1173182175906747, 0.1173182175906747]), ([0.004, 0.016, 1.0], [1.0, 0.07616859108383567, 0.07616859108383567]), ([0.004, 0.04, 1.0], [1.0, 0.09656431107041, 0.09656431107041]), ([0.004, 0.016, 1.0], [1.0, 0.21308589022926125, 0.21308589022926125])]"

        # Convert the expected values from string to their actual types
        self.expected_all_recommended_xs = eval(self.expected_all_recommended_xs, {'nan': float('nan'), 'inf': float('inf')})
        self.expected_all_intermediate_budgets = eval(self.expected_all_intermediate_budgets, {'nan': float('nan'), 'inf': float('inf')})
        self.expected_all_est_objectives = eval(self.expected_all_est_objectives, {'nan': float('nan'), 'inf': float('inf')})
        self.expected_objective_curves = eval(self.expected_objective_curves, {'nan': float('nan'), 'inf': float('inf')})
        self.expected_progress_curves = eval(self.expected_progress_curves, {'nan': float('nan'), 'inf': float('inf')})
        
        # Number of macro-replications and post-replications
        self.num_macroreps = 24
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
