import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class test_PARAMESTI1_STRONG(unittest.TestCase):
    def setUp(self):
        # Expected values
        self.expected_problem_name = "PARAMESTI-1"
        self.expected_solver_name = "STRONG"
        self.expected_all_recommended_xs = "[[(1, 1), (2.7235752627097733, 2.014538473284743), (1.3528970774056805, 3.4709866571661605), (2.0719033088006547, 3.588045019706422), (2.0719033088006547, 3.588045019706422)], [(1, 1), (2.757989242641407, 1.9536633697259695), (1.4452592814365657, 3.4625503929835544), (2.1604738643213395, 3.593983311084327), (2.1604738643213395, 3.593983311084327)], [(1, 1), (2.8048422954247303, 1.8617100954764256), (1.4981778515222532, 3.3758527674334085), (2.175544816637799, 3.5606430237701425), (2.175544816637799, 3.5606430237701425)], [(1, 1), (2.7004220444998888, 2.052884072715899), (1.5320420740412382, 3.6761179574752836), (1.5320420740412382, 3.6761179574752836)], [(1, 1), (2.772876978697469, 1.9256928315616006), (1.6943491754237288, 3.60996649625078), (2.3460984344237055, 3.722763223125858), (2.4199177459686094, 4.798821674575626), (2.4199177459686094, 4.798821674575626)], [(1, 1), (2.712570773694229, 2.0330059753400995), (1.3801142052728328, 3.5245016988574944), (2.044141482445509, 3.641852456177532), (2.044141482445509, 3.641852456177532)], [(1, 1), (2.798182339850576, 1.8755228567259157), (2.517707684434785, 3.855758693102329), (2.517707684434785, 3.855758693102329)], [(1, 1), (2.8136032050021926, 1.84311530338725), (1.9165776781031663, 3.6306681568971886), (2.2341516313761094, 4.222735620154041), (2.2341516313761094, 4.222735620154041)], [(1, 1), (2.8304457624213777, 1.8058959677487039), (1.801443558738547, 3.5208754034317904), (2.5826918884195216, 3.6872100376751584), (2.3028313185428746, 4.342685104624084), (2.3028313185428746, 4.342685104624084)], [(1, 1), (2.7023867099154253, 2.04970447741416), (1.6796825368233423, 3.7684470917517574), (1.6796825368233423, 3.7684470917517574)]]"
        self.expected_all_intermediate_budgets = "[[10, 60, 115, 640, 1000], [10, 60, 115, 640, 1000], [10, 60, 115, 640, 1000], [10, 60, 115, 1000], [10, 60, 115, 640, 835, 1000], [10, 60, 115, 640, 1000], [10, 60, 115, 1000], [10, 60, 115, 899, 1000], [10, 60, 115, 640, 735, 1000], [10, 60, 115, 1000]]"
        self.expected_all_est_objectives = "[[-9.16944227603262, -6.656118848660844, -5.581815741113141, -4.929373945803076, -4.929373945803076], [-8.940090362495347, -6.854706285259254, -5.2404035598942755, -4.912346655109802, -4.912346655109802], [-9.121210005202611, -7.003656321196936, -5.290613139000511, -4.936450930189247, -4.936450930189247], [-8.779886386724968, -6.419730021798727, -4.977529214214554, -4.977529214214554], [-8.99288952739613, -6.353502113820264, -4.923128905460677, -4.776226784303009, -4.681999241738291, -4.681999241738291], [-8.87740808504234, -6.315623213185667, -5.340647046924687, -4.72988907361193, -4.72988907361193], [-9.024638576352391, -6.84445456607904, -5.116058130936463, -5.116058130936463], [-8.921050660074993, -6.990641765487221, -4.824733405250829, -4.768264047711256, -4.768264047711256], [-8.550164686658025, -6.370750733809734, -4.692021515602302, -4.881550401730862, -4.550712640576805, -4.550712640576805], [-8.983830735669818, -6.4259441883097335, -4.887416385535807, -4.887416385535807]]"
        self.expected_objective_curves = "[([10, 60, 115, 640, 1000], [-9.265122221743944, -6.656118848660844, -5.581815741113141, -4.929373945803076, -4.929373945803076]), ([10, 60, 115, 640, 1000], [-9.265122221743944, -6.854706285259254, -5.2404035598942755, -4.912346655109802, -4.912346655109802]), ([10, 60, 115, 640, 1000], [-9.265122221743944, -7.003656321196936, -5.290613139000511, -4.936450930189247, -4.936450930189247]), ([10, 60, 115, 1000], [-9.265122221743944, -6.419730021798727, -4.977529214214554, -4.977529214214554]), ([10, 60, 115, 640, 835, 1000], [-9.265122221743944, -6.353502113820264, -4.923128905460677, -4.776226784303009, -4.681999241738291, -4.681999241738291]), ([10, 60, 115, 640, 1000], [-9.265122221743944, -6.315623213185667, -5.340647046924687, -4.72988907361193, -4.72988907361193]), ([10, 60, 115, 1000], [-9.265122221743944, -6.84445456607904, -5.116058130936463, -5.116058130936463]), ([10, 60, 115, 899, 1000], [-9.265122221743944, -6.990641765487221, -4.824733405250829, -4.768264047711256, -4.768264047711256]), ([10, 60, 115, 640, 735, 1000], [-9.265122221743944, -6.370750733809734, -4.692021515602302, -4.881550401730862, -4.550712640576805, -4.550712640576805]), ([10, 60, 115, 1000], [-9.265122221743944, -6.4259441883097335, -4.887416385535807, -4.887416385535807])]"
        self.expected_progress_curves = "[([0.01, 0.06, 0.115, 0.64, 1.0], [1.0, 0.4398852372953998, 0.20924811495016538, 0.06917842967152513, 0.06917842967152513]), ([0.01, 0.06, 0.115, 0.64, 1.0], [1.0, 0.48251904761316444, 0.1359519265125531, 0.06552292006791167, 0.06552292006791167]), ([0.01, 0.06, 0.115, 0.64, 1.0], [1.0, 0.514496435950339, 0.14673118681429562, 0.07069775443400725, 0.07069775443400725]), ([0.01, 0.06, 0.115, 1.0], [1.0, 0.38913602285209736, 0.07951665959393328, 0.07951665959393328]), ([0.01, 0.06, 0.115, 0.64, 0.835, 1.0], [1.0, 0.3749178622531163, 0.06783771109463434, 0.036299980027695924, 0.01607070850264397, 0.01607070850264397]), ([0.01, 0.06, 0.115, 0.64, 1.0], [1.0, 0.36678581778765185, 0.1574727330894989, 0.026351953093051828, 0.026351953093051828]), ([0.01, 0.06, 0.115, 1.0], [1.0, 0.4803181538484493, 0.10925678646303949, 0.10925678646303949]), ([0.01, 0.06, 0.115, 0.899, 1.0], [1.0, 0.5117024017002267, 0.046713640187009485, 0.03459049725638834, 0.03459049725638834]), ([0.01, 0.06, 0.115, 0.64, 0.735, 1.0], [1.0, 0.3786208880107898, 0.018222343722826613, 0.05891141609996084, -0.012114599425479246, -0.012114599425479246]), ([0.01, 0.06, 0.115, 1.0], [1.0, 0.3904701132701585, 0.06017075679719158, 0.06017075679719158])]"

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
