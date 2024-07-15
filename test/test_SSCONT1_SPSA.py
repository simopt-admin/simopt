import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class test_SSCONT1_SPSA(unittest.TestCase):
    def setUp(self):
        # Expected values
        self.expected_problem_name = "SSCONT-1"
        self.expected_solver_name = "SPSA"
        self.expected_all_recommended_xs = "[[(600, 600), (599.2872982749515, 599.2872982749515), (598.7845545608287, 598.8881253296626), (598.4127241074534, 598.5579971331124), (597.8471457340399, 598.0221481731817), (597.6158505168473, 597.8286983148453), (597.4079067970073, 597.6441746009053), (597.402583500057, 597.6494978978557), (597.402583500057, 597.6494978978557)], [(600, 600), (599.0426358459758, 599.0426358459758), (598.7822142856272, 598.9953442839411), (598.6262024702149, 598.8791661765002), (598.2712656902249, 598.5521358017697), (598.2712656902249, 598.5521358017697)], [(600, 600), (599.4392346690771, 599.4392346690771), (599.0214634818176, 599.0214634818176), (598.366993350482, 598.366993350482), (598.366993350482, 598.366993350482)], [(600, 600), (600, 600)], [(600, 600), (599.492399063192, 599.492399063192), (599.1314948928496, 599.1314948928496), (598.3581971271973, 598.6335017585271), (598.1262783865739, 598.6258333918236), (598.1262783865739, 598.6258333918236)], [(600, 600), (598.7254532688579, 598.7254532688579), (597.7426423939182, 597.7426423939182), (596.9407092249073, 596.9407092249073), (596.2432541374054, 596.2432541374054), (595.6393761828489, 595.6393761828489), (595.0991611466105, 595.0991611466105), (594.0409370142887, 593.7796640514524), (593.7542061300892, 593.4000081014597), (593.7542061300892, 593.4000081014597)], [(600, 600), (599.9672777198275, 600.0327222801725), (599.9672777198275, 600.0327222801725)], [(600, 600), (599.9, 599.9), (599.68054058265, 599.6810261434726), (599.6495132230045, 599.6562054867165), (599.6220719009358, 599.6350357917096), (599.6193597647665, 599.6377479278789), (599.6193597647665, 599.6377479278789)], [(600, 600), (599.7736378818547, 599.7736378818547), (599.7736378818547, 599.7736378818547)], [(600, 600), (599.9, 599.9), (599.8071867387588, 599.9035526671404), (599.8071867387588, 599.9035526671404)]]"
        self.expected_all_intermediate_budgets = "[[0, 210, 330, 450, 630, 810, 930, 990, 1000], [0, 330, 570, 690, 930, 1000], [0, 210, 270, 390, 1000], [0, 1000], [0, 210, 270, 630, 930, 1000], [0, 210, 270, 330, 390, 450, 510, 810, 930, 1000], [0, 210, 1000], [0, 210, 570, 690, 810, 870, 1000], [0, 330, 1000], [0, 210, 450, 1000]]"
        self.expected_all_est_objectives = "[[618.5809976153716, 618.3692233528762, 617.9299406028401, 617.6746457035173, 617.1732892507664, 616.9857516330617, 616.8577169250245, 616.8577169250245, 616.8577169250245], [619.371245290233, 618.2272994305168, 618.0887271690285, 618.0249728242222, 617.7547609969663, 617.7547609969663], [620.2040298994102, 619.64783408728, 619.5306295828523, 619.4342585968005, 619.4342585968005], [620.3929887875448, 620.3929887875448], [617.140803174291, 616.977964795912, 616.7649833217964, 616.1071648472699, 615.9963528483212, 615.9963528483212], [617.6250759903628, 617.1869781633847, 616.9055776556821, 616.4670237125466, 615.9587677574567, 615.2456251770844, 614.9413427463967, 614.3346396838655, 613.890611420458, 613.890611420458], [622.8299886318688, 622.8155391889641, 622.8155391889641], [617.1638109984892, 617.0908993444739, 616.8045016641678, 616.774009802673, 616.7534117021602, 616.7534117021602, 616.7534117021602], [625.4509909440814, 625.4314168403478, 625.4314168403478], [616.3517529689802, 616.5177270489065, 616.4784096549038, 616.4784096549038]]"
        self.expected_objective_curves = "[([0, 210, 330, 450, 630, 810, 930, 990, 1000], [624.4131899421741, 618.3692233528762, 617.9299406028401, 617.6746457035173, 617.1732892507664, 616.9857516330617, 616.8577169250245, 616.8577169250245, 616.8577169250245]), ([0, 330, 570, 690, 930, 1000], [624.4131899421741, 618.2272994305168, 618.0887271690285, 618.0249728242222, 617.7547609969663, 617.7547609969663]), ([0, 210, 270, 390, 1000], [624.4131899421741, 619.64783408728, 619.5306295828523, 619.4342585968005, 619.4342585968005]), ([0, 1000], [624.4131899421741, 624.4131899421741]), ([0, 210, 270, 630, 930, 1000], [624.4131899421741, 616.977964795912, 616.7649833217964, 616.1071648472699, 615.9963528483212, 615.9963528483212]), ([0, 210, 270, 330, 390, 450, 510, 810, 930, 1000], [624.4131899421741, 617.1869781633847, 616.9055776556821, 616.4670237125466, 615.9587677574567, 615.2456251770844, 614.9413427463967, 614.3346396838655, 619.8562164573261, 619.8562164573261]), ([0, 210, 1000], [624.4131899421741, 622.8155391889641, 622.8155391889641]), ([0, 210, 570, 690, 810, 870, 1000], [624.4131899421741, 617.0908993444739, 616.8045016641678, 616.774009802673, 616.7534117021602, 616.7534117021602, 616.7534117021602]), ([0, 330, 1000], [624.4131899421741, 625.4314168403478, 625.4314168403478]), ([0, 210, 450, 1000], [624.4131899421741, 616.5177270489065, 616.4784096549038, 616.4784096549038])]"
        self.expected_progress_curves = "[([0.0, 0.21, 0.33, 0.45, 0.63, 0.81, 0.93, 0.99, 1.0], [1.0, -0.3263115551130956, -0.4227094717340149, -0.47873237820290476, -0.5887519897757842, -0.6299059746142494, -0.658002409334106, -0.658002409334106, -0.658002409334106]), ([0.0, 0.33, 0.57, 0.69, 0.93, 1.0], [1.0, -0.35745589308901554, -0.3878647295567011, -0.4018552311512943, -0.4611515663514869, -0.4611515663514869]), ([0.0, 0.21, 0.27, 0.39, 1.0], [1.0, -0.045728238432594395, -0.07144805111472345, -0.0925960754278282, -0.0925960754278282]), ([0.0, 1.0], [1.0, 1.0]), ([0.0, 0.21, 0.27, 0.63, 0.93, 1.0], [1.0, -0.6316147484694252, -0.6783522322015098, -0.8227064788772377, -0.8470234952735806, -0.8470234952735806]), ([0.0, 0.21, 0.27, 0.33, 0.39, 0.45, 0.51, 0.81, 0.93, 1.0], [1.0, -0.5857480415053443, -0.6474996642958196, -0.7437376486935136, -0.8552713139166864, -1.0117660977339487, -1.0785390187745243, -1.2116763004700426, 0.0, 0.0]), ([0.0, 0.21, 1.0], [1.0, 0.6494052996967737, 0.6494052996967737]), ([0.0, 0.21, 0.57, 0.69, 0.81, 0.87, 1.0], [1.0, -0.6068319515237444, -0.6696801733223512, -0.6763714261014434, -0.6808915534581989, -0.6808915534581989, -0.6808915534581989]), ([0.0, 0.33, 1.0], [1.0, 1.2234436740874854, 1.2234436740874854]), ([0.0, 0.21, 0.45, 1.0], [1.0, -0.7326111111947785, -0.7412390731817161, -0.7412390731817161])]"

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
