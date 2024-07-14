import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class test_EXAMPLE1_ADAM(unittest.TestCase):
    def setUp(self):
        # Expected values
        self.expected_problem_name = "EXAMPLE-1"
        self.expected_solver_name = "ADAM"
        self.expected_all_recommended_xs = "[[(2.0, 2.0), (1.50000000125, 1.50000000125), (1.0087124618718393, 1.0087124618718393), (0.5369748288298138, 0.5369748288298138), (0.10119094158679959, 0.10119094158679959), (-0.055936685066587694, -0.055936685066587694), (-0.019650871240695242, -0.019650871240695242), (-0.019650871240695242, -0.019650871240695242)], [(2.0, 2.0), (1.50000000125, 1.50000000125), (1.0087124618718393, 1.0087124618718393), (0.5369748288298138, 0.5369748288298138), (0.10119094158679959, 0.10119094158679959), (-0.055936685066587694, -0.055936685066587694), (-0.019650871240695242, -0.019650871240695242), (-0.019650871240695242, -0.019650871240695242)], [(2.0, 2.0), (1.50000000125, 1.50000000125), (1.0087124618718393, 1.0087124618718393), (0.5369748288298138, 0.5369748288298138), (0.10119094158679959, 0.10119094158679959), (-0.055936685066587694, -0.055936685066587694), (-0.019650871240695242, -0.019650871240695242), (-0.019650871240695242, -0.019650871240695242)], [(2.0, 2.0), (1.50000000125, 1.50000000125), (1.0087124618718393, 1.0087124618718393), (0.5369748288298138, 0.5369748288298138), (0.10119094158679959, 0.10119094158679959), (-0.055936685066587694, -0.055936685066587694), (-0.019650871240695242, -0.019650871240695242), (-0.019650871240695242, -0.019650871240695242)], [(2.0, 2.0), (1.50000000125, 1.50000000125), (1.0087124618718393, 1.0087124618718393), (0.5369748288298138, 0.5369748288298138), (0.10119094158679959, 0.10119094158679959), (-0.055936685066587694, -0.055936685066587694), (-0.019650871240695242, -0.019650871240695242), (-0.019650871240695242, -0.019650871240695242)], [(2.0, 2.0), (1.50000000125, 1.50000000125), (1.0087124618718393, 1.0087124618718393), (0.5369748288298138, 0.5369748288298138), (0.10119094158679959, 0.10119094158679959), (-0.055936685066587694, -0.055936685066587694), (-0.019650871240695242, -0.019650871240695242), (-0.019650871240695242, -0.019650871240695242)], [(2.0, 2.0), (1.50000000125, 1.50000000125), (1.0087124618718393, 1.0087124618718393), (0.5369748288298138, 0.5369748288298138), (0.10119094158679959, 0.10119094158679959), (-0.055936685066587694, -0.055936685066587694), (-0.019650871240695242, -0.019650871240695242), (-0.019650871240695242, -0.019650871240695242)], [(2.0, 2.0), (1.50000000125, 1.50000000125), (1.0087124618718393, 1.0087124618718393), (0.5369748288298138, 0.5369748288298138), (0.10119094158679959, 0.10119094158679959), (-0.055936685066587694, -0.055936685066587694), (-0.019650871240695242, -0.019650871240695242), (-0.019650871240695242, -0.019650871240695242)], [(2.0, 2.0), (1.50000000125, 1.50000000125), (1.0087124618718393, 1.0087124618718393), (0.5369748288298138, 0.5369748288298138), (0.10119094158679959, 0.10119094158679959), (-0.055936685066587694, -0.055936685066587694), (-0.019650871240695242, -0.019650871240695242), (-0.019650871240695242, -0.019650871240695242)], [(2.0, 2.0), (1.50000000125, 1.50000000125), (1.0087124618718393, 1.0087124618718393), (0.5369748288298138, 0.5369748288298138), (0.10119094158679959, 0.10119094158679959), (-0.055936685066587694, -0.055936685066587694), (-0.019650871240695242, -0.019650871240695242), (-0.019650871240695242, -0.019650871240695242)]]"
        self.expected_all_intermediate_budgets = "[[0, 60, 90, 120, 150, 480, 840, 1000], [0, 60, 90, 120, 150, 480, 840, 1000], [0, 60, 90, 120, 150, 480, 840, 1000], [0, 60, 90, 120, 150, 480, 840, 1000], [0, 60, 90, 120, 150, 480, 840, 1000], [0, 60, 90, 120, 150, 480, 840, 1000], [0, 60, 90, 120, 150, 480, 840, 1000], [0, 60, 90, 120, 150, 480, 840, 1000], [0, 60, 90, 120, 150, 480, 840, 1000], [0, 60, 90, 120, 150, 480, 840, 1000]]"
        self.expected_all_est_objectives = "[[7.984539704940337, 4.484539712440336, 2.0195413664114295, 0.5612236385339512, 0.0050189182587814865, -0.00920246958718745, -0.014687981578627923, -0.014687981578627923], [8.081590387702734, 4.581590395202731, 2.116592049173825, 0.6582743212963469, 0.10206960102117726, 0.08784821317520829, 0.08236270118376783, 0.08236270118376783], [7.9253347189439385, 4.425334726443936, 1.9603363804150302, 0.5020186525375524, -0.05418606773761721, -0.06840745558358616, -0.07389296757502664, -0.07389296757502664], [8.073099810658121, 4.573099818158121, 2.108101472129214, 0.649783744251736, 0.09357902397656628, 0.07935763613059736, 0.07387212413915688, 0.07387212413915688], [7.880122723414122, 4.3801227309141195, 1.915124384885213, 0.45680665700773504, -0.09939806326743458, -0.11361945111340355, -0.11910496310484402, -0.11910496310484402], [8.025785950362149, 4.525785957862148, 2.0607876118332404, 0.6024698839557625, 0.046265163680592795, 0.03204377583462384, 0.02655826384318338, 0.02655826384318338], [8.015084462897443, 4.515084470397443, 2.0500861243685358, 0.5917683964910578, 0.035563676215888185, 0.02134228836991923, 0.01585677637847875, 0.01585677637847875], [7.994852045957048, 4.494852053457047, 2.0298537074281398, 0.5715359795506617, 0.015331259275492058, 0.0011098714295231195, -0.004375640561917353, -0.004375640561917353], [7.910902809206077, 4.410902816706076, 1.9459044706771693, 0.4875867427996913, -0.06861797747547835, -0.08283936532144727, -0.08832487731288775, -0.08832487731288775], [7.943417039435916, 4.443417046935915, 1.9784187009070084, 0.5201009730295305, -0.03610374724563913, -0.050325135091608075, -0.05581064708304855, -0.05581064708304855]]"
        self.expected_objective_curves = "[([0, 60, 90, 120, 150, 480, 840, 1000], [8.090508544469758, 4.484539712440336, 2.0195413664114295, 0.5612236385339512, 0.0050189182587814865, -0.00920246958718745, -0.014687981578627923, -0.014687981578627923]), ([0, 60, 90, 120, 150, 480, 840, 1000], [8.090508544469758, 4.581590395202731, 2.116592049173825, 0.6582743212963469, 0.10206960102117726, 0.08784821317520829, 0.08236270118376783, 0.08236270118376783]), ([0, 60, 90, 120, 150, 480, 840, 1000], [8.090508544469758, 4.425334726443936, 1.9603363804150302, 0.5020186525375524, -0.05418606773761721, -0.06840745558358616, -0.07389296757502664, -0.07389296757502664]), ([0, 60, 90, 120, 150, 480, 840, 1000], [8.090508544469758, 4.573099818158121, 2.108101472129214, 0.649783744251736, 0.09357902397656628, 0.07935763613059736, 0.07387212413915688, 0.07387212413915688]), ([0, 60, 90, 120, 150, 480, 840, 1000], [8.090508544469758, 4.3801227309141195, 1.915124384885213, 0.45680665700773504, -0.09939806326743458, -0.11361945111340355, -0.11910496310484402, -0.11910496310484402]), ([0, 60, 90, 120, 150, 480, 840, 1000], [8.090508544469758, 4.525785957862148, 2.0607876118332404, 0.6024698839557625, 0.046265163680592795, 0.03204377583462384, 0.02655826384318338, 0.02655826384318338]), ([0, 60, 90, 120, 150, 480, 840, 1000], [8.090508544469758, 4.515084470397443, 2.0500861243685358, 0.5917683964910578, 0.035563676215888185, 0.02134228836991923, 0.01585677637847875, 0.01585677637847875]), ([0, 60, 90, 120, 150, 480, 840, 1000], [8.090508544469758, 4.494852053457047, 2.0298537074281398, 0.5715359795506617, 0.015331259275492058, 0.0011098714295231195, -0.004375640561917353, -0.004375640561917353]), ([0, 60, 90, 120, 150, 480, 840, 1000], [8.090508544469758, 4.410902816706076, 1.9459044706771693, 0.4875867427996913, -0.06861797747547835, -0.08283936532144727, -0.08832487731288775, -0.08832487731288775]), ([0, 60, 90, 120, 150, 480, 840, 1000], [8.090508544469758, 4.443417046935915, 1.9784187009070084, 0.5201009730295305, -0.03610374724563913, -0.050325135091608075, -0.05581064708304855, -0.05581064708304855])]"
        self.expected_progress_curves = "[([0.0, 0.06, 0.09, 0.12, 0.15, 0.48, 0.84, 1.0], [1.0, 0.5542963940759607, 0.24961859385117152, 0.06936815349111446, 0.000620346450559298, -0.0011374401913805245, -0.0018154583853283051, -0.0018154583853283051]), ([0.0, 0.06, 0.09, 0.12, 0.15, 0.48, 0.84, 1.0], [1.0, 0.5662920161346919, 0.2616142159099028, 0.08136377554984578, 0.012615968509290634, 0.010858181867350805, 0.010180163673403027, 0.010180163673403027]), ([0.0, 0.06, 0.09, 0.12, 0.15, 0.48, 0.84, 1.0], [1.0, 0.5469785616219218, 0.2423007613971327, 0.06205032103707568, -0.006697486003479463, -0.008455272645419287, -0.00913329083936707, -0.00913329083936707]), ([0.0, 0.06, 0.09, 0.12, 0.15, 0.48, 0.84, 1.0], [1.0, 0.5652425670181201, 0.26056476679333096, 0.08031432643327392, 0.01156651939271876, 0.00980873275077894, 0.009130714556831157, 0.009130714556831157]), ([0.0, 0.06, 0.09, 0.12, 0.15, 0.48, 0.84, 1.0], [1.0, 0.5413902855226744, 0.2367124852978853, 0.05646204493782827, -0.012285762102726883, -0.014043548744666708, -0.014721566938614488, -0.014721566938614488]), ([0.0, 0.06, 0.09, 0.12, 0.15, 0.48, 0.84, 1.0], [1.0, 0.55939449701907, 0.25471669679428066, 0.07446625643422365, 0.005718449393668486, 0.003960662751728661, 0.0032826445577808826, 0.0032826445577808826]), ([0.0, 0.06, 0.09, 0.12, 0.15, 0.48, 0.84, 1.0], [1.0, 0.558071775782712, 0.25339397555792287, 0.07314353519786582, 0.004395728157310658, 0.0026379415153708337, 0.0019599233214230524, 0.0019599233214230524]), ([0.0, 0.06, 0.09, 0.12, 0.15, 0.48, 0.84, 1.0], [1.0, 0.5555710161791361, 0.2508932159543469, 0.07064277559428984, 0.001894968553734696, 0.00013718191179487335, -0.0005408362821529072, -0.0005408362821529072]), ([0.0, 0.06, 0.09, 0.12, 0.15, 0.48, 0.84, 1.0], [1.0, 0.5451947541321287, 0.24051695390733951, 0.06026651354728247, -0.008481293493272675, -0.010239080135212497, -0.010917098329160279, -0.010917098329160279]), ([0.0, 0.06, 0.09, 0.12, 0.15, 0.48, 0.84, 1.0], [1.0, 0.5492135658114098, 0.24453576558662068, 0.06428532522656365, -0.0044624818139915, -0.006220268455931323, -0.006898286649879104, -0.006898286649879104])]"

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
