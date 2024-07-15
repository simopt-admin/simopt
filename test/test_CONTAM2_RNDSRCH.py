import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class test_CONTAM2_RNDSRCH(unittest.TestCase):
    def setUp(self):
        # Expected values
        self.expected_problem_name = "CONTAM-2"
        self.expected_solver_name = "RNDSRCH"
        self.expected_all_recommended_xs = "[[(1, 1, 1, 1, 1), (0.03326209446380745, 0.9980069071951875, 0.9012693649772621, 0.7880875551892006, 0.9467520953445779), (0.5586587815082229, 0.9848623184606812, 0.6708517455349591, 0.7924703773655554, 0.6446912384349336), (0.20506481515557543, 0.7982577744493301, 0.8046505840884806, 0.9233324616339879, 0.7252097422824303), (0.16707844979882186, 0.8670421935023686, 0.7435598696257112, 0.7425617141306486, 0.8992466493610534), (0.11935584988119473, 0.7380309655588216, 0.7859860873979299, 0.8617839960500299, 0.8710377624202182), (0.08857969740046587, 0.6197503839871101, 0.8498400358405728, 0.8730402261932305, 0.7883400965888846), (0.08857969740046587, 0.6197503839871101, 0.8498400358405728, 0.8730402261932305, 0.7883400965888846)], [(1, 1, 1, 1, 1), (0.4025584488949174, 0.7805415192508689, 0.9456483676784814, 0.982317050295404, 0.8975987021579711), (0.17406964190455285, 0.7718715038032441, 0.9319150868892526, 0.8981351942783503, 0.7945680197491656), (0.17406964190455285, 0.7718715038032441, 0.9319150868892526, 0.8981351942783503, 0.7945680197491656)], [(1, 1, 1, 1, 1), (1, 1, 1, 1, 1)], [(1, 1, 1, 1, 1), (1, 1, 1, 1, 1)], [(1, 1, 1, 1, 1), (1, 1, 1, 1, 1)], [(1, 1, 1, 1, 1), (0.8420821309911748, 0.7750584253138287, 0.6871811414448725, 0.8463317337532065, 0.8028906113471015), (0.6066071342617003, 0.8107883747769479, 0.7804731839193084, 0.8536461618632083, 0.7855266435978799), (0.13917897198098392, 0.917781073576413, 0.788741525741815, 0.8139095756442267, 0.9133195916587663), (0.10283039984962046, 0.973400427835828, 0.6092631508891321, 0.9601696726668841, 0.8431757421653147), (0.10283039984962046, 0.973400427835828, 0.6092631508891321, 0.9601696726668841, 0.8431757421653147)], [(1, 1, 1, 1, 1), (0.15532444168522114, 0.8048985780726431, 0.9763238320768246, 0.7371860820182379, 0.8733646698901084), (0.15265302098165928, 0.8318161377724625, 0.9754705994617858, 0.695260625475601, 0.7859104761084028), (0.15265302098165928, 0.8318161377724625, 0.9754705994617858, 0.695260625475601, 0.7859104761084028)], [(1, 1, 1, 1, 1), (0.25482071354116975, 0.8143541534397901, 0.89158782536403, 0.9921094680574652, 0.8520586530743632), (0.33400845119584305, 0.8158981571222695, 0.7327990199956569, 0.9490010925550544, 0.6155955477254172), (0.33400845119584305, 0.8158981571222695, 0.7327990199956569, 0.9490010925550544, 0.6155955477254172)], [(1, 1, 1, 1, 1), (0.7007941377268129, 0.9827784801409404, 0.46256385888282275, 0.7666013702873805, 0.9127250697572751), (0.1427668749111495, 0.8876552981399704, 0.8522797872485118, 0.6898583284324343, 0.8831793891036216), (0.13590465329311974, 0.8833809140471812, 0.5627404810045893, 0.8196071736696857, 0.8298511464635466), (0.13590465329311974, 0.8833809140471812, 0.5627404810045893, 0.8196071736696857, 0.8298511464635466)], [(1, 1, 1, 1, 1), (0.040689602136480915, 0.931800068778548, 0.6429362743466034, 0.8138730023721198, 0.8935573834134117), (0.040689602136480915, 0.931800068778548, 0.6429362743466034, 0.8138730023721198, 0.8935573834134117)]]"
        self.expected_all_intermediate_budgets = "[[0, 30, 440, 2020, 6060, 8550, 9720, 10000], [0, 1840, 3540, 10000], [0, 10000], [0, 10000], [0, 10000], [0, 3720, 4920, 5150, 8440, 10000], [0, 910, 4040, 10000], [0, 400, 640, 10000], [0, 280, 3210, 7010, 10000], [0, 70, 10000]]"
        self.expected_all_est_objectives = "[[5.0, 3.6673780171700368, 3.6515344613043514, 3.4565153776098043, 3.419488876418604, 3.3761946613081957, 3.219550440010264, 3.219550440010264], [5.0, 4.008664088277642, 3.5705594466245647, 3.5705594466245647], [5.0, 5.0], [5.0, 5.0], [5.0, 5.0], [5.0, 3.9535440428501833, 3.837041498419045, 3.5729307386022047, 3.4888393934067783, 3.4888393934067783], [5.0, 3.547097603743034, 3.4411108597999123, 3.4411108597999123], [5.0, 3.8049308134768176, 3.447302268594242, 3.447302268594242], [5.0, 3.8254629167952316, 3.455739677835687, 3.2314843684781214, 3.2314843684781214], [5.0, 3.3228563310471633, 3.3228563310471633]]"
        self.expected_objective_curves = "[([0, 30, 440, 2020, 6060, 8550, 9720, 10000], [5.0, 3.6673780171700368, 3.6515344613043514, 3.4565153776098043, 3.419488876418604, 3.3761946613081957, 3.219550440010264, 3.219550440010264]), ([0, 1840, 3540, 10000], [5.0, 4.008664088277642, 3.5705594466245647, 3.5705594466245647]), ([0, 10000], [5.0, 5.0]), ([0, 10000], [5.0, 5.0]), ([0, 10000], [5.0, 5.0]), ([0, 3720, 4920, 5150, 8440, 10000], [5.0, 3.9535440428501833, 3.837041498419045, 3.5729307386022047, 3.4888393934067783, 3.4888393934067783]), ([0, 910, 4040, 10000], [5.0, 3.547097603743034, 3.4411108597999123, 3.4411108597999123]), ([0, 400, 640, 10000], [5.0, 3.8049308134768176, 3.447302268594242, 3.447302268594242]), ([0, 280, 3210, 7010, 10000], [5.0, 3.8254629167952316, 3.455739677835687, 3.2314843684781214, 3.2314843684781214]), ([0, 70, 10000], [5.0, 3.3228563310471633, 3.3228563310471633])]"
        self.expected_progress_curves = "[([0.0, 0.003, 0.044, 0.202, 0.606, 0.855, 0.972, 1.0], [1.0, 0.2515250008893003, 0.24262637426054218, 0.13309275529316666, 0.11229660244320132, 0.08798015109107307, 0.0, 0.0]), ([0.0, 0.184, 0.354, 1.0], [1.0, 0.4432103363107499, 0.19714627951399202, 0.19714627951399202]), ([0.0, 1.0], [1.0, 1.0]), ([0.0, 1.0], [1.0, 1.0]), ([0.0, 1.0], [1.0, 1.0]), ([0.0, 0.372, 0.492, 0.515, 0.844, 1.0], [1.0, 0.41225183758878886, 0.3468174961453783, 0.1984781296438287, 0.1512477294768556, 0.1512477294768556]), ([0.0, 0.091, 0.404, 1.0], [1.0, 0.18396879703497937, 0.12444071697876428, 0.12444071697876428]), ([0.0, 0.04, 0.064, 1.0], [1.0, 0.32878234049490745, 0.12791815825733974, 0.12791815825733974]), ([0.0, 0.028, 0.321, 0.701, 1.0], [1.0, 0.3403143174628663, 0.13265707893841402, 0.006702761334011638, 0.006702761334011638]), ([0.0, 0.007, 1.0], [1.0, 0.058022363204433984, 0.058022363204433984])]"

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
