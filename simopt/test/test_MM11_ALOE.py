import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class test_MM11_ALOE(unittest.TestCase):
    def setUp(self):
        # Expected values
        self.expected_problem_name = "MM1-1"
        self.expected_solver_name = "ALOE"
        self.expected_all_recommended_xs = "[[(5,), (4.068363045037877,), (3.1954320291559246,), (2.54789745330792,), (2.7223159323474206,), (2.7223159323474206,)], [(5,), (4.070245430133759,), (3.203625828952615,), (2.6816920968594635,), (2.837137038961862,), (2.837137038961862,)], [(5,), (4.0698290513184405,), (3.20057733045431,), (2.5600315617711913,), (2.7043929379029126,), (2.899527251791042,), (2.899527251791042,)], [(5,), (4.070754629536534,), (3.2062270171567695,), (2.706137838725003,), (2.8410880882508316,), (2.8410880882508316,)], [(5,), (4.0740564892409665,), (3.2126838001722193,), (2.586211377512199,), (2.688745836222323,), (2.688745836222323,)], [(5,), (4.072039930037047,), (3.2059748709238622,), (2.5717788385211637,), (2.7368416976658168,), (2.764129255958074,), (2.764129255958074,)], [(5,), (4.067976139171115,), (3.1950912750175346,), (2.5419313584484087,), (2.6651435290553054,), (2.6651435290553054,)], [(5,), (4.066584889852279,), (3.1910395194526626,), (2.5353142333928718,), (2.6426848084176577,), (2.821478796038285,), (2.821478796038285,)], [(5,), (4.06990648820933,), (3.2017545501170557,), (2.565896832903403,), (2.6966927874536357,), (2.920604432561989,), (2.7790004170607814,), (2.7790004170607814,)], [(5,), (4.066667179797703,), (3.1922253533293192,), (2.538857510529519,), (2.679986733110487,), (2.8405555369814506,), (2.8405555369814506,)]]"
        self.expected_all_intermediate_budgets = "[[0, 60, 90, 120, 240, 1000], [0, 60, 90, 240, 480, 1000], [0, 60, 90, 120, 240, 480, 1000], [0, 60, 90, 240, 720, 1000], [0, 60, 90, 120, 240, 1000], [0, 60, 90, 120, 240, 960, 1000], [0, 60, 90, 120, 240, 1000], [0, 60, 90, 120, 240, 720, 1000], [0, 60, 90, 120, 240, 480, 720, 1000], [0, 60, 90, 120, 240, 480, 1000]]"
        self.expected_all_est_objectives = "[[2.7852321578182706, 2.0434955300959756, 1.6067949860630397, 1.5953185527727127, 1.551094392839577, 1.551094392839577], [2.7857037031168543, 2.0454590444287932, 1.6112733855379604, 1.5591639731106854, 1.5478056708472974, 1.5478056708472974], [2.7866293625352507, 2.04797163822759, 1.6205137429033964, 1.6163253621532803, 1.5759105545236445, 1.5664687724992692, 1.5664687724992692], [2.7889080044387127, 2.051164328901086, 1.6227059425210009, 1.5749298888776202, 1.5649659079370986, 1.5649659079370986], [2.7833651638972787, 2.0438598971642126, 1.608137408927407, 1.5763337097688268, 1.5521300465336114, 1.5521300465336114], [2.787955763524055, 2.050284006246868, 1.6215173623533616, 1.620189560285288, 1.5752097928232494, 1.5716967873961198, 1.5716967873961198], [2.7843462630059106, 2.042523510862577, 1.6072035528477624, 1.5934064648951312, 1.5584508725578103, 1.5584508725578103], [2.7907221687784363, 2.052665196346358, 1.628996218505635, 1.6431966479141764, 1.604964304544481, 1.5790413250028876, 1.5790413250028876], [2.789502875694011, 2.05027100272252, 1.6206037335904169, 1.609361043544037, 1.5736959250822544, 1.565914453421025, 1.56433775547408, 1.56433775547408], [2.7891645344327056, 2.0495789353187406, 1.6225178001355536, 1.637931354858589, 1.5896712431907034, 1.57215273101062, 1.57215273101062]]"
        self.expected_objective_curves = "[([0, 60, 90, 120, 240, 1000], [2.7854035060729516, 2.0434955300959756, 1.6067949860630397, 1.5953185527727127, 1.551094392839577, 1.551094392839577]), ([0, 60, 90, 240, 480, 1000], [2.7854035060729516, 2.0454590444287932, 1.6112733855379604, 1.5591639731106854, 1.552416814468584, 1.552416814468584]), ([0, 60, 90, 120, 240, 480, 1000], [2.7854035060729516, 2.04797163822759, 1.6205137429033964, 1.6163253621532803, 1.5759105545236445, 1.5664687724992692, 1.5664687724992692]), ([0, 60, 90, 240, 720, 1000], [2.7854035060729516, 2.051164328901086, 1.6227059425210009, 1.5749298888776202, 1.5649659079370986, 1.5649659079370986]), ([0, 60, 90, 120, 240, 1000], [2.7854035060729516, 2.0438598971642126, 1.608137408927407, 1.5763337097688268, 1.5521300465336114, 1.5521300465336114]), ([0, 60, 90, 120, 240, 960, 1000], [2.7854035060729516, 2.050284006246868, 1.6215173623533616, 1.620189560285288, 1.5752097928232494, 1.5716967873961198, 1.5716967873961198]), ([0, 60, 90, 120, 240, 1000], [2.7854035060729516, 2.042523510862577, 1.6072035528477624, 1.5934064648951312, 1.5584508725578103, 1.5584508725578103]), ([0, 60, 90, 120, 240, 720, 1000], [2.7854035060729516, 2.052665196346358, 1.628996218505635, 1.6431966479141764, 1.604964304544481, 1.5790413250028876, 1.5790413250028876]), ([0, 60, 90, 120, 240, 480, 720, 1000], [2.7854035060729516, 2.05027100272252, 1.6206037335904169, 1.609361043544037, 1.5736959250822544, 1.565914453421025, 1.56433775547408, 1.56433775547408]), ([0, 60, 90, 120, 240, 480, 1000], [2.7854035060729516, 2.0495789353187406, 1.6225178001355536, 1.637931354858589, 1.5896712431907034, 1.57215273101062, 1.57215273101062])]"
        self.expected_progress_curves = "[([0.0, 0.06, 0.09, 0.12, 0.24, 1.0], [1.0, 0.39828387359834183, 0.04410280497326267, 0.03479497272456759, -0.0010725352009161786, -0.0010725352009161786]), ([0.0, 0.06, 0.09, 0.24, 0.48, 1.0], [1.0, 0.3998763598321249, 0.0477349605394297, 0.005472207192538373, 0.0, 0.0]), ([0.0, 0.06, 0.09, 0.12, 0.24, 0.48, 1.0], [1.0, 0.4019141707962702, 0.05522924853812047, 0.05183230939949421, 0.019054333850506037, 0.011396682645780056, 0.011396682645780056]), ([0.0, 0.06, 0.09, 0.24, 0.72, 1.0], [1.0, 0.40450356668775517, 0.05700720740217919, 0.018258975998955967, 0.010177801231727564, 0.010177801231727564]), ([0.0, 0.06, 0.09, 0.12, 0.24, 1.0], [1.0, 0.3985793894143016, 0.045191561951345285, 0.019397529156719533, -0.00023257991097970873, -0.00023257991097970873]), ([0.0, 0.06, 0.09, 0.12, 0.24, 0.96, 1.0], [1.0, 0.40378959089206146, 0.05604322281440343, 0.054966323868847214, 0.018485988948515823, 0.01563680537577314, 0.01563680537577314]), ([0.0, 0.06, 0.09, 0.12, 0.24, 1.0], [1.0, 0.39749552832258417, 0.04443416847256455, 0.03324419533937655, 0.004893855002907509, 0.004893855002907509]), ([0.0, 0.06, 0.09, 0.12, 0.24, 0.72, 1.0], [1.0, 0.4057208283625906, 0.062108865049796805, 0.0736259637380752, 0.04261805130071767, 0.02159351006429743, 0.02159351006429743]), ([0.0, 0.06, 0.09, 0.12, 0.24, 0.48, 0.72, 1.0], [1.0, 0.4037790445297718, 0.05530223447351879, 0.046183977056035434, 0.017258183529930905, 0.010947108386772498, 0.009668345235733681, 0.009668345235733681]), ([0.0, 0.06, 0.09, 0.12, 0.24, 0.48, 1.0], [1.0, 0.4032177510393459, 0.05685461663479428, 0.06935560697636822, 0.03021478575218347, 0.016006593320448285, 0.016006593320448285])]"

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
