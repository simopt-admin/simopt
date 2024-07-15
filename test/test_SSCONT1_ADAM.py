import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class test_SSCONT1_ADAM(unittest.TestCase):
    def setUp(self):
        # Expected values
        self.expected_problem_name = "SSCONT-1"
        self.expected_solver_name = "ADAM"
        self.expected_all_recommended_xs = "[[(600, 600), (599.5000000110613, 599.5000000081036), (599.0000848587781, 599.0036855781585), (598.500233148688, 598.5283960523874), (598.0004142824321, 598.1962889559256), (597.5005732251103, 597.8765810889952), (597.0009056240991, 597.5277050755282), (597.0009056240991, 597.5277050755282)], [(600, 600), (599.5000000106548, 599.5000000069507), (598.9999793245573, 599.0073258918422), (598.4999507534643, 598.5869982260906), (597.9999369081033, 598.1837008487945), (597.4999946200007, 597.7854826371071), (597.0001798175526, 597.3569500216134), (597.0001798175526, 597.3569500216134)], [(600, 600), (599.5000000106439, 600.4999999973123), (598.0001122516425, 601.4439703205642), (597.5002662928774, 601.5451399240679), (597.000541764238, 601.6034739070757), (597.000541764238, 601.6034739070757)], [(600, 600), (599.500000010782, 599.5000000048257), (599.0000794382208, 598.9994997723954), (598.5001772433972, 598.5143649336619), (598.0003482521668, 598.0588411071055), (597.5006141233453, 597.8652770976229), (597.000958092979, 597.8274615873223), (597.000958092979, 597.8274615873223)], [(600, 600), (599.5000000108216, 600.4999999946197), (599.0001384485261, 600.9713236765305), (598.500317967374, 601.2182769831927), (598.0005232924763, 601.3940818069407), (597.5007521919945, 601.5582746713177), (597.0010282598391, 601.8000046600027), (597.0010282598391, 601.8000046600027)], [(600, 600), (599.5000000110132, 599.5000000060979), (598.9999825055289, 599.0203657389802), (598.4999610281761, 598.5362534599152), (598.0000645677142, 598.0477005399413), (597.0006560599502, 597.0778790205106), (597.0006560599502, 597.0778790205106)], [(600, 600), (599.5000000113104, 599.5000000324784), (599.0000219944855, 599.0285304279078), (598.500095037853, 598.6241906499826), (598.0000981418592, 598.1880238534483), (597.5003232253913, 597.8077081611476), (597.0009277934951, 597.4286114173113), (597.0009277934951, 597.4286114173113)], [(600, 600), (599.5000000108464, 599.5000000070715), (599.0000496244954, 599.007577540642), (598.4999905786697, 598.5577909296176), (597.9999672369455, 598.088792655785), (597.4999378361583, 597.7335799885267), (597.4999378361583, 597.7335799885267)], [(600, 600), (599.5000000116346, 599.5000000022575), (598.9997940694826, 599.0579904833044), (598.4992443011204, 598.6169396732552), (597.998643610039, 598.1550342095941), (597.4981017223334, 597.7249841407158), (596.997713328104, 597.3196267624867), (596.997713328104, 597.3196267624867)], [(600, 600), (599.5000000108854, 600.4999999113443), (598.9999355896637, 600.1564754116206), (598.4999561931775, 599.8369245005092), (598.0001133522595, 599.4965467140914), (597.5004350266794, 599.1401685856582), (597.0008392492098, 598.7431310510749), (597.0008392492098, 598.7431310510749)]]"
        self.expected_all_intermediate_budgets = "[[0, 180, 330, 480, 630, 780, 930, 1000], [0, 180, 330, 480, 630, 780, 930, 1000], [0, 180, 630, 780, 930, 1000], [0, 180, 330, 480, 630, 780, 930, 1000], [0, 180, 330, 480, 630, 780, 930, 1000], [0, 180, 330, 480, 630, 930, 1000], [0, 180, 330, 480, 630, 780, 930, 1000], [0, 180, 330, 480, 630, 780, 1000], [0, 180, 330, 480, 630, 780, 930, 1000], [0, 180, 330, 480, 630, 780, 930, 1000]]"
        self.expected_all_est_objectives = "[[618.5809976153716, 618.531757149935, 618.1283744102988, 617.6994108978113, 617.3382879798855, 616.9234623271001, 616.8273343936545, 616.8273343936545], [619.371245290233, 618.4090044791473, 618.1921734669342, 617.8733569055988, 617.56288464254, 617.2687985789329, 616.8349930886011, 616.8349930886011], [620.2040298994102, 620.3298105858942, 619.9558807673388, 619.7806561682831, 619.6565552416639, 619.6565552416639], [620.3929887875448, 619.9074934621298, 619.6409871932873, 618.9013254496462, 618.5994949688534, 618.2842032593941, 618.0442396902184, 618.0442396902184], [617.140803174291, 616.7666205278667, 616.8475687559796, 616.7215383518055, 616.4955778007785, 616.0258010406919, 615.8111142751699, 615.8111142751699], [617.6250759903628, 617.5791502039027, 617.287224563433, 617.0000910692144, 616.9688138482924, 616.3161545013147, 616.3161545013147], [622.8299886318688, 622.4531513499861, 622.1881268669069, 622.3171551557701, 621.9735655078604, 622.0033367354589, 621.8159738691305, 621.8159738691305], [617.1638109984892, 617.0750675759077, 616.9412611383488, 616.3221502243505, 615.9220089162174, 615.6127552567112, 615.6127552567112], [625.4509909440814, 625.4068859122034, 624.9020879871041, 624.3509920652422, 623.4957096944308, 623.1151937020346, 622.6399022742665, 622.6399022742665], [616.3517529689802, 616.6000497968287, 615.9213241724925, 615.8908618237317, 615.1410630285519, 614.714221406568, 614.4575096177534, 614.4575096177534]]"
        self.expected_objective_curves = "[([0, 180, 330, 480, 630, 780, 930, 1000], [624.4131899421741, 618.531757149935, 618.1283744102988, 617.6994108978113, 617.3382879798855, 616.9234623271001, 616.8273343936545, 616.8273343936545]), ([0, 180, 330, 480, 630, 780, 930, 1000], [624.4131899421741, 618.4090044791473, 618.1921734669342, 617.8733569055988, 617.56288464254, 617.2687985789329, 616.8349930886011, 616.8349930886011]), ([0, 180, 630, 780, 930, 1000], [624.4131899421741, 620.3298105858942, 619.9558807673388, 619.7806561682831, 619.6565552416639, 619.6565552416639]), ([0, 180, 330, 480, 630, 780, 930, 1000], [624.4131899421741, 619.9074934621298, 619.6409871932873, 618.9013254496462, 618.5994949688534, 618.2842032593941, 618.0442396902184, 618.0442396902184]), ([0, 180, 330, 480, 630, 780, 930, 1000], [624.4131899421741, 616.7666205278667, 616.8475687559796, 616.7215383518055, 616.4955778007785, 616.0258010406919, 615.8111142751699, 615.8111142751699]), ([0, 180, 330, 480, 630, 930, 1000], [624.4131899421741, 617.5791502039027, 617.287224563433, 617.0000910692144, 616.9688138482924, 616.3161545013147, 616.3161545013147]), ([0, 180, 330, 480, 630, 780, 930, 1000], [624.4131899421741, 622.4531513499861, 622.1881268669069, 622.3171551557701, 621.9735655078604, 622.0033367354589, 621.8159738691305, 621.8159738691305]), ([0, 180, 330, 480, 630, 780, 1000], [624.4131899421741, 617.0750675759077, 616.9412611383488, 616.3221502243505, 615.9220089162174, 615.6127552567112, 615.6127552567112]), ([0, 180, 330, 480, 630, 780, 930, 1000], [624.4131899421741, 625.4068859122034, 624.9020879871041, 624.3509920652422, 623.4957096944308, 623.1151937020346, 622.6399022742665, 622.6399022742665]), ([0, 180, 330, 480, 630, 780, 930, 1000], [624.4131899421741, 616.6000497968287, 615.9213241724925, 615.8908618237317, 615.1410630285519, 614.714221406568, 623.0835454380884, 623.0835454380884])]"
        self.expected_progress_curves = "[([0.0, 0.18, 0.33, 0.48, 0.63, 0.78, 0.93, 1.0], [1.0, -3.423312226814787, -3.7266886092964926, -4.0493037979197615, -4.320897383134535, -4.632879760011062, -4.705175725699794, -4.705175725699794]), ([0.0, 0.18, 0.33, 0.48, 0.63, 0.78, 0.93, 1.0], [1.0, -3.51563214421414, -3.678706568653786, -3.9184823586154525, -4.151982562696198, -4.373159022045631, -4.699415768866922, -4.699415768866922]), ([0.0, 0.18, 0.63, 0.78, 0.93, 1.0], [1.0, -2.071030898659577, -2.3522563069595672, -2.4840393501092786, -2.577373264729243, -2.577373264729243]), ([0.0, 0.18, 0.33, 0.48, 0.63, 0.78, 0.93, 1.0], [1.0, -2.3886474664464785, -2.589081693808433, -3.1453670327605736, -3.3723679189864235, -3.609492735800612, -3.7899647104060783, -3.7899647104060783]), ([0.0, 0.18, 0.33, 0.48, 0.63, 0.78, 0.93, 1.0], [1.0, -4.750837453779144, -4.689957851852336, -4.7847428893468384, -4.954683463938541, -5.307993509325164, -5.469455287163062, -5.469455287163062]), ([0.0, 0.18, 0.33, 0.48, 0.63, 0.93, 1.0], [1.0, -4.139749547546097, -4.359301194300303, -4.575248760237198, -4.598771755162447, -5.089624268726942, -5.089624268726942]), ([0.0, 0.18, 0.33, 0.48, 0.63, 0.78, 0.93, 1.0], [1.0, -0.47410724157117545, -0.6734270464248958, -0.5763873576458829, -0.8347945084700358, -0.8124041420923677, -0.9533161420688037, -0.9533161420688037]), ([0.0, 0.18, 0.33, 0.48, 0.63, 0.78, 1.0], [1.0, -4.518860374873308, -4.619493617178165, -5.085115001011141, -5.386053565344389, -5.618637281184116, -5.618637281184116]), ([0.0, 0.18, 0.33, 0.48, 0.63, 0.78, 0.93, 1.0], [1.0, 1.7473395836074526, 1.3676907951168564, 0.9532221757463738, 0.3099807919153838, 0.023802049231154792, -0.33365547141261426, -0.33365547141261426]), ([0.0, 0.18, 0.33, 0.48, 0.63, 0.78, 0.93, 1.0], [1.0, -4.876112089613185, -5.386568547900048, -5.409478693181139, -5.973387913183827, -6.294407268863019, 0.0, 0.0])]"

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
