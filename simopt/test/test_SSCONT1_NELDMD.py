import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class test_SSCONT1_NELDMD(unittest.TestCase):
    def setUp(self):
        # Expected values
        self.expected_problem_name = "SSCONT-1"
        self.expected_solver_name = "NELDMD"
        self.expected_all_recommended_xs = "[[(600, 600), (386.50898996914805, 420.1320551157649), (378.6017084362248, 415.67079758302384), (384.72863772603625, 415.3550700801035), (384.72863772603625, 415.3550700801035)], [(600, 600), (97.68228084475709, 746.8016557398353), (192.60441917479966, 497.7546679304735), (404.7040984405264, 380.4423000756949), (463.3489817843795, 135.08147325196518), (602.8941510777272, 25.141427368450763), (602.8941510777272, 25.141427368450763)], [(600, 600), (397.50753683570093, 261.5403268770166), (641.516908702544, 0), (609.6900641937873, 81.53934371392694), (638.8655626517414, 97.69360570859973), (638.8655626517414, 97.69360570859973)], [(600, 600), (403.21990585682386, 165.71476201389407), (634.6937675396191, 104.83279038197128), (535.8854112122359, 167.27440056284132), (539.2395043970282, 158.0383625323855), (539.2395043970282, 158.0383625323855)], [(600, 600), (379.3932928263026, 480.793598757748), (351.91642412280794, 403.0532414570137), (338.6952203778805, 408.62149229505553), (335.13130413823876, 387.7943402603615), (331.0643317938014, 395.08922233330065), (335.64144749404636, 395.9525379507494), (335.64144749404636, 395.9525379507494)], [(600, 600), (373.36516544516064, 405.7402513642893), (339.442132732995, 438.03790108331475), (329.73435016980113, 365.27341915119587), (346.69586652588396, 349.1245942916831), (377.8987490657163, 350.1694346195193), (436.85751218995745, 346.5714285597352), (436.85751218995745, 346.5714285597352)], [(600, 600), (410.5130965152081, 560.9679254242766), (131.84419720317976, 612.57542465047), (168.07645281200163, 556.4496670513751), (386.1361911982283, 503.7126032319561), (252.8959678714591, 543.830183393133), (379.95868927732135, 453.896587461982), (460.91726563621205, 410.14209394750617), (463.72969406997515, 397.99116611778754), (462.3234798530936, 404.06663003264686), (504.20898224942044, 376.1139193605496), (504.20898224942044, 376.1139193605496)], [(600, 600), (325.1173485120577, 320.0626123028588), (505.6377521001699, 427.65031195988803), (411.27550420033987, 255.30062391977617), (391.7869883311563, 330.7690401213455), (434.09598331377543, 435.81098819179374), (434.09598331377543, 435.81098819179374)], [(600, 600), (617.0189059187588, 71.65654768451077), (580.8391424867718, 175.82935290011238), (590.4159241547297, 133.27541871135293), (585.6275333207508, 154.55238580573265), (607.4421242508008, 114.21048187327025), (608.7726380478678, 136.5934640957985), (596.8674572350426, 139.97717939513353), (596.8674572350426, 139.97717939513353)], [(600, 600), (752.6553518700596, 1.0000002248489182e-07), (676.4634009641036, 116.08518829901602), (685.987394827348, 101.57453977413903), (680.1462105255633, 106.29891928370029), (679.7651018202796, 110.01095891396784), (679.2493274292785, 108.5134840606375), (679.5072146247791, 109.26222148730267), (679.5072146247791, 109.26222148730267)]]"
        self.expected_all_intermediate_budgets = "[[0, 510, 630, 750, 1000], [0, 150, 270, 390, 510, 570, 1000], [0, 270, 330, 540, 600, 1000], [0, 150, 270, 450, 690, 1000], [0, 210, 390, 510, 630, 780, 960, 1000], [0, 150, 270, 330, 510, 630, 780, 1000], [0, 150, 210, 270, 390, 450, 510, 690, 750, 870, 930, 1000], [0, 240, 300, 360, 420, 540, 1000], [0, 210, 390, 450, 570, 630, 690, 750, 1000], [0, 150, 420, 600, 660, 720, 840, 960, 1000]]"
        self.expected_all_est_objectives = "[[618.5809976153716, 531.1444951247202, 530.1364995728987, 530.3231123326633, 530.3231123326633], [619.371245290233, 548.332940129753, 541.0318624791117, 527.3543164453117, 528.4834974637095, 527.6341320229454, 527.6341320229454], [620.2040298994102, 533.0586082668939, 533.0168219505572, 519.1923884743609, 520.4311467807754, 520.4311467807754], [620.3929887875448, 541.1619522590145, 516.1767385555041, 521.1693989681909, 521.8596473125608, 521.8596473125608], [617.140803174291, 532.6667375301192, 526.7604330470449, 527.5683854750222, 526.7401317499002, 527.5835903022498, 527.4895877336083, 527.4895877336083], [617.6250759903628, 528.8195081311449, 529.429614265211, 531.2991127132857, 530.3987414502466, 527.3496986321633, 524.9445217806227, 524.9445217806227], [622.8299886318688, 547.7415327166732, 538.5473977797091, 535.6284315555572, 535.02527887644, 531.400388170159, 529.994446511586, 531.8332060038596, 531.6220618557606, 532.8284844056692, 531.6477083039185, 531.6477083039185], [617.1638109984892, 539.7762638951599, 545.9728696725527, 532.990322448582, 533.7926164793174, 538.8239584900106, 538.8239584900106], [625.4509909440814, 520.6197506321978, 523.5295041592052, 522.233631875064, 520.1716655029037, 520.279738767584, 522.0032336129273, 521.0412373327464, 521.0412373327464], [616.3517529689802, 530.573410838247, 522.5942511189746, 521.7261752324249, 520.9234075273365, 520.6150065414831, 520.1255712985693, 520.7072236244998, 520.7072236244998]]"
        self.expected_objective_curves = "[([0, 510, 630, 750, 1000], [624.4131899421741, 531.1444951247202, 530.1364995728987, 530.3231123326633, 530.3231123326633]), ([0, 150, 270, 390, 510, 570, 1000], [624.4131899421741, 548.332940129753, 541.0318624791117, 527.3543164453117, 528.4834974637095, 527.6341320229454, 527.6341320229454]), ([0, 270, 330, 540, 600, 1000], [624.4131899421741, 533.0586082668939, 533.0168219505572, 519.1923884743609, 520.4311467807754, 520.4311467807754]), ([0, 150, 270, 450, 690, 1000], [624.4131899421741, 541.1619522590145, 520.4787581473157, 521.1693989681909, 521.8596473125608, 521.8596473125608]), ([0, 210, 390, 510, 630, 780, 960, 1000], [624.4131899421741, 532.6667375301192, 526.7604330470449, 527.5683854750222, 526.7401317499002, 527.5835903022498, 527.4895877336083, 527.4895877336083]), ([0, 150, 270, 330, 510, 630, 780, 1000], [624.4131899421741, 528.8195081311449, 529.429614265211, 531.2991127132857, 530.3987414502466, 527.3496986321633, 524.9445217806227, 524.9445217806227]), ([0, 150, 210, 270, 390, 450, 510, 690, 750, 870, 930, 1000], [624.4131899421741, 547.7415327166732, 538.5473977797091, 535.6284315555572, 535.02527887644, 531.400388170159, 529.994446511586, 531.8332060038596, 531.6220618557606, 532.8284844056692, 531.6477083039185, 531.6477083039185]), ([0, 240, 300, 360, 420, 540, 1000], [624.4131899421741, 539.7762638951599, 545.9728696725527, 532.990322448582, 533.7926164793174, 538.8239584900106, 538.8239584900106]), ([0, 210, 390, 450, 570, 630, 690, 750, 1000], [624.4131899421741, 520.6197506321978, 523.5295041592052, 522.233631875064, 520.1716655029037, 520.279738767584, 522.0032336129273, 521.0412373327464, 521.0412373327464]), ([0, 150, 420, 600, 660, 720, 840, 960, 1000], [624.4131899421741, 530.573410838247, 522.5942511189746, 521.7261752324249, 520.9234075273365, 520.6150065414831, 520.1255712985693, 520.7072236244998, 520.7072236244998])]"
        self.expected_progress_curves = "[([0.0, 0.51, 0.63, 0.75, 1.0], [1.0, 0.10261986132233869, 0.09292148192665439, 0.09471696737398802, 0.09471696737398802]), ([0.0, 0.15, 0.27, 0.39, 0.51, 0.57, 1.0], [1.0, 0.26799763563834855, 0.19775067777695574, 0.06615284443529465, 0.07701720380973626, 0.06884507618950293, 0.06884507618950293]), ([0.0, 0.27, 0.33, 0.54, 0.6, 1.0], [1.0, 0.12103640634133425, 0.12063436136340842, -0.012376742247399002, -0.0004580904106377613, -0.0004580904106377613]), ([0.0, 0.15, 0.27, 0.45, 0.69, 1.0], [1.0, 0.19900233016641122, 0.0, 0.006644966532730167, 0.013286156872157994, 0.013286156872157994]), ([0.0, 0.21, 0.39, 0.51, 0.63, 0.78, 0.96, 1.0], [1.0, 0.11726604141021972, 0.060438824663300236, 0.06821249902726836, 0.06024349673593215, 0.068358791521152, 0.06745435045173793, 0.06745435045173793]), ([0.0, 0.15, 0.27, 0.33, 0.51, 0.63, 0.78, 1.0], [1.0, 0.08025011384381049, 0.08612021986671434, 0.10410750681089749, 0.09544462919189789, 0.06610841437425799, 0.04296712413958592, 0.04296712413958592]), ([0.0, 0.15, 0.21, 0.27, 0.39, 0.45, 0.51, 0.69, 0.75, 0.87, 0.93, 1.0], [1.0, 0.26230743843549026, 0.17384652343177764, 0.14576183413542207, 0.13995863043573095, 0.1050819236150724, 0.09155472541623168, 0.10924625901601899, 0.10721474602794905, 0.11882228098123326, 0.10746150206168102, 0.10746150206168102]), ([0.0, 0.24, 0.3, 0.36, 0.42, 0.54, 1.0], [1.0, 0.18566999804196582, 0.2452903343480656, 0.12037939771452354, 0.12809863008901673, 0.17650743864077573, 0.17650743864077573]), ([0.0, 0.21, 0.39, 0.45, 0.57, 0.63, 0.69, 0.75, 1.0], [1.0, 0.0013565522266996354, 0.02935260201268938, 0.016884430861295357, -0.0029546767044260213, -0.0019148551283226054, 0.01466766536637727, 0.005411865689907979, 0.005411865689907979]), ([0.0, 0.15, 0.42, 0.6, 0.66, 0.72, 0.84, 0.96, 1.0], [1.0, 0.09712520207793814, 0.020354111097989017, 0.012001961848132226, 0.004278172039256663, 0.001310907192299195, -0.003398169813864227, 0.00219816929999657, 0.00219816929999657])]"

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
