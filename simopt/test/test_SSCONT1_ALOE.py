import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class test_SSCONT1_ALOE(unittest.TestCase):
    def setUp(self):
        # Expected values
        self.expected_problem_name = "SSCONT-1"
        self.expected_solver_name = "ALOE"
        self.expected_all_recommended_xs = "[[(600, 600), (599.5478845471159, 597.6234948266642), (598.9854931466446, 597.7286013820234), (598.2840709663377, 597.5139554739116), (597.411952419828, 597.4969254527642), (596.3275423764765, 597.1705808410154), (594.9803976786561, 596.752406551361), (594.9803976786561, 596.752406551361)], [(600, 600), (599.531122964027, 599.8394100357302), (598.9408094549597, 599.6802392753373), (598.2055754970447, 599.3567076404452), (597.2884980466814, 598.5707240416492), (596.1486891857113, 598.2679161287234), (594.7305408946756, 597.0258174722401), (594.7305408946756, 597.0258174722401)], [(600, 600), (599.530473384621, 600.0406952841537), (598.945001440669, 600.0247097817362), (598.2157850030947, 600.0462266858261), (597.3077727087264, 600.4056016764685), (596.1757185511776, 600.6281605255718), (594.7675020013168, 600.620059042839), (594.7675020013168, 600.620059042839)], [(600, 600), (599.5364840109689, 597.7313724173246), (598.9616412231529, 597.9097128292206), (598.2440936270514, 597.8325866327859), (596.2327582189782, 596.260043317129), (594.8477444327689, 595.5850078833494), (594.8477444327689, 595.5850078833494)], [(600, 600), (599.5384149694582, 600.6518052743465), (598.9667974757788, 601.0970571205572), (598.2518848163102, 601.7187171569632), (597.3591929524073, 601.8150508842651), (596.2450595209932, 601.8458886552901), (594.855234429275, 602.8368380047193), (594.855234429275, 602.8368380047193)], [(600, 600), (599.5459986101321, 599.6019366179091), (598.9778900877561, 599.0092963953332), (598.268462476813, 598.1128362511927), (597.3874136647937, 597.0994084320916), (596.293950047366, 595.9809368384783), (594.9238067718225, 592.6876740708262), (594.9238067718225, 592.6876740708262)], [(600, 600), (599.557802471425, 599.6644445718039), (599.0058115189174, 598.8271434616643), (598.3113834033122, 597.9532148655614), (597.4563611622083, 598.0828747862189), (596.393545576457, 598.0216610726663), (595.073453208189, 597.3116744487444), (595.073453208189, 597.3116744487444)], [(600, 600), (599.5388270369399, 599.5305308550694), (598.9644526495208, 598.2649849386937), (597.3277323381794, 595.472665863474), (596.1950491111667, 593.7773553323472), (594.7997571144944, 589.8942858072046), (594.7997571144944, 589.8942858072046)], [(600, 600), (599.5701319575628, 598.5320268617802), (599.0163418710342, 597.6799022987589), (598.3274916047529, 596.3512375416697), (597.4705971433023, 595.6055070851947), (596.4088439531092, 594.7677070335094), (595.0797881503252, 593.4464465618212), (595.0797881503252, 593.4464465618212)], [(600, 600), (599.5403430208623, 599.5359596729973), (598.9684497424306, 599.227521342894), (598.2552552267211, 598.8684196619994), (597.3658999962784, 598.594840156999), (596.2594228747997, 598.4662060388578), (594.8843077799439, 598.2793029697884), (594.8843077799439, 598.2793029697884)]]"
        self.expected_all_intermediate_budgets = "[[0, 180, 330, 480, 630, 780, 930, 1000], [0, 180, 330, 480, 630, 780, 930, 1000], [0, 180, 330, 480, 630, 780, 930, 1000], [0, 180, 330, 480, 780, 930, 1000], [0, 180, 330, 480, 630, 780, 930, 1000], [0, 180, 330, 480, 630, 780, 930, 1000], [0, 180, 330, 480, 630, 780, 930, 1000], [0, 180, 330, 630, 780, 930, 1000], [0, 180, 330, 480, 630, 780, 930, 1000], [0, 180, 330, 480, 630, 780, 930, 1000]]"
        self.expected_all_est_objectives = "[[618.5809976153716, 617.8328760145909, 617.5866951493366, 617.4002141660097, 616.9986616131891, 616.9688874875611, 616.2988253152213, 616.2988253152213], [619.371245290233, 619.0263408513159, 618.3871009232229, 617.9436221163096, 617.3218034677584, 616.7210969426787, 615.7254182564449, 615.7254182564449], [620.2040298994102, 620.0135885617987, 619.7469350147244, 619.433539461188, 619.3182757282156, 618.8150547917851, 618.194022546352, 618.194022546352], [620.3929887875448, 619.1415054491042, 618.9489461024248, 618.5957964430839, 616.9476853937638, 615.8688364205649, 615.8688364205649], [617.140803174291, 616.9450216642482, 616.9850553850752, 616.3853211582018, 615.9554916001213, 615.4508141460359, 615.0332682800488, 615.0332682800488], [617.6250759903628, 617.2738072504757, 617.2724589361178, 617.187161568923, 616.4979939660135, 615.5423792680523, 614.0177838269694, 614.0177838269694], [622.8299886318688, 622.4761536080149, 622.4453805276625, 622.0787286210466, 621.7553948360692, 621.1851357587685, 620.6611130349339, 620.6611130349339], [617.1638109984892, 616.9212623515181, 616.374105502157, 616.634560563609, 615.8495487283104, 614.6233786732573, 614.6233786732573], [625.4509909440814, 624.5267028976504, 623.8127784315343, 623.6566958639733, 623.0271075822885, 622.0380157690374, 621.1287228778968, 621.1287228778968], [616.3517529689802, 615.817983100305, 615.3058035688673, 615.034236712977, 614.5005086219048, 614.1052419941436, 613.2840165089626, 613.2840165089626]]"
        self.expected_objective_curves = "[([0, 180, 330, 480, 630, 780, 930, 1000], [624.4131899421741, 617.8328760145909, 617.5866951493366, 617.4002141660097, 616.9986616131891, 616.9688874875611, 616.2988253152213, 616.2988253152213]), ([0, 180, 330, 480, 630, 780, 930, 1000], [624.4131899421741, 619.0263408513159, 618.3871009232229, 617.9436221163096, 617.3218034677584, 616.7210969426787, 615.7254182564449, 615.7254182564449]), ([0, 180, 330, 480, 630, 780, 930, 1000], [624.4131899421741, 620.0135885617987, 619.7469350147244, 619.433539461188, 619.3182757282156, 618.8150547917851, 618.194022546352, 618.194022546352]), ([0, 180, 330, 480, 780, 930, 1000], [624.4131899421741, 619.1415054491042, 618.9489461024248, 618.5957964430839, 616.9476853937638, 615.8688364205649, 615.8688364205649]), ([0, 180, 330, 480, 630, 780, 930, 1000], [624.4131899421741, 616.9450216642482, 616.9850553850752, 616.3853211582018, 615.9554916001213, 615.4508141460359, 615.0332682800488, 615.0332682800488]), ([0, 180, 330, 480, 630, 780, 930, 1000], [624.4131899421741, 617.2738072504757, 617.2724589361178, 617.187161568923, 616.4979939660135, 615.5423792680523, 614.0177838269694, 614.0177838269694]), ([0, 180, 330, 480, 630, 780, 930, 1000], [624.4131899421741, 622.4761536080149, 622.4453805276625, 622.0787286210466, 621.7553948360692, 621.1851357587685, 620.6611130349339, 620.6611130349339]), ([0, 180, 330, 630, 780, 930, 1000], [624.4131899421741, 616.9212623515181, 616.374105502157, 616.634560563609, 615.8495487283104, 614.6233786732573, 614.6233786732573]), ([0, 180, 330, 480, 630, 780, 930, 1000], [624.4131899421741, 624.5267028976504, 623.8127784315343, 623.6566958639733, 623.0271075822885, 622.0380157690374, 621.1287228778968, 621.1287228778968]), ([0, 180, 330, 480, 630, 780, 930, 1000], [624.4131899421741, 615.817983100305, 615.3058035688673, 615.034236712977, 614.5005086219048, 614.1052419941436, 622.4723209735103, 622.4723209735103])]"
        self.expected_progress_curves = "[([0.0, 0.18, 0.33, 0.48, 0.63, 0.78, 0.93, 1.0], [1.0, -2.390395762838828, -2.5172363013960113, -2.6133174827316346, -2.8202106626957786, -2.835551278733615, -3.180789511277018, -3.180789511277018]), ([0.0, 0.18, 0.33, 0.48, 0.63, 0.78, 0.93, 1.0], [1.0, -1.7754831355599785, -2.1048407266255533, -2.333335701852337, -2.653717272473913, -2.963221177569169, -3.4762278268121287, -3.4762278268121287]), ([0.0, 0.18, 0.33, 0.48, 0.63, 0.78, 0.93, 1.0], [1.0, -1.26682040436984, -1.4042091469276872, -1.5656809199305248, -1.6250686142228632, -1.8843447140292822, -2.204321103708289, -2.204321103708289]), ([0.0, 0.18, 0.33, 0.48, 0.78, 0.93, 1.0], [1.0, -1.7161465189992133, -1.815359474530163, -1.9973138800271761, -2.8464752999528002, -3.402334036744087, -3.402334036744087]), ([0.0, 0.18, 0.33, 0.48, 0.63, 0.78, 0.93, 1.0], [1.0, -2.847847741657306, -2.8272210422389956, -3.1362239870829387, -3.357686417066789, -3.6177129630282123, -3.8328464278465826, -3.8328464278465826]), ([0.0, 0.18, 0.33, 0.48, 0.63, 0.78, 0.93, 1.0], [1.0, -2.6784465138898197, -2.679141210121064, -2.7230892398809163, -3.0781712232792806, -3.5705355783130415, -4.356057664398244, -4.356057664398244]), ([0.0, 0.18, 0.33, 0.48, 0.63, 0.78, 0.93, 1.0], [1.0, 0.0019747002845254555, -0.013880610325974808, -0.20279182099274826, -0.36938410012018275, -0.6632004712960456, -0.9331943412044101, -0.9331943412044101]), ([0.0, 0.18, 0.33, 0.63, 0.78, 0.93, 1.0], [1.0, -2.860089326799703, -3.1420026646886217, -3.0078075872994763, -3.4122716948579908, -4.044035134250421, -4.044035134250421]), ([0.0, 0.18, 0.33, 0.48, 0.63, 0.78, 0.93, 1.0], [1.0, 1.058485635717291, 0.6906480961189603, 0.6102291857849483, 0.28584444273958765, -0.22376843129797294, -0.6922662566646028, -0.6922662566646028]), ([0.0, 0.18, 0.33, 0.48, 0.63, 0.78, 0.93, 1.0], [1.0, -3.4285353522789968, -3.6924272170607355, -3.832347459114633, -4.1073418557943695, -4.310996318894636, 0.0, 0.0])]"

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
