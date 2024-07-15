import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class test_RMITD1_RNDSRCH(unittest.TestCase):
    def setUp(self):
        # Expected values
        self.expected_problem_name = "RMITD-1"
        self.expected_solver_name = "RNDSRCH"
        self.expected_all_recommended_xs = "[[(100, 50, 30), (175.08312738903115, 53.49231714531826, 0.5331458316404217), (166.41214546135774, 30.63243575662995, 8.620546616863855), (140.5243724186601, 11.884035093681725, 2.6596140473149075), (156.02410134231047, 34.125807811079554, 7.328905985786684), (137.76632194765727, 29.455936077710877, 0.5305126566315612), (137.76632194765727, 29.455936077710877, 0.5305126566315612)], [(100, 50, 30), (74.67673442623597, 61.8070442825242, 3.8593686657838253), (124.39646354747576, 11.170866834824045, 6.5803749413969905), (196.81240057968054, 142.57180552346065, 5.375621960994175), (196.01101488114594, 99.49375942691742, 0.8546394709881885), (199.60318298951307, 16.12808610187888, 0.49209946355705353), (199.60318298951307, 16.12808610187888, 0.49209946355705353)], [(100, 50, 30), (128.07894722568363, 103.57650517111483, 2.1196934489757377), (180.031163628791, 153.77454384814604, 0.6767062332376131), (180.031163628791, 153.77454384814604, 0.6767062332376131)], [(100, 50, 30), (73.54768586762219, 59.0969318738584, 4.8556434479481165), (103.95805487019835, 16.78715792757665, 9.87193543775067), (87.09110541151603, 22.306904997638487, 6.649804763300203), (108.62152208417575, 13.049186327082749, 2.4801144180502277), (41.685374656356394, 18.023337272194716, 1.889385607324589), (64.31580995621357, 9.641142656411434, 0.5292650102840556), (64.31580995621357, 9.641142656411434, 0.5292650102840556)], [(100, 50, 30), (114.29500025076793, 11.828055945279925, 8.327346791533785), (87.86185678915732, 18.224624216258956, 14.876355951251938), (74.29960240943295, 28.197069760642602, 4.3169785518971135), (52.95440093952124, 13.73577054055414, 11.679573270806868), (44.775899479488636, 15.334883935203743, 2.184397087980666), (58.52546793718295, 10.655472058881585, 3.73067981004282), (39.32439879967714, 7.681895698847787, 2.380161288909975), (39.32439879967714, 7.681895698847787, 2.380161288909975)], [(100, 50, 30), (187.47880607740763, 64.3202458458513, 30.492683020997347), (179.14687107842167, 40.23227327696813, 29.360415997674348), (147.9360727525091, 43.32850845817704, 3.5323898621688348), (142.13722244942147, 32.85186896873376, 3.2918003584943887), (139.32074694398685, 5.425412796550864, 2.718040944391982), (139.32074694398685, 5.425412796550864, 2.718040944391982)], [(100, 50, 30), (73.67178507236096, 35.18890201097625, 21.669836181058997), (46.37794728544844, 11.930589536568762, 10.87036199426169), (50.79043208724118, 36.552024400518526, 5.98351588579158), (34.21166621056073, 14.629569007770707, 2.800535453136865), (42.42803021916894, 30.018310398750142, 1.4640068878683805), (43.212950646014356, 23.720527331780104, 2.164707204852975), (43.212950646014356, 23.720527331780104, 2.164707204852975)], [(100, 50, 30), (67.62310142293691, 45.48901400103116, 4.946184258164449), (96.00618052512537, 48.3663751883912, 12.99321020547946), (88.9967418534966, 40.91598124022691, 7.602371177937186), (88.66113830393105, 22.17612625393883, 11.437107803979522), (78.82402976867702, 40.25406319947103, 3.155018122923507), (81.32844067092884, 14.062872465019458, 3.4907240714110923), (81.32844067092884, 14.062872465019458, 3.4907240714110923)], [(100, 50, 30), (34.36180929356635, 24.692619623631444, 2.866542012486779), (141.02101398919964, 70.6293035975879, 17.605200563995567), (127.77508585183367, 56.28112715354061, 8.987842749215497), (82.43973803414627, 43.01925733406225, 2.012136536306795), (82.3727822242162, 21.680640128807436, 2.1555115581365314), (108.56938105598822, 2.615016918611601, 1.3671675893410247), (97.80383080784159, 18.756835605348883, 0.15735808590671094), (100.70187625149039, 42.4540891382961, 0.37706584633088114), (101.35199508658029, 36.76466128021701, 0.6297145343806184), (101.35199508658029, 36.76466128021701, 0.6297145343806184)], [(100, 50, 30), (194.97353982983537, 134.56862023805107, 35.928910102977724), (188.17971645420909, 36.052812891776, 10.18242335830444), (158.4908694415588, 116.80386408586143, 10.775367506145603), (94.2937604182172, 42.742687997054105, 3.896170950123006), (137.61937181112108, 106.25602940592313, 6.744730193844038), (122.08009944126492, 73.34863461007272, 4.840569618818928), (104.4283428977, 65.34399161849876, 0.4907858330019409), (104.4283428977, 65.34399161849876, 0.4907858330019409)]]"
        self.expected_all_intermediate_budgets = "[[0, 60, 470, 640, 3820, 6890, 10000], [0, 40, 80, 100, 560, 3000, 10000], [0, 40, 1520, 10000], [0, 80, 90, 120, 2410, 2530, 3830, 10000], [0, 170, 370, 620, 1100, 2410, 3180, 9900, 10000], [0, 40, 60, 170, 2210, 2610, 10000], [0, 40, 510, 1960, 4970, 5590, 8240, 10000], [0, 80, 940, 1420, 1790, 2090, 4280, 10000], [0, 50, 170, 290, 340, 510, 780, 1940, 3680, 7980, 10000], [0, 80, 130, 190, 240, 980, 2350, 4640, 10000]]"
        self.expected_all_est_objectives = "[[4506.047142126555, 3342.3641612574866, 3582.8865673791165, 5051.45927131579, 4159.1376986865325, 5268.433600189716, 5268.433600189716], [3503.43007812232, 4552.7282278195335, 4337.38770112416, -29.90138566484151, 663.4575630274444, 831.3184702252797, 831.3184702252797], [3854.4221570058726, 3696.372863281683, 285.76880844162883, 285.76880844162883], [2842.6366699906334, 3705.1499342547754, 3726.335401953453, 4465.06574285847, 3835.2877204083693, 4649.505251918164, 5286.408602012684, 5286.408602012684], [3152.340974881656, 4258.8432178762605, 4327.68444475099, 5197.298802196925, 3935.400854411654, 4838.431104494418, 5134.01576353541, 4446.814740961055, 4446.814740961055], [2569.18394230087, -1141.3388938651267, -633.474544886249, 1796.065670009543, 2136.9356263281015, 2280.0229212829395, 2280.0229212829395], [3403.2774806763955, 3803.7963853636793, 3946.8228898532934, 4510.65138039327, 4403.915654356277, 4734.093527716102, 4922.533575949079, 4922.533575949079], [3335.297892703174, 4890.155723174274, 4552.038391251517, 5023.2760055684985, 4787.482170841365, 5367.092251680084, 5421.575219169749, 5421.575219169749], [2349.1541419308, 3759.8137777335673, 1707.2443264779367, 2597.057630905399, 4165.6069751203595, 4433.729680100104, 3739.7221051993947, 4164.613521136326, 3924.099468246922, 3967.3200858562104, 3967.3200858562104], [4134.983467487309, 502.17860280432365, 2102.7341617634943, 2672.50324736627, 6037.763559936185, 3554.2930608757656, 4909.585514257262, 5597.325809261068, 5597.325809261068]]"
        self.expected_objective_curves = "[([0, 60, 470, 640, 3820, 6890, 10000], [3456.9776372884635, 3342.3641612574866, 3582.8865673791165, 5051.45927131579, 4159.1376986865325, 5268.433600189716, 5268.433600189716]), ([0, 40, 80, 100, 560, 3000, 10000], [3456.9776372884635, 4552.7282278195335, 4337.38770112416, -29.90138566484151, 663.4575630274444, 831.3184702252797, 831.3184702252797]), ([0, 40, 1520, 10000], [3456.9776372884635, 3696.372863281683, 285.76880844162883, 285.76880844162883]), ([0, 80, 90, 120, 2410, 2530, 3830, 10000], [3456.9776372884635, 3705.1499342547754, 3726.335401953453, 4465.06574285847, 3835.2877204083693, 4649.505251918164, 5286.408602012684, 5286.408602012684]), ([0, 170, 370, 620, 1100, 2410, 3180, 9900, 10000], [3456.9776372884635, 4258.8432178762605, 4327.68444475099, 5197.298802196925, 3935.400854411654, 4838.431104494418, 5134.01576353541, 4446.814740961055, 4446.814740961055]), ([0, 40, 60, 170, 2210, 2610, 10000], [3456.9776372884635, -1141.3388938651267, -633.474544886249, 1796.065670009543, 2136.9356263281015, 2280.0229212829395, 2280.0229212829395]), ([0, 40, 510, 1960, 4970, 5590, 8240, 10000], [3456.9776372884635, 3803.7963853636793, 3946.8228898532934, 4510.65138039327, 4403.915654356277, 4734.093527716102, 4922.533575949079, 4922.533575949079]), ([0, 80, 940, 1420, 1790, 2090, 4280, 10000], [3456.9776372884635, 4890.155723174274, 4552.038391251517, 5023.2760055684985, 4787.482170841365, 5367.092251680084, 5421.575219169749, 5421.575219169749]), ([0, 50, 170, 290, 340, 510, 780, 1940, 3680, 7980, 10000], [3456.9776372884635, 3759.8137777335673, 1707.2443264779367, 2597.057630905399, 4165.6069751203595, 4433.729680100104, 3739.7221051993947, 4164.613521136326, 3924.099468246922, 3967.3200858562104, 3967.3200858562104]), ([0, 80, 130, 190, 240, 980, 2350, 4640, 10000], [3456.9776372884635, 502.17860280432365, 2102.7341617634943, 2672.50324736627, 5073.831958353406, 3554.2930608757656, 4909.585514257262, 5597.325809261068, 5597.325809261068])]"
        self.expected_progress_curves = "[([0.0, 0.006, 0.047, 0.064, 0.382, 0.689, 1.0], [1.0, 1.0708867054611864, 0.922127226646045, 0.01383716933933791, 0.5657245972935949, -0.12035817902761708, -0.12035817902761708]), ([0.0, 0.004, 0.008, 0.01, 0.056, 0.3, 1.0], [1.0, 0.32229479412260653, 0.4554796604954403, 3.1565820603161514, 2.7277500130135817, 2.62393057485463, 2.62393057485463]), ([0.0, 0.004, 0.152, 1.0], [1.0, 0.8519376650856575, 2.9613448085775076, 2.9613448085775076]), ([0.0, 0.008, 0.009, 0.012, 0.241, 0.253, 0.383, 1.0], [1.0, 0.8465091791307129, 0.8334062870379213, 0.37651271828495436, 0.7660209221132974, 0.26243966503782423, -0.13147544642071696, -0.13147544642071696]), ([0.0, 0.017, 0.037, 0.062, 0.11, 0.241, 0.318, 0.99, 1.0], [1.0, 0.5040582381845956, 0.4614809781446265, -0.0763623798600468, 0.7041024593928309, 0.14559187602253545, -0.03722277535947951, 0.3878006875593874, 0.3878006875593874]), ([0.0, 0.004, 0.006, 0.017, 0.221, 0.261, 1.0], [1.0, 3.843989388063673, 3.5298829516567287, 2.0272489893739833, 1.8164260649598385, 1.72792873215091, 1.72792873215091]), ([0.0, 0.004, 0.051, 0.196, 0.497, 0.559, 0.824, 1.0], [1.0, 0.7854978376488592, 0.697038102825372, 0.3483186893357196, 0.4143331252972056, 0.21012309285448502, 0.09357576649495172, 0.09357576649495172]), ([0.0, 0.008, 0.094, 0.142, 0.179, 0.209, 0.428, 1.0], [1.0, 0.11360097986945025, 0.32272144763061206, 0.03126809393168369, 0.17710302269126918, -0.1813770662613079, -0.21507395953106165, -0.21507395953106165]), ([0.0, 0.005, 0.017, 0.029, 0.034, 0.051, 0.078, 0.194, 0.368, 0.798, 1.0], [1.0, 0.8127004167910189, 2.082183650075576, 1.5318475481555305, 0.5617234474376412, 0.3958936002544113, 0.8251268130794236, 0.5623378837360079, 0.7110921962030641, 0.6843608963907092, 0.6843608963907092]), ([0.0, 0.008, 0.013, 0.019, 0.024, 0.098, 0.235, 0.464, 1.0], [1.0, 2.8274986162871865, 1.8375791547088767, 1.485185572813696, -0.0, 0.9398118789556716, 0.10158394727112019, -0.32377304750799213, -0.32377304750799213])]"

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
