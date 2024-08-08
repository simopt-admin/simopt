import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class test_IRONORECONT1_RNDSRCH(unittest.TestCase):
    def setUp(self):
        # Expected values
        self.expected_problem_name = "IRONORECONT-1"
        self.expected_solver_name = "RNDSRCH"
        self.expected_all_recommended_xs = "[[(80, 40, 100), (80, 40, 100)], [(80, 40, 100), (89.51286085366789, 110.69874262239097, 28.284950503085465), (48.98708330069736, 98.01083372144204, 99.73877755543312), (48.98708330069736, 98.01083372144204, 99.73877755543312)], [(80, 40, 100), (35.62536771908235, 43.36685849769248, 99.27384048105371), (35.62536771908235, 43.36685849769248, 99.27384048105371)], [(80, 40, 100), (80, 40, 100)], [(80, 40, 100), (82.31961648623096, 97.24186772912596, 31.03681099951429), (82.31961648623096, 97.24186772912596, 31.03681099951429)], [(80, 40, 100), (80, 40, 100)], [(80, 40, 100), (80, 40, 100)], [(80, 40, 100), (126.52678555423257, 17.98040104682185, 97.12547689041833), (94.51644538396839, 85.06518197933129, 22.02606581205134), (94.51644538396839, 85.06518197933129, 22.02606581205134)], [(80, 40, 100), (91.31252825949757, 90.48184614967685, 89.29705684153677), (91.31252825949757, 90.48184614967685, 89.29705684153677)], [(80, 40, 100), (89.00817259485456, 116.40115198323831, 91.11061514985424), (113.00855572188193, 94.53529333943227, 31.416681866883046), (113.00855572188193, 94.53529333943227, 31.416681866883046)], [(80, 40, 100), (102.55613491829448, 99.43943688071552, 24.879815221920225), (102.55613491829448, 99.43943688071552, 24.879815221920225)], [(80, 40, 100), (55.52468209970047, 16.470337806363542, 100.87410951510024), (112.00769473338623, 38.06209895699656, 103.97917717163797), (75.47832405343891, 103.74713500871242, 47.759642472392336), (90.48889705035457, 59.447754166834045, 102.62143057829438), (90.48889705035457, 59.447754166834045, 102.62143057829438)], [(80, 40, 100), (76.01477491889344, 104.88547484354753, 86.91900650604643), (76.01477491889344, 104.88547484354753, 86.91900650604643)], [(80, 40, 100), (87.93491619141716, 88.43583482144255, 24.98583066635066), (87.93491619141716, 88.43583482144255, 24.98583066635066)], [(80, 40, 100), (115.6881171570172, 71.43447612461716, 11.31110340597059), (115.6881171570172, 71.43447612461716, 11.31110340597059)], [(80, 40, 100), (117.60077953644348, 71.35494602482068, 36.18235762513837), (82.6866556416487, 91.25287287543664, 29.992277892432096), (82.6866556416487, 91.25287287543664, 29.992277892432096)], [(80, 40, 100), (92.06709267142114, 189.04870295267295, 27.75479162585811), (67.77903631557045, 98.9987944865486, 25.095485853487492), (67.77903631557045, 98.9987944865486, 25.095485853487492)], [(80, 40, 100), (122.90553955989999, 12.114538906429312, 100.51390627262943), (122.90553955989999, 12.114538906429312, 100.51390627262943)], [(80, 40, 100), (91.69839857915929, 109.04423698747999, 43.83541304780066), (91.69839857915929, 109.04423698747999, 43.83541304780066)], [(80, 40, 100), (80, 40, 100)], [(80, 40, 100), (1681.8832778499736, 294.4977282728228, 88.52484884136608), (87.73404534279683, 411.1098442611314, 36.025941558752145), (85.48762306598165, 87.22556722374529, 70.12579053559712), (106.74899435157948, 87.24697341270542, 37.67063735188248), (106.74899435157948, 87.24697341270542, 37.67063735188248)], [(80, 40, 100), (80, 40, 100)], [(80, 40, 100), (100.518049174063, 26.234215635176728, 101.22698216967835), (48.862860950902146, 103.12908171335523, 97.55164507645053), (105.03729919357474, 86.66758205594304, 22.25102128643694), (88.2494030239024, 109.60740845772456, 102.76494616624827), (88.2494030239024, 109.60740845772456, 102.76494616624827)], [(80, 40, 100), (63.31492358210308, 109.64854280216943, 87.69418521579908), (84.44770687380792, 82.5306970815239, 92.75820954988112), (98.09257016137323, 95.73006440738105, 43.770344891029964), (98.09257016137323, 95.73006440738105, 43.770344891029964)]]"
        self.expected_all_intermediate_budgets = "[[0, 1000], [0, 140, 180, 1000], [0, 360, 1000], [0, 1000], [0, 960, 1000], [0, 1000], [0, 1000], [0, 230, 310, 1000], [0, 330, 1000], [0, 70, 850, 1000], [0, 320, 1000], [0, 160, 340, 410, 800, 1000], [0, 380, 1000], [0, 240, 1000], [0, 390, 1000], [0, 260, 450, 1000], [0, 20, 820, 1000], [0, 820, 1000], [0, 630, 1000], [0, 1000], [0, 20, 40, 60, 360, 1000], [0, 1000], [0, 320, 430, 480, 580, 1000], [0, 130, 460, 560, 1000]]"
        self.expected_all_est_objectives = "[[149895.8055235529, 149895.8055235529], [122793.09736468189, 197938.90796272215, 187639.80947999496, 187639.80947999496], [99852.80349485856, 100352.59644273225, 100352.59644273225], [126011.12695446546, 126011.12695446546], [136147.71179130895, 203080.36067948042, 203080.36067948042], [132850.26196652921, 132850.26196652921], [134982.68434045353, 134982.68434045353], [161256.2908821113, 137192.46049918345, 208229.85949044532, 208229.85949044532], [146337.47315675917, 227189.19430184594, 227189.19430184594], [134867.2205665852, 176493.37803660682, 219325.4955932965, 219325.4955932965], [149243.01256369415, 264096.61980393203, 264096.61980393203], [112822.77485929335, 102123.36089691252, 88753.69050467688, 158010.06189744166, 127456.63099497948, 127456.63099497948], [132809.38556277155, 178024.8589837799, 178024.8589837799], [118379.15455996453, 187119.98448827662, 187119.98448827662], [127606.7164810152, 146875.0911867372, 146875.0911867372], [145498.2552215891, 152686.44023600186, 197733.25585602038, 197733.25585602038], [161264.15011124164, 115002.1466821434, 173009.51418091674, 173009.51418091674], [132500.94479520118, 120572.65567399841, 120572.65567399841], [112031.98326897933, 209897.6484755537, 209897.6484755537], [130863.18264271188, 130863.18264271188], [147610.26102665017, 0.0, 99912.27582929395, 200090.81299157804, 234493.28222547378, 234493.28222547378], [132677.02997009846, 132677.02997009846], [132803.08586581453, 123491.27493544853, 184759.5667756141, 221590.78397498463, 213684.45147367337, 213684.45147367337], [137521.1409071744, 143617.35918330486, 178765.42839979695, 242392.12397328235, 242392.12397328235]]"
        self.expected_objective_curves = "[([0, 1000], [121270.73497283501, 121270.73497283501]), ([0, 140, 180, 1000], [121270.73497283501, 197938.90796272215, 187639.80947999496, 187639.80947999496]), ([0, 360, 1000], [121270.73497283501, 100352.59644273225, 100352.59644273225]), ([0, 1000], [121270.73497283501, 121270.73497283501]), ([0, 960, 1000], [121270.73497283501, 203080.36067948042, 203080.36067948042]), ([0, 1000], [121270.73497283501, 121270.73497283501]), ([0, 1000], [121270.73497283501, 121270.73497283501]), ([0, 230, 310, 1000], [121270.73497283501, 137192.46049918345, 208229.85949044532, 208229.85949044532]), ([0, 330, 1000], [121270.73497283501, 227189.19430184594, 227189.19430184594]), ([0, 70, 850, 1000], [121270.73497283501, 176493.37803660682, 219325.4955932965, 219325.4955932965]), ([0, 320, 1000], [121270.73497283501, 240934.85571613206, 240934.85571613206]), ([0, 160, 340, 410, 800, 1000], [121270.73497283501, 102123.36089691252, 88753.69050467688, 158010.06189744166, 127456.63099497948, 127456.63099497948]), ([0, 380, 1000], [121270.73497283501, 178024.8589837799, 178024.8589837799]), ([0, 240, 1000], [121270.73497283501, 187119.98448827662, 187119.98448827662]), ([0, 390, 1000], [121270.73497283501, 146875.0911867372, 146875.0911867372]), ([0, 260, 450, 1000], [121270.73497283501, 152686.44023600186, 197733.25585602038, 197733.25585602038]), ([0, 20, 820, 1000], [121270.73497283501, 115002.1466821434, 173009.51418091674, 173009.51418091674]), ([0, 820, 1000], [121270.73497283501, 120572.65567399841, 120572.65567399841]), ([0, 630, 1000], [121270.73497283501, 209897.6484755537, 209897.6484755537]), ([0, 1000], [121270.73497283501, 121270.73497283501]), ([0, 20, 40, 60, 360, 1000], [121270.73497283501, 0.0, 99912.27582929395, 200090.81299157804, 234493.28222547378, 234493.28222547378]), ([0, 1000], [121270.73497283501, 121270.73497283501]), ([0, 320, 430, 480, 580, 1000], [121270.73497283501, 123491.27493544853, 184759.5667756141, 221590.78397498463, 213684.45147367337, 213684.45147367337]), ([0, 130, 460, 560, 1000], [121270.73497283501, 143617.35918330486, 178765.42839979695, 242392.12397328235, 242392.12397328235])]"
        self.expected_progress_curves = "[([0.0, 1.0], [1.0, 1.0]), ([0.0, 0.14, 0.18, 1.0], [1.0, 0.35930525780275135, 0.4453719787108569, 0.4453719787108569]), ([0.0, 0.36, 1.0], [1.0, 1.1748071050885527, 1.1748071050885527]), ([0.0, 1.0], [1.0, 1.0]), ([0.0, 0.96, 1.0], [1.0, 0.3163395577681713, 0.3163395577681713]), ([0.0, 1.0], [1.0, 1.0]), ([0.0, 1.0], [1.0, 1.0]), ([0.0, 0.23, 0.31, 1.0], [1.0, 0.8669465381314784, 0.2733066187470291, 0.2733066187470291]), ([0.0, 0.33, 1.0], [1.0, 0.11486869521878872, 0.11486869521878872]), ([0.0, 0.07, 0.85, 1.0], [1.0, 0.53851962709662, 0.18058345298998904, 0.18058345298998904]), ([0.0, 0.32, 1.0], [1.0, -0.0, -0.0]), ([0.0, 0.16, 0.34, 0.41, 0.8, 1.0], [1.0, 1.1600093157162568, 1.2717359578307816, 0.6929795940805039, 0.94830617578836, 0.94830617578836]), ([0.0, 0.38, 1.0], [1.0, 0.5257214638906377, 0.5257214638906377]), ([0.0, 0.24, 1.0], [1.0, 0.4497160125656952, 0.4497160125656952]), ([0.0, 0.39, 1.0], [1.0, 0.7860314682892414, 0.7860314682892414]), ([0.0, 0.26, 0.45, 1.0], [1.0, 0.7374676296618626, 0.36102383564734136, 0.36102383564734136]), ([0.0, 0.02, 0.82, 1.0], [1.0, 1.052384860656261, 0.5676333149259373, 0.5676333149259373]), ([0.0, 0.82, 1.0], [1.0, 1.005833655856914, 1.005833655856914]), ([0.0, 0.63, 1.0], [1.0, 0.259369366922933, 0.259369366922933]), ([0.0, 1.0], [1.0, 1.0]), ([0.0, 0.02, 0.04, 0.06, 0.36, 1.0], [1.0, 2.0134260312912375, 1.1784867428170813, 0.34132238193746045, 0.05383045018545463, 0.05383045018545463]), ([0.0, 1.0], [1.0, 1.0]), ([0.0, 0.32, 0.43, 0.48, 0.58, 1.0], [1.0, 0.9814435609536044, 0.46944137132821073, 0.16165306376707728, 0.22772410036686053, 0.22772410036686053]), ([0.0, 0.13, 0.46, 0.56, 1.0], [1.0, 0.8132554346978597, 0.5195327298622843, -0.01217798825661709, -0.01217798825661709])]"

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
        self.myexperiment.has_run = True
        self.myexperiment.has_postreplicated= True
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
