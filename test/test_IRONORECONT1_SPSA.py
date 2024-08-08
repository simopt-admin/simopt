import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class test_IRONORECONT1_SPSA(unittest.TestCase):
    def setUp(self):
        # Expected values
        self.expected_problem_name = "IRONORECONT-1"
        self.expected_solver_name = "SPSA"
        self.expected_all_recommended_xs = "[[(80, 40, 100), (79.9, 40.1, 99.9), (79.840173975191, 40.15982622854404, 99.840173975191), (79.84017403301645, 40.1598262863695, 99.84017403301645), (79.84017408493288, 40.15982633828593, 99.84017408493288), (79.84017408493288, 40.15982633828593, 99.84017408493288)], [(80, 40, 100), (80.06129025754778, 39.93870974245221, 99.78429614545477), (80.09129718066005, 39.90870281933994, 99.64861717997532), (80.09129718066005, 39.90870281933994, 99.64861717997532)], [(80, 40, 100), (80.33334426584184, 40.01185870179294, 99.66665573415816), (80.40503045448851, 39.94017251314627, 99.59496954551149), (80.4692621253788, 39.87594084225598, 99.5307378746212), (80.55866495203203, 39.84851023696141, 99.44133504796797), (80.58005459226503, 39.82712059672841, 99.36304204235721), (80.59859120535364, 39.80858398363979, 99.29394302000642), (80.61504505119117, 39.79213013780226, 99.23180828458527), (80.61504505119117, 39.79213013780226, 99.23180828458527)], [(80, 40, 100), (79.88944512188709, 40.11055487811291, 99.88944512188709), (79.80790898027158, 40.192091019728416, 99.80790898027158), (79.80790898027158, 40.192091019728416, 99.80790898027158)], [(80, 40, 100), (0.0, 120.0, 20.0), (0.0, 120.0, 20.0), (20.0, 100.0, 0.0), (120.0, 0.0, 100.0), (120.0, 0.0, 100.0)], [(80, 40, 100), (80.13705796125292, 39.862942038747086, 99.86294203874708), (80.23814058672366, 39.76185941327634, 99.76185941327634), (80.36987752490744, 39.729603412142836, 99.63012247509256), (80.58861475418223, 39.58210760329671, 99.41138607558304), (80.58861475418223, 39.58210760329671, 99.41138607558304)], [(80, 40, 100), (80.14163977030795, 39.858360229692046, 99.85836022969205), (80.30557029889549, 39.81336717992019, 99.69442970110451), (80.37767127777165, 39.74126620104403, 99.62232872222835), (80.37767127777165, 39.74126620104403, 99.62232872222835)], [(80, 40, 100), (120.0, 0.0, 60.0), (120.0, 0.0, 60.0)], [(80, 40, 100), (80, 40, 100)], [(80, 40, 100), (80, 40, 100)], [(80, 40, 100), (80.07053474554621, 40.070534745546205, 99.92946525445379), (80.07053474554621, 40.070534745546205, 99.92946525445379)], [(80, 40, 100), (80.05247622790796, 40.19532625653434, 99.80467374346566), (80.05247622790796, 40.19532625653434, 99.80467374346566)], [(80, 40, 100), (119.80000000000001, 0.0, 60.00000000000001), (119.80000000000001, 0.0, 60.00000000000001)], [(80, 40, 100), (179.8, 139.8, 0.0), (179.8, 139.8, 0.0)], [(80, 40, 100), (80, 40, 100)], [(80, 40, 100), (80.09350857777503, 40.093508577775026, 99.90649142222497), (80.19655380360317, 39.990463351946886, 99.80344619639683), (80.28014230057518, 39.906874854974866, 99.71985769942482), (80.39448545934593, 39.95131652907139, 99.53302050229749), (80.44125569893268, 39.90454628948464, 99.48625026271074), (80.48466118670879, 39.86114080170854, 99.44284477493463), (80.48466118670879, 39.86114080170854, 99.44284477493463)], [(80, 40, 100), (79.92935079964043, 40.07064920035956, 99.92935079964043), (80.00310315414062, 39.99689684585937, 99.85559844514025), (80.02704900382258, 39.9729509961774, 99.75989103750197), (80.02704900382258, 39.9729509961774, 99.75989103750197)], [(80, 40, 100), (79.89338736918243, 40.10661263081756, 99.89338736918243), (79.99200098661892, 40.11006385461273, 99.66723324904224), (80.05947499218301, 40.04258984904864, 99.59975924347815), (80.11993249732956, 39.98213234390209, 99.5393017383316), (80.17491672641287, 39.92714811481876, 99.48431750924829), (80.19275172680253, 39.90931311442912, 99.40100015133311), (80.26359533902368, 39.89240683555032, 99.33015653911195), (80.26359533902368, 39.89240683555032, 99.33015653911195)], [(80, 40, 100), (79.91981990702958, 40.08018009297042, 99.91981990702958), (79.86068569670098, 40.13931430329902, 99.86068569670098), (79.81271705262971, 40.18728294737029, 99.81271705262971), (79.77185903741803, 40.22814096258197, 99.77185903741803), (79.73599344243864, 40.26400655756136, 99.73599344243864), (79.67656077632142, 40.36301013992734, 99.50780214422758), (79.67656077632142, 40.36301013992734, 99.50780214422758)], [(80, 40, 100), (0.0, 120.0, 20.0), (20.0, 100.0, 0.0), (50.707590873209874, 69.29240912679012, 30.707590873209874), (50.707590873209874, 69.29240912679012, 30.707590873209874)], [(80, 40, 100), (80.09421933311828, 40.09421933311828, 99.90578066688172), (80.19260318682308, 39.99583547941348, 99.80739681317692), (80.26058061697931, 39.927858816558086, 99.73942015032152), (80.26058091901577, 39.92785911859455, 99.73942045235798), (80.26058091901577, 39.92785911859455, 99.73942045235798)], [(80, 40, 100), (80, 40, 100)], [(80, 40, 100), (0.0, 120.0, 20.0), (20.0, 100.0, 0.0), (20.0, 100.0, 0.0)], [(80, 40, 100), (120.0, 0.0, 60.0), (60.00000000000001, 59.99999999999999, 7.105427357601002e-15), (60.000000000000014, 60.0, 0.0), (0.0, 120.00000000000001, 60.00000000000002), (0.0, 120.00000000000001, 60.00000000000002)]]"
        self.expected_all_intermediate_budgets = "[[0, 210, 330, 390, 450, 1000], [0, 270, 390, 1000], [0, 390, 450, 510, 630, 750, 870, 990, 1000], [0, 210, 270, 1000], [0, 210, 270, 570, 690, 1000], [0, 210, 270, 390, 750, 1000], [0, 210, 330, 390, 1000], [0, 210, 1000], [0, 1000], [0, 1000], [0, 210, 1000], [0, 390, 1000], [0, 270, 1000], [0, 270, 1000], [0, 1000], [0, 210, 270, 330, 570, 630, 690, 1000], [0, 210, 270, 390, 1000], [0, 210, 390, 450, 510, 570, 690, 810, 1000], [0, 210, 270, 330, 390, 450, 990, 1000], [0, 210, 810, 930, 1000], [0, 210, 270, 390, 450, 1000], [0, 1000], [0, 210, 330, 1000], [0, 210, 510, 570, 750, 1000]]"
        self.expected_all_est_objectives = "[[149895.8055235529, 149490.01490668004, 149055.9839961229, 149055.9839961229, 149055.9839961229, 149055.9839961229], [122793.09736468189, 123777.93057210676, 123146.51073289948, 123146.51073289948], [99852.80349485856, 102406.22573755184, 102070.64177815235, 101588.43733685819, 103428.73599471145, 103411.91603451176, 103702.52248035539, 103749.21753314632, 103749.21753314632], [126011.12695446546, 125345.33865181894, 124448.63770230618, 124448.63770230618], [136147.71179130895, 65822.07046203454, 65822.07046203454, 133596.77111131456, 130997.4391892812, 130997.4391892812], [132850.26196652921, 133244.21392665227, 133023.7769571096, 132848.328269142, 131372.08599689155, 131372.08599689155], [134982.68434045353, 134233.36659889162, 133557.24319635433, 133138.57078580675, 133138.57078580675], [161256.2908821113, 48055.83014443732, 48055.83014443732], [146337.47315675917, 146337.47315675917], [134867.2205665852, 134867.2205665852], [149243.01256369415, 148637.102825817, 148637.102825817], [112822.77485929335, 111700.91594605673, 111700.91594605673], [132809.38556277155, 30735.99701553667, 30735.99701553667], [118379.15455996453, 2779.92150146073, 2779.92150146073], [127606.7164810152, 127606.7164810152], [145498.2552215891, 145013.4657293224, 145416.88581258664, 144468.82053017514, 143567.8791966509, 143021.28655879266, 143060.84199689285, 143060.84199689285], [161264.15011124164, 160863.83247654935, 160602.18750837655, 160395.33388187754, 160395.33388187754], [132500.94479520118, 132561.1757691837, 133160.13177872173, 133501.59454174797, 134536.37816839627, 134199.40507453054, 134643.49069652145, 134143.36391626124, 134143.36391626124], [112031.98326897933, 111807.46351074848, 112673.97798211314, 112457.94394533597, 112324.18191206649, 112132.90622378167, 110651.4176913473, 110651.4176913473], [130863.18264271188, 56865.40336828781, 123511.30595080237, 39673.26608669524, 39673.26608669524], [147610.26102665017, 147201.6157522236, 146192.11809600273, 145184.13558234138, 145184.13558234138, 145184.13558234138], [132677.02997009846, 132677.02997009846], [132803.08586581453, 66141.02420632845, 134373.2812266687, 134373.2812266687], [137521.1409071744, 31436.09282245306, 32916.72297617469, 32916.72297617469, 62850.243533222005, 62850.243533222005]]"
        self.expected_objective_curves = "[([0, 210, 330, 390, 450, 1000], [121270.73497283501, 149490.01490668004, 149055.9839961229, 149055.9839961229, 149055.9839961229, 149055.9839961229]), ([0, 270, 390, 1000], [121270.73497283501, 123777.93057210676, 123146.51073289948, 123146.51073289948]), ([0, 390, 450, 510, 630, 750, 870, 990, 1000], [121270.73497283501, 102406.22573755184, 102070.64177815235, 101588.43733685819, 103428.73599471145, 103411.91603451176, 103702.52248035539, 103749.21753314632, 103749.21753314632]), ([0, 210, 270, 1000], [121270.73497283501, 125345.33865181894, 124448.63770230618, 124448.63770230618]), ([0, 210, 270, 570, 690, 1000], [121270.73497283501, 65822.07046203454, 65822.07046203454, 133596.77111131456, 130997.4391892812, 130997.4391892812]), ([0, 210, 270, 390, 750, 1000], [121270.73497283501, 133244.21392665227, 133023.7769571096, 132848.328269142, 131372.08599689155, 131372.08599689155]), ([0, 210, 330, 390, 1000], [121270.73497283501, 134233.36659889162, 133557.24319635433, 133138.57078580675, 133138.57078580675]), ([0, 210, 1000], [121270.73497283501, 48055.83014443732, 48055.83014443732]), ([0, 1000], [121270.73497283501, 121270.73497283501]), ([0, 1000], [121270.73497283501, 121270.73497283501]), ([0, 210, 1000], [121270.73497283501, 148637.102825817, 148637.102825817]), ([0, 390, 1000], [121270.73497283501, 111700.91594605673, 111700.91594605673]), ([0, 270, 1000], [121270.73497283501, 30735.99701553667, 30735.99701553667]), ([0, 270, 1000], [121270.73497283501, 2779.92150146073, 2779.92150146073]), ([0, 1000], [121270.73497283501, 121270.73497283501]), ([0, 210, 270, 330, 570, 630, 690, 1000], [121270.73497283501, 145013.4657293224, 145416.88581258664, 144468.82053017514, 143567.8791966509, 143021.28655879266, 143060.84199689285, 143060.84199689285]), ([0, 210, 270, 390, 1000], [121270.73497283501, 160863.83247654935, 160602.18750837655, 160395.33388187754, 160395.33388187754]), ([0, 210, 390, 450, 510, 570, 690, 810, 1000], [121270.73497283501, 132561.1757691837, 133160.13177872173, 133501.59454174797, 134536.37816839627, 134199.40507453054, 134643.49069652145, 134143.36391626124, 134143.36391626124]), ([0, 210, 270, 330, 390, 450, 990, 1000], [121270.73497283501, 111807.46351074848, 112673.97798211314, 112457.94394533597, 112324.18191206649, 112132.90622378167, 110651.4176913473, 110651.4176913473]), ([0, 210, 810, 930, 1000], [121270.73497283501, 56865.40336828781, 123511.30595080237, 39673.26608669524, 39673.26608669524]), ([0, 210, 270, 390, 450, 1000], [121270.73497283501, 147201.6157522236, 146192.11809600273, 145184.13558234138, 145184.13558234138, 145184.13558234138]), ([0, 1000], [121270.73497283501, 121270.73497283501]), ([0, 210, 330, 1000], [121270.73497283501, 66141.02420632845, 134373.2812266687, 134373.2812266687]), ([0, 210, 510, 570, 750, 1000], [121270.73497283501, 31436.09282245306, 32916.72297617469, 32916.72297617469, 62850.243533222005, 62850.243533222005])]"
        self.expected_progress_curves = "[([0.0, 0.21, 0.33, 0.39, 0.45, 1.0], [nan, inf, inf, inf, inf, inf]), ([0.0, 0.27, 0.39, 1.0], [nan, inf, inf, inf]), ([0.0, 0.39, 0.45, 0.51, 0.63, 0.75, 0.87, 0.99, 1.0], [nan, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf]), ([0.0, 0.21, 0.27, 1.0], [nan, inf, inf, inf]), ([0.0, 0.21, 0.27, 0.57, 0.69, 1.0], [nan, -inf, -inf, inf, inf, inf]), ([0.0, 0.21, 0.27, 0.39, 0.75, 1.0], [nan, inf, inf, inf, inf, inf]), ([0.0, 0.21, 0.33, 0.39, 1.0], [nan, inf, inf, inf, inf]), ([0.0, 0.21, 1.0], [nan, -inf, -inf]), ([0.0, 1.0], [nan, nan]), ([0.0, 1.0], [nan, nan]), ([0.0, 0.21, 1.0], [nan, inf, inf]), ([0.0, 0.39, 1.0], [nan, -inf, -inf]), ([0.0, 0.27, 1.0], [nan, -inf, -inf]), ([0.0, 0.27, 1.0], [nan, -inf, -inf]), ([0.0, 1.0], [nan, nan]), ([0.0, 0.21, 0.27, 0.33, 0.57, 0.63, 0.69, 1.0], [nan, inf, inf, inf, inf, inf, inf, inf]), ([0.0, 0.21, 0.27, 0.39, 1.0], [nan, inf, inf, inf, inf]), ([0.0, 0.21, 0.39, 0.45, 0.51, 0.57, 0.69, 0.81, 1.0], [nan, inf, inf, inf, inf, inf, inf, inf, inf]), ([0.0, 0.21, 0.27, 0.33, 0.39, 0.45, 0.99, 1.0], [nan, -inf, -inf, -inf, -inf, -inf, -inf, -inf]), ([0.0, 0.21, 0.81, 0.93, 1.0], [nan, -inf, inf, -inf, -inf]), ([0.0, 0.21, 0.27, 0.39, 0.45, 1.0], [nan, inf, inf, inf, inf, inf]), ([0.0, 1.0], [nan, nan]), ([0.0, 0.21, 0.33, 1.0], [nan, -inf, inf, inf]), ([0.0, 0.21, 0.51, 0.57, 0.75, 1.0], [nan, -inf, -inf, -inf, -inf, -inf])]"

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
