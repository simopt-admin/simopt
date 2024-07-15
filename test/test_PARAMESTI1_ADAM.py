import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class test_PARAMESTI1_ADAM(unittest.TestCase):
    def setUp(self):
        # Expected values
        self.expected_problem_name = "PARAMESTI-1"
        self.expected_solver_name = "ADAM"
        self.expected_all_recommended_xs = "[[(1, 1), (1.499999998791309, 1.4999999977642131), (1.9501941923588508, 1.987094219932607), (2.3107878532807606, 2.4582134482168443), (2.5529285525385585, 2.910688626601515), (2.6735728102941936, 3.342105880932273), (2.6899542500362976, 3.7502815215869814), (2.6899542500362976, 3.7502815215869814)], [(1, 1), (1.4999999985559338, 1.4999999977293828), (1.932993291938198, 1.986711774230398), (2.2440348916506148, 2.456897788889597), (2.4083163624171253, 2.9077290626998678), (2.44254407028723, 3.336642733544894), (2.3797113794507667, 3.741323114809531), (2.3797113794507667, 3.741323114809531)], [(1, 1), (1.499999998835872, 1.4999999978958247), (1.9404733052245104, 1.988477408601106), (2.2721224509352327, 2.462940275831444), (2.4668453889578603, 2.921263524147326), (2.532846259123281, 3.3615417496061113), (2.497651319590837, 3.7820494001259655), (2.497651319590837, 3.7820494001259655)], [(1, 1), (1.4999999986312154, 1.4999999976417437), (1.9469294217038196, 1.9857183716205578), (2.2982941289497942, 2.453463546710798), (2.5254015181952383, 2.8999730046233427), (2.6284990850479986, 3.322281927910226), (2.6280624990716146, 3.717724608129799), (2.6280624990716146, 3.717724608129799)], [(1, 1), (1.499999998718335, 1.4999999976014837), (1.9595288332775824, 1.9852468651195), (2.348494517090238, 2.4518252940297622), (2.6410910913081436, 2.896258221564911), (2.6410910913081436, 2.896258221564911)], [(1, 1), (1.4999999985606116, 1.499999997627275), (1.9426900620229344, 1.9855500305407883), (2.2817412903056398, 2.452879240372902), (2.4889684793692197, 2.898649139654859), (2.569371442317222, 3.3198248303688933), (2.5476548391529574, 3.7136805732869904), (2.5476548391529574, 3.7136805732869904)], [(1, 1), (1.4999999988836588, 1.4999999977117935), (1.9619539124383973, 1.9865159869509958), (2.3578282363369873, 2.456222844873722), (2.662726547849891, 2.9062081922205154), (2.662726547849891, 2.9062081922205154)], [(1, 1), (1.499999998917367, 1.4999999978230236), (1.9546640245311062, 1.9877242918214684), (2.3283490653507437, 2.460372846851994), (2.5928458019910563, 2.9155312257844384), (2.740909309994596, 3.351023321210268), (2.7847195560387896, 3.7648783142492475), (2.7847195560387896, 3.7648783142492475)], [(1, 1), (1.4999999990221815, 1.499999997835602), (1.9613678832195345, 1.987856522556788), (2.3551064045551033, 2.4608247224132196), (2.655646424640037, 2.916542164127999), (2.8509694100967384, 3.3528813645206785), (2.9451902560983765, 3.767915346468817), (2.9451902560983765, 3.767915346468817)], [(1, 1), (1.4999999986853543, 1.4999999976895564), (1.9464715710867926, 1.9862658938665052), (2.296292244742408, 2.4553593193822), (2.5206706851870417, 2.904259882762772), (2.6204538832727606, 3.330226864240831), (2.616769236305944, 3.730788548055702), (2.616769236305944, 3.730788548055702)]]"
        self.expected_all_intermediate_budgets = "[[0, 180, 330, 480, 630, 780, 930, 1000], [0, 180, 330, 480, 630, 780, 930, 1000], [0, 180, 330, 480, 630, 780, 930, 1000], [0, 180, 330, 480, 630, 780, 930, 1000], [0, 180, 330, 480, 630, 1000], [0, 180, 330, 480, 630, 780, 930, 1000], [0, 180, 330, 480, 630, 1000], [0, 180, 330, 480, 630, 780, 930, 1000], [0, 180, 330, 480, 630, 780, 930, 1000], [0, 180, 330, 480, 630, 780, 930, 1000]]"
        self.expected_all_est_objectives = "[[-9.16944227603262, -7.018017019163427, -6.045399102140761, -5.7212537204920135, -5.648727110217291, -5.587087892913641, -5.450109809520928, -5.450109809520928], [-8.940090362495347, -6.8817800264007865, -5.984458403025596, -5.658405109009439, -5.487402548197918, -5.288838750431085, -5.047284182131535, -5.047284182131535], [-9.121210005202611, -6.983022006353776, -6.023652187523378, -5.676467342085714, -5.526834517889321, -5.363838608584933, -5.145260518756914, -5.145260518756914], [-8.779886386724968, -6.741647515064164, -5.8409285218667994, -5.550080879197394, -5.478317264188384, -5.401334578889208, -5.249821223855754, -5.249821223855754], [-8.99288952739613, -6.817445425647, -5.786154251036526, -5.427931736233523, -5.3935083049195995, -5.3935083049195995], [-8.87740808504234, -6.766447332604166, -5.806702036011494, -5.461009177903708, -5.33492981969938, -5.2148171630214275, -5.044442300285056, -5.044442300285056], [-9.024638576352391, -6.912889654296439, -5.959515437505122, -5.684964263965716, -5.732991320083336, -5.732991320083336], [-8.921050660074993, -6.85559432013702, -5.945223949177388, -5.679317431735191, -5.67459623098717, -5.684048487899454, -5.604281884301284, -5.604281884301284], [-8.550164686658025, -6.502194651028625, -5.550339283606514, -5.245116437761885, -5.25406069371343, -5.3417764085036685, -5.3805742813393245, -5.3805742813393245], [-8.983830735669818, -6.868664620931846, -5.910780630697875, -5.578583905595107, -5.4777230604013765, -5.382247436745347, -5.222565523580854, -5.222565523580854]]"
        self.expected_objective_curves = "[([0, 180, 330, 480, 630, 780, 930, 1000], [-9.265122221743944, -7.018017019163427, -6.045399102140761, -5.7212537204920135, -5.648727110217291, -5.587087892913641, -5.450109809520928, -5.450109809520928]), ([0, 180, 330, 480, 630, 780, 930, 1000], [-9.265122221743944, -6.8817800264007865, -5.984458403025596, -5.658405109009439, -5.487402548197918, -5.288838750431085, -5.047284182131535, -5.047284182131535]), ([0, 180, 330, 480, 630, 780, 930, 1000], [-9.265122221743944, -6.983022006353776, -6.023652187523378, -5.676467342085714, -5.526834517889321, -5.363838608584933, -5.145260518756914, -5.145260518756914]), ([0, 180, 330, 480, 630, 780, 930, 1000], [-9.265122221743944, -6.741647515064164, -5.8409285218667994, -5.550080879197394, -5.478317264188384, -5.401334578889208, -5.249821223855754, -5.249821223855754]), ([0, 180, 330, 480, 630, 1000], [-9.265122221743944, -6.817445425647, -5.786154251036526, -5.427931736233523, -5.3935083049195995, -5.3935083049195995]), ([0, 180, 330, 480, 630, 780, 930, 1000], [-9.265122221743944, -6.766447332604166, -5.806702036011494, -5.461009177903708, -5.33492981969938, -5.2148171630214275, -5.044442300285056, -5.044442300285056]), ([0, 180, 330, 480, 630, 1000], [-9.265122221743944, -6.912889654296439, -5.959515437505122, -5.684964263965716, -5.732991320083336, -5.732991320083336]), ([0, 180, 330, 480, 630, 780, 930, 1000], [-9.265122221743944, -6.85559432013702, -5.945223949177388, -5.679317431735191, -5.67459623098717, -5.684048487899454, -5.604281884301284, -5.604281884301284]), ([0, 180, 330, 480, 630, 780, 930, 1000], [-9.265122221743944, -6.502194651028625, -5.550339283606514, -5.245116437761885, -5.25406069371343, -5.3417764085036685, -5.3805742813393245, -5.3805742813393245]), ([0, 180, 330, 480, 630, 780, 930, 1000], [-9.265122221743944, -6.868664620931846, -5.910780630697875, -5.578583905595107, -5.4777230604013765, -5.382247436745347, -5.222565523580854, -5.222565523580854])]"
        self.expected_progress_curves = "[([0.0, 0.18, 0.33, 0.48, 0.63, 0.78, 0.93, 1.0], [1.0, 0.5175794672015666, 0.3087726640307372, 0.2391834042401568, 0.2236130046297531, 0.21037996862469321, 0.18097278292812005, 0.18097278292812005]), ([0.0, 0.18, 0.33, 0.48, 0.63, 0.78, 0.93, 1.0], [1.0, 0.4883313827950448, 0.29568958963685216, 0.22569072904086157, 0.18897898701745505, 0.1463502516101624, 0.09449202823675577, 0.09449202823675577]), ([0.0, 0.18, 0.33, 0.48, 0.63, 0.78, 0.93, 1.0], [1.0, 0.5100665511593927, 0.3041039203847378, 0.2295684255964922, 0.1974444526333831, 0.16245162126959636, 0.1155261108699032, 0.1155261108699032]), ([0.0, 0.18, 0.33, 0.48, 0.63, 0.78, 0.93, 1.0], [1.0, 0.45824698767915856, 0.26487582904317497, 0.20243510549422258, 0.1870285097779065, 0.1705014562088817, 0.13797376085337335, 0.13797376085337335]), ([0.0, 0.18, 0.33, 0.48, 0.63, 1.0], [1.0, 0.474519687491516, 0.25311659637029366, 0.17621147583868518, 0.16882126996130004, 0.16882126996130004]), ([0.0, 0.18, 0.33, 0.48, 0.63, 0.78, 0.93, 1.0], [1.0, 0.46357114481128237, 0.2575279044675151, 0.18331271747873615, 0.15624532824837492, 0.13045890233299237, 0.093881917881017, 0.093881917881017]), ([0.0, 0.18, 0.33, 0.48, 0.63, 1.0], [1.0, 0.49501016367604095, 0.2903347007326113, 0.231392590115265, 0.2417032947384378, 0.2417032947384378]), ([0.0, 0.18, 0.33, 0.48, 0.63, 0.78, 0.93, 1.0], [1.0, 0.4827096956782322, 0.2872665277758532, 0.2301802980471656, 0.2291667254838103, 0.2311959864238567, 0.21407126642269256, 0.21407126642269256]), ([0.0, 0.18, 0.33, 0.48, 0.63, 0.78, 0.93, 1.0], [1.0, 0.40683996938937, 0.20249058113131424, 0.13696371227554058, 0.13888391285924032, 0.15771519045821875, 0.1660445247783605, 0.1660445247783605]), ([0.0, 0.18, 0.33, 0.48, 0.63, 0.78, 0.93, 1.0], [1.0, 0.48551569758060636, 0.27987205241566493, 0.2085542872598835, 0.18690094293893977, 0.16640372670691334, 0.13212236171402186, 0.13212236171402186])]"

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
