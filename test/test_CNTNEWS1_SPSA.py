import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class test_CNTNEWS1_SPSA(unittest.TestCase):
    def setUp(self):
        # Expected values
        self.expected_problem_name = "CNTNEWS-1"
        self.expected_solver_name = "SPSA"
        self.expected_all_recommended_xs = "[[(0,), (0.09999999999999999,), (0.14916782386404526,), (0.1651214576715408,), (0.17531301861842097,), (0.17829510504001253,), (0.1809670896609604,), (0.1833971769453942,), (0.18563244571068954,), (0.18563244571069762,), (0.18563244571069762,)], [(0,), (0.09999999999999999,), (0.1540846062504531,), (0.17402664850983296,), (0.17062946152754227,), (0.17062946152755235,), (0.17062946152756042,), (0.17062946152756042,)], [(0,), (0.09999999999999999,), (0.15900138863685542,), (0.19090865625182796,), (0.20789459116326917,), (0.21385876400645962,), (0.21920273324834202,), (0.22163282053278507,), (0.2238680892980804,), (0.2238680892980804,)], [(0,), (0.09999999999999999,), (0.14425104147764842,), (0.14823944992951418,), (0.15163663691180487,), (0.1546187233334001,), (0.1572907079543413,), (0.15972079523876587,), (0.15972079523876587,)], [(0,), (0.09999999999999999,), (0.14425104147764842,), (0.15222785838137995,), (0.14543348441680265,), (0.14841557083840518,), (0.14841557083840518,)], [(0,), (0.09999999999999999,), (0.13441747670483828,), (0.15435951896420422,), (0.14756514499961879,), (0.14348433999868518,), (0.14380596010119867,), (0.14403748414410464,), (0.14576050339814944,), (0.14576050339815624,), (0.14576050339815624,)], [(0,), (0.09999999999999999,), (0.14916782386404526,), (0.1611330492196704,), (0.1679274231842436,), (0.17389159602742674,), (0.17923556526931247,), (0.18409573983818933,), (0.18856627736876563,), (0.18856627736876966,), (0.18856627736876966,)], [(0,), (0.09999999999999999,), (0.14916782386405625,), (0.16910986612341292,), (0.17930142707028088,), (0.18526559991347133,), (0.19060956915537372,), (0.1954697437242475,), (0.19994028125482669,), (0.20408919868289538,), (0.20796739972576508,), (0.20979073822575334,), (0.21151375747979353,), (0.21314886596389587,), (0.21470619973870425,), (0.21470619973870425,)], [(0,), (0.09999999999999999,), (0.1540846062504531,), (0.1820034654135923,), (0.20238658730732012,), (0.2143149329937083,), (0.2250028714775131,), (0.2347232206152422,), (0.236958489380549,), (0.23903294809457257,), (0.23895044465745666,), (0.23815505990404412,), (0.23815505990404412,)], [(0,), (0.09999999999999999,), (0.14425104147764295,), (0.15222785838139768,), (0.15902223234596685,), (0.15902223234597052,), (0.159022232345975,), (0.159022232345975,)]]"
        self.expected_all_intermediate_budgets = "[[0, 210, 270, 330, 390, 450, 510, 570, 630, 930, 1000], [0, 210, 270, 330, 390, 570, 690, 1000], [0, 210, 270, 330, 390, 450, 510, 570, 630, 1000], [0, 210, 270, 330, 390, 450, 510, 570, 1000], [0, 210, 270, 330, 390, 450, 1000], [0, 210, 270, 330, 390, 570, 690, 810, 870, 990, 1000], [0, 210, 270, 330, 390, 450, 510, 570, 630, 810, 1000], [0, 210, 270, 330, 390, 450, 510, 570, 630, 690, 750, 810, 870, 930, 990, 1000], [0, 210, 270, 330, 390, 450, 510, 570, 630, 690, 810, 870, 1000], [0, 210, 270, 330, 390, 450, 630, 1000]]"
        self.expected_all_est_objectives = "[[0.0, 0.3528961132849871, 0.433564071695458, 0.4453426026667063, 0.44790583734543843, 0.4482535291098946, 0.4482915292989911, 0.4480933729136198, 0.4477313528945558, 0.44773135289455396, 0.44773135289455396], [0.0, 0.3653863363301532, 0.4820584522164391, 0.5000073720450766, 0.49780558849158935, 0.4978055884915966, 0.49780558849160256, 0.49780558849160256], [0.0, 0.3487492736590919, 0.4377188655397367, 0.4457690254707469, 0.4344733659314021, 0.428630422569088, 0.4226044607656039, 0.4195581868072364, 0.41669704278765834, 0.41669704278765834], [0.0, 0.36444198176302317, 0.46138118598072964, 0.46712417341958135, 0.471653391198235, 0.4753836163309769, 0.4785899978761065, 0.481365148059892, 0.481365148059892], [0.0, 0.3362590251873797, 0.40593634406117607, 0.41099348390791407, 0.4068844212363845, 0.40897848556342387, 0.40897848556342387], [0.0, 0.35920029599611353, 0.4342688250105092, 0.46225463577866227, 0.4538319452951886, 0.44831546937801037, 0.4487786023256298, 0.4491034978665007, 0.45144981838124937, 0.4514498183812581, 0.4514498183812581], [0.0, 0.3580609717723564, 0.45380658902269794, 0.466324323336392, 0.4708875462107065, 0.4729886306213358, 0.4746063596992872, 0.47553510686790085, 0.47567954254363604, 0.47567954254363615, 0.47567954254363615], [0.0, 0.35337259733556264, 0.4460815523287945, 0.4576400256566333, 0.45639804282312246, 0.4536677256240786, 0.4506582957714473, 0.4476824009354082, 0.44434145920083395, 0.4403889327853159, 0.4365606245719081, 0.43468503803445985, 0.43289309801025816, 0.4311644920875476, 0.42948257161075454, 0.42948257161075454], [0.0, 0.34031957586493206, 0.43142662716742164, 0.44450986236850254, 0.4362213148690918, 0.42649950570882667, 0.4136892221122319, 0.3997984695120931, 0.39623303098399487, 0.39279175681124096, 0.39293366272308033, 0.39428457656584437, 0.39428457656584437], [0.0, 0.33135452468318005, 0.4050572620037609, 0.41544665771498984, 0.4219788076800103, 0.4219788076800131, 0.4219788076800169, 0.4219788076800169]]"
        self.expected_objective_curves = "[([0, 210, 270, 330, 390, 450, 510, 570, 630, 930, 1000], [0.0, 0.3528961132849871, 0.433564071695458, 0.4453426026667063, 0.44790583734543843, 0.4482535291098946, 0.4482915292989911, 0.4480933729136198, 0.4477313528945558, 0.44773135289455396, 0.44773135289455396]), ([0, 210, 270, 330, 390, 570, 690, 1000], [0.0, 0.3653863363301532, 0.4820584522164391, 0.5053034727420339, 0.49780558849158935, 0.4978055884915966, 0.49780558849160256, 0.49780558849160256]), ([0, 210, 270, 330, 390, 450, 510, 570, 630, 1000], [0.0, 0.3487492736590919, 0.4377188655397367, 0.4457690254707469, 0.4344733659314021, 0.428630422569088, 0.4226044607656039, 0.4195581868072364, 0.41669704278765834, 0.41669704278765834]), ([0, 210, 270, 330, 390, 450, 510, 570, 1000], [0.0, 0.36444198176302317, 0.46138118598072964, 0.46712417341958135, 0.471653391198235, 0.4753836163309769, 0.4785899978761065, 0.481365148059892, 0.481365148059892]), ([0, 210, 270, 330, 390, 450, 1000], [0.0, 0.3362590251873797, 0.40593634406117607, 0.41099348390791407, 0.4068844212363845, 0.40897848556342387, 0.40897848556342387]), ([0, 210, 270, 330, 390, 570, 690, 810, 870, 990, 1000], [0.0, 0.35920029599611353, 0.4342688250105092, 0.46225463577866227, 0.4538319452951886, 0.44831546937801037, 0.4487786023256298, 0.4491034978665007, 0.45144981838124937, 0.4514498183812581, 0.4514498183812581]), ([0, 210, 270, 330, 390, 450, 510, 570, 630, 810, 1000], [0.0, 0.3580609717723564, 0.45380658902269794, 0.466324323336392, 0.4708875462107065, 0.4729886306213358, 0.4746063596992872, 0.47553510686790085, 0.47567954254363604, 0.47567954254363615, 0.47567954254363615]), ([0, 210, 270, 330, 390, 450, 510, 570, 630, 690, 750, 810, 870, 930, 990, 1000], [0.0, 0.35337259733556264, 0.4460815523287945, 0.4576400256566333, 0.45639804282312246, 0.4536677256240786, 0.4506582957714473, 0.4476824009354082, 0.44434145920083395, 0.4403889327853159, 0.4365606245719081, 0.43468503803445985, 0.43289309801025816, 0.4311644920875476, 0.42948257161075454, 0.42948257161075454]), ([0, 210, 270, 330, 390, 450, 510, 570, 630, 690, 810, 870, 1000], [0.0, 0.34031957586493206, 0.43142662716742164, 0.44450986236850254, 0.4362213148690918, 0.42649950570882667, 0.4136892221122319, 0.3997984695120931, 0.39623303098399487, 0.39279175681124096, 0.39293366272308033, 0.39428457656584437, 0.39428457656584437]), ([0, 210, 270, 330, 390, 450, 630, 1000], [0.0, 0.33135452468318005, 0.4050572620037609, 0.41544665771498984, 0.4219788076800103, 0.4219788076800131, 0.4219788076800169, 0.4219788076800169])]"
        self.expected_progress_curves = "[([0.0, 0.21, 0.27, 0.33, 0.39, 0.45, 0.51, 0.57, 0.63, 0.93, 1.0], [1.0, 0.301615499751084, 0.14197290324818346, 0.1186630872531895, 0.11359042336504574, 0.11290233831672934, 0.11282713560955163, 0.11321928883243766, 0.11393572962216643, 0.11393572962217005, 0.11393572962217005]), ([0.0, 0.21, 0.27, 0.33, 0.39, 0.57, 0.69, 1.0], [1.0, 0.2768972389059966, 0.04600209929184828, -0.0, 0.01483837862771302, 0.014838378627698628, 0.014838378627686874, 0.014838378627686874]), ([0.0, 0.21, 0.27, 0.33, 0.39, 0.45, 0.51, 0.57, 0.63, 1.0], [1.0, 0.30982213170513, 0.13375052982626204, 0.11781919278769798, 0.1401734019880599, 0.15173663809765425, 0.16366206930592242, 0.16969067216083028, 0.17535290124477468, 0.17535290124477468]), ([0.0, 0.21, 0.27, 0.33, 0.39, 0.45, 0.51, 0.57, 1.0], [1.0, 0.278766124868734, 0.08692259034548008, 0.07555716788422658, 0.06659380621549348, 0.05921165799375297, 0.05286620082178831, 0.047374154292351034, 0.047374154292351034]), ([0.0, 0.21, 0.27, 0.33, 0.39, 0.45, 1.0], [1.0, 0.3345404428695749, 0.19664841830918203, 0.18664029424208353, 0.19477216527244015, 0.19062799362114338, 0.19062799362114338]), ([0.0, 0.21, 0.27, 0.33, 0.39, 0.57, 0.69, 0.81, 0.87, 0.99, 1.0], [1.0, 0.2891394669288342, 0.14057819026268425, 0.08519402554224836, 0.10186260380823149, 0.11277975798341068, 0.11186321382212427, 0.11122024270002244, 0.106576853842993, 0.10657685384297574, 0.10657685384297574]), ([0.0, 0.21, 0.27, 0.33, 0.39, 0.45, 0.51, 0.57, 0.63, 0.81, 1.0], [1.0, 0.29139419955035084, 0.1019127840936609, 0.07714007820710432, 0.06810942015610752, 0.06395135569787665, 0.06074985567815026, 0.058911856893828086, 0.05862601742601005, 0.05862601742600983, 0.05862601742600983]), ([0.0, 0.21, 0.27, 0.33, 0.39, 0.45, 0.51, 0.57, 0.63, 0.69, 0.75, 0.81, 0.87, 0.93, 0.99, 1.0], [1.0, 0.3006725336400658, 0.11720069939727731, 0.09432637940672457, 0.09678427431643342, 0.10218759597624269, 0.10814328402307247, 0.11403260597823416, 0.12064435894411933, 0.12846644335226648, 0.13604269885004378, 0.13975450104144838, 0.14330076605023154, 0.14672169231723356, 0.1500502276777093, 0.1500502276777093]), ([0.0, 0.21, 0.27, 0.33, 0.39, 0.45, 0.51, 0.57, 0.63, 0.69, 0.81, 0.87, 1.0], [1.0, 0.32650457750035883, 0.1462029246973485, 0.12031108759976314, 0.13671419572493165, 0.1559537412350974, 0.18130540471581646, 0.20879532582156415, 0.21585135990886287, 0.2226616716490134, 0.22238083860611085, 0.21970736827464193, 0.21970736827464193]), ([0.0, 0.21, 0.27, 0.33, 0.39, 0.45, 0.63, 1.0], [1.0, 0.3442464923403718, 0.19838812940327924, 0.17782742425939652, 0.1649002422442529, 0.1649002422442473, 0.16490024224423983, 0.16490024224423983])]"

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
