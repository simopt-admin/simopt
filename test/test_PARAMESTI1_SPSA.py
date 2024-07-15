import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class test_PARAMESTI1_SPSA(unittest.TestCase):
    def setUp(self):
        # Expected values
        self.expected_problem_name = "PARAMESTI-1"
        self.expected_solver_name = "SPSA"
        self.expected_all_recommended_xs = "[[(1, 1), (1.0730633963883887, 1.0730633963883887), (1.1805715652878455, 1.0490897761977567), (1.2524913870438796, 1.0361844323726817), (1.3285720514085386, 1.0487953467548081), (1.35137088137727, 1.0259965167860767), (1.3886982244185369, 1.0207014419585891), (1.420997980750457, 1.0162059758668371), (1.4372085985583554, 0.9999953580589386), (1.4372085985583554, 0.9999953580589386)], [(1, 1), (1.1193169611955687, 1.1193169611955687), (1.2136590735199597, 1.2136590735199597), (1.2915792151512941, 1.2915792151512941), (1.3163133639980131, 1.2668450663045752), (1.3380252596958442, 1.245133170606744), (1.3380252596958442, 1.245133170606744)], [(1, 1), (1.0745392310831705, 1.0745392310831705), (1.1373809936793897, 1.1373809936793897), (1.1911524060782865, 1.1911524060782865), (1.2382474132767798, 1.2382474132767798), (1.3008384843888667, 1.2445103621215379), (1.3494254815774904, 1.2506909348353838), (1.403264355534035, 1.2425407030788789), (1.403264355534035, 1.2425407030788789)], [(1, 1), (1.2937596768617543, 1.094638966275609), (1.3529983175239348, 1.0973875248786358), (1.3752074143280788, 1.0751784280744918), (1.3752074143280788, 1.0751784280744918)], [(1, 1), (1.0757440256108675, 1.0757440256108675), (1.1392822819785051, 1.1392822819785051), (1.2295567245212893, 1.1309320806415393), (1.3233583762926437, 1.1552821600558068), (1.347157472816242, 1.1314830635322084), (1.3692444110565622, 1.1093961252918882), (1.3898901791159632, 1.0887503572324873), (1.409303419094182, 1.0693371172542685), (1.4435233097041769, 1.0668667429837626), (1.4435233097041769, 1.0668667429837626)], [(1, 1), (1.0796222219458675, 1.0796222219458675), (1.147553802987517, 1.147553802987517), (1.2058742010337637, 1.2058742010337637), (1.257008861935299, 1.257008861935299), (1.3026370414566526, 1.3026370414566526), (1.343914574172541, 1.343914574172541), (1.3612590029340996, 1.3265701454109822), (1.3612590029340996, 1.3265701454109822)], [(1, 1), (1.124972388127519, 0.8750276118724808), (1.2171416936373776, 0.7828583063626222), (1.3206108928739768, 0.7367952268816601), (1.3745167329233416, 0.6828893868322954), (1.4724842473176216, 0.6842563637316473), (1.5053672227876733, 0.6513733882615957), (1.5590271782392189, 0.6431935427271658), (1.5590271782392189, 0.6431935427271658)], [(1, 1), (1.1, 1.1), (1.3981068340246214, 1.2156178806199394), (1.4546376721887448, 1.2128205093609161), (1.5014368713169837, 1.2105899293501292), (1.5223370586025773, 1.1896897420645356), (1.5223370586025773, 1.1896897420645356)], [(1, 1), (1.1, 1.1), (1.1985654273148325, 1.1985654273148325), (1.28622805223397, 1.28622805223397), (1.425766006028837, 1.1466900984391029), (1.5511660400211136, 1.0271143689670301), (1.6425912473413469, 0.9356891616467969), (1.7274391282551531, 0.8508412807329906), (1.785443054184837, 0.7502220737849563), (1.785443054184837, 0.7502220737849563)], [(1, 1), (1.1, 1.1), (1.1690717861244158, 1.0309282138755844), (1.2251016190343056, 0.9748983809656946), (1.304950299655502, 0.9592985531012898), (1.3403841081177335, 0.9238647446390583), (1.3726100598347155, 0.8916387929220764), (1.4022524779347332, 0.8619963748220588), (1.4753759802929316, 0.8801001198361371), (1.4952891759580946, 0.8601869241709741), (1.4952891759580946, 0.8601869241709741)]]"
        self.expected_all_intermediate_budgets = "[[0, 210, 330, 450, 630, 690, 810, 930, 990, 1000], [0, 210, 270, 330, 390, 450, 1000], [0, 210, 270, 330, 390, 510, 630, 810, 1000], [0, 450, 570, 630, 1000], [0, 210, 270, 390, 570, 630, 690, 750, 810, 930, 1000], [0, 210, 270, 330, 390, 450, 510, 570, 1000], [0, 210, 270, 390, 450, 690, 750, 930, 1000], [0, 210, 570, 690, 810, 870, 1000], [0, 210, 270, 330, 390, 570, 630, 690, 810, 1000], [0, 210, 270, 330, 450, 510, 570, 630, 870, 930, 1000]]"
        self.expected_all_est_objectives = "[[-9.16944227603262, -8.76199918901914, -8.478958825757433, -8.31419158516531, -8.11131104639232, -8.109237540666536, -8.04531697247482, -7.994271738823127, -7.9992309411239555, -7.9992309411239555], [-8.940090362495347, -8.316515494500386, -7.887602329950944, -7.572060863540566, -7.5632514571935925, -7.5581831664713715, -7.5581831664713715], [-9.121210005202611, -8.707614720876107, -8.387872565278455, -8.133927458999755, -7.9256054984010635, -7.765947418936601, -7.651152490997057, -7.5616229304636775, -7.5616229304636775], [-8.779886386724968, -7.775671702387292, -7.651509542951069, -7.653423420098762, -7.653423420098762], [-8.99288952739613, -8.571080297750942, -8.245730297536934, -7.999604906463446, -7.720310120440221, -7.70966613327603, -7.702438140547024, -7.697976418780367, -7.695795051399041, -7.634617738570967, -7.634617738570967], [-8.87740808504234, -8.444311106025948, -8.107250933956582, -7.840039244548475, -7.621634345310226, -7.4386824804625915, -7.282477738705977, -7.274670186423022, -7.274670186423022], [-9.024638576352391, -8.882127270935003, -8.845624561056535, -8.718851005159365, -8.753880278277734, -8.57855425668448, -8.621371031845321, -8.571915464036836, -8.571915464036836], [-8.921050660074993, -8.391427379336571, -7.473382441260676, -7.385771299561927, -7.321958598189855, -7.331719815012605, -7.331719815012605], [-8.550164686658025, -8.030700364956498, -7.577743523325219, -7.219540952575464, -7.172468400719894, -7.198964317519387, -7.272797028412522, -7.377116995134969, -7.572568527460646, -7.572568527460646], [-8.983830735669818, -8.44483638080425, -8.375791123184792, -8.341465697321043, -8.183989286094437, -8.184903347817244, -8.192075277340896, -8.204005319994922, -8.037564480249985, -8.053025925953289, -8.053025925953289]]"
        self.expected_objective_curves = "[([0, 210, 330, 450, 630, 690, 810, 930, 990, 1000], [-9.265122221743944, -8.76199918901914, -8.478958825757433, -8.31419158516531, -8.11131104639232, -8.109237540666536, -8.04531697247482, -7.994271738823127, -7.9992309411239555, -7.9992309411239555]), ([0, 210, 270, 330, 390, 450, 1000], [-9.265122221743944, -8.316515494500386, -7.887602329950944, -7.572060863540566, -7.5632514571935925, -7.5581831664713715, -7.5581831664713715]), ([0, 210, 270, 330, 390, 510, 630, 810, 1000], [-9.265122221743944, -8.707614720876107, -8.387872565278455, -8.133927458999755, -7.9256054984010635, -7.765947418936601, -7.651152490997057, -7.5616229304636775, -7.5616229304636775]), ([0, 450, 570, 630, 1000], [-9.265122221743944, -7.775671702387292, -7.651509542951069, -7.653423420098762, -7.653423420098762]), ([0, 210, 270, 390, 570, 630, 690, 750, 810, 930, 1000], [-9.265122221743944, -8.571080297750942, -8.245730297536934, -7.999604906463446, -7.720310120440221, -7.70966613327603, -7.702438140547024, -7.697976418780367, -7.695795051399041, -7.634617738570967, -7.634617738570967]), ([0, 210, 270, 330, 390, 450, 510, 570, 1000], [-9.265122221743944, -8.444311106025948, -8.107250933956582, -7.840039244548475, -7.621634345310226, -7.4386824804625915, -7.282477738705977, -7.274670186423022, -7.274670186423022]), ([0, 210, 270, 390, 450, 690, 750, 930, 1000], [-9.265122221743944, -8.882127270935003, -8.845624561056535, -8.718851005159365, -8.753880278277734, -8.57855425668448, -8.621371031845321, -8.571915464036836, -8.571915464036836]), ([0, 210, 570, 690, 810, 870, 1000], [-9.265122221743944, -8.391427379336571, -7.473382441260676, -7.385771299561927, -7.321958598189855, -7.331719815012605, -7.331719815012605]), ([0, 210, 270, 330, 390, 570, 630, 690, 810, 1000], [-9.265122221743944, -8.030700364956498, -7.577743523325219, -7.219540952575464, -7.172468400719894, -7.198964317519387, -7.272797028412522, -7.377116995134969, -7.572568527460646, -7.572568527460646]), ([0, 210, 270, 330, 450, 510, 570, 630, 870, 930, 1000], [-9.265122221743944, -8.44483638080425, -8.375791123184792, -8.341465697321043, -8.183989286094437, -8.184903347817244, -8.192075277340896, -8.204005319994922, -8.037564480249985, -8.053025925953289, -8.053025925953289])]"
        self.expected_progress_curves = "[([0.0, 0.21, 0.33, 0.45, 0.63, 0.69, 0.81, 0.93, 0.99, 1.0], [1.0, 0.8919868632623278, 0.8312222481070737, 0.795849137892697, 0.7522936614971535, 0.7518485102261562, 0.7381257016418433, 0.7271670385642127, 0.7282317065706211, 0.7282317065706211]), ([0.0, 0.21, 0.27, 0.33, 0.39, 0.45, 1.0], [1.0, 0.7963480471463962, 0.7042666808051201, 0.6365245554382775, 0.6346333050870859, 0.6335452173964058, 0.6335452173964058]), ([0.0, 0.21, 0.27, 0.33, 0.39, 0.51, 0.63, 0.81, 1.0], [1.0, 0.8803113155098719, 0.8116673637783721, 0.7571490736102752, 0.7124254037481677, 0.6781491555011473, 0.6535043679604678, 0.634283684280806, 0.634283684280806]), ([0.0, 0.45, 0.57, 0.63, 1.0], [1.0, 0.6802368165496725, 0.6535810217787641, 0.6539919031350266, 0.6539919031350266]), ([0.0, 0.21, 0.27, 0.39, 0.57, 0.63, 0.69, 0.75, 0.81, 0.93, 1.0], [1.0, 0.850999377166385, 0.7811515034357877, 0.7283119914447418, 0.6683514967930363, 0.6660663888448405, 0.6645146448057159, 0.6635567785743544, 0.6630884709877224, 0.6499545990996577, 0.6499545990996577]), ([0.0, 0.21, 0.27, 0.33, 0.39, 0.45, 0.51, 0.57, 1.0], [1.0, 0.8237838907956675, 0.7514220148994681, 0.6940555838902775, 0.6471672549703373, 0.6078901726090364, 0.5743553053240823, 0.5726791383480683, 0.5726791383480683]), ([0.0, 0.21, 0.27, 0.39, 0.45, 0.69, 0.75, 0.93, 1.0], [1.0, 0.917776600750076, 0.9099400042578716, 0.8827235809633892, 0.8902438521914114, 0.8526039265424847, 0.8617960602539295, 0.851178675120105, 0.851178675120105]), ([0.0, 0.21, 0.57, 0.69, 0.81, 0.87, 1.0], [1.0, 0.8124305302246723, 0.6153397453113171, 0.5965309180214986, 0.5828312668614772, 0.5849268569633488, 0.5849268569633488]), ([0.0, 0.21, 0.27, 0.33, 0.39, 0.57, 0.63, 0.69, 0.81, 1.0], [1.0, 0.7349877303627964, 0.6377445391568307, 0.5608437003209715, 0.5507379137648869, 0.5564261985344275, 0.5722769987934077, 0.5946729657779372, 0.6366335434369375, 0.6366335434369375]), ([0.0, 0.21, 0.27, 0.33, 0.45, 0.51, 0.57, 0.63, 0.87, 0.93, 1.0], [1.0, 0.8238966595873215, 0.8090736553373498, 0.8017044897928499, 0.7678966137219465, 0.768092849368956, 0.7696325574600681, 0.7721937626616999, 0.7364613552528422, 0.7397807009005031, 0.7397807009005031])]"

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
