import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class test_MM11_ADAM(unittest.TestCase):
    def setUp(self):
        # Expected values
        self.expected_problem_name = "MM1-1"
        self.expected_solver_name = "ADAM"
        self.expected_all_recommended_xs = "[[(5,), (4.500000005366897,), (4.003004560324342,), (3.5123680289566863,), (3.033809830342704,), (2.7040480453692344,), (2.841378779661173,), (2.841378779661173,)], [(5,), (4.500000005377763,), (4.003063613212165,), (3.512698763886283,), (3.0349726782257616,), (2.8392016843721914,), (2.8392016843721914,)], [(5,), (4.500000005375356,), (4.003021944059995,), (3.512528800915692,), (3.03440158276607,), (2.7037094220298337,), (2.8428708705357058,), (2.8428708705357058,)], [(5,), (4.50000000538071,), (4.003092018928721,), (3.512809639456232,), (3.035197131139862,), (2.872500415114987,), (2.8857068749001282,), (2.8857068749001282,)], [(5,), (4.500000005399897,), (4.003102733658987,), (3.5128936519448044,), (3.0354861202091614,), (2.7346144472620924,), (2.8758091523718488,), (2.8758091523718488,)], [(5,), (4.500000005388163,), (4.003058472726469,), (3.5126780187496123,), (3.0349493734745496,), (2.686477889201981,), (2.831428970373492,), (2.831428970373492,)], [(5,), (4.50000000536467,), (4.003002495195689,), (3.5123763945691593,), (3.033685402293352,), (2.577816876095712,), (2.6278661195002955,), (2.774297909432402,), (2.7587563844921235,), (2.7587563844921235,)], [(5,), (4.500000005356673,), (4.002969569343121,), (3.512261056149438,), (3.0334024427985837,), (2.577291436683673,), (2.6015203636852884,), (2.753612451462658,), (2.7617575873358655,), (2.7617575873358655,)], [(5,), (4.500000005375804,), (4.003037907171925,), (3.512604308481135,), (3.0346011103126522,), (2.6880116603792086,), (2.834310483874307,), (2.834310483874307,)], [(5,), (4.500000005357146,), (4.002973607736823,), (3.512297952023572,), (3.0336148245939447,), (2.6621236542037057,), (2.8068534006951555,), (2.8068534006951555,)]]"
        self.expected_all_intermediate_budgets = "[[0, 60, 90, 120, 150, 420, 450, 1000], [0, 60, 90, 120, 150, 450, 1000], [0, 60, 90, 120, 150, 420, 450, 1000], [0, 60, 90, 120, 150, 450, 660, 1000], [0, 60, 90, 120, 150, 420, 450, 1000], [0, 60, 90, 120, 150, 420, 450, 1000], [0, 60, 90, 120, 150, 180, 420, 450, 690, 1000], [0, 60, 90, 120, 150, 180, 420, 450, 690, 1000], [0, 60, 90, 120, 150, 420, 450, 1000], [0, 60, 90, 120, 150, 420, 450, 1000]]"
        self.expected_all_est_objectives = "[[2.7852321578182706, 2.3577222923567365, 2.0007947090608287, 1.7277595315142087, 1.5670475808635072, 1.5536274548033373, 1.5455346393846563, 1.5455346393846563], [2.7857037031168543, 2.358264583497712, 2.001600818463035, 1.729985862726112, 1.5689877932554779, 1.5478331564209389, 1.5478331564209389], [2.7866293625352507, 2.35980239336231, 2.0046780793778654, 1.7358468907673676, 1.5821271702679323, 1.576026391369124, 1.5649350507059543, 1.5649350507059543], [2.7889080044387127, 2.362302779280393, 2.0072424575131644, 1.7377589470462727, 1.5827974447301625, 1.5655141984346137, 1.5659938798921655, 1.5659938798921655], [2.7833651638972787, 2.3549506392236124, 1.9973644207333896, 1.72438765925803, 1.5637066548134007, 1.546201397075579, 1.5435656983349022, 1.5435656983349022], [2.787955763524055, 2.361043297403672, 2.005396134757599, 1.7361086135219, 1.5820419866805162, 1.5844296210523061, 1.567191382599853, 1.567191382599853], [2.7843462630059106, 2.356852454095064, 2.000138788267423, 1.7282668447527374, 1.5675295393684667, 1.58063903766362, 1.5663710685177055, 1.5463057856633415, 1.547127331680449, 1.547127331680449], [2.7907221687784363, 2.3649491744090856, 2.0117334854436093, 1.7452409673170575, 1.594053693287159, 1.625785476715929, 1.6172073691478284, 1.5841956544451268, 1.583280184592205, 1.583280184592205], [2.789502875694011, 2.362519357738991, 2.006812063682465, 1.7369781349621172, 1.580988291079133, 1.5752024844909869, 1.5625408955175755, 1.5625408955175755], [2.7891645344327056, 2.3629750270472005, 2.008371136708882, 1.7398882929189385, 1.5874299336702535, 1.5938290757196913, 1.5732166386438513, 1.5732166386438513]]"
        self.expected_objective_curves = "[([0, 60, 90, 120, 150, 420, 450, 1000], [2.7854035060729516, 2.3577222923567365, 2.0007947090608287, 1.7277595315142087, 1.5670475808635072, 1.5536274548033373, 1.5455346393846563, 1.5455346393846563]), ([0, 60, 90, 120, 150, 450, 1000], [2.7854035060729516, 2.358264583497712, 2.001600818463035, 1.729985862726112, 1.5689877932554779, 1.5478331564209389, 1.5478331564209389]), ([0, 60, 90, 120, 150, 420, 450, 1000], [2.7854035060729516, 2.35980239336231, 2.0046780793778654, 1.7358468907673676, 1.5821271702679323, 1.576026391369124, 1.5649350507059543, 1.5649350507059543]), ([0, 60, 90, 120, 150, 450, 660, 1000], [2.7854035060729516, 2.362302779280393, 2.0072424575131644, 1.7377589470462727, 1.5827974447301625, 1.5655141984346137, 1.5659938798921655, 1.5659938798921655]), ([0, 60, 90, 120, 150, 420, 450, 1000], [2.7854035060729516, 2.3549506392236124, 1.9973644207333896, 1.72438765925803, 1.5637066548134007, 1.546201397075579, 1.5537685325350565, 1.5537685325350565]), ([0, 60, 90, 120, 150, 420, 450, 1000], [2.7854035060729516, 2.361043297403672, 2.005396134757599, 1.7361086135219, 1.5820419866805162, 1.5844296210523061, 1.567191382599853, 1.567191382599853]), ([0, 60, 90, 120, 150, 180, 420, 450, 690, 1000], [2.7854035060729516, 2.356852454095064, 2.000138788267423, 1.7282668447527374, 1.5675295393684667, 1.58063903766362, 1.5663710685177055, 1.5463057856633415, 1.547127331680449, 1.547127331680449]), ([0, 60, 90, 120, 150, 180, 420, 450, 690, 1000], [2.7854035060729516, 2.3649491744090856, 2.0117334854436093, 1.7452409673170575, 1.594053693287159, 1.625785476715929, 1.6172073691478284, 1.5841956544451268, 1.583280184592205, 1.583280184592205]), ([0, 60, 90, 120, 150, 420, 450, 1000], [2.7854035060729516, 2.362519357738991, 2.006812063682465, 1.7369781349621172, 1.580988291079133, 1.5752024844909869, 1.5625408955175755, 1.5625408955175755]), ([0, 60, 90, 120, 150, 420, 450, 1000], [2.7854035060729516, 2.3629750270472005, 2.008371136708882, 1.7398882929189385, 1.5874299336702535, 1.5938290757196913, 1.5732166386438513, 1.5732166386438513])]"
        self.expected_progress_curves = "[([0.0, 0.06, 0.09, 0.12, 0.15, 0.42, 0.45, 1.0], [1.0, 0.6527532727593042, 0.36295346115552474, 0.14126831627666409, 0.010781642786828548, -0.00011454508417699583, -0.006685335612667952, -0.006685335612667952]), ([0.0, 0.06, 0.09, 0.12, 0.15, 0.45, 1.0], [1.0, 0.6531935745959903, 0.36360796465658296, 0.14307593887567824, 0.012356957253904318, -0.00481910325838521, -0.00481910325838521]), ([0.0, 0.06, 0.09, 0.12, 0.15, 0.42, 0.45, 1.0], [1.0, 0.6544421668312209, 0.3661064816530522, 0.14783467678681414, 0.023025196866093365, 0.018071798310607616, 0.00906641854998784, 0.00906641854998784]), ([0.0, 0.06, 0.09, 0.12, 0.15, 0.45, 0.66, 1.0], [1.0, 0.6564723023598511, 0.3681885743106948, 0.14938713049264923, 0.023569412056982958, 0.009536645314494047, 0.009926112541276302, 0.009926112541276302]), ([0.0, 0.06, 0.09, 0.12, 0.15, 0.42, 0.45, 1.0], [1.0, 0.6505028875456053, 0.36016831100865493, 0.138530595824887, 0.008069048453371435, -0.006143975789954042, 0.0, 0.0]), ([0.0, 0.06, 0.09, 0.12, 0.15, 0.42, 0.45, 1.0], [1.0, 0.655449692655043, 0.366689491550596, 0.14804717704878745, 0.02295603385168873, 0.02489462314404325, 0.01089839956902083, 0.01089839956902083]), ([0.0, 0.06, 0.09, 0.12, 0.15, 0.18, 0.42, 0.45, 0.69, 1.0], [1.0, 0.6520470259569957, 0.36242090012283384, 0.14168021854432333, 0.011172958814153708, 0.021816939032980955, 0.010232362878140677, -0.0060592196811999535, -0.005392182746752094, -0.005392182746752094]), ([0.0, 0.06, 0.09, 0.12, 0.15, 0.18, 0.42, 0.45, 0.69, 1.0], [1.0, 0.6586209869827723, 0.3718349695714142, 0.15546199880309722, 0.03270868529852033, 0.05847263655886809, 0.051507823320851756, 0.02470465889959892, 0.023961362490686488, 0.023961362490686488]), ([0.0, 0.06, 0.09, 0.12, 0.15, 0.42, 0.45, 1.0], [1.0, 0.6566481486643582, 0.36783912513139533, 0.14875316661460786, 0.022100507966160837, 0.017402844524917072, 0.007122534818348128, 0.007122534818348128]), ([0.0, 0.06, 0.09, 0.12, 0.15, 0.42, 0.45, 1.0], [1.0, 0.6570181197337088, 0.36910498154170684, 0.1511160078941648, 0.02733066359629589, 0.03252631180938304, 0.01579047893787046, 0.01579047893787046])]"

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
