import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class test_MM11_NELDMD(unittest.TestCase):
    def setUp(self):
        # Expected values
        self.expected_problem_name = "MM1-1"
        self.expected_solver_name = "NELDMD"
        self.expected_all_recommended_xs = "[[(5,), (3.402793206539716,), (2.604189809809574,), (3.003491508174645,), (2.8038406589921094,), (2.8287970151399264,), (2.816318837066018,), (2.8225579261029723,), (2.8221679830381623,), (2.8223629545705675,), (2.8223629545705675,)], [(5,), (2.970106601179852,), (2.8432382637535927,), (2.8749553481101575,), (2.859096805931875,), (2.85711448815959,), (2.8581056470457327,), (2.8576100676026615,), (2.8576720150330455,), (2.8576720150330455,)], [(5,), (2.5398676148660257,), (2.8473841630077725,), (2.808944594490054,), (2.8281643787489132,), (2.8185544866194836,), (2.8209569596518413,), (2.8197557231356623,), (2.820356341393752,), (2.820356341393752,)], [(5,), (2.6455592511292894,), (2.9398643447381283,), (2.8662880713359185,), (2.9030762080370236,), (2.8938791738617473,), (2.8984776909493855,), (2.897328061677476,), (2.897328061677476,)], [(5,), (2.691382899199186,), (2.979960036799288,), (2.835671467999237,), (2.8717436101992497,), (2.867234592424248,), (2.866670965202373,), (2.8669527788133107,), (2.8669527788133107,)], [(5,), (2.6451804034362896,), (2.8414153698165987,), (2.8168859990190604,), (2.8199521703687527,), (2.8203354417874644,), (2.8201438060781086,), (2.8201438060781086,)], [(5,), (2.5766612978753325,), (2.818995168087799,), (2.697828232981566,), (2.7584117005346824,), (2.7432658336464035,), (2.739479366924334,), (2.7413726002853687,), (2.7408992919451096,), (2.7411359461152394,), (2.7410176190301745,), (2.7410767825727067,), (2.7410767825727067,)], [(5,), (2.587000339321399,), (2.8886252969062243,), (2.7378128181138117,), (2.7755159378119147,), (2.7566643779628635,), (2.7590208229439948,), (2.7590208229439948,)], [(5,), (2.768364681234213,), (2.838103284945644,), (2.829385959481715,), (2.831565290847697,), (2.8310204580062015,), (2.8310204580062015,)], [(5,), (3.126465518801295,), (2.6580818985016186,), (2.892273708651457,), (2.775177803576538,), (2.8337257561139975,), (2.8044517798452677,), (2.8190887679796326,), (2.81177027391245,), (2.8154295209460414,), (2.8154295209460414,)]]"
        self.expected_all_intermediate_budgets = "[[0, 120, 240, 300, 360, 540, 600, 660, 900, 960, 1000], [0, 120, 300, 420, 480, 660, 720, 780, 960, 1000], [0, 180, 360, 540, 600, 660, 780, 840, 900, 1000], [0, 180, 360, 480, 540, 660, 720, 840, 1000], [0, 180, 360, 420, 540, 720, 900, 960, 1000], [0, 180, 300, 480, 660, 840, 900, 1000], [0, 180, 240, 300, 360, 480, 600, 660, 780, 840, 900, 960, 1000], [0, 180, 360, 420, 540, 600, 780, 1000], [0, 180, 480, 660, 780, 900, 1000], [0, 180, 300, 360, 420, 480, 540, 600, 660, 720, 1000]]"
        self.expected_all_est_objectives = "[[2.7852321578182706, 1.6800692645834387, 1.5757661165678583, 1.5615362172844625, 1.5454764118579556, 1.5453483143002529, 1.5453244748299075, 1.5453146671557314, 1.5453137976580544, 1.5453141942819526, 1.5453141942819526], [2.7857037031168543, 1.558199008139838, 1.547898177089963, 1.5489801650906794, 1.5483091610640054, 1.5482441689967117, 1.5482763003715518, 1.5482601486900052, 1.5482621592889183, 1.5482621592889183], [2.7866293625352507, 1.6250013469401086, 1.5649415445159398, 1.565526366780859, 1.5650445111663702, 1.5652346282335277, 1.565177275911252, 1.5652053592677373, 1.5651911945247576, 1.5651911945247576], [2.7889080044387127, 1.5869879498482913, 1.569817266348574, 1.5653392821646883, 1.5669010604490508, 1.5663740644136999, 1.5666225057145808, 1.5665575229065742, 1.5665575229065742], [2.7833651638972787, 1.551712605861189, 1.5542996751332294, 1.5420425169752634, 1.5433391073874367, 1.5431070611953772, 1.5430793461978096, 1.5430931689897809, 1.5430931689897809], [2.787955763524055, 1.594766378682147, 1.5669538565371, 1.5677179196727766, 1.5675912439125466, 1.5675760047398029, 1.5675836111858568, 1.5675836111858568], [2.7843462630059106, 1.5810161393809539, 1.5455010577616408, 1.5532450921768188, 1.5471489552661633, 1.5482376001017806, 1.548551123546784, 1.5483924493049583, 1.5484317063677622, 1.5484120498770297, 1.5484218734327277, 1.5484169604451825, 1.5484169604451825], [2.7907221687784363, 1.622216250540972, 1.5789706856220218, 1.5861986783351893, 1.581939861017691, 1.5838439245459641, 1.5835789545436671, 1.5835789545436671], [2.789502875694011, 1.5650816035822153, 1.5625358258254465, 1.5625667845295426, 1.5625522798550078, 1.5625555361591261, 1.5625555361591261], [2.7891645344327056, 1.605837430504597, 1.59483924111534, 1.5728275684243656, 1.5753762053144607, 1.5722747908242116, 1.573339079876476, 1.5726948202938191, 1.5729861911372063, 1.5728340439885082, 1.5728340439885082]]"
        self.expected_objective_curves = "[([0, 120, 240, 300, 360, 540, 600, 660, 900, 960, 1000], [2.7854035060729516, 1.6800692645834387, 1.5757661165678583, 1.5615362172844625, 1.5454764118579556, 1.5453483143002529, 1.5453244748299075, 1.5453146671557314, 1.5453137976580544, 1.5453141942819526, 1.5453141942819526]), ([0, 120, 300, 420, 480, 660, 720, 780, 960, 1000], [2.7854035060729516, 1.558199008139838, 1.547898177089963, 1.5489801650906794, 1.5483091610640054, 1.5482441689967117, 1.5482763003715518, 1.5482601486900052, 1.5482621592889183, 1.5482621592889183]), ([0, 180, 360, 540, 600, 660, 780, 840, 900, 1000], [2.7854035060729516, 1.6250013469401086, 1.5649415445159398, 1.565526366780859, 1.5650445111663702, 1.5652346282335277, 1.565177275911252, 1.5652053592677373, 1.5651911945247576, 1.5651911945247576]), ([0, 180, 360, 480, 540, 660, 720, 840, 1000], [2.7854035060729516, 1.5869879498482913, 1.569817266348574, 1.5653392821646883, 1.5669010604490508, 1.5663740644136999, 1.5666225057145808, 1.5665575229065742, 1.5665575229065742]), ([0, 180, 360, 420, 540, 720, 900, 960, 1000], [2.7854035060729516, 1.551712605861189, 1.5542996751332294, 1.552395869037029, 1.5433391073874367, 1.5431070611953772, 1.5430793461978096, 1.5430931689897809, 1.5430931689897809]), ([0, 180, 300, 480, 660, 840, 900, 1000], [2.7854035060729516, 1.594766378682147, 1.5669538565371, 1.5677179196727766, 1.5675912439125466, 1.5675760047398029, 1.5675836111858568, 1.5675836111858568]), ([0, 180, 240, 300, 360, 480, 600, 660, 780, 840, 900, 960, 1000], [2.7854035060729516, 1.5810161393809539, 1.5455010577616408, 1.5532450921768188, 1.5471489552661633, 1.5482376001017806, 1.548551123546784, 1.5483924493049583, 1.5484317063677622, 1.5484120498770297, 1.5484218734327277, 1.5484169604451825, 1.5484169604451825]), ([0, 180, 360, 420, 540, 600, 780, 1000], [2.7854035060729516, 1.622216250540972, 1.5789706856220218, 1.5861986783351893, 1.581939861017691, 1.5838439245459641, 1.5835789545436671, 1.5835789545436671]), ([0, 180, 480, 660, 780, 900, 1000], [2.7854035060729516, 1.5650816035822153, 1.5625358258254465, 1.5625667845295426, 1.5625522798550078, 1.5625555361591261, 1.5625555361591261]), ([0, 180, 300, 360, 420, 480, 540, 600, 660, 720, 1000], [2.7854035060729516, 1.605837430504597, 1.59483924111534, 1.5728275684243656, 1.5753762053144607, 1.5722747908242116, 1.573339079876476, 1.5726948202938191, 1.5729861911372063, 1.5728340439885082, 1.5728340439885082])]"
        self.expected_progress_curves = "[([0.0, 0.12, 0.24, 0.3, 0.36, 0.54, 0.6, 0.66, 0.9, 0.96, 1.0], [1.0, 0.10354631367355441, 0.018953854646845476, 0.007413050797808795, -0.00561185265300331, -0.005715742972783202, -0.0057350773788560246, -0.005743031647654967, -0.005743736831995154, -0.005743415160104198, -0.005743415160104198]), ([0.0, 0.12, 0.3, 0.42, 0.48, 0.66, 0.72, 0.78, 0.96, 1.0], [1.0, 0.004706490802246331, -0.0036477405426930264, -0.0027702212409330667, -0.0033144222714206195, -0.003367132461805124, -0.0033410731140160127, -0.0033541725312957794, -0.003352541885342952, -0.003352541885342952]), ([0.0, 0.18, 0.36, 0.54, 0.6, 0.66, 0.78, 0.84, 0.9, 1.0], [1.0, 0.05888485660771644, 0.010174856263721038, 0.010649161732197342, 0.010258364789814145, 0.010412554481302566, 0.010366040315004762, 0.01038881661876935, 0.010377328658310555, 0.010377328658310555]), ([0.0, 0.18, 0.36, 0.48, 0.54, 0.66, 0.72, 0.84, 1.0], [1.0, 0.02805504181176008, 0.01412918848858482, 0.010497431434224155, 0.011764072643452072, 0.01133666569192843, 0.01153815779418192, 0.011485455113310606, 0.011485455113310606]), ([0.0, 0.18, 0.36, 0.42, 0.54, 0.72, 0.9, 0.96, 1.0], [1.0, -0.0005541435067527424, 0.0015440343101013401, 0.0, -0.0073452599785709, -0.00753345523794281, -0.007555932793421883, -0.007544722163774453, -0.007544722163774453]), ([0.0, 0.18, 0.3, 0.48, 0.66, 0.84, 0.9, 1.0], [1.0, 0.03436354193796749, 0.01180689158995602, 0.01242656588290155, 0.012323828676395199, 0.012311469326553423, 0.012317638344348151, 0.012317638344348151]), ([0.0, 0.18, 0.24, 0.3, 0.36, 0.48, 0.6, 0.66, 0.78, 0.84, 0.9, 0.96, 1.0], [1.0, 0.023211754318672614, -0.005591864209343322, 0.0006887411839810973, -0.004255378160900118, -0.003372459999716329, -0.0031181846525197444, -0.003246873427073508, -0.0032150349683124544, -0.003230976873408587, -0.0032230097242986087, -0.003226994279947466, -0.003226994279947466]), ([0.0, 0.18, 0.36, 0.42, 0.54, 0.6, 0.78, 1.0], [1.0, 0.05662607384313301, 0.021552840215067214, 0.027414922894898143, 0.023960915645002766, 0.025505158738946994, 0.02529026144687998, 0.02529026144687998]), ([0.0, 0.18, 0.48, 0.66, 0.78, 0.9, 1.0], [1.0, 0.010288447665808498, 0.008223758299497121, 0.008248866582013934, 0.008237102928570833, 0.008239743872568695, 0.008239743872568695]), ([0.0, 0.18, 0.3, 0.36, 0.42, 0.48, 0.54, 0.6, 0.66, 0.72, 1.0], [1.0, 0.04334244157322368, 0.03442263519173524, 0.01657061868363863, 0.018637626878512323, 0.01612230223891427, 0.016985467251276063, 0.016462956633089176, 0.016699265666898236, 0.016575870527948573, 0.016575870527948573])]"

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
