import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class test_EXAMPLE1_NELDMD(unittest.TestCase):
    def setUp(self):
        # Expected values
        self.expected_problem_name = "EXAMPLE-1"
        self.expected_solver_name = "NELDMD"
        self.expected_all_recommended_xs = "[[(2.0, 2.0), (0.0014398540752765588, 0.0005734645090497872), (-0.00021385740037241883, -0.00016352846523951176), (-0.00021385740037241883, -0.00016352846523951176)], [(2.0, 2.0), (2.0, 2.0)], [(2.0, 2.0), (0.39716772151346813, 0.20467360046640937), (0.08716502555664843, -0.11946174822433622), (0.12726996114380834, -0.07309676690981598), (0.06745384401303463, -0.12782602561979162), (0.11775565662196757, -0.06246069234573895), (0.05793953949119385, -0.11718995105571459), (0.12863510614367288, -0.01382391386259707), (0.04435065520836494, -0.07159941268598959), (0.07969843853460445, -0.019916394089430828), (0.07969843853460445, -0.019916394089430828)], [(2.0, 2.0), (0.4932156251752103, -0.09115699609092676), (-0.32304954230884053, -0.28196743697500054), (0.047869876567697416, 0.1708294596286205), (0.01774782665620792, 0.08370579105183301), (-0.03614761318123169, 0.005325261832183507), (-0.01911115323496152, -0.013215443677410676), (-0.01911115323496152, -0.013215443677410676)], [(2.0, 2.0), (-0.3155053294698361, 0.3495098184288207), (-0.002107063551796063, 0.012737973639889748), (-0.002107063551796063, 0.012737973639889748)], [(2.0, 2.0), (0.007988206696347255, 0.08078197950820071), (-0.003408463630263689, -0.026835299918143277), (-0.003408463630263689, -0.026835299918143277)], [(2.0, 2.0), (0.14253442599595245, -0.3582480888806312), (0.18524773522605542, 0.026430324798442123), (-0.08393384426712736, 0.10132952658846828), (-0.06096447413800525, -0.004202477055053333), (0.020283598187474123, -0.014416977158996293), (-0.012747101152268084, 0.01026449860608806), (-0.0006927579058337947, 0.013881242521373408), (0.0067818343292115925, -0.0011720532976327788), (0.0067818343292115925, -0.0011720532976327788)], [(2.0, 2.0), (2.0, 2.0)], [(2.0, 2.0), (0.5417562561029352, -0.357146561384323), (-0.30012615593996594, -0.26387956575141014), (0.22228773229234422, 0.3079195129134994), (0.2514185221395622, -0.16756329390163918), (-0.031636514362006374, -0.09685072812274001), (-0.015895199914400252, 0.0016367166665950533), (-0.003960263110404868, 0.0023129955340262503), (-0.003960263110404868, 0.0023129955340262503)], [(2.0, 2.0), (0.24882147837578839, 0.12877784771062784), (-0.02496856670110592, -0.20126953428536853), (0.12548123723269694, -0.049878610748196106), (0.07126211165127411, 0.0046524590951069755), (-0.017517571710983032, -0.05741023521265347), (-0.0002372928205658792, 0.0008866468628783014), (-0.0002372928205658792, 0.0008866468628783014)]]"
        self.expected_all_intermediate_budgets = "[[0, 750, 870, 1000], [0, 1000], [0, 150, 480, 540, 600, 660, 720, 780, 840, 960, 1000], [0, 360, 420, 480, 660, 720, 900, 1000], [0, 300, 750, 1000], [0, 390, 450, 1000], [0, 360, 420, 510, 630, 810, 870, 930, 990, 1000], [0, 1000], [0, 210, 270, 330, 390, 450, 750, 990, 1000], [0, 270, 390, 450, 510, 630, 750, 1000]]"
        self.expected_all_est_objectives = "[[7.984539704940337, -0.015457893018363457, -0.015460222583118077, -0.015460222583118077], [8.081590387702734, 8.081590387702734], [7.9253347189439385, 0.12496820068401976, -0.05279643008695753, -0.05312450071384893, -0.05377576715817695, -0.05689754830128356, -0.05757480619117103, -0.05792718992898891, -0.06757182454167351, -0.06791677719768373, -0.06791677719768373], [8.073099810658121, 0.32467106151141484, 0.2569664529583223, 0.1045740400177332, 0.08042145546475206, 0.07443481901040137, 0.0736396947876811, 0.0736396947876811], [7.880122723414122, 0.10182344951613623, -0.11971058089661887, -0.11971058089661887], [8.025785950362149, 0.03237549002163351, 0.026517701308162164, 0.026517701308162164], [8.015084462897443, 0.16374221867806235, 0.05009974837277587, 0.03239702606954348, 0.01881879081776376, 0.01570373648327594, 0.015352311416862535, 0.015277631704895272, 0.015131829883243393, 0.015131829883243393], [7.994852045957048, 7.994852045957048], [7.910902809206077, 0.33195631654139063, 0.07061094390652903, 0.05512907156663513, 0.002191539944084133, -0.07871625821504216, -0.08884185457215904, -0.08907615716168044, -0.08907615716168044], [7.943417039435916, 0.02191290159800928, -0.015450105809528596, -0.038349543856466876, -0.051483026631454994, -0.05298016013826365, -0.05658211811354312, -0.05658211811354312]]"
        self.expected_objective_curves = "[([0, 750, 870, 1000], [8.090508544469758, -0.015457893018363457, -0.015460222583118077, -0.015460222583118077]), ([0, 1000], [8.090508544469758, 8.090508544469758]), ([0, 150, 480, 540, 600, 660, 720, 780, 840, 960, 1000], [8.090508544469758, 0.12496820068401976, -0.05279643008695753, -0.05312450071384893, -0.05377576715817695, -0.05689754830128356, -0.05757480619117103, -0.05792718992898891, -0.06757182454167351, -0.06791677719768373, -0.06791677719768373]), ([0, 360, 420, 480, 660, 720, 900, 1000], [8.090508544469758, 0.32467106151141484, 0.2569664529583223, 0.1045740400177332, 0.08042145546475206, 0.07443481901040137, 0.0736396947876811, 0.0736396947876811]), ([0, 300, 750, 1000], [8.090508544469758, 0.10182344951613623, -0.11971058089661887, -0.11971058089661887]), ([0, 390, 450, 1000], [8.090508544469758, 0.03237549002163351, 0.026517701308162164, 0.026517701308162164]), ([0, 360, 420, 510, 630, 810, 870, 930, 990, 1000], [8.090508544469758, 0.16374221867806235, 0.05009974837277587, 0.03239702606954348, 0.01881879081776376, 0.01570373648327594, 0.015352311416862535, 0.015277631704895272, 0.015131829883243393, 0.015131829883243393]), ([0, 1000], [8.090508544469758, 8.090508544469758]), ([0, 210, 270, 330, 390, 450, 750, 990, 1000], [8.090508544469758, 0.33195631654139063, 0.07061094390652903, 0.05512907156663513, 0.002191539944084133, -0.07871625821504216, -0.08884185457215904, -0.08907615716168044, -0.08907615716168044]), ([0, 270, 390, 450, 510, 630, 750, 1000], [8.090508544469758, 0.02191290159800928, -0.015450105809528596, -0.038349543856466876, -0.051483026631454994, -0.05298016013826365, -0.05658211811354312, -0.05658211811354312])]"
        self.expected_progress_curves = "[([0.0, 0.75, 0.87, 1.0], [1.0, -0.0019106206900837712, -0.0019109086280720716, -0.0019109086280720716]), ([0.0, 1.0], [1.0, 1.0]), ([0.0, 0.15, 0.48, 0.54, 0.6, 0.66, 0.72, 0.78, 0.84, 0.96, 1.0], [1.0, 0.015446272628862296, -0.006525724532241718, -0.006566274594711604, -0.006646772185283113, -0.007032629406241182, -0.007116339581709744, -0.007159894784189415, -0.00835198729106615, -0.008394623999761798, -0.008394623999761798]), ([0.0, 0.36, 0.42, 0.48, 0.66, 0.72, 0.9, 1.0], [1.0, 0.040129870665959896, 0.03176147105536041, 0.012925521237996153, 0.009940222548769681, 0.00920026455707547, 0.009101985911382207, 0.009101985911382207]), ([0.0, 0.3, 0.75, 1.0], [1.0, 0.012585543783367898, -0.014796422281569268, -0.014796422281569268]), ([0.0, 0.39, 0.45, 1.0], [1.0, 0.004001663164148522, 0.0032776309625540483, 0.0032776309625540483]), ([0.0, 0.36, 0.42, 0.51, 0.63, 0.81, 0.87, 0.93, 0.99, 1.0], [1.0, 0.02023880424549923, 0.006192410291318634, 0.004004325054657826, 0.002326033118230533, 0.0019410073417461724, 0.0018975706326095606, 0.001888340098885162, 0.0018703187568582086, 0.0018703187568582086]), ([0.0, 1.0], [1.0, 1.0]), ([0.0, 0.21, 0.27, 0.33, 0.39, 0.45, 0.75, 0.99, 1.0], [1.0, 0.041030340023347274, 0.00872762738193941, 0.006814042808757484, 0.0002708778974817539, -0.009729457398428733, -0.010980997558291513, -0.01100995773900618, -0.01100995773900618]), ([0.0, 0.27, 0.39, 0.45, 0.51, 0.63, 0.75, 1.0], [1.0, 0.0027084702373854825, -0.0019096581784206219, -0.004740065923628569, -0.006363385731376065, -0.006548433865072434, -0.00699364172258611, -0.00699364172258611])]"

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
