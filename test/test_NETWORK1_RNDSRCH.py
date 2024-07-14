import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class test_NETWORK1_RNDSRCH(unittest.TestCase):
    def setUp(self):
        # Expected values
        self.expected_problem_name = "NETWORK-1"
        self.expected_solver_name = "RNDSRCH"
        self.expected_all_recommended_xs = "[[(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1), [0.006426160247395138, 0.3485885002807809, 0.0032379922927322623, 0.15731403695221596, 0.10178880026186957, 0.10297813159597309, 0.12770554370690224, 0.09780860181911108, 0.04789356776694945, 0.00625866507607038], [0.006426160247395138, 0.3485885002807809, 0.0032379922927322623, 0.15731403695221596, 0.10178880026186957, 0.10297813159597309, 0.12770554370690224, 0.09780860181911108, 0.04789356776694945, 0.00625866507607038]], [(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1), [0.05696871650250251, 0.030326407757629723, 0.18925393406500357, 0.1225490060298385, 0.1168968924789166, 0.14828626256930166, 0.06969998091980603, 0.10346901684482862, 0.12557177241263548, 0.03697801041953715], [0.05696871650250251, 0.030326407757629723, 0.18925393406500357, 0.1225490060298385, 0.1168968924789166, 0.14828626256930166, 0.06969998091980603, 0.10346901684482862, 0.12557177241263548, 0.03697801041953715]], [(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1), [0.002757855835346458, 0.2108154120029565, 0.25348366236505254, 0.2455678713127596, 0.012535768167107511, 0.001592617451398355, 0.007231097553653179, 0.08938654128713562, 0.08979586164021913, 0.08683331238437113], [0.002757855835346458, 0.2108154120029565, 0.25348366236505254, 0.2455678713127596, 0.012535768167107511, 0.001592617451398355, 0.007231097553653179, 0.08938654128713562, 0.08979586164021913, 0.08683331238437113]], [(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1), [0.01636341458470823, 0.11086500817761781, 0.23229609052389602, 0.2507841351605212, 0.08007181744699308, 0.16925353278183647, 0.020512872386398363, 0.018993718515662443, 0.08326081995091446, 0.01759859047145199], [0.01636341458470823, 0.11086500817761781, 0.23229609052389602, 0.2507841351605212, 0.08007181744699308, 0.16925353278183647, 0.020512872386398363, 0.018993718515662443, 0.08326081995091446, 0.01759859047145199]], [(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1), [0.040786653696496304, 0.2794808152748242, 0.12715883123131236, 0.20462961499882648, 0.14356305547350107, 0.06206932141198861, 0.04010698342877502, 0.0381037464861489, 0.00323473644508852, 0.060866241553038464], [0.040786653696496304, 0.2794808152748242, 0.12715883123131236, 0.20462961499882648, 0.14356305547350107, 0.06206932141198861, 0.04010698342877502, 0.0381037464861489, 0.00323473644508852, 0.060866241553038464]], [(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1), [0.011817151682923185, 0.23833876505263468, 0.07546946058429202, 0.13040711782443612, 0.14320188412214377, 0.11168007736943254, 0.09079089477499441, 0.06767678140042133, 0.019382238835409762, 0.11123562835331237], [0.011817151682923185, 0.23833876505263468, 0.07546946058429202, 0.13040711782443612, 0.14320188412214377, 0.11168007736943254, 0.09079089477499441, 0.06767678140042133, 0.019382238835409762, 0.11123562835331237]], [(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1), [0.018553447380570614, 0.3507417361430201, 0.041485754300258304, 0.05496883118914075, 0.16104817371160662, 0.10058499416641549, 0.11898145478784736, 0.05483643323860349, 0.050420202387136494, 0.04837897269540069], [0.056823900278784444, 0.024156748359902246, 0.09563010739607714, 0.23013065306077868, 0.13388361966162285, 0.09482882762205073, 0.10655401733666452, 0.08831351786978256, 0.10792468772236896, 0.0617539206919679], [0.022713834989403492, 0.07435402943638098, 0.22313400301326186, 0.13424684338805387, 0.18333775320871742, 0.08827362179787486, 0.11094674408474303, 0.0963622643292233, 0.026800585143976357, 0.039830320608364764], [0.022713834989403492, 0.07435402943638098, 0.22313400301326186, 0.13424684338805387, 0.18333775320871742, 0.08827362179787486, 0.11094674408474303, 0.0963622643292233, 0.026800585143976357, 0.039830320608364764]], [(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1), [0.0023774656723509796, 0.32600837542258665, 0.08059568594704276, 0.16194549719371615, 0.17253375484871408, 0.02857497242201638, 0.09551525676142868, 0.05525157174817948, 0.07071962804146327, 0.006477791942501524], [0.0023774656723509796, 0.32600837542258665, 0.08059568594704276, 0.16194549719371615, 0.17253375484871408, 0.02857497242201638, 0.09551525676142868, 0.05525157174817948, 0.07071962804146327, 0.006477791942501524]], [(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1), [0.05464909630877593, 0.1262036820980385, 0.24273424153668657, 0.17121924251324253, 0.013715256313726642, 0.09771544927687897, 0.08848281649307277, 0.09230357843328262, 0.10230467223237838, 0.010671964793917302], [0.05464909630877593, 0.1262036820980385, 0.24273424153668657, 0.17121924251324253, 0.013715256313726642, 0.09771544927687897, 0.08848281649307277, 0.09230357843328262, 0.10230467223237838, 0.010671964793917302]], [(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1), (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)]]"
        self.expected_all_intermediate_budgets = "[[0, 710, 1000], [0, 40, 1000], [0, 840, 1000], [0, 360, 1000], [0, 130, 1000], [0, 350, 1000], [0, 110, 640, 900, 1000], [0, 540, 1000], [0, 780, 1000], [0, 1000]]"
        self.expected_all_est_objectives = "[[374.26387546766983, 351.0838700529034, 351.0838700529034], [374.92363611669754, 368.09151963091995, 368.09151963091995], [375.25419698945274, 373.9117132708454, 373.9117132708454], [374.55016054738235, 387.1171823421017, 387.1171823421017], [375.60459477947575, 369.5814470619176, 369.5814470619176], [375.2281664459391, 357.91447303829773, 357.91447303829773], [376.8541839253335, 354.878879547984, 330.7371373304416, 317.25393053621804, 317.25393053621804], [373.64831006309873, 347.6897683702463, 347.6897683702463], [377.2963176264956, 357.2205110618623, 357.2205110618623], [376.0586471727726, 376.0586471727726]]"
        self.expected_objective_curves = "[([0, 710, 1000], [374.48332808289985, 351.0838700529034, 351.0838700529034]), ([0, 40, 1000], [374.48332808289985, 368.09151963091995, 368.09151963091995]), ([0, 840, 1000], [374.48332808289985, 373.9117132708454, 373.9117132708454]), ([0, 360, 1000], [374.48332808289985, 387.1171823421017, 387.1171823421017]), ([0, 130, 1000], [374.48332808289985, 369.5814470619176, 369.5814470619176]), ([0, 350, 1000], [374.48332808289985, 357.91447303829773, 357.91447303829773]), ([0, 110, 640, 900, 1000], [374.48332808289985, 354.878879547984, 330.7371373304416, 317.9572932282748, 317.9572932282748]), ([0, 540, 1000], [374.48332808289985, 347.6897683702463, 347.6897683702463]), ([0, 780, 1000], [374.48332808289985, 357.2205110618623, 357.2205110618623]), ([0, 1000], [374.48332808289985, 374.48332808289985])]"
        self.expected_progress_curves = "[([0.0, 0.71, 1.0], [1.0, 0.5860410501076948, 0.5860410501076948]), ([0.0, 0.04, 1.0], [1.0, 0.8869227521721893, 0.8869227521721893]), ([0.0, 0.84, 1.0], [1.0, 0.9898875834201964, 0.9898875834201964]), ([0.0, 0.36, 1.0], [1.0, 1.223505050224979, 1.223505050224979]), ([0.0, 0.13, 1.0], [1.0, 0.913281003459927, 0.913281003459927]), ([0.0, 0.35, 1.0], [1.0, 0.7068809958594429, 0.7068809958594429]), ([0.0, 0.11, 0.64, 0.9, 1.0], [1.0, 0.6531784232639879, 0.22608775115810448, 0.0, 0.0]), ([0.0, 0.54, 1.0], [1.0, 0.5259961222901657, 0.5259961222901657]), ([0.0, 0.78, 1.0], [1.0, 0.6946041401022648, 0.6946041401022648]), ([0.0, 1.0], [1.0, 1.0])]"

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
