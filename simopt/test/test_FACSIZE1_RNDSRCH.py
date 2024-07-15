import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class test_FACSIZE1_RNDSRCH(unittest.TestCase):
    def setUp(self):
        # Expected values
        self.expected_problem_name = "FACSIZE-1"
        self.expected_solver_name = "RNDSRCH"
        self.expected_all_recommended_xs = "[[(300, 300, 300), (289.8915086187877, 195.03698226817045, 212.9793556453907), (289.8915086187877, 195.03698226817045, 212.9793556453907)], [(300, 300, 300), (380.927949319142, 227.6318357963392, 290.7130595057053), (221.8705975458371, 451.4169891215547, 211.149308573062), (212.9283349015932, 299.0174639377216, 281.77804395323756), (398.14575586829335, 229.4098837465824, 164.94492952979186), (356.936183203668, 188.95044489766755, 228.44368280537333), (334.62232455609757, 225.11087331417042, 201.39873978276938), (373.97224895725583, 192.1043516615712, 172.33971889199702), (373.97224895725583, 192.1043516615712, 172.33971889199702)], [(300, 300, 300), (217.71567896416917, 205.77047481676706, 406.5343135354902), (317.73985356050315, 227.79275595882498, 220.31893894315053), (250.64041365878714, 224.88687352391975, 256.6246316494979), (208.51974831378504, 211.92050242140425, 244.1953580172491), (208.51974831378504, 211.92050242140425, 244.1953580172491)], [(300, 300, 300), (190.20498923989018, 474.0379345263219, 193.32111016188117), (237.85025044134346, 216.5066264927754, 396.99080077925316), (194.9605027171918, 332.9019112349397, 200.38632260364227), (219.73676588685467, 237.2873240604807, 202.67779636082452), (220.6489322008942, 242.10349595988814, 195.22303227384361), (203.44781664835222, 253.34246266593482, 172.0981958215241), (203.44781664835222, 253.34246266593482, 172.0981958215241)], [(300, 300, 300), (218.38483452962876, 236.36409120816987, 414.52735391378417), (214.55410145789597, 396.98550216843745, 209.0336858353265), (251.73379592593736, 192.64639356909197, 219.44809358445386), (199.39430208017, 200.57275004544425, 191.33531933032947), (199.82844464071093, 173.65927835678093, 202.92545955199498), (199.82844464071093, 173.65927835678093, 202.92545955199498)], [(300, 300, 300), (190.27113654655494, 353.51049361957394, 301.40561613205386), (236.160970145895, 252.2290312214704, 182.12199371577316), (236.160970145895, 252.2290312214704, 182.12199371577316)], [(300, 300, 300), (256.9312040061072, 193.3304141300806, 386.43317647300904), (153.45887974518723, 156.80969447211004, 366.90608264783685), (251.80855077884598, 148.4823947697158, 245.34435671209596), (160.327459900841, 162.80050614091945, 255.4448822726978), (160.327459900841, 162.80050614091945, 255.4448822726978)], [(300, 300, 300), (223.33067077931395, 272.4287310368244, 314.77278889001514), (331.7142259966948, 218.57542759744746, 202.05499582108524), (331.7142259966948, 218.57542759744746, 202.05499582108524)], [(300, 300, 300), (276.7918561983725, 274.4581344529287, 271.0922413370609), (253.2060207603967, 307.43904225567366, 189.91293747120812), (211.38549249416735, 217.35983410353742, 202.62911801108402), (191.213402041859, 202.64370663337124, 183.20912544847488), (191.213402041859, 202.64370663337124, 183.20912544847488)], [(300, 300, 300), (270.26477222338156, 338.7828797005254, 276.22655957311554), (406.7379863459669, 241.94662382482272, 224.61629501669952), (228.73215177759894, 308.58981561460513, 320.85399379539217), (294.9631195928099, 268.00060409145647, 290.44914405899385), (339.59912111493975, 231.45331994977818, 197.86058593467408), (352.3466244656597, 218.42418077276466, 191.34295722525428), (209.95456005981617, 249.07148405405155, 240.54051120480395), (209.95456005981617, 249.07148405405155, 240.54051120480395)]]"
        self.expected_all_intermediate_budgets = "[[0, 710, 10000], [0, 230, 380, 1120, 2510, 4340, 4400, 5330, 10000], [0, 790, 1020, 5090, 8070, 10000], [0, 140, 160, 1480, 1670, 2920, 8980, 10000], [0, 1120, 1610, 1870, 3260, 5910, 10000], [0, 750, 1040, 10000], [0, 430, 530, 590, 3400, 10000], [0, 1170, 1790, 10000], [0, 210, 2630, 4060, 4100, 10000], [0, 50, 770, 3190, 3270, 4180, 5770, 9150, 10000]]"
        self.expected_all_est_objectives = "[[900.0, 697.907846532349, 697.907846532349], [900.0, 899.2728446211863, 884.4368952404539, 793.7238427925523, 792.5005691446677, 774.3303109067091, 761.1319376530373, 738.416319510824, 738.416319510824], [900.0, 830.0204673164263, 765.8515484624787, 732.1519188322047, 664.6356087524383, 664.6356087524383], [900.0, 857.5640339280933, 851.3476777133723, 728.2487365557737, 659.70188630816, 657.975460434626, 628.8884751358111, 628.8884751358111], [900.0, 869.2762796515829, 820.5732894616596, 663.8282830794832, 591.3023714559439, 576.4131825494869, 576.4131825494869], [900.0, 845.1872462981829, 670.5119950831386, 670.5119950831386], [900.0, 836.6947946091968, 677.1746568651344, 645.6353022606579, 578.5728483144584, 578.5728483144584], [900.0, 810.5321907061536, 752.3446494152274, 752.3446494152274], [900.0, 822.3422319883621, 750.5580004872785, 631.3744446087886, 577.066234123705, 577.066234123705], [900.0, 885.2742114970225, 873.3009051874889, 858.1759611875963, 853.4128677432601, 768.9130269993921, 762.1137624636788, 699.5665553186717, 699.5665553186717]]"
        self.expected_objective_curves = "[([0, 710, 10000], [900.0, 697.907846532349, 697.907846532349]), ([0, 230, 380, 1120, 2510, 4340, 4400, 5330, 10000], [900.0, 899.2728446211863, 884.4368952404539, 793.7238427925523, 792.5005691446677, 774.3303109067091, 761.1319376530373, 738.416319510824, 738.416319510824]), ([0, 790, 1020, 5090, 8070, 10000], [900.0, 830.0204673164263, 765.8515484624787, 732.1519188322047, 664.6356087524383, 664.6356087524383]), ([0, 140, 160, 1480, 1670, 2920, 8980, 10000], [900.0, 857.5640339280933, 851.3476777133723, 728.2487365557737, 659.70188630816, 657.975460434626, 628.8884751358111, 628.8884751358111]), ([0, 1120, 1610, 1870, 3260, 5910, 10000], [900.0, 869.2762796515829, 820.5732894616596, 663.8282830794832, 591.3023714559439, 576.4131825494869, 576.4131825494869]), ([0, 750, 1040, 10000], [900.0, 845.1872462981829, 670.5119950831386, 670.5119950831386]), ([0, 430, 530, 590, 3400, 10000], [900.0, 836.6947946091968, 677.1746568651344, 645.6353022606579, 578.5728483144584, 578.5728483144584]), ([0, 1170, 1790, 10000], [900.0, 810.5321907061536, 752.3446494152274, 752.3446494152274]), ([0, 210, 2630, 4060, 4100, 10000], [900.0, 822.3422319883621, 750.5580004872785, 631.3744446087886, 577.066234123705, 577.066234123705]), ([0, 50, 770, 3190, 3270, 4180, 5770, 9150, 10000], [900.0, 885.2742114970225, 873.3009051874889, 858.1759611875963, 853.4128677432601, 768.9130269993921, 762.1137624636788, 699.5665553186717, 699.5665553186717])]"
        self.expected_progress_curves = "[([0.0, 0.071, 1.0], [1.0, 0.3754623409572072, 0.3754623409572072]), ([0.0, 0.023, 0.038, 0.112, 0.251, 0.434, 0.44, 0.533, 1.0], [1.0, 0.9977528275578628, 0.9519043918965387, 0.6715683350614221, 0.6677879781929853, 0.6116353253095365, 0.5708475906370934, 0.5006481359090366, 0.5006481359090366]), ([0.0, 0.079, 0.102, 0.509, 0.807, 1.0], [1.0, 0.7837379988624666, 0.5854328906398144, 0.4812888779269734, 0.2726391232437751, 0.2726391232437751]), ([0.0, 0.014, 0.016, 0.148, 0.167, 0.292, 0.898, 1.0], [1.0, 0.8688575560455378, 0.8496467727889804, 0.4692266366181847, 0.25739214104854774, 0.25205686229048135, 0.16216758457519462, 0.16216758457519462]), ([0.0, 0.112, 0.161, 0.187, 0.326, 0.591, 1.0], [1.0, 0.905052620528598, 0.7545428112179281, 0.2701441956712741, 0.04601296500199369, 0.0, 0.0]), ([0.0, 0.075, 0.104, 1.0], [1.0, 0.8306088173378703, 0.2907992769144324, 0.2907992769144324]), ([0.0, 0.043, 0.053, 0.059, 0.34, 1.0], [1.0, 0.8043640779634523, 0.3113893053787864, 0.21392132181576679, 0.0066741463140776655, 0.0066741463140776655]), ([0.0, 0.117, 0.179, 1.0], [1.0, 0.7235121937328955, 0.5436916999644034, 0.5436916999644034]), ([0.0, 0.021, 0.263, 0.406, 0.41, 1.0], [1.0, 0.7600094817721852, 0.5381703102427031, 0.16985012706120842, 0.0020181649529588696, 0.0020181649529588696]), ([0.0, 0.005, 0.077, 0.319, 0.327, 0.418, 0.577, 0.915, 1.0], [1.0, 0.9544920011915209, 0.9174901653198704, 0.8707486320304133, 0.8560289549994892, 0.594893963748522, 0.5738817834956812, 0.38058834948682335, 0.38058834948682335])]"

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
