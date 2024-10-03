import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class test_DUALSOURCING1_RNDSRCH(unittest.TestCase):
    def setUp(self):
        # Expected values
        self.expected_problem_name = "DUALSOURCING-1"
        self.expected_solver_name = "RNDSRCH"
        self.expected_all_recommended_xs = "[[(50, 80), (48, 83), (54, 85), (52, 83), (49, 87), (54, 90), (49, 90), (49, 90)], [(50, 80), (55, 87), (54, 87), (52, 88), (50, 89), (50, 89)], [(50, 80), (53, 84), (55, 88), (48, 87), (48, 87)], [(50, 80), (47, 84), (52, 90), (51, 90), (51, 90)], [(50, 80), (46, 86), (49, 87), (47, 90), (53, 89), (50, 88), (50, 88)], [(50, 80), (50, 82), (55, 89), (49, 86), (52, 90), (50, 90), (50, 90)], [(50, 80), (56, 88), (49, 86), (51, 86), (51, 90), (50, 90), (50, 90)], [(50, 80), (47, 85), (51, 88), (52, 89), (52, 89)], [(50, 80), (54, 90), (48, 90), (48, 90)], [(50, 80), (54, 83), (52, 83), (54, 86), (48, 88), (52, 90), (52, 90)], [(50, 80), (50, 83), (53, 86), (47, 90), (52, 88), (51, 90), (51, 90)], [(50, 80), (53, 90), (50, 90), (50, 90)], [(50, 80), (50, 82), (51, 88), (49, 90), (49, 90)], [(50, 80), (55, 86), (54, 86), (49, 89), (51, 89), (49, 90), (51, 90), (51, 90)], [(50, 80), (52, 83), (51, 85), (52, 90), (50, 90), (50, 90)], [(50, 80), (49, 82), (49, 84), (51, 90), (51, 90)], [(50, 80), (48, 87), (54, 87), (52, 86), (50, 90), (50, 90)], [(50, 80), (47, 86), (54, 89), (50, 89), (50, 89)], [(50, 80), (53, 83), (46, 88), (52, 87), (48, 90), (48, 90)], [(50, 80), (46, 89), (49, 84), (47, 90), (47, 90)], [(50, 80), (48, 85), (51, 87), (53, 90), (53, 90)], [(50, 80), (48, 82), (51, 85), (50, 87), (52, 90), (52, 90)], [(50, 80), (51, 81), (51, 90), (51, 90)], [(50, 80), (46, 88), (56, 90), (51, 85), (50, 85), (46, 90), (54, 90), (50, 87), (50, 87)]]"
        self.expected_all_intermediate_budgets = "[[0, 20, 30, 170, 290, 520, 590, 1000], [0, 110, 210, 240, 610, 1000], [0, 50, 230, 690, 1000], [0, 60, 70, 240, 1000], [0, 30, 50, 330, 600, 780, 1000], [0, 60, 120, 150, 250, 520, 1000], [0, 20, 30, 140, 150, 220, 1000], [0, 60, 220, 630, 1000], [0, 140, 470, 1000], [0, 120, 140, 170, 180, 550, 1000], [0, 110, 130, 170, 350, 400, 1000], [0, 40, 130, 1000], [0, 50, 140, 450, 1000], [0, 70, 100, 190, 470, 540, 900, 1000], [0, 50, 110, 130, 490, 1000], [0, 40, 50, 90, 1000], [0, 50, 170, 310, 320, 1000], [0, 50, 240, 320, 1000], [0, 100, 160, 200, 340, 1000], [0, 30, 40, 200, 1000], [0, 30, 290, 380, 1000], [0, 20, 120, 180, 200, 1000], [0, 40, 110, 1000], [0, 30, 40, 50, 130, 380, 610, 780, 1000]]"
        self.expected_all_est_objectives = "[[3295.6687, 3289.1849000000007, 3283.9767500000003, 3284.02175, 3266.177525, 3262.6564000000003, 3253.321725, 3253.321725], [3293.9677249999995, 3280.4072250000004, 3274.2230750000003, 3260.9720999999995, 3254.2430249999998, 3254.2430249999998], [3291.3866, 3279.8244250000002, 3273.886475, 3266.6751, 3266.6751], [3291.3193, 3288.5852749999995, 3250.3023749999998, 3248.080675, 3248.080675], [3298.841925, 3298.904075, 3269.3992, 3268.3267250000004, 3265.4206499999996, 3263.2492, 3263.2492], [3301.858575, 3292.7115000000003, 3278.3033750000004, 3277.2083249999996, 3260.0329500000003, 3257.9074249999994, 3257.9074249999994], [3292.7081999999996, 3282.025675, 3267.613175, 3266.1359250000005, 3249.3984750000004, 3248.8652249999996, 3248.8652249999996], [3298.6563, 3291.847825, 3263.0922499999997, 3260.92975, 3260.92975], [3295.815025000001, 3264.1634999999997, 3258.0775, 3258.0775], [3297.5789, 3294.264925, 3285.5824, 3280.9518, 3268.9954, 3255.4451749999994, 3255.4451749999994], [3297.3286, 3283.6958, 3276.9807999999994, 3266.5272250000003, 3264.6113, 3254.0228500000003, 3254.0228500000003], [3294.936775, 3258.17495, 3251.3071750000004, 3251.3071750000004], [3294.569275, 3285.437025, 3259.327375, 3252.342525, 3252.342525], [3291.4617249999997, 3282.144925, 3276.067575, 3253.6408250000004, 3252.1333999999997, 3249.43875, 3248.059475, 3248.059475], [3298.947075, 3287.48, 3276.5307499999994, 3257.4833999999996, 3255.0105500000004, 3255.0105500000004], [3294.804725, 3287.7612, 3278.6729499999997, 3251.8084, 3251.8084], [3298.940925, 3274.4538749999997, 3278.726025, 3274.34945, 3255.092625, 3255.092625], [3294.60075, 3282.4127500000004, 3266.4228749999997, 3254.9552499999995, 3254.9552499999995], [3292.9795499999996, 3285.2570499999997, 3282.1686250000002, 3263.9058250000003, 3254.8136250000007, 3254.8136250000007], [3295.500875, 3281.32805, 3279.5721750000002, 3265.58055, 3265.58055], [3297.530325, 3282.0586749999998, 3266.5066250000004, 3259.813675, 3259.813675], [3301.7639750000003, 3300.6249250000005, 3279.034925, 3270.294125, 3259.7942250000006, 3259.7942250000006], [3298.922975, 3294.3079749999997, 3255.291475, 3255.291475], [3296.4926, 3286.1313250000003, 3276.443975, 3274.1289749999996, 3273.913175, 3276.6189, 3263.795625, 3265.27875, 3265.27875]]"
        self.expected_objective_curves = "[([0, 20, 30, 170, 290, 520, 590, 1000], [3294.62375, 3289.1849000000007, 3283.9767500000003, 3284.02175, 3266.177525, 3262.6564000000003, 3253.321725, 3253.321725]), ([0, 110, 210, 240, 610, 1000], [3294.62375, 3280.4072250000004, 3274.2230750000003, 3260.9720999999995, 3254.2430249999998, 3254.2430249999998]), ([0, 50, 230, 690, 1000], [3294.62375, 3279.8244250000002, 3273.886475, 3266.6751, 3266.6751]), ([0, 60, 70, 240, 1000], [3294.62375, 3288.5852749999995, 3250.3023749999998, 3251.1229, 3251.1229]), ([0, 30, 50, 330, 600, 780, 1000], [3294.62375, 3298.904075, 3269.3992, 3268.3267250000004, 3265.4206499999996, 3263.2492, 3263.2492]), ([0, 60, 120, 150, 250, 520, 1000], [3294.62375, 3292.7115000000003, 3278.3033750000004, 3277.2083249999996, 3260.0329500000003, 3257.9074249999994, 3257.9074249999994]), ([0, 20, 30, 140, 150, 220, 1000], [3294.62375, 3282.025675, 3267.613175, 3266.1359250000005, 3251.1229, 3248.8652249999996, 3248.8652249999996]), ([0, 60, 220, 630, 1000], [3294.62375, 3291.847825, 3263.0922499999997, 3260.92975, 3260.92975]), ([0, 140, 470, 1000], [3294.62375, 3264.1634999999997, 3258.0775, 3258.0775]), ([0, 120, 140, 170, 180, 550, 1000], [3294.62375, 3294.264925, 3285.5824, 3280.9518, 3268.9954, 3255.4451749999994, 3255.4451749999994]), ([0, 110, 130, 170, 350, 400, 1000], [3294.62375, 3283.6958, 3276.9807999999994, 3266.5272250000003, 3264.6113, 3251.1229, 3251.1229]), ([0, 40, 130, 1000], [3294.62375, 3258.17495, 3251.3071750000004, 3251.3071750000004]), ([0, 50, 140, 450, 1000], [3294.62375, 3285.437025, 3259.327375, 3252.342525, 3252.342525]), ([0, 70, 100, 190, 470, 540, 900, 1000], [3294.62375, 3282.144925, 3276.067575, 3253.6408250000004, 3252.1333999999997, 3249.43875, 3251.1229, 3251.1229]), ([0, 50, 110, 130, 490, 1000], [3294.62375, 3287.48, 3276.5307499999994, 3257.4833999999996, 3255.0105500000004, 3255.0105500000004]), ([0, 40, 50, 90, 1000], [3294.62375, 3287.7612, 3278.6729499999997, 3251.1229, 3251.1229]), ([0, 50, 170, 310, 320, 1000], [3294.62375, 3274.4538749999997, 3278.726025, 3274.34945, 3255.092625, 3255.092625]), ([0, 50, 240, 320, 1000], [3294.62375, 3282.4127500000004, 3266.4228749999997, 3254.9552499999995, 3254.9552499999995]), ([0, 100, 160, 200, 340, 1000], [3294.62375, 3285.2570499999997, 3282.1686250000002, 3263.9058250000003, 3254.8136250000007, 3254.8136250000007]), ([0, 30, 40, 200, 1000], [3294.62375, 3281.32805, 3279.5721750000002, 3265.58055, 3265.58055]), ([0, 30, 290, 380, 1000], [3294.62375, 3282.0586749999998, 3266.5066250000004, 3259.813675, 3259.813675]), ([0, 20, 120, 180, 200, 1000], [3294.62375, 3300.6249250000005, 3279.034925, 3270.294125, 3259.7942250000006, 3259.7942250000006]), ([0, 40, 110, 1000], [3294.62375, 3294.3079749999997, 3251.1229, 3251.1229]), ([0, 30, 40, 50, 130, 380, 610, 780, 1000], [3294.62375, 3286.1313250000003, 3276.443975, 3274.1289749999996, 3273.913175, 3276.6189, 3263.795625, 3265.27875, 3265.27875])]"
        self.expected_progress_curves = "[([0.0, 0.02, 0.03, 0.17, 0.29, 0.52, 0.59, 1.0], [1.0, 0.8749714086046708, 0.7552461618566109, 0.7562806244015868, 0.34607657091758187, 0.2651327502796006, 0.05054671345502271, 0.05054671345502271]), ([0.0, 0.11, 0.21, 0.24, 0.61, 1.0], [1.0, 0.6731897192813542, 0.5310281293354108, 0.22641396662363092, 0.07172560995934311, 0.07172560995934311]), ([0.0, 0.05, 0.23, 0.69, 1.0], [1.0, 0.6597922799209709, 0.5232903494989123, 0.35751485315804077, 0.35751485315804077]), ([0.0, 0.06, 0.07, 0.24, 1.0], [1.0, 0.8611871951927217, -0.0188622751049711, 0.0, 0.0]), ([0.0, 0.03, 0.05, 0.33, 0.6, 0.78, 1.0], [1.0, 1.0983963531747007, 0.42013661802010405, 0.39548250206606, 0.32867748561234106, 0.2787600702055306, 0.2787600702055306]), ([0.0, 0.06, 0.12, 0.15, 0.25, 0.52, 1.0], [1.0, 0.9560410888522904, 0.6248262964976626, 0.5996532251668542, 0.20482473330981665, 0.1559630444002708, 0.1559630444002708]), ([0.0, 0.02, 0.03, 0.14, 0.15, 0.22, 1.0], [1.0, 0.7103947394131326, 0.37907937431107613, 0.34512026776489496, 0.0, -0.0518995605833052, -0.0518995605833052]), ([0.0, 0.06, 0.22, 0.63, 1.0], [1.0, 0.936186879106952, 0.27515209472917773, 0.22544042242852405, 0.22544042242852405]), ([0.0, 0.14, 0.47, 1.0], [1.0, 0.29977805031395266, 0.15987273811890965, 0.15987273811890965]), ([0.0, 0.12, 0.14, 0.17, 0.18, 0.55, 1.0], [1.0, 0.9917513106065686, 0.7921569348644831, 0.6857084401798984, 0.4108540407831065, 0.09936070214718794, 0.09936070214718794]), ([0.0, 0.11, 0.13, 0.17, 0.35, 0.4, 1.0], [1.0, 0.7487876673674159, 0.5944228675991229, 0.354115494294946, 0.3100721020393871, 0.0, 0.0]), ([0.0, 0.04, 0.13, 1.0], [1.0, 0.16211292423022028, 0.004236124121724606, 0.004236124121724606]), ([0.0, 0.05, 0.14, 0.45, 1.0], [1.0, 0.7888150461427776, 0.18860493530586034, 0.028036808476160083, 0.028036808476160083]), ([0.0, 0.07, 0.1, 0.19, 0.47, 0.54, 0.9, 1.0], [1.0, 0.7131360651573465, 0.5734295996515008, 0.05788220230180649, 0.023229431149043227, -0.03871533544746899, 0.0, 0.0]), ([0.0, 0.05, 0.11, 0.13, 0.49, 1.0], [1.0, 0.8357790709836675, 0.5840770927464496, 0.14621553371944968, 0.08936951806690087, 0.08936951806690087]), ([0.0, 0.04, 0.05, 0.09, 1.0], [1.0, 0.8422433124869921, 0.6333221074990397, 0.0, 0.0]), ([0.0, 0.05, 0.17, 0.31, 0.32, 1.0], [1.0, 0.53633377278834, 0.6345421986007139, 0.5339332449825722, 0.09125626280866576, 0.09125626280866576]), ([0.0, 0.05, 0.24, 0.32, 1.0], [1.0, 0.7192928414042556, 0.3517166905933971, 0.08809827853937675, 0.08809827853937675]), ([0.0, 0.1, 0.16, 0.2, 0.34, 1.0], [1.0, 0.7846777706642418, 0.7136808820977095, 0.29385460284110143, 0.08484259502977089, 0.08484259502977089]), ([0.0, 0.03, 0.04, 0.2, 1.0], [1.0, 0.6943576964588044, 0.6539935426549175, 0.3323532758555331, 0.3323532758555331]), ([0.0, 0.03, 0.29, 0.38, 1.0], [1.0, 0.7111533452794524, 0.3536419403299124, 0.1997840272086628, 0.1997840272086628]), ([0.0, 0.02, 0.12, 0.18, 0.2, 1.0], [1.0, 1.1379553502977593, 0.6416432092706201, 0.4407092045327825, 0.19933690950867938, 0.19933690950867938]), ([0.0, 0.04, 0.11, 1.0], [1.0, 0.9927409464412653, 0.0, 0.0]), ([0.0, 0.03, 0.04, 0.05, 0.13, 0.38, 0.61, 0.78, 1.0], [1.0, 0.8047756538090671, 0.5820823041388866, 0.5288649532135485, 0.5239041306089446, 0.5861034899318033, 0.29132131900871483, 0.3254154803871645, 0.3254154803871645])]"

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
