import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class test_AMUSEMENTPARK1_RNDSRCH(unittest.TestCase):
    def setUp(self):
        # Expected values
        self.expected_problem_name = "AMUSEMENTPARK-1"
        self.expected_solver_name = "RNDSRCH"
        self.expected_all_recommended_xs = "[[(344, 1, 1, 1, 1, 1, 1), (138, 4, 2, 32, 27, 12, 135), (16, 5, 67, 38, 166, 46, 12), (16, 5, 67, 38, 166, 46, 12)], [(344, 1, 1, 1, 1, 1, 1), (103, 56, 23, 63, 23, 69, 13), (37, 106, 5, 21, 112, 20, 49), (40, 75, 17, 8, 41, 39, 130), (65, 28, 33, 37, 114, 54, 19), (65, 28, 33, 37, 114, 54, 19)], [(344, 1, 1, 1, 1, 1, 1), (51, 47, 44, 74, 70, 17, 47), (100, 29, 14, 5, 39, 134, 29), (100, 29, 14, 5, 39, 134, 29)], [(344, 1, 1, 1, 1, 1, 1), (13, 67, 122, 96, 18, 17, 17), (48, 24, 17, 30, 183, 24, 24), (48, 24, 17, 30, 183, 24, 24)], [(344, 1, 1, 1, 1, 1, 1), (2, 107, 28, 69, 34, 86, 24), (74, 35, 6, 47, 99, 51, 38), (65, 44, 14, 23, 124, 32, 48), (65, 44, 14, 23, 124, 32, 48)], [(344, 1, 1, 1, 1, 1, 1), (3, 17, 98, 26, 142, 1, 63), (73, 73, 9, 16, 34, 59, 86), (5, 36, 68, 45, 56, 75, 65)], [(344, 1, 1, 1, 1, 1, 1), (39, 3, 80, 27, 52, 21, 128), (40, 10, 18, 25, 62, 109, 86), (40, 10, 18, 25, 62, 109, 86)], [(344, 1, 1, 1, 1, 1, 1), (41, 23, 24, 168, 28, 38, 28), (16, 55, 46, 12, 103, 112, 6), (29, 11, 115, 20, 55, 51, 69), (29, 11, 115, 20, 55, 51, 69)], [(344, 1, 1, 1, 1, 1, 1), (32, 64, 29, 37, 32, 40, 116), (51, 10, 24, 62, 41, 79, 83), (33, 45, 49, 9, 177, 18, 19), (33, 45, 49, 9, 177, 18, 19)], [(344, 1, 1, 1, 1, 1, 1), (11, 3, 62, 16, 57, 50, 151), (12, 83, 13, 42, 50, 86, 64), (46, 39, 28, 36, 49, 108, 44)], [(344, 1, 1, 1, 1, 1, 1), (138, 11, 58, 32, 44, 54, 13), (45, 25, 44, 50, 141, 19, 26), (45, 25, 44, 50, 141, 19, 26)], [(344, 1, 1, 1, 1, 1, 1), (55, 1, 116, 63, 19, 78, 18), (65, 41, 70, 80, 45, 45, 4), (69, 13, 170, 20, 27, 23, 28), (17, 17, 17, 81, 94, 32, 92), (17, 17, 17, 81, 94, 32, 92)], [(344, 1, 1, 1, 1, 1, 1), (14, 88, 92, 85, 34, 1, 36), (20, 31, 103, 4, 75, 39, 78), (55, 21, 25, 80, 99, 49, 21), (55, 21, 25, 80, 99, 49, 21)], [(344, 1, 1, 1, 1, 1, 1), (9, 103, 6, 42, 68, 113, 9), (22, 94, 47, 49, 50, 48, 40), (37, 7, 14, 82, 14, 117, 79)], [(344, 1, 1, 1, 1, 1, 1), (4, 9, 42, 7, 122, 13, 153), (22, 14, 18, 11, 125, 83, 77), (22, 14, 18, 11, 125, 83, 77)], [(344, 1, 1, 1, 1, 1, 1), (185, 91, 19, 19, 10, 3, 23), (10, 138, 31, 26, 42, 74, 29), (69, 38, 11, 65, 64, 87, 16), (69, 38, 11, 65, 64, 87, 16)], [(344, 1, 1, 1, 1, 1, 1), (53, 53, 24, 132, 11, 60, 17), (25, 31, 59, 66, 89, 42, 38), (38, 56, 8, 50, 45, 86, 67)], [(344, 1, 1, 1, 1, 1, 1), (114, 23, 8, 130, 59, 3, 13), (51, 135, 17, 35, 91, 17, 4), (62, 66, 7, 15, 75, 24, 101), (5, 9, 91, 14, 129, 15, 87), (5, 9, 91, 14, 129, 15, 87)], [(344, 1, 1, 1, 1, 1, 1), (91, 18, 25, 194, 15, 1, 6), (17, 16, 22, 28, 82, 179, 6), (17, 16, 22, 28, 82, 179, 6)], [(344, 1, 1, 1, 1, 1, 1), (126, 41, 8, 105, 25, 17, 28), (3, 35, 8, 133, 81, 37, 53), (14, 65, 118, 40, 52, 40, 21), (15, 55, 59, 74, 19, 114, 14), (42, 13, 11, 10, 152, 100, 22), (42, 13, 11, 10, 152, 100, 22)], [(344, 1, 1, 1, 1, 1, 1), (67, 31, 21, 83, 92, 21, 35), (50, 12, 21, 65, 82, 77, 43), (50, 12, 21, 65, 82, 77, 43)], [(344, 1, 1, 1, 1, 1, 1), (12, 62, 37, 19, 12, 43, 165), (60, 15, 21, 7, 101, 40, 106), (60, 15, 21, 7, 101, 40, 106)], [(344, 1, 1, 1, 1, 1, 1), (13, 5, 162, 75, 12, 51, 32), (49, 24, 34, 25, 41, 100, 77), (49, 24, 34, 25, 41, 100, 77)], [(344, 1, 1, 1, 1, 1, 1), (3, 171, 6, 13, 104, 47, 6), (41, 26, 72, 17, 140, 35, 19), (41, 26, 72, 17, 140, 35, 19)]]"
        self.expected_all_intermediate_budgets = "[[0, 20, 30, 100], [0, 20, 40, 60, 70, 100], [0, 20, 30, 100], [0, 20, 40, 100], [0, 20, 30, 50, 100], [0, 20, 30, 100], [0, 20, 30, 100], [0, 20, 50, 80, 100], [0, 20, 50, 90, 100], [0, 20, 30, 100], [0, 20, 30, 100], [0, 20, 30, 50, 60, 100], [0, 20, 30, 70, 100], [0, 20, 90, 100], [0, 20, 50, 100], [0, 20, 30, 50, 100], [0, 20, 40, 100], [0, 20, 30, 40, 80, 100], [0, 20, 30, 100], [0, 20, 40, 60, 80, 90, 100], [0, 20, 70, 100], [0, 20, 40, 100], [0, 20, 40, 100], [0, 20, 30, 100]]"
        self.expected_all_est_objectives = "[[1696.43, 660.17, 437.83, 437.83], [1692.15, 503.78, 462.48, 457.25, 411.28, 411.28], [1693.035, 459.575, 433.76, 433.76], [1695.61, 565.765, 375.95, 375.95], [1689.125, 559.7, 435.43, 395.31, 395.31], [1695.015, 870.48, 453.305, 448.115], [1696.11, 518.37, 373.12, 373.12], [1693.415, 500.37, 428.155, 433.745, 433.745], [1696.255, 442.125, 419.305, 410.145, 410.145], [1694.79, 491.105, 414.635, 395.73], [1688.89, 500.8, 407.365, 407.365], [1697.96, 674.27, 578.09, 531.99, 409.725, 409.725], [1697.86, 826.23, 472.75, 432.39, 432.39], [1694.48, 471.895, 457.77, 435.835], [1691.175, 485.375, 336.855, 336.855], [1688.535, 700.01, 454.19, 436.13, 436.13], [1692.745, 523.36, 418.49, 412.775], [1693.745, 661.445, 561.59, 446.4, 447.315, 447.315], [1693.26, 911.745, 384.91, 384.91], [1692.22, 545.515, 515.41, 486.28, 461.675, 349.065, 349.065], [1694.03, 449.77, 398.99, 398.99], [1699.125, 487.73, 409.5, 409.5], [1704.06, 571.885, 399.935, 399.935], [1693.61, 592.08, 408.33, 408.33]]"
        self.expected_objective_curves = "[([0, 20, 30, 100], [1697.795, 660.17, 437.83, 437.83]), ([0, 20, 40, 60, 70, 100], [1697.795, 503.78, 462.48, 457.25, 411.28, 411.28]), ([0, 20, 30, 100], [1697.795, 459.575, 433.76, 433.76]), ([0, 20, 40, 100], [1697.795, 565.765, 375.95, 375.95]), ([0, 20, 30, 50, 100], [1697.795, 559.7, 435.43, 395.31, 395.31]), ([0, 20, 30, 100], [1697.795, 870.48, 453.305, 448.115]), ([0, 20, 30, 100], [1697.795, 518.37, 373.12, 373.12]), ([0, 20, 50, 80, 100], [1697.795, 500.37, 428.155, 433.745, 433.745]), ([0, 20, 50, 90, 100], [1697.795, 442.125, 419.305, 410.145, 410.145]), ([0, 20, 30, 100], [1697.795, 491.105, 414.635, 395.73]), ([0, 20, 30, 100], [1697.795, 500.8, 407.365, 407.365]), ([0, 20, 30, 50, 60, 100], [1697.795, 674.27, 578.09, 531.99, 409.725, 409.725]), ([0, 20, 30, 70, 100], [1697.795, 826.23, 472.75, 432.39, 432.39]), ([0, 20, 90, 100], [1697.795, 471.895, 457.77, 435.835]), ([0, 20, 50, 100], [1697.795, 485.375, 338.295, 338.295]), ([0, 20, 30, 50, 100], [1697.795, 700.01, 454.19, 436.13, 436.13]), ([0, 20, 40, 100], [1697.795, 523.36, 418.49, 412.775]), ([0, 20, 30, 40, 80, 100], [1697.795, 661.445, 561.59, 446.4, 447.315, 447.315]), ([0, 20, 30, 100], [1697.795, 911.745, 384.91, 384.91]), ([0, 20, 40, 60, 80, 90, 100], [1697.795, 545.515, 515.41, 486.28, 461.675, 349.065, 349.065]), ([0, 20, 70, 100], [1697.795, 449.77, 398.99, 398.99]), ([0, 20, 40, 100], [1697.795, 487.73, 409.5, 409.5]), ([0, 20, 40, 100], [1697.795, 571.885, 399.935, 399.935]), ([0, 20, 30, 100], [1697.795, 592.08, 408.33, 408.33])]"
        self.expected_progress_curves = "[([0.0, 0.2, 0.3, 1.0], [1.0, 0.23675983817579987, 0.07321441706509745, 0.07321441706509745]), ([0.0, 0.2, 0.4, 0.6, 0.7, 1.0], [1.0, 0.12172489885987492, 0.09134608311879368, 0.08749908054431775, 0.05368517837440232, 0.05368517837440232]), ([0.0, 0.2, 0.3, 1.0], [1.0, 0.08920926811327692, 0.07022066936373665, 0.07022066936373665]), ([0.0, 0.2, 0.4, 1.0], [1.0, 0.16731886723059947, 0.027697682971680744, 0.027697682971680744]), ([0.0, 0.2, 0.3, 0.5, 1.0], [1.0, 0.16285766826038986, 0.07144906215520411, 0.04193821257815372, 0.04193821257815372]), ([0.0, 0.2, 0.3, 1.0], [1.0, 0.391456417800662, 0.08459727841118057, 0.08077969841853622]), ([0.0, 0.2, 0.3, 1.0], [1.0, 0.1324567855829349, 0.02561603530709819, 0.02561603530709819]), ([0.0, 0.2, 0.5, 0.8, 1.0], [1.0, 0.11921662375873482, 0.06609783008458989, 0.07020963589554982, 0.07020963589554982]), ([0.0, 0.2, 0.5, 0.9, 1.0], [1.0, 0.07637366678926075, 0.059588083854358216, 0.052850312614931935, 0.052850312614931935]), ([0.0, 0.2, 0.3, 1.0], [1.0, 0.11240161824200073, 0.05615299742552407, 0.04224714968738507]), ([0.0, 0.2, 0.3, 1.0], [1.0, 0.11953291651342406, 0.05080544317763883, 0.05080544317763883]), ([0.0, 0.2, 0.3, 0.5, 0.6, 1.0], [1.0, 0.2471312982714233, 0.1763847002574476, 0.1424751746965796, 0.05254137550570063, 0.05254137550570063]), ([0.0, 0.2, 0.3, 0.7, 1.0], [1.0, 0.3589076866495035, 0.09890033100404559, 0.06921294593600587, 0.06921294593600587]), ([0.0, 0.2, 0.9, 1.0], [1.0, 0.09827142331739608, 0.08788157410812797, 0.07174696579624859]), ([0.0, 0.2, 0.5, 1.0], [1.0, 0.10818683339463037, 0.0, 0.0]), ([0.0, 0.2, 0.3, 0.5, 1.0], [1.0, 0.2660647296800294, 0.08524825303420373, 0.07196395733725633, 0.07196395733725633]), ([0.0, 0.2, 0.4, 1.0], [1.0, 0.13612725266642148, 0.058988598749540266, 0.05478484737035672]), ([0.0, 0.2, 0.3, 0.4, 0.8, 1.0], [1.0, 0.23769768297168078, 0.16424788525193088, 0.07951820522250824, 0.0801912467819051, 0.0801912467819051]), ([0.0, 0.2, 0.3, 1.0], [1.0, 0.4218094887826407, 0.03428834130194925, 0.03428834130194925]), ([0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0], [1.0, 0.15242368517837437, 0.13027951452739975, 0.10885251930856929, 0.09075395365943362, 0.007922030158146364, 0.007922030158146364]), ([0.0, 0.2, 0.7, 1.0], [1.0, 0.08199705774181681, 0.044645090106656855, 0.044645090106656855]), ([0.0, 0.2, 0.4, 1.0], [1.0, 0.10991908789996323, 0.052375873482898114, 0.052375873482898114]), ([0.0, 0.2, 0.4, 1.0], [1.0, 0.1718205222508275, 0.045340198602427356, 0.045340198602427356]), ([0.0, 0.2, 0.3, 1.0], [1.0, 0.18667524825303422, 0.0515152629643251, 0.0515152629643251])]"

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
