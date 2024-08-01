import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class test_IRONORECONT1_ADAM(unittest.TestCase):
    def setUp(self):
        # Expected values
        self.expected_problem_name = "IRONORECONT-1"
        self.expected_solver_name = "ADAM"
        self.expected_all_recommended_xs = "[[(80, 40, 100), (81.45294547748918, 41.7475570769722, 101.49827049751283), (81.45294547748918, 41.7475570769722, 101.49827049751283)], [(80, 40, 100), (80.4999999998809, 40.49999999998566, 100.49999999999933), (81.094007612972, 41.09400761327011, 101.43039479653247), (81.30614703045066, 41.30614703083772, 101.59609527945749), (81.30614703045066, 41.30614703083772, 101.59609527945749)], [(80, 40, 100), (80.49999999989515, 40.0, 99.50000000000394), (81.84903432706177, 40.0, 100.03137279727117), (81.84903432706177, 40.0, 100.03137279727117)], [(80, 40, 100), (80.49999999998296, 40.49999999999757, 99.50000000000118), (80.9997688888276, 40.8587948184709, 99.01686120373093), (80.9997688888276, 40.8587948184709, 99.01686120373093)], [(80, 40, 100), (80.0, 40.49999999995532, 100.49999999999932), (80.0, 40.97604082271071, 100.97466517188901), (80.0, 41.331409421545025, 101.44673659626437), (80.0, 41.73814148302175, 101.92500444002127), (80.0, 41.73814148302175, 101.92500444002127)], [(80, 40, 100), (81.09400761134064, 40.0, 100.06308899785508), (81.09400761134064, 40.0, 100.06308899785508)], [(80, 40, 100), (80.49999999996709, 40.49999999999158, 99.50000000000408), (80.49999999996709, 40.49999999999158, 99.50000000000408)], [(80, 40, 100), (80.0, 40.0, 99.80804962510106), (80.0, 40.0, 99.80804962510106)], [(80, 40, 100), (79.50000000327958, 40.499999999934495, 100.49999999999861), (79.50000000327958, 40.499999999934495, 100.49999999999861)], [(80, 40, 100), (81.33459744537683, 41.38365186859791, 100.25582947635012), (81.33459744537683, 41.38365186859791, 100.25582947635012)], [(80, 40, 100), (81.5053729189032, 40.0, 100.11712147027491), (81.5053729189032, 40.0, 100.11712147027491)], [(80, 40, 100), (80.499999999017, 40.49999999999651, 100.49999999999956), (80.99934436108269, 40.999999999993015, 100.99469914430222), (80.99934436108269, 40.999999999993015, 100.99469914430222)], [(80, 40, 100), (80.49999999992438, 40.0, 99.50000000000247), (81.30614703061131, 40.0, 99.22515254724311), (81.30614703061131, 40.0, 99.22515254724311)], [(80, 40, 100), (80.0, 40.49999999998677, 100.49999999999913), (80.0, 41.15365029055579, 101.23522997590756), (80.0, 41.15365029055579, 101.23522997590756)], [(80, 40, 100), (80.0, 40.499999999988944, 100.49999999999955), (80.0, 40.99999999997789, 100.98399911710735), (80.0, 40.99999999997789, 100.98399911710735)], [(80, 40, 100), (80.49999999982079, 40.0, 100.49999999999929), (80.99999999964156, 40.0, 100.99797585011436), (81.38650144862774, 40.0, 101.49733284963541), (81.38650144862774, 40.0, 101.49733284963541)], [(80, 40, 100), (79.50000000436582, 40.49999999999807, 100.49999999999837), (79.50000000436582, 40.49999999999807, 100.49999999999837)], [(80, 40, 100), (80.0, 40.85019083423836, 100.87601035639386), (80.0, 40.85019083423836, 100.87601035639386)], [(80, 40, 100), (80.0, 40.0, 100.49999999999952), (80.0, 40.0, 100.94590813980153), (80.0, 40.0, 100.94590813980153)], [(80, 40, 100), (80.0, 41.82517393591782, 99.7542392866934), (80.0, 41.82517393591782, 99.7542392866934)], [(80, 40, 100), (80.49999999888888, 40.0, 99.50000000000058), (80.49999999888888, 40.0, 99.50000000000058)], [(80, 40, 100), (80.0, 40.0, 100.4999999999994), (80.0, 40.0, 100.67488099869274), (80.0, 40.0, 100.67488099869274)], [(80, 40, 100), (80.0, 40.499999999993534, 100.49999999999945), (80.0, 41.88132235699524, 101.46066792856044), (80.0, 41.88132235699524, 101.46066792856044)], [(80, 40, 100), (80.4999999998785, 40.0, 100.49999999997281), (80.8350291268316, 40.0, 100.88491716761614), (81.09400761296519, 40.0, 101.31531858225544), (81.30614703044179, 40.0, 101.76689256724201), (81.30614703044179, 40.0, 101.76689256724201)]]"
        self.expected_all_intermediate_budgets = "[[0, 870, 1000], [0, 240, 660, 870, 1000], [0, 240, 870, 1000], [0, 240, 450, 1000], [0, 240, 450, 660, 870, 1000], [0, 660, 1000], [0, 240, 1000], [0, 870, 1000], [0, 240, 1000], [0, 660, 1000], [0, 870, 1000], [0, 240, 450, 1000], [0, 240, 870, 1000], [0, 240, 660, 1000], [0, 240, 450, 1000], [0, 240, 450, 660, 1000], [0, 240, 1000], [0, 450, 1000], [0, 240, 450, 1000], [0, 870, 1000], [0, 240, 1000], [0, 240, 450, 1000], [0, 240, 870, 1000], [0, 240, 450, 660, 870, 1000]]"
        self.expected_all_est_objectives = "[[149895.8055235529, 152828.75196002246, 152828.75196002246], [122793.09736468189, 124694.54881332583, 120907.1051610305, 119718.44332001486, 119718.44332001486], [99852.80349485856, 102719.53385606415, 99910.96572005817, 99910.96572005817], [126011.12695446546, 125333.55811445268, 123471.72144938468, 123471.72144938468], [136147.71179130895, 137300.23966324652, 135469.4527007973, 135948.35320359637, 134215.5859256513, 134215.5859256513], [132850.26196652921, 131661.4251168007, 131661.4251168007], [134982.68434045353, 133257.021184012, 133257.021184012], [161256.2908821113, 159855.6492244887, 159855.6492244887], [146337.47315675917, 144899.0747268912, 144899.0747268912], [134867.2205665852, 135863.95245457996, 135863.95245457996], [149243.01256369415, 149398.92178068135, 149398.92178068135], [112822.77485929335, 113405.65080250782, 107175.15726811437, 107175.15726811437], [132809.38556277155, 132912.90864149397, 133260.0689322136, 133260.0689322136], [118379.15455996453, 119480.42299985937, 117674.50118531213, 117674.50118531213], [127606.7164810152, 127411.05368491581, 127110.57580328781, 127110.57580328781], [145498.2552215891, 143807.63526754544, 144825.46263338544, 144035.30761219538, 144035.30761219538], [161264.15011124164, 163085.86886891746, 163085.86886891746], [132500.94479520118, 127665.63459549358, 127665.63459549358], [112031.98326897933, 113987.51061603778, 110619.02701203969, 110619.02701203969], [130863.18264271188, 129075.85035586855, 129075.85035586855], [147610.26102665017, 145842.96248323342, 145842.96248323342], [132677.02997009846, 133302.9035278589, 133500.36843053735, 133500.36843053735], [132803.08586581453, 130899.30453112493, 126347.61655854268, 126347.61655854268], [137521.1409071744, 136868.13025232955, 136848.59275571187, 135415.11620997443, 134525.7589189693, 134525.7589189693]]"
        self.expected_objective_curves = "[([0, 870, 1000], [121270.73497283501, 152828.75196002246, 152828.75196002246]), ([0, 240, 660, 870, 1000], [121270.73497283501, 124694.54881332583, 120907.1051610305, 119718.44332001486, 119718.44332001486]), ([0, 240, 870, 1000], [121270.73497283501, 102719.53385606415, 99910.96572005817, 99910.96572005817]), ([0, 240, 450, 1000], [121270.73497283501, 125333.55811445268, 123471.72144938468, 123471.72144938468]), ([0, 240, 450, 660, 870, 1000], [121270.73497283501, 137300.23966324652, 135469.4527007973, 135948.35320359637, 134215.5859256513, 134215.5859256513]), ([0, 660, 1000], [121270.73497283501, 131661.4251168007, 131661.4251168007]), ([0, 240, 1000], [121270.73497283501, 133257.021184012, 133257.021184012]), ([0, 870, 1000], [121270.73497283501, 159855.6492244887, 159855.6492244887]), ([0, 240, 1000], [121270.73497283501, 144899.0747268912, 144899.0747268912]), ([0, 660, 1000], [121270.73497283501, 135863.95245457996, 135863.95245457996]), ([0, 870, 1000], [121270.73497283501, 149398.92178068135, 149398.92178068135]), ([0, 240, 450, 1000], [121270.73497283501, 113405.65080250782, 107175.15726811437, 107175.15726811437]), ([0, 240, 870, 1000], [121270.73497283501, 132912.90864149397, 133260.0689322136, 133260.0689322136]), ([0, 240, 660, 1000], [121270.73497283501, 119480.42299985937, 117674.50118531213, 117674.50118531213]), ([0, 240, 450, 1000], [121270.73497283501, 127411.05368491581, 127110.57580328781, 127110.57580328781]), ([0, 240, 450, 660, 1000], [121270.73497283501, 143807.63526754544, 144825.46263338544, 144035.30761219538, 144035.30761219538]), ([0, 240, 1000], [121270.73497283501, 122403.15716005917, 122403.15716005917]), ([0, 450, 1000], [121270.73497283501, 127665.63459549358, 127665.63459549358]), ([0, 240, 450, 1000], [121270.73497283501, 113987.51061603778, 110619.02701203969, 110619.02701203969]), ([0, 870, 1000], [121270.73497283501, 129075.85035586855, 129075.85035586855]), ([0, 240, 1000], [121270.73497283501, 145842.96248323342, 145842.96248323342]), ([0, 240, 450, 1000], [121270.73497283501, 133302.9035278589, 133500.36843053735, 133500.36843053735]), ([0, 240, 870, 1000], [121270.73497283501, 130899.30453112493, 126347.61655854268, 126347.61655854268]), ([0, 240, 450, 660, 870, 1000], [121270.73497283501, 136868.13025232955, 136848.59275571187, 135415.11620997443, 134525.7589189693, 134525.7589189693])]"
        self.expected_progress_curves = "[([0.0, 0.87, 1.0], [1.0, -26.867713422803668, -26.867713422803668]), ([0.0, 0.24, 0.66, 0.87, 1.0], [1.0, -2.023442916535761, 1.321107989499793, 2.3707711402451284, 2.3707711402451284]), ([0.0, 0.24, 0.87, 1.0], [1.0, 17.381877117971612, 19.86201938972492, 19.86201938972492]), ([0.0, 0.24, 0.45, 1.0], [1.0, -2.5877283114494833, -0.9436094606595629, -0.9436094606595629]), ([0.0, 0.24, 0.45, 0.66, 0.87, 1.0], [1.0, -13.155060604829439, -11.538360593911312, -11.96125985198129, -10.43111738612896, -10.43111738612896]), ([0.0, 0.66, 1.0], [1.0, -8.175632781830043, -8.175632781830043]), ([0.0, 0.24, 1.0], [1.0, -9.58464444303962, -9.58464444303962]), ([0.0, 0.87, 1.0], [1.0, -33.07290557087598, -33.07290557087598]), ([0.0, 0.24, 1.0], [1.0, -19.865309794022078, -19.865309794022078]), ([0.0, 0.66, 1.0], [1.0, -11.886728683333585, -11.886728683333585]), ([0.0, 0.87, 1.0], [1.0, -23.838957700745233, -23.838957700745233]), ([0.0, 0.24, 0.45, 1.0], [1.0, 7.945363892601218, 13.44728146776449, 13.44728146776449]), ([0.0, 0.24, 0.87, 1.0], [1.0, -9.280771429599696, -9.58733579634934, -9.58733579634934]), ([0.0, 0.24, 0.66, 1.0], [1.0, 2.580958050075059, 4.175700571831886, 4.175700571831886]), ([0.0, 0.24, 0.45, 1.0], [1.0, -4.422287536711193, -4.156946672660706, -4.156946672660706]), ([0.0, 0.24, 0.45, 0.66, 1.0], [1.0, -18.901500119804066, -19.80030568659973, -19.10254911656385, -19.10254911656385]), ([0.0, 0.24, 1.0], [1.0, -0.0, -0.0]), ([0.0, 0.45, 1.0], [1.0, -4.647098489242786, -4.647098489242786]), ([0.0, 0.24, 0.45, 1.0], [1.0, 7.43154508889495, 10.406127927345906, 10.406127927345906]), ([0.0, 0.87, 1.0], [1.0, -5.89240768247905, -5.89240768247905]), ([0.0, 0.24, 1.0], [1.0, -20.69882203617967, -20.69882203617967]), ([0.0, 0.24, 0.45, 1.0], [1.0, -9.62516143781819, -9.79953536382054, -9.79953536382054]), ([0.0, 0.24, 0.87, 1.0], [1.0, -7.50263238120745, -3.4832056833435283, -3.4832056833435283]), ([0.0, 0.24, 0.45, 0.66, 0.87, 1.0], [1.0, -12.773480823196776, -12.756227985131526, -11.490378055741463, -10.705019643447242, -10.705019643447242])]"

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
