import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class test_PARAMESTI1_ALOE(unittest.TestCase):
    def setUp(self):
        # Expected values
        self.expected_problem_name = "PARAMESTI-1"
        self.expected_solver_name = "ALOE"
        self.expected_all_recommended_xs = "[[(1, 1), (3.12608491500362, 2.1484016294747956), (1.591487330009341, 2.8223380842330785), (1.591487330009341, 2.8223380842330785)], [(1, 1), (3.3539357528102585, 2.471791819987909), (0.8715817527857763, 3.1491986542305432), (0.8715817527857763, 3.1491986542305432)], [(1, 1), (3.2080063671815324, 2.2200195816463557), (1.141266346996694, 2.9557636463521284), (1.141266346996694, 2.9557636463521284)], [(1, 1), (3.4706087501058813, 2.419418168703624), (1.2649033311399727, 3.0536425841367123), (1.2649033311399727, 3.0536425841367123)], [(1, 1), (3.623532316196335, 2.3966415959399936), (1.7436758741788376, 3.012290239508519), (1.7436758741788376, 3.012290239508519)], [(1, 1), (3.354131820692981, 2.4111437495417363), (1.2233571357185258, 3.0386052304086864), (1.2233571357185258, 3.0386052304086864)], [(1, 1), (3.3010166604566833, 2.122171109146829), (1.9814484083007677, 2.7737834230034832), (2.541483088230164, 3.32029896916271), (2.541483088230164, 3.32029896916271)], [(1, 1), (3.373184966861831, 2.1793338105117357), (1.5584142308274003, 2.8798153627866108), (1.5584142308274003, 2.8798153627866108)], [(1, 1), (3.6268794214258655, 2.186167884801846), (1.7041570551439678, 2.89254542374119), (1.7041570551439678, 2.89254542374119)], [(1, 1), (2.9550377674284314, 2.111403240149154), (1.612095497476963, 2.7539021977896287), (1.612095497476963, 2.7539021977896287)], [(1, 1), (3.7472353760743924, 2.4309547474174087), (1.6282620365395415, 3.074635821513148), (1.6282620365395415, 3.074635821513148)], [(1, 1), (3.2339056360961056, 2.1809314224541376), (1.5907460275075942, 2.882790291456895), (1.5907460275075942, 2.882790291456895)], [(1, 1), (2.938501859598577, 2.198322441044798), (1.191555909232325, 2.915213435310172), (1.191555909232325, 2.915213435310172)], [(1, 1), (2.9493974575244777, 2.145555676563607), (1.4780275987620832, 2.8170616317770127), (1.4780275987620832, 2.8170616317770127)], [(1, 1), (3.3274632289995827, 2.110361809177099), (2.0221766666612107, 2.7519809521322274), (2.538417497062129, 3.2879496330311917), (2.538417497062129, 3.2879496330311917)], [(1, 1), (2.986332457618304, 2.2151475577954827), (1.0329604558501118, 2.94664880004806), (1.0329604558501118, 2.94664880004806)], [(1, 1), (3.4828202596067497, 2.1848737012382804), (1.686278757076782, 2.8901338512342702), (1.686278757076782, 2.8901338512342702)], [(1, 1), (3.05434626391029, 1.9811364195270027), (1.6608197831848943, 2.673550793118574), (1.6608197831848943, 2.673550793118574)], [(1, 1), (3.181891763827056, 2.1963694037928), (1.2828130919677765, 2.911568717800467), (1.2828130919677765, 2.911568717800467)], [(1, 1), (3.161927082543114, 2.168472535739398), (1.4745674623591112, 2.859606724708518), (1.4745674623591112, 2.859606724708518)], [(1, 1), (3.0275139861221763, 2.134457365468209), (1.7079541815595736, 2.7965045298644826), (2.547088558235464, 3.3540633390259207), (2.547088558235464, 3.3540633390259207)], [(1, 1), (3.4967237047552944, 2.1889315171472434), (1.8631931052352546, 2.8976964857543934), (2.7800490702100733, 3.5050542826471727), (2.7800490702100733, 3.5050542826471727)], [(1, 1), (3.5540233338914864, 2.462620596392078), (1.2413823385598342, 3.132419717003843), (1.2413823385598342, 3.132419717003843)], [(1, 1), (3.191873225848376, 2.4497303006559177), (0.9727763990724889, 3.108869288815846), (0.9727763990724889, 3.108869288815846)]]"
        self.expected_all_intermediate_budgets = "[[0, 630, 780, 1000], [0, 480, 630, 1000], [0, 630, 780, 1000], [0, 480, 630, 1000], [0, 480, 630, 1000], [0, 480, 630, 1000], [0, 630, 780, 930, 1000], [0, 630, 780, 1000], [0, 630, 780, 1000], [0, 630, 780, 1000], [0, 480, 630, 1000], [0, 630, 780, 1000], [0, 630, 780, 1000], [0, 630, 780, 1000], [0, 630, 780, 930, 1000], [0, 630, 780, 1000], [0, 630, 780, 1000], [0, 780, 930, 1000], [0, 630, 780, 1000], [0, 630, 780, 1000], [0, 630, 780, 930, 1000], [0, 630, 780, 930, 1000], [0, 480, 630, 1000], [0, 480, 630, 1000]]"
        self.expected_all_est_objectives = "[[-9.16944227603262, -7.333017640572527, -5.545476099612057, -5.545476099612057], [-8.940090362495347, -7.77968455057781, -6.9913540616408, -6.9913540616408], [-9.121210005202611, -7.5152513475418345, -6.29962322035692, -6.29962322035692], [-8.779886386724968, -7.849461896281971, -5.709528947481635, -5.709528947481635], [-8.99288952739613, -7.724294709422449, -5.127242865574378, -5.127242865574378], [-8.87740808504234, -7.302807191448271, -5.911173974260237, -5.911173974260237], [-9.024638576352391, -7.686669866249672, -5.2793181442166235, -5.342933319679849, -5.342933319679849], [-8.921050660074993, -7.963973792729178, -5.409164060493447, -5.409164060493447], [-8.550164686658025, -7.7104330294818535, -5.019920730425201, -5.019920730425201], [-8.983830735669818, -6.817103012895079, -5.444471426961226, -5.444471426961226], [-9.025843710458552, -8.776767850439107, -5.244107524885294, -5.244107524885294], [-9.203733926294058, -7.287600420586198, -5.457891040613999, -5.457891040613999], [-9.33623207280299, -6.603631807834111, -6.378982255461353, -6.378982255461353], [-9.573886675373538, -7.519233695495506, -6.013550417911575, -6.013550417911575], [-8.941889405024408, -8.044581133247968, -5.338101653406268, -5.477202562252321, -5.477202562252321], [-9.587195496910567, -6.8990917803616005, -7.061587702570883, -7.061587702570883], [-9.346621843523279, -8.329397679276928, -5.531634372674202, -5.531634372674202], [-9.398264139884379, -7.240948695555619, -5.683438121621459, -5.683438121621459], [-8.499265659696825, -7.015788765350918, -5.577577391658465, -5.577577391658465], [-9.254478807791063, -7.317677459431239, -5.743521222892926, -5.743521222892926], [-9.605611629911163, -7.290429796682888, -5.658490888763109, -5.514671933091987, -5.514671933091987], [-8.501695309087717, -7.991675083785679, -4.985283242739755, -5.510939340118978, -5.510939340118978], [-9.152042163756049, -8.27120584469354, -5.944127145729939, -5.944127145729939], [-9.213076761398039, -7.021553338797141, -6.9239952167982, -6.9239952167982]]"
        self.expected_objective_curves = "[([0, 630, 780, 1000], [-9.265122221743944, -7.333017640572527, -5.545476099612057, -5.545476099612057]), ([0, 480, 630, 1000], [-9.265122221743944, -7.77968455057781, -6.9913540616408, -6.9913540616408]), ([0, 630, 780, 1000], [-9.265122221743944, -7.5152513475418345, -6.29962322035692, -6.29962322035692]), ([0, 480, 630, 1000], [-9.265122221743944, -7.849461896281971, -5.709528947481635, -5.709528947481635]), ([0, 480, 630, 1000], [-9.265122221743944, -7.724294709422449, -5.127242865574378, -5.127242865574378]), ([0, 480, 630, 1000], [-9.265122221743944, -7.302807191448271, -5.911173974260237, -5.911173974260237]), ([0, 630, 780, 930, 1000], [-9.265122221743944, -7.686669866249672, -5.2793181442166235, -5.342933319679849, -5.342933319679849]), ([0, 630, 780, 1000], [-9.265122221743944, -7.963973792729178, -5.409164060493447, -5.409164060493447]), ([0, 630, 780, 1000], [-9.265122221743944, -7.7104330294818535, -5.019920730425201, -5.019920730425201]), ([0, 630, 780, 1000], [-9.265122221743944, -6.817103012895079, -5.444471426961226, -5.444471426961226]), ([0, 480, 630, 1000], [-9.265122221743944, -8.776767850439107, -5.244107524885294, -5.244107524885294]), ([0, 630, 780, 1000], [-9.265122221743944, -7.287600420586198, -5.457891040613999, -5.457891040613999]), ([0, 630, 780, 1000], [-9.265122221743944, -6.603631807834111, -6.378982255461353, -6.378982255461353]), ([0, 630, 780, 1000], [-9.265122221743944, -7.519233695495506, -6.013550417911575, -6.013550417911575]), ([0, 630, 780, 930, 1000], [-9.265122221743944, -8.044581133247968, -5.338101653406268, -5.477202562252321, -5.477202562252321]), ([0, 630, 780, 1000], [-9.265122221743944, -6.8990917803616005, -7.061587702570883, -7.061587702570883]), ([0, 630, 780, 1000], [-9.265122221743944, -8.329397679276928, -5.531634372674202, -5.531634372674202]), ([0, 780, 930, 1000], [-9.265122221743944, -7.240948695555619, -5.683438121621459, -5.683438121621459]), ([0, 630, 780, 1000], [-9.265122221743944, -7.015788765350918, -5.577577391658465, -5.577577391658465]), ([0, 630, 780, 1000], [-9.265122221743944, -7.317677459431239, -5.743521222892926, -5.743521222892926]), ([0, 630, 780, 930, 1000], [-9.265122221743944, -7.290429796682888, -5.658490888763109, -5.514671933091987, -5.514671933091987]), ([0, 630, 780, 930, 1000], [-9.265122221743944, -7.991675083785679, -4.985283242739755, -5.510939340118978, -5.510939340118978]), ([0, 480, 630, 1000], [-9.265122221743944, -8.27120584469354, -5.944127145729939, -5.944127145729939]), ([0, 480, 630, 1000], [-9.265122221743944, -7.021553338797141, -6.9239952167982, -6.9239952167982])]"
        self.expected_progress_curves = "[([0.0, 0.63, 0.78, 1.0], [1.0, 0.5852054810782226, 0.20144652684715847, 0.20144652684715847]), ([0.0, 0.48, 0.63, 1.0], [1.0, 0.6810983162070474, 0.5118553212384509, 0.5118553212384509]), ([0.0, 0.63, 0.78, 1.0], [1.0, 0.6243283854749591, 0.3633508539700472, 0.3633508539700472]), ([0.0, 0.48, 0.63, 1.0], [1.0, 0.6960784890326034, 0.23666626742070038, 0.23666626742070038]), ([0.0, 0.48, 0.63, 1.0], [1.0, 0.6692069296128179, 0.11165798496368633, 0.11165798496368633]), ([0.0, 0.48, 0.63, 1.0], [1.0, 0.5787197406928294, 0.279956497510051, 0.279956497510051]), ([0.0, 0.63, 0.78, 0.93, 1.0], [1.0, 0.6611294275586085, 0.14430631707729302, 0.15796356232143755, 0.15796356232143755]), ([0.0, 0.63, 0.78, 1.0], [1.0, 0.7206625138625851, 0.17218233108857808, 0.17218233108857808]), ([0.0, 0.63, 0.78, 1.0], [1.0, 0.6662310302135622, 0.08861749644526934, 0.08861749644526934]), ([0.0, 0.63, 0.78, 1.0], [1.0, 0.47444617649522236, 0.17976230488043055, 0.17976230488043055]), ([0.0, 0.48, 0.63, 1.0], [1.0, 0.8951574782841607, 0.13674711347682245, 0.13674711347682245]), ([0.0, 0.63, 0.78, 1.0], [1.0, 0.5754550699987291, 0.18264329913067096, 0.18264329913067096]), ([0.0, 0.63, 0.78, 1.0], [1.0, 0.4286170393616457, 0.3803880749926702, 0.3803880749926702]), ([0.0, 0.63, 0.78, 1.0], [1.0, 0.6251833371787747, 0.30193521859332334, 0.30193521859332334]), ([0.0, 0.63, 0.78, 0.93, 1.0], [1.0, 0.7379677277510507, 0.15692627442974086, 0.1867891996195351, 0.1867891996195351]), ([0.0, 0.63, 0.78, 1.0], [1.0, 0.49204796249049504, 0.5269334539573618, 0.5269334539573618]), ([0.0, 0.63, 0.78, 1.0], [1.0, 0.7991136632983995, 0.19847491106497697, 0.19847491106497697]), ([0.0, 0.78, 0.93, 1.0], [1.0, 0.56543962879047, 0.23106494973510078, 0.23106494973510078]), ([0.0, 0.63, 0.78, 1.0], [1.0, 0.5171010937857576, 0.20833820347839135, 0.20833820347839135]), ([0.0, 0.63, 0.78, 1.0], [1.0, 0.5819121691536601, 0.2439639104487679, 0.2439639104487679]), ([0.0, 0.63, 0.78, 0.93, 1.0], [1.0, 0.5760624955533871, 0.22570914469599926, 0.19483332404385467, 0.19483332404385467]), ([0.0, 0.63, 0.78, 0.93, 1.0], [1.0, 0.7266095748080785, 0.08118133580409294, 0.19403199107168642, 0.19403199107168642]), ([0.0, 0.48, 0.63, 1.0], [1.0, 0.786620729806902, 0.2870310601605027, 0.2870310601605027]), ([0.0, 0.48, 0.63, 1.0], [1.0, 0.5183386631661716, 0.49739436508012896, 0.49739436508012896])]"

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
