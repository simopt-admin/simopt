import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class test_EXAMPLE1_ALOE(unittest.TestCase):
    def setUp(self):
        # Expected values
        self.expected_problem_name = "EXAMPLE-1"
        self.expected_solver_name = "ALOE"
        self.expected_all_recommended_xs = "[[(2.0, 2.0), (-1.2000000000000002, -1.2000000000000002), (0.7199999999999998, 0.7199999999999998), (-0.7199999999999986, -0.7199999999999986), (-0.7199999999999986, -0.7199999999999986)], [(2.0, 2.0), (-1.2000000000000002, -1.2000000000000002), (0.7199999999999998, 0.7199999999999998), (-0.7199999999999986, -0.7199999999999986), (-0.7199999999999986, -0.7199999999999986)], [(2.0, 2.0), (-1.2000000000000002, -1.2000000000000002), (0.7199999999999998, 0.7199999999999998), (-0.7199999999999986, -0.7199999999999986), (-0.7199999999999986, -0.7199999999999986)], [(2.0, 2.0), (-1.2000000000000002, -1.2000000000000002), (0.7199999999999998, 0.7199999999999998), (-0.7199999999999986, -0.7199999999999986), (-0.7199999999999986, -0.7199999999999986)], [(2.0, 2.0), (-1.2000000000000002, -1.2000000000000002), (0.7199999999999998, 0.7199999999999998), (-0.7199999999999986, -0.7199999999999986), (-0.7199999999999986, -0.7199999999999986)], [(2.0, 2.0), (-1.2000000000000002, -1.2000000000000002), (0.7199999999999998, 0.7199999999999998), (-0.7199999999999986, -0.7199999999999986), (-0.7199999999999986, -0.7199999999999986)], [(2.0, 2.0), (-1.2000000000000002, -1.2000000000000002), (0.7199999999999998, 0.7199999999999998), (-0.7199999999999986, -0.7199999999999986), (-0.7199999999999986, -0.7199999999999986)], [(2.0, 2.0), (-1.2000000000000002, -1.2000000000000002), (0.7199999999999998, 0.7199999999999998), (-0.7199999999999986, -0.7199999999999986), (-0.7199999999999986, -0.7199999999999986)], [(2.0, 2.0), (-1.2000000000000002, -1.2000000000000002), (0.7199999999999998, 0.7199999999999998), (-0.7199999999999986, -0.7199999999999986), (-0.7199999999999986, -0.7199999999999986)], [(2.0, 2.0), (-1.2000000000000002, -1.2000000000000002), (0.7199999999999998, 0.7199999999999998), (-0.7199999999999986, -0.7199999999999986), (-0.7199999999999986, -0.7199999999999986)], [(2.0, 2.0), (-1.2000000000000002, -1.2000000000000002), (0.7199999999999998, 0.7199999999999998), (-0.7199999999999986, -0.7199999999999986), (-0.7199999999999986, -0.7199999999999986)], [(2.0, 2.0), (-1.2000000000000002, -1.2000000000000002), (0.7199999999999998, 0.7199999999999998), (-0.7199999999999986, -0.7199999999999986), (-0.7199999999999986, -0.7199999999999986)], [(2.0, 2.0), (-1.2000000000000002, -1.2000000000000002), (0.7199999999999998, 0.7199999999999998), (-0.7199999999999986, -0.7199999999999986), (-0.7199999999999986, -0.7199999999999986)], [(2.0, 2.0), (-1.2000000000000002, -1.2000000000000002), (0.7199999999999998, 0.7199999999999998), (-0.7199999999999986, -0.7199999999999986), (-0.7199999999999986, -0.7199999999999986)], [(2.0, 2.0), (-1.2000000000000002, -1.2000000000000002), (0.7199999999999998, 0.7199999999999998), (-0.7199999999999986, -0.7199999999999986), (-0.7199999999999986, -0.7199999999999986)], [(2.0, 2.0), (-1.2000000000000002, -1.2000000000000002), (0.7199999999999998, 0.7199999999999998), (-0.7199999999999986, -0.7199999999999986), (-0.7199999999999986, -0.7199999999999986)], [(2.0, 2.0), (-1.2000000000000002, -1.2000000000000002), (0.7199999999999998, 0.7199999999999998), (-0.7199999999999986, -0.7199999999999986), (-0.7199999999999986, -0.7199999999999986)], [(2.0, 2.0), (-1.2000000000000002, -1.2000000000000002), (0.7199999999999998, 0.7199999999999998), (-0.7199999999999986, -0.7199999999999986), (-0.7199999999999986, -0.7199999999999986)], [(2.0, 2.0), (-1.2000000000000002, -1.2000000000000002), (0.7199999999999998, 0.7199999999999998), (-0.7199999999999986, -0.7199999999999986), (-0.7199999999999986, -0.7199999999999986)], [(2.0, 2.0), (-1.2000000000000002, -1.2000000000000002), (0.7199999999999998, 0.7199999999999998), (-0.7199999999999986, -0.7199999999999986), (-0.7199999999999986, -0.7199999999999986)], [(2.0, 2.0), (-1.2000000000000002, -1.2000000000000002), (0.7199999999999998, 0.7199999999999998), (-0.7199999999999986, -0.7199999999999986), (-0.7199999999999986, -0.7199999999999986)], [(2.0, 2.0), (-1.2000000000000002, -1.2000000000000002), (0.7199999999999998, 0.7199999999999998), (-0.7199999999999986, -0.7199999999999986), (-0.7199999999999986, -0.7199999999999986)], [(2.0, 2.0), (-1.2000000000000002, -1.2000000000000002), (0.7199999999999998, 0.7199999999999998), (-0.7199999999999986, -0.7199999999999986), (-0.7199999999999986, -0.7199999999999986)], [(2.0, 2.0), (-1.2000000000000002, -1.2000000000000002), (0.7199999999999998, 0.7199999999999998), (-0.7199999999999986, -0.7199999999999986), (-0.7199999999999986, -0.7199999999999986)]]"
        self.expected_all_intermediate_budgets = "[[0, 90, 150, 180, 1000], [0, 90, 150, 180, 1000], [0, 90, 150, 180, 1000], [0, 90, 150, 180, 1000], [0, 90, 150, 180, 1000], [0, 90, 150, 180, 1000], [0, 90, 150, 180, 1000], [0, 90, 150, 180, 1000], [0, 90, 150, 180, 1000], [0, 90, 150, 180, 1000], [0, 90, 150, 180, 1000], [0, 90, 150, 180, 1000], [0, 90, 150, 180, 1000], [0, 90, 150, 180, 1000], [0, 90, 150, 180, 1000], [0, 90, 150, 180, 1000], [0, 90, 150, 180, 1000], [0, 90, 150, 180, 1000], [0, 90, 150, 180, 1000], [0, 90, 150, 180, 1000], [0, 90, 150, 180, 1000], [0, 90, 150, 180, 1000], [0, 90, 150, 180, 1000], [0, 90, 150, 180, 1000]]"
        self.expected_all_est_objectives = "[[7.984539704940337, 2.864539704940336, 1.021339704940335, 1.0213397049403314, 1.0213397049403314], [8.081590387702734, 2.961590387702732, 1.1183903877027306, 1.1183903877027273, 1.1183903877027273], [7.9253347189439385, 2.8053347189439375, 0.9621347189439362, 0.9621347189439328, 0.9621347189439328], [8.073099810658121, 2.9530998106581205, 1.1098998106581195, 1.1098998106581164, 1.1098998106581164], [7.880122723414122, 2.7601227234141197, 0.9169227234141186, 0.9169227234141155, 0.9169227234141155], [8.025785950362149, 2.905785950362148, 1.062585950362146, 1.0625859503621427, 1.0625859503621427], [8.015084462897443, 2.8950844628974424, 1.0518844628974415, 1.0518844628974382, 1.0518844628974382], [7.994852045957048, 2.874852045957047, 1.0316520459570455, 1.0316520459570422, 1.0316520459570422], [7.910902809206077, 2.7909028092060764, 0.9477028092060751, 0.9477028092060716, 0.9477028092060716], [7.943417039435916, 2.823417039435916, 0.9802170394359142, 0.9802170394359109, 0.9802170394359109], [8.091417684005862, 2.9714176840058597, 1.1282176840058586, 1.1282176840058553, 1.1282176840058553], [7.909472416682599, 2.7894724166825973, 0.946272416682596, 0.9462724166825925, 0.9462724166825925], [7.977013328578315, 2.857013328578314, 1.0138133285783133, 1.0138133285783097, 1.0138133285783097], [8.026689092895502, 2.9066890928955007, 1.0634890928954994, 1.0634890928954963, 1.0634890928954963], [7.9661065902554125, 2.8461065902554115, 1.0029065902554104, 1.0029065902554068, 1.0029065902554068], [7.989698471543214, 2.8696984715432143, 1.0264984715432126, 1.0264984715432095, 1.0264984715432095], [7.920874622531411, 2.80087462253141, 0.9576746225314083, 0.9576746225314051, 0.9576746225314051], [7.936832268065668, 2.8168322680656686, 0.973632268065667, 0.9736322680656637, 0.9736322680656637], [8.053894643913893, 2.9338946439138915, 1.09069464391389, 1.0906946439138867, 1.0906946439138867], [8.098435148416002, 2.9784351484160005, 1.135235148415999, 1.1352351484159957, 1.1352351484159957], [7.934581643563343, 2.8145816435633413, 0.9713816435633399, 0.9713816435633364, 0.9713816435633364], [8.017557178997036, 2.8975571789970345, 1.0543571789970327, 1.0543571789970294, 1.0543571789970294], [7.988890794564379, 2.8688907945643773, 1.025690794564376, 1.0256907945643727, 1.0256907945643727], [7.9800797052490156, 2.8600797052490146, 1.0168797052490135, 1.01687970524901, 1.01687970524901]]"
        self.expected_objective_curves = "[([0, 90, 150, 180, 1000], [8.090508544469758, 2.864539704940336, 1.021339704940335, 1.0213397049403314, 1.0213397049403314]), ([0, 90, 150, 180, 1000], [8.090508544469758, 2.961590387702732, 1.1183903877027306, 1.1183903877027273, 1.1183903877027273]), ([0, 90, 150, 180, 1000], [8.090508544469758, 2.8053347189439375, 0.9621347189439362, 0.9621347189439328, 0.9621347189439328]), ([0, 90, 150, 180, 1000], [8.090508544469758, 2.9530998106581205, 1.1098998106581195, 1.1098998106581164, 1.1098998106581164]), ([0, 90, 150, 180, 1000], [8.090508544469758, 2.7601227234141197, 0.9169227234141186, 0.9169227234141155, 0.9169227234141155]), ([0, 90, 150, 180, 1000], [8.090508544469758, 2.905785950362148, 1.062585950362146, 1.0625859503621427, 1.0625859503621427]), ([0, 90, 150, 180, 1000], [8.090508544469758, 2.8950844628974424, 1.0518844628974415, 1.0518844628974382, 1.0518844628974382]), ([0, 90, 150, 180, 1000], [8.090508544469758, 2.874852045957047, 1.0316520459570455, 1.0316520459570422, 1.0316520459570422]), ([0, 90, 150, 180, 1000], [8.090508544469758, 2.7909028092060764, 0.9477028092060751, 0.9477028092060716, 0.9477028092060716]), ([0, 90, 150, 180, 1000], [8.090508544469758, 2.823417039435916, 0.9802170394359142, 0.9802170394359109, 0.9802170394359109]), ([0, 90, 150, 180, 1000], [8.090508544469758, 2.9714176840058597, 1.1282176840058586, 1.1282176840058553, 1.1282176840058553]), ([0, 90, 150, 180, 1000], [8.090508544469758, 2.7894724166825973, 0.946272416682596, 0.9462724166825925, 0.9462724166825925]), ([0, 90, 150, 180, 1000], [8.090508544469758, 2.857013328578314, 1.0138133285783133, 1.0138133285783097, 1.0138133285783097]), ([0, 90, 150, 180, 1000], [8.090508544469758, 2.9066890928955007, 1.0634890928954994, 1.0634890928954963, 1.0634890928954963]), ([0, 90, 150, 180, 1000], [8.090508544469758, 2.8461065902554115, 1.0029065902554104, 1.0029065902554068, 1.0029065902554068]), ([0, 90, 150, 180, 1000], [8.090508544469758, 2.8696984715432143, 1.0264984715432126, 1.0264984715432095, 1.0264984715432095]), ([0, 90, 150, 180, 1000], [8.090508544469758, 2.80087462253141, 0.9576746225314083, 0.9576746225314051, 0.9576746225314051]), ([0, 90, 150, 180, 1000], [8.090508544469758, 2.8168322680656686, 0.973632268065667, 0.9736322680656637, 0.9736322680656637]), ([0, 90, 150, 180, 1000], [8.090508544469758, 2.9338946439138915, 1.09069464391389, 1.0906946439138867, 1.0906946439138867]), ([0, 90, 150, 180, 1000], [8.090508544469758, 2.9784351484160005, 1.135235148415999, 1.1352351484159957, 1.1352351484159957]), ([0, 90, 150, 180, 1000], [8.090508544469758, 2.8145816435633413, 0.9713816435633399, 0.9713816435633364, 0.9713816435633364]), ([0, 90, 150, 180, 1000], [8.090508544469758, 2.8975571789970345, 1.0543571789970327, 1.0543571789970294, 1.0543571789970294]), ([0, 90, 150, 180, 1000], [8.090508544469758, 2.8688907945643773, 1.025690794564376, 1.0256907945643727, 1.0256907945643727]), ([0, 90, 150, 180, 1000], [8.090508544469758, 2.8600797052490146, 1.0168797052490135, 1.01687970524901, 1.01687970524901])]"
        self.expected_progress_curves = "[([0.0, 0.09, 0.15, 0.18, 1.0], [1.0, 0.3540617612842623, 0.12623924680710813, 0.12623924680710769, 0.12623924680710769]), ([0.0, 0.09, 0.15, 0.18, 1.0], [1.0, 0.36605738334299365, 0.13823486886583944, 0.13823486886583905, 0.13823486886583905]), ([0.0, 0.09, 0.15, 0.18, 1.0], [1.0, 0.3467439288302236, 0.11892141435306937, 0.11892141435306894, 0.11892141435306894]), ([0.0, 0.09, 0.15, 0.18, 1.0], [1.0, 0.36500793422642175, 0.13718541974926757, 0.13718541974926718, 0.13718541974926718]), ([0.0, 0.09, 0.15, 0.18, 1.0], [1.0, 0.3411556527309761, 0.11333313825382192, 0.11333313825382153, 0.11333313825382153]), ([0.0, 0.09, 0.15, 0.18, 1.0], [1.0, 0.35915986422737156, 0.1313373497502173, 0.13133734975021688, 0.13133734975021688]), ([0.0, 0.09, 0.15, 0.18, 1.0], [1.0, 0.35783714299101366, 0.13001462851385948, 0.13001462851385906, 0.13001462851385906]), ([0.0, 0.09, 0.15, 0.18, 1.0], [1.0, 0.3553363833874377, 0.12751386891028352, 0.12751386891028313, 0.12751386891028313]), ([0.0, 0.09, 0.15, 0.18, 1.0], [1.0, 0.34496012134043036, 0.11713760686327615, 0.11713760686327573, 0.11713760686327573]), ([0.0, 0.09, 0.15, 0.18, 1.0], [1.0, 0.34897893301971156, 0.12115641854255732, 0.1211564185425569, 0.1211564185425569]), ([0.0, 0.09, 0.15, 0.18, 1.0], [1.0, 0.3672720531315628, 0.13944953865440862, 0.1394495386544082, 0.1394495386544082]), ([0.0, 0.09, 0.15, 0.18, 1.0], [1.0, 0.34478332250070143, 0.11696080802354725, 0.11696080802354682, 0.11696080802354682]), ([0.0, 0.09, 0.15, 0.18, 1.0], [1.0, 0.35313148893850643, 0.12530897446135228, 0.12530897446135184, 0.12530897446135184]), ([0.0, 0.09, 0.15, 0.18, 1.0], [1.0, 0.3592714941117464, 0.1314489796345922, 0.13144897963459182, 0.13144897963459182]), ([0.0, 0.09, 0.15, 0.18, 1.0], [1.0, 0.3517833983626232, 0.123960883885469, 0.12396088388546855, 0.12396088388546855]), ([0.0, 0.09, 0.15, 0.18, 1.0], [1.0, 0.3546993932173507, 0.12687687874019643, 0.12687687874019604, 0.12687687874019604]), ([0.0, 0.09, 0.15, 0.18, 1.0], [1.0, 0.3461926536677276, 0.11837013919057336, 0.11837013919057296, 0.11837013919057296]), ([0.0, 0.09, 0.15, 0.18, 1.0], [1.0, 0.3481650445806779, 0.12034253010352364, 0.12034253010352322, 0.12034253010352322]), ([0.0, 0.09, 0.15, 0.18, 1.0], [1.0, 0.36263414441597075, 0.1348116299388165, 0.1348116299388161, 0.1348116299388161]), ([0.0, 0.09, 0.15, 0.18, 1.0], [1.0, 0.36813942313328385, 0.14031690865612959, 0.1403169086561292, 0.1403169086561292]), ([0.0, 0.09, 0.15, 0.18, 1.0], [1.0, 0.347886863735808, 0.12006434925865381, 0.12006434925865338, 0.12006434925865338]), ([0.0, 0.09, 0.15, 0.18, 1.0], [1.0, 0.3581427747181172, 0.13032026024096288, 0.13032026024096247, 0.13032026024096247]), ([0.0, 0.09, 0.15, 0.18, 1.0], [1.0, 0.3545995630306081, 0.1267770485534539, 0.12677704855345348, 0.12677704855345348]), ([0.0, 0.09, 0.15, 0.18, 1.0], [1.0, 0.35351049807666457, 0.1256879835995104, 0.12568798359950994, 0.12568798359950994])]"

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
