import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class test_FACSIZE2_RNDSRCH(unittest.TestCase):
    def setUp(self):
        # Expected values
        self.expected_problem_name = "FACSIZE-2"
        self.expected_solver_name = "RNDSRCH"
        self.expected_all_recommended_xs = "[[(100, 100, 100), [115.6462517228025, 67.32898403527885, 192.94242761843483], [139.59332113978707, 132.72747399921406, 187.89481955629827], [139.59332113978707, 132.72747399921406, 187.89481955629827]], [(100, 100, 100), [85.51334282075412, 108.05785690341946, 159.85981322611704], [111.94696947582293, 221.24355205303496, 146.38163509019188], [193.07904193653746, 144.79805380524957, 103.75268316840702], [174.29099191271848, 177.04697100579972, 125.28524002044689], [202.61429556267652, 147.1833935040389, 129.27141166023293], [202.61429556267652, 147.1833935040389, 129.27141166023293]], [(100, 100, 100), [130.23611195131934, 127.5900994750533, 123.57708618604435], [143.38104154990444, 123.02091801267837, 209.77163574018053], [202.9846610317029, 125.0475615984511, 154.51891374307075], [149.98298922936942, 207.51450400869754, 101.79967309216315], [154.79572808777718, 217.56350057041462, 119.11991536080424], [183.20262236663734, 156.23222256458885, 154.11687899760685], [183.20262236663734, 156.23222256458885, 154.11687899760685]], [(100, 100, 100), [134.88010283444572, 140.23529311384567, 149.33646518317622], [129.7767278723324, 188.73960018573257, 179.88256817114868], [129.7767278723324, 188.73960018573257, 179.88256817114868]], [(100, 100, 100), [100.1269501462592, 131.88544228490724, 242.38508670034304], [126.8167221867196, 141.80684147771055, 207.59284826445213], [152.90481767249344, 171.68763773772605, 161.89283707032675], [152.90481767249344, 171.68763773772605, 161.89283707032675]], [(100, 100, 100), [167.0290317251437, 123.85734281556849, 84.39079789754139], [146.86641785041778, 166.3188830936159, 176.07253455163146], [146.86641785041778, 166.3188830936159, 176.07253455163146]], [(100, 100, 100), [135.62609423190068, 112.43768348056781, 206.21176690143704], [138.79616194628218, 193.19949664303462, 163.71978017355175], [138.79616194628218, 193.19949664303462, 163.71978017355175]], [(100, 100, 100), [81.39324547485334, 150.4271587563793, 241.51037836777013], [97.40375326014605, 223.38757767915172, 148.56049206587073], [151.54239268061187, 223.8076726328572, 115.71030005992911], [178.62339565848615, 172.09138960927007, 109.26936432440476], [153.0967413783358, 127.57417362542547, 139.45692719589937], [201.80272007709507, 147.0601377283476, 145.22338593529167], [201.80272007709507, 147.0601377283476, 145.22338593529167]], [(100, 100, 100), [195.09946102758124, 123.75768349543176, 115.28776946474234], [160.50667871846568, 136.86294748156638, 154.29282306994008], [141.35691451426555, 126.7568940216289, 221.52372917089042], [149.116179327519, 161.74594537428501, 165.8252220348562], [149.116179327519, 161.74594537428501, 165.8252220348562]], [(100, 100, 100), [141.85285929250406, 85.81791069594341, 201.91756449613095], [197.68606338619747, 110.78972172556963, 153.87472510476192], [197.68606338619747, 110.78972172556963, 153.87472510476192]], [(100, 100, 100), [105.64754099461449, 133.98624953560997, 182.17996423911126], [221.7142734715177, 131.79738619687419, 138.09968187118267], [155.990470421039, 175.47709902265035, 160.55202376908173], [155.990470421039, 175.47709902265035, 160.55202376908173]], [(100, 100, 100), [84.57984591196474, 85.56378651346725, 153.5750344497168], [262.02021248177726, 90.3442083605554, 124.07990063275659], [124.7515710881741, 126.63963165624145, 220.66602597444628], [172.84160013102294, 142.84562242959845, 175.72801410020955], [172.84160013102294, 142.84562242959845, 175.72801410020955]], [(100, 100, 100), [120.13031437227163, 176.4483188049054, 200.75179723938314], [177.00682424880088, 139.1397383858137, 176.21725323917076], [177.00682424880088, 139.1397383858137, 176.21725323917076]], [(100, 100, 100), [100.7172200943301, 100.15997589409237, 237.6444013626397], [98.88471129071432, 127.52841588247347, 228.44281921072545], [139.59727472072308, 179.71363521191205, 148.86698940869735], [115.32412771773957, 148.0315052416532, 219.36403799981807], [115.32412771773957, 148.0315052416532, 219.36403799981807]], [(100, 100, 100), [164.04239882734115, 83.73423337860048, 197.42439875013073], [144.4095301062759, 111.54497501006227, 194.3486447968795], [136.87696900926755, 142.7982227648679, 134.87166668132568], [111.63730598533517, 140.63984080056824, 171.7168663668232], [145.78209149713513, 134.81948386003558, 203.50449167399998], [145.78209149713513, 134.81948386003558, 203.50449167399998]], [(100, 100, 100), [105.86186801532018, 130.44031363250343, 138.03301570258736], [135.0190416173918, 118.68444438706256, 121.82427159032049], [166.61038963472495, 195.28160316836403, 106.88777948558747], [143.20125542251884, 165.2615699857489, 131.20798284450086], [163.45695988252936, 156.25312947683295, 177.65377565566104], [163.45695988252936, 156.25312947683295, 177.65377565566104]], [(100, 100, 100), [167.16582602134253, 117.08220545041809, 146.940081395101], [172.87574276750777, 124.8260858617318, 148.03207951846355], [156.72621668760038, 184.0278841037769, 141.07713914570508], [156.72621668760038, 184.0278841037769, 141.07713914570508]], [(100, 100, 100), [111.51160723399705, 106.7274878032779, 276.4927076665878], [138.69017056365396, 156.13731915982495, 153.88354601053928], [161.4784137549601, 163.65097359274571, 164.83318707563515], [161.4784137549601, 163.65097359274571, 164.83318707563515]], [(100, 100, 100), [186.4636109174302, 120.31665887354525, 147.30447941444152], [145.580588788903, 129.21357466290326, 210.97481702053028], [145.8010365782808, 160.06207042208652, 192.43213884203809], [179.5226216410998, 140.40470523857016, 178.9692038962604], [179.5226216410998, 140.40470523857016, 178.9692038962604]], [(100, 100, 100), [126.78258921270691, 115.56555378195718, 185.31288230909954], [172.4693781867695, 188.50852844998565, 124.63623190399636], [183.56507613359392, 153.33710294079907, 146.32611301164872], [183.56507613359392, 153.33710294079907, 146.32611301164872]], [(100, 100, 100), [157.52285701799073, 157.8809351984492, 63.76155797913763], [204.88584568171203, 119.81271019229756, 116.5134666335771], [139.1290509930911, 99.57826561580396, 252.49375415004337], [144.50187761252525, 127.94753713838006, 125.72947311022558], [148.38550120223877, 189.35085504431697, 138.87195402881275], [148.38550120223877, 189.35085504431697, 138.87195402881275]], [(100, 100, 100), [221.16482025065523, 192.83418476337346, 80.37773466635694], [165.13620138827946, 158.1992981735277, 151.1423776246641], [157.37294224872534, 138.7475029005391, 160.82081488583455], [158.19041137667506, 148.5341464623582, 183.28468900248762], [171.83566867416238, 155.5086401863483, 170.0938188888864], [171.83566867416238, 155.5086401863483, 170.0938188888864]], [(100, 100, 100), [65.35590011487419, 132.875658720298, 151.9120312523335], [76.55300454306999, 186.22353631874907, 174.28210628001906], [99.61279440193002, 173.84543867312635, 189.0204579840077], [126.52140816591051, 118.62087800939166, 173.856467046343], [171.44606336038123, 122.92923768267066, 177.68377658404071], [171.44606336038123, 122.92923768267066, 177.68377658404071]], [(100, 100, 100), [79.01630036891217, 194.50668395901803, 136.02221391457607], [144.97459788218055, 120.35084481653196, 111.76339798764945], [160.11823569531393, 167.46370257633976, 137.19985867793915], [126.68667776762254, 154.96235183257824, 182.9575188586404], [126.68667776762254, 154.96235183257824, 182.9575188586404]]]"
        self.expected_all_intermediate_budgets = "[[0, 50, 170, 10000], [0, 110, 120, 280, 350, 1300, 10000], [0, 50, 620, 1130, 1840, 3710, 8680, 10000], [0, 30, 730, 10000], [0, 60, 4830, 6610, 10000], [0, 220, 690, 10000], [0, 30, 140, 10000], [0, 40, 70, 190, 240, 660, 4120, 10000], [0, 70, 160, 690, 4160, 10000], [0, 20, 540, 10000], [0, 90, 350, 1500, 10000], [0, 20, 120, 510, 4840, 10000], [0, 60, 1390, 10000], [0, 140, 230, 380, 470, 10000], [0, 20, 120, 150, 560, 2550, 10000], [0, 50, 110, 1120, 3520, 7380, 10000], [0, 20, 110, 2600, 10000], [0, 510, 2290, 3140, 10000], [0, 40, 120, 490, 4810, 10000], [0, 180, 410, 4490, 10000], [0, 110, 200, 310, 1210, 2600, 10000], [0, 80, 230, 510, 2790, 3850, 10000], [0, 40, 90, 300, 640, 2050, 10000], [0, 180, 190, 500, 2240, 10000]]"
        self.expected_all_est_objectives = "[[0.275, 0.215, 0.685, 0.685], [0.24, 0.255, 0.535, 0.5, 0.645, 0.65, 0.65], [0.21, 0.525, 0.705, 0.7, 0.42, 0.595, 0.835, 0.835], [0.185, 0.62, 0.71, 0.71], [0.29, 0.545, 0.72, 0.85, 0.85], [0.25, 0.345, 0.81, 0.81], [0.235, 0.58, 0.72, 0.72], [0.3, 0.34, 0.495, 0.56, 0.56, 0.62, 0.76, 0.76], [0.28, 0.525, 0.74, 0.72, 0.83, 0.83], [0.2, 0.345, 0.525, 0.525], [0.265, 0.475, 0.585, 0.745, 0.745], [0.215, 0.245, 0.315, 0.62, 0.81, 0.81], [0.24, 0.67, 0.77, 0.77], [0.245, 0.335, 0.38, 0.705, 0.6, 0.6], [0.255, 0.395, 0.605, 0.585, 0.555, 0.73, 0.73], [0.205, 0.385, 0.46, 0.53, 0.605, 0.84, 0.84], [0.26, 0.59, 0.675, 0.8, 0.8], [0.23, 0.475, 0.745, 0.845, 0.845], [0.245, 0.56, 0.64, 0.76, 0.7, 0.7], [0.21, 0.56, 0.685, 0.785, 0.785], [0.26, 0.2, 0.495, 0.48, 0.555, 0.72, 0.72], [0.27, 0.335, 0.805, 0.725, 0.81, 0.85, 0.85], [0.21, 0.19, 0.275, 0.475, 0.555, 0.635, 0.635], [0.24, 0.255, 0.425, 0.71, 0.715, 0.715]]"
        self.expected_objective_curves = "[([0, 50, 170, 10000], [0.25, 0.215, 0.685, 0.685]), ([0, 110, 120, 280, 350, 1300, 10000], [0.25, 0.255, 0.535, 0.5, 0.645, 0.65, 0.65]), ([0, 50, 620, 1130, 1840, 3710, 8680, 10000], [0.25, 0.525, 0.705, 0.7, 0.42, 0.595, 0.835, 0.835]), ([0, 30, 730, 10000], [0.25, 0.62, 0.71, 0.71]), ([0, 60, 4830, 6610, 10000], [0.25, 0.545, 0.72, 0.835, 0.835]), ([0, 220, 690, 10000], [0.25, 0.345, 0.81, 0.81]), ([0, 30, 140, 10000], [0.25, 0.58, 0.72, 0.72]), ([0, 40, 70, 190, 240, 660, 4120, 10000], [0.25, 0.34, 0.495, 0.56, 0.56, 0.62, 0.76, 0.76]), ([0, 70, 160, 690, 4160, 10000], [0.25, 0.525, 0.74, 0.72, 0.83, 0.83]), ([0, 20, 540, 10000], [0.25, 0.345, 0.525, 0.525]), ([0, 90, 350, 1500, 10000], [0.25, 0.475, 0.585, 0.745, 0.745]), ([0, 20, 120, 510, 4840, 10000], [0.25, 0.245, 0.315, 0.62, 0.81, 0.81]), ([0, 60, 1390, 10000], [0.25, 0.67, 0.77, 0.77]), ([0, 140, 230, 380, 470, 10000], [0.25, 0.335, 0.38, 0.705, 0.6, 0.6]), ([0, 20, 120, 150, 560, 2550, 10000], [0.25, 0.395, 0.605, 0.585, 0.555, 0.73, 0.73]), ([0, 50, 110, 1120, 3520, 7380, 10000], [0.25, 0.385, 0.46, 0.53, 0.605, 0.84, 0.84]), ([0, 20, 110, 2600, 10000], [0.25, 0.59, 0.675, 0.8, 0.8]), ([0, 510, 2290, 3140, 10000], [0.25, 0.475, 0.745, 0.845, 0.845]), ([0, 40, 120, 490, 4810, 10000], [0.25, 0.56, 0.64, 0.76, 0.7, 0.7]), ([0, 180, 410, 4490, 10000], [0.25, 0.56, 0.685, 0.785, 0.785]), ([0, 110, 200, 310, 1210, 2600, 10000], [0.25, 0.2, 0.495, 0.48, 0.555, 0.72, 0.72]), ([0, 80, 230, 510, 2790, 3850, 10000], [0.25, 0.335, 0.805, 0.725, 0.81, 0.85, 0.85]), ([0, 40, 90, 300, 640, 2050, 10000], [0.25, 0.19, 0.275, 0.475, 0.555, 0.635, 0.635]), ([0, 180, 190, 500, 2240, 10000], [0.25, 0.255, 0.425, 0.71, 0.715, 0.715])]"
        self.expected_progress_curves = "[([0.0, 0.005, 0.017, 1.0], [1.0, 1.0598290598290598, 0.2564102564102563, 0.2564102564102563]), ([0.0, 0.011, 0.012, 0.028, 0.035, 0.13, 1.0], [1.0, 0.9914529914529915, 0.5128205128205128, 0.5726495726495726, 0.32478632478632474, 0.31623931623931617, 0.31623931623931617]), ([0.0, 0.005, 0.062, 0.113, 0.184, 0.371, 0.868, 1.0], [1.0, 0.5299145299145298, 0.22222222222222224, 0.2307692307692308, 0.7094017094017094, 0.41025641025641024, -0.0, -0.0]), ([0.0, 0.003, 0.073, 1.0], [1.0, 0.3675213675213675, 0.2136752136752137, 0.2136752136752137]), ([0.0, 0.006, 0.483, 0.661, 1.0], [1.0, 0.49572649572649563, 0.19658119658119658, -0.0, -0.0]), ([0.0, 0.022, 0.069, 1.0], [1.0, 0.8376068376068376, 0.04273504273504258, 0.04273504273504258]), ([0.0, 0.003, 0.014, 1.0], [1.0, 0.43589743589743596, 0.19658119658119658, 0.19658119658119658]), ([0.0, 0.004, 0.007, 0.019, 0.024, 0.066, 0.412, 1.0], [1.0, 0.8461538461538461, 0.5811965811965811, 0.47008547008546997, 0.47008547008546997, 0.3675213675213675, 0.12820512820512814, 0.12820512820512814]), ([0.0, 0.007, 0.016, 0.069, 0.416, 1.0], [1.0, 0.5299145299145298, 0.16239316239316237, 0.19658119658119658, 0.008547008547008555, 0.008547008547008555]), ([0.0, 0.002, 0.054, 1.0], [1.0, 0.8376068376068376, 0.5299145299145298, 0.5299145299145298]), ([0.0, 0.009, 0.035, 0.15, 1.0], [1.0, 0.6153846153846154, 0.4273504273504274, 0.1538461538461538, 0.1538461538461538]), ([0.0, 0.002, 0.012, 0.051, 0.484, 1.0], [1.0, 1.0085470085470085, 0.888888888888889, 0.3675213675213675, 0.04273504273504258, 0.04273504273504258]), ([0.0, 0.006, 0.139, 1.0], [1.0, 0.28205128205128194, 0.11111111111111102, 0.11111111111111102]), ([0.0, 0.014, 0.023, 0.038, 0.047, 1.0], [1.0, 0.8547008547008547, 0.7777777777777778, 0.22222222222222224, 0.4017094017094017, 0.4017094017094017]), ([0.0, 0.002, 0.012, 0.015, 0.056, 0.255, 1.0], [1.0, 0.7521367521367521, 0.39316239316239315, 0.4273504273504274, 0.47863247863247854, 0.17948717948717946, 0.17948717948717946]), ([0.0, 0.005, 0.011, 0.112, 0.352, 0.738, 1.0], [1.0, 0.7692307692307692, 0.641025641025641, 0.5213675213675213, 0.39316239316239315, -0.008547008547008555, -0.008547008547008555]), ([0.0, 0.002, 0.011, 0.26, 1.0], [1.0, 0.4188034188034188, 0.27350427350427337, 0.05982905982905969, 0.05982905982905969]), ([0.0, 0.051, 0.229, 0.314, 1.0], [1.0, 0.6153846153846154, 0.1538461538461538, -0.01709401709401711, -0.01709401709401711]), ([0.0, 0.004, 0.012, 0.049, 0.481, 1.0], [1.0, 0.47008547008546997, 0.33333333333333326, 0.12820512820512814, 0.2307692307692308, 0.2307692307692308]), ([0.0, 0.018, 0.041, 0.449, 1.0], [1.0, 0.47008547008546997, 0.2564102564102563, 0.08547008547008536, 0.08547008547008536]), ([0.0, 0.011, 0.02, 0.031, 0.121, 0.26, 1.0], [1.0, 1.0854700854700856, 0.5811965811965811, 0.6068376068376068, 0.47863247863247854, 0.19658119658119658, 0.19658119658119658]), ([0.0, 0.008, 0.023, 0.051, 0.279, 0.385, 1.0], [1.0, 0.8547008547008547, 0.05128205128205114, 0.18803418803418803, 0.04273504273504258, -0.025641025641025664, -0.025641025641025664]), ([0.0, 0.004, 0.009, 0.03, 0.064, 0.205, 1.0], [1.0, 1.1025641025641026, 0.9572649572649572, 0.6153846153846154, 0.47863247863247854, 0.34188034188034183, 0.34188034188034183]), ([0.0, 0.018, 0.019, 0.05, 0.224, 1.0], [1.0, 0.9914529914529915, 0.7008547008547008, 0.2136752136752137, 0.20512820512820512, 0.20512820512820512])]"

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
