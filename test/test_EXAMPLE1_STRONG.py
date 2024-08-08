import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class test_EXAMPLE1_STRONG(unittest.TestCase):
    def setUp(self):
        # Expected values
        self.expected_problem_name = "EXAMPLE-1"
        self.expected_solver_name = "STRONG"
        self.expected_all_recommended_xs = "[[(2.0, 2.0), (0.5857864376269049, 0.5857864376269049), (0.22756040634011176, 0.22756040634011176), (0.04112054594755016, 0.04112054594755016), (0.04112054594755016, 0.04112054594755016)], [(2.0, 2.0), (0.5857864376269051, 0.5857864376269051), (0.20780029998524557, 0.20780029998524557), (0.03511191530186958, 0.03511191530186958), (0.0010585468290958966, 0.0010585468290958966), (0.0010585468290958966, 0.0010585468290958966)], [(2.0, 2.0), (0.5857864376269049, 0.5857864376269049), (0.2184892307486237, 0.2184892307486237), (0.04052845254242407, 0.04052845254242407), (0.0013631895459503274, 0.0013631895459503274), (0.0013631895459503274, 0.0013631895459503274)], [(2.0, 2.0), (0.5857864376269049, 0.5857864376269049), (0.21501284100736973, 0.21501284100736973), (0.03831776207503851, 0.03831776207503851), (0.0013353677809041509, 0.0013353677809041509), (0.0013353677809041509, 0.0013353677809041509)], [(2.0, 2.0), (0.5857864376269049, 0.5857864376269049), (0.2207947963157782, 0.2207947963157782), (0.03934224605155248, 0.03934224605155248), (0.03934224605155248, 0.03934224605155248)], [(2.0, 2.0), (0.5857864376269051, 0.5857864376269051), (0.21657065566905398, 0.21657065566905398), (0.03517178460167489, 0.03517178460167489), (0.0005563622318154118, 0.0005563622318154118), (0.0005563622318154118, 0.0005563622318154118)], [(2.0, 2.0), (0.5857864376269049, 0.5857864376269049), (0.2191083357420765, 0.2191083357420765), (0.03872622418508648, 0.03872622418508648), (0.03872622418508648, 0.03872622418508648)], [(2.0, 2.0), (0.5857864376269051, 0.5857864376269051), (0.21522620438211115, 0.21522620438211115), (0.03877774483103974, 0.03877774483103974), (0.0009323933150718611, 0.0009323933150718611), (0.0009323933150718611, 0.0009323933150718611)], [(2.0, 2.0), (0.5857864376269051, 0.5857864376269051), (0.2144952323609805, 0.2144952323609805), (0.03748673560418814, 0.03748673560418814), (0.03748673560418814, 0.03748673560418814)], [(2.0, 2.0), (0.5857864376269049, 0.5857864376269049), (0.21031827369793593, 0.21031827369793593), (0.03778266696479224, 0.03778266696479224), (0.03778266696479224, 0.03778266696479224)], [(2.0, 2.0), (0.5857864376269049, 0.5857864376269049), (0.21601550093478156, 0.21601550093478156), (0.03703399730294751, 0.03703399730294751), (0.0012689898505106323, 0.0012689898505106323), (0.0012689898505106323, 0.0012689898505106323)], [(2.0, 2.0), (0.5857864376269049, 0.5857864376269049), (0.21379194297334086, 0.21379194297334086), (0.21379194297334086, 0.21379194297334086)], [(2.0, 2.0), (0.5857864376269049, 0.5857864376269049), (0.21650519639535015, 0.21650519639535015), (0.038796013187748, 0.038796013187748), (0.0014047142381751407, 0.0014047142381751407), (0.0014047142381751407, 0.0014047142381751407)], [(2.0, 2.0), (0.5857864376269051, 0.5857864376269051), (0.2185856029815056, 0.2185856029815056), (0.04004946889409508, 0.04004946889409508), (0.0014947584542047906, 0.0014947584542047906), (0.0014947584542047906, 0.0014947584542047906)], [(2.0, 2.0), (0.5857864376269049, 0.5857864376269049), (0.21260930308316184, 0.21260930308316184), (0.21260930308316184, 0.21260930308316184)], [(2.0, 2.0), (0.5857864376269049, 0.5857864376269049), (0.22027860969473728, 0.22027860969473728), (0.03927414506343607, 0.03927414506343607), (0.03927414506343607, 0.03927414506343607)], [(2.0, 2.0), (0.5857864376269049, 0.5857864376269049), (0.22232388318691493, 0.22232388318691493), (0.039536592188373854, 0.039536592188373854), (0.039536592188373854, 0.039536592188373854)], [(2.0, 2.0), (0.5857864376269049, 0.5857864376269049), (0.2178615650195479, 0.2178615650195479), (0.04039518711484241, 0.04039518711484241), (0.04039518711484241, 0.04039518711484241)], [(2.0, 2.0), (0.5857864376269049, 0.5857864376269049), (0.2112754849109617, 0.2112754849109617), (0.03719342954953597, 0.03719342954953597), (0.0009764905044987518, 0.0009764905044987518), (0.0009764905044987518, 0.0009764905044987518)], [(2.0, 2.0), (0.5857864376269049, 0.5857864376269049), (0.2198143241123911, 0.2198143241123911), (0.03862954467374591, 0.03862954467374591), (0.03862954467374591, 0.03862954467374591)], [(2.0, 2.0), (0.5857864376269049, 0.5857864376269049), (0.20965686186994742, 0.20965686186994742), (0.036826747808163585, 0.036826747808163585), (0.036826747808163585, 0.036826747808163585)], [(2.0, 2.0), (0.5857864376269049, 0.5857864376269049), (0.21887819715292461, 0.21887819715292461), (0.03785377025889006, 0.03785377025889006), (0.03785377025889006, 0.03785377025889006)], [(2.0, 2.0), (0.5857864376269049, 0.5857864376269049), (0.22020740557090163, 0.22020740557090163), (0.038761110676857, 0.038761110676857), (0.0011793278358463305, 0.0011793278358463305), (0.0011793278358463305, 0.0011793278358463305)], [(2.0, 2.0), (0.5857864376269049, 0.5857864376269049), (0.20856466154738412, 0.20856466154738412), (0.033656161715546445, 0.033656161715546445), (0.033656161715546445, 0.033656161715546445)]]"
        self.expected_all_intermediate_budgets = "[[10, 60, 550, 735, 1000], [10, 60, 550, 735, 940, 1000], [10, 60, 550, 735, 835, 1000], [10, 60, 550, 735, 940, 1000], [10, 60, 550, 735, 1000], [10, 60, 550, 735, 940, 1000], [10, 60, 550, 735, 1000], [10, 60, 550, 735, 940, 1000], [10, 60, 550, 735, 1000], [10, 60, 550, 735, 1000], [10, 60, 550, 735, 940, 1000], [10, 60, 550, 1000], [10, 60, 550, 735, 940, 1000], [10, 60, 550, 735, 940, 1000], [10, 60, 550, 1000], [10, 60, 550, 735, 1000], [10, 60, 550, 735, 1000], [10, 60, 550, 735, 1000], [10, 60, 550, 735, 940, 1000], [10, 60, 550, 735, 1000], [10, 60, 550, 735, 1000], [10, 60, 550, 735, 1000], [10, 60, 550, 735, 940, 1000], [10, 60, 550, 735, 1000]]"
        self.expected_all_est_objectives = "[[7.984539704940337, 0.6708312059555748, 0.0881071820076889, -0.012078496461615526, -0.012078496461615526], [8.081590387702734, 0.767881888717971, 0.16795231705064714, 0.08405608089506242, 0.08159262874550982, 0.08159262874550982], [7.9253347189439385, 0.611626219959176, 0.020809806850187313, -0.07138017012509631, -0.07466156448458701, -0.07466156448458701], [8.073099810658121, 0.7593913116733596, 0.16556085425424108, 0.07603631243899867, 0.07310337707234067, 0.07310337707234067], [7.880122723414122, 0.5664142244293586, -0.022376592425628807, -0.11678165193711894, -0.11678165193711894], [8.025785950362149, 0.7120774513773865, 0.11959164815599455, 0.028260059226279804, 0.025786569440012643, 0.025786569440012643], [8.015084462897443, 0.7013759639126814, 0.11110138848076702, 0.01808390377670916, 0.01808390377670916], [7.994852045957048, 0.681143546972286, 0.08749668406250644, -0.002140527054591628, -0.005146215328366099, -0.005146215328366099], [7.910902809206077, 0.5971943102213154, 0.002919218617257564, -0.08628668010140789, -0.08628668010140789], [7.943417039435916, 0.629708540451154, 0.031884591938474476, -0.05372790071814053, -0.05372790071814053], [8.091417684005862, 0.7777091850210985, 0.18474307729406847, 0.09416071791832868, 0.09142090467634068, 0.09142090467634068], [7.909472416682599, 0.595763917697836, 0.0008864064432290242, 0.0008864064432290242], [7.977013328578315, 0.6633048295935531, 0.07076232871069196, -0.01997641014315845, -0.022982724977504435, -0.022982724977504435], [8.026689092895502, 0.7129805939107401, 0.12224842455707684, 0.02989701281289824, 0.02669356150117288, 0.02669356150117288], [7.9661065902554125, 0.6523980912706503, 0.0565120217704264, 0.0565120217704264], [7.989698471543214, 0.6759899725584526, 0.08674380332130609, -0.0072166115158590486, -0.0072166115158590486], [7.920874622531411, 0.6071661235466483, 0.019730440602026868, -0.07599909322485153, -0.07599909322485153], [7.936832268065668, 0.6231237690809069, 0.031759591091200946, -0.0599041896502462, -0.0599041896502462], [8.053894643913893, 0.7401861449291299, 0.1431693049626145, 0.05666134631720311, 0.05389655098130123, 0.05389655098130123], [8.098435148416002, 0.784726649431239, 0.19507182258597425, 0.10141963185940142, 0.10141963185940142], [7.934581643563343, 0.6208731445785798, 0.02249364302164878, -0.06270593772840749, -0.06270593772840749], [8.017557178997036, 0.7038486800122726, 0.11337250937486232, 0.020422994842658908, 0.020422994842658908], [7.988890794564379, 0.6751822955796161, 0.08587339750091165, -0.008104358033816352, -0.01110642380733466, -0.01110642380733466], [7.9800797052490156, 0.6663712062642534, 0.06707814134176371, -0.017654820308140015, -0.017654820308140015]]"
        self.expected_objective_curves = "[([10, 60, 550, 735, 1000], [8.090508544469758, 0.6708312059555748, 0.0881071820076889, -0.012078496461615526, -0.012078496461615526]), ([10, 60, 550, 735, 940, 1000], [8.090508544469758, 0.767881888717971, 0.16795231705064714, 0.08405608089506242, 0.08159262874550982, 0.08159262874550982]), ([10, 60, 550, 735, 835, 1000], [8.090508544469758, 0.611626219959176, 0.020809806850187313, -0.07138017012509631, -0.07466156448458701, -0.07466156448458701]), ([10, 60, 550, 735, 940, 1000], [8.090508544469758, 0.7593913116733596, 0.16556085425424108, 0.07603631243899867, 0.07310337707234067, 0.07310337707234067]), ([10, 60, 550, 735, 1000], [8.090508544469758, 0.5664142244293586, -0.022376592425628807, -0.11678165193711894, -0.11678165193711894]), ([10, 60, 550, 735, 940, 1000], [8.090508544469758, 0.7120774513773865, 0.11959164815599455, 0.028260059226279804, 0.025786569440012643, 0.025786569440012643]), ([10, 60, 550, 735, 1000], [8.090508544469758, 0.7013759639126814, 0.11110138848076702, 0.01808390377670916, 0.01808390377670916]), ([10, 60, 550, 735, 940, 1000], [8.090508544469758, 0.681143546972286, 0.08749668406250644, -0.002140527054591628, -0.005146215328366099, -0.005146215328366099]), ([10, 60, 550, 735, 1000], [8.090508544469758, 0.5971943102213154, 0.002919218617257564, -0.08628668010140789, -0.08628668010140789]), ([10, 60, 550, 735, 1000], [8.090508544469758, 0.629708540451154, 0.031884591938474476, -0.05372790071814053, -0.05372790071814053]), ([10, 60, 550, 735, 940, 1000], [8.090508544469758, 0.7777091850210985, 0.18474307729406847, 0.09416071791832868, 0.09142090467634068, 0.09142090467634068]), ([10, 60, 550, 1000], [8.090508544469758, 0.595763917697836, 0.0008864064432290242, 0.0008864064432290242]), ([10, 60, 550, 735, 940, 1000], [8.090508544469758, 0.6633048295935531, 0.07076232871069196, -0.01997641014315845, -0.022982724977504435, -0.022982724977504435]), ([10, 60, 550, 735, 940, 1000], [8.090508544469758, 0.7129805939107401, 0.12224842455707684, 0.02989701281289824, 0.02669356150117288, 0.02669356150117288]), ([10, 60, 550, 1000], [8.090508544469758, 0.6523980912706503, 0.0565120217704264, 0.0565120217704264]), ([10, 60, 550, 735, 1000], [8.090508544469758, 0.6759899725584526, 0.08674380332130609, -0.0072166115158590486, -0.0072166115158590486]), ([10, 60, 550, 735, 1000], [8.090508544469758, 0.6071661235466483, 0.019730440602026868, -0.07599909322485153, -0.07599909322485153]), ([10, 60, 550, 735, 1000], [8.090508544469758, 0.6231237690809069, 0.031759591091200946, -0.0599041896502462, -0.0599041896502462]), ([10, 60, 550, 735, 940, 1000], [8.090508544469758, 0.7401861449291299, 0.1431693049626145, 0.05666134631720311, 0.05389655098130123, 0.05389655098130123]), ([10, 60, 550, 735, 1000], [8.090508544469758, 0.784726649431239, 0.19507182258597425, 0.10141963185940142, 0.10141963185940142]), ([10, 60, 550, 735, 1000], [8.090508544469758, 0.6208731445785798, 0.02249364302164878, -0.06270593772840749, -0.06270593772840749]), ([10, 60, 550, 735, 1000], [8.090508544469758, 0.7038486800122726, 0.11337250937486232, 0.020422994842658908, 0.020422994842658908]), ([10, 60, 550, 735, 940, 1000], [8.090508544469758, 0.6751822955796161, 0.08587339750091165, -0.008104358033816352, -0.01110642380733466, -0.01110642380733466]), ([10, 60, 550, 735, 1000], [8.090508544469758, 0.6663712062642534, 0.06707814134176371, -0.017654820308140015, -0.017654820308140015])]"
        self.expected_progress_curves = "[([0.01, 0.06, 0.55, 0.735, 1.0], [1.0, 0.0829158268937395, 0.010890190835768202, -0.001492921785475617, -0.001492921785475617]), ([0.01, 0.06, 0.55, 0.735, 0.94, 1.0], [1.0, 0.09491144895247089, 0.0207591792441095, 0.010389468156796979, 0.010084981468968624, 0.010084981468968624]), ([0.01, 0.06, 0.55, 0.735, 0.835, 1.0], [1.0, 0.07559799443970072, 0.0025721259344583217, -0.008822704992245266, -0.00922829066605729, -0.00922829066605729]), ([0.01, 0.06, 0.55, 0.735, 0.94, 1.0], [1.0, 0.09386199983589896, 0.02046359055728452, 0.009398211746648863, 0.00903569617046017, 0.00903569617046017]), ([0.01, 0.06, 0.55, 0.735, 1.0], [1.0, 0.0700097183404533, -0.002765783176995006, -0.014434401903814151, -0.014434401903814151]), ([0.01, 0.06, 0.55, 0.735, 0.94, 1.0], [1.0, 0.08801392983684875, 0.01478172200160904, 0.0034929892318817067, 0.003187261875848209, 0.003187261875848209]), ([0.01, 0.06, 0.55, 0.735, 1.0], [1.0, 0.08669120860049086, 0.013732312112408562, 0.0022351998860529426, 0.0022351998860529426]), ([0.01, 0.06, 0.55, 0.735, 0.94, 1.0], [1.0, 0.08419044899691497, 0.010814732297923908, -0.0002645726214645405, -0.0006360805751677722, -0.0006360805751677722]), ([0.01, 0.06, 0.55, 0.735, 1.0], [1.0, 0.07381418694990759, 0.00036082016367846084, -0.010665173842550217, -0.010665173842550217]), ([0.01, 0.06, 0.55, 0.735, 1.0], [1.0, 0.07783299862918869, 0.003940987363553196, -0.006640855815530418, -0.006640855815530418]), ([0.01, 0.06, 0.55, 0.735, 0.94, 1.0], [1.0, 0.09612611874103999, 0.022834544488596953, 0.011638417708945125, 0.01129977234111336, 0.01129977234111336]), ([0.01, 0.06, 0.55, 1.0], [1.0, 0.07363738811017863, 0.00010956127644595649, 0.00010956127644595649]), ([0.01, 0.06, 0.55, 0.735, 0.94, 1.0], [1.0, 0.08198555454798365, 0.008746338789674887, -0.002469116747526738, -0.002840702145134524, -0.002840702145134524]), ([0.01, 0.06, 0.55, 0.735, 0.94, 1.0], [1.0, 0.08812555972122367, 0.015110103881002557, 0.0036953193545953607, 0.003299367568114019, 0.003299367568114019]), ([0.01, 0.06, 0.55, 1.0], [1.0, 0.08063746397210036, 0.006984977700697816, 0.006984977700697816]), ([0.01, 0.06, 0.55, 0.735, 1.0], [1.0, 0.08355345882682783, 0.010721675015175597, -0.0008919849075237602, -0.0008919849075237602]), ([0.01, 0.06, 0.55, 0.735, 1.0], [1.0, 0.07504671927720476, 0.0024387145126388314, -0.00939361139131365, -0.00939361139131365]), ([0.01, 0.06, 0.55, 0.735, 1.0], [1.0, 0.07701911019015502, 0.003925537055752833, -0.007404255161586044, -0.007404255161586044]), ([0.01, 0.06, 0.55, 0.735, 0.94, 1.0], [1.0, 0.0914882100254479, 0.01769595868735315, 0.0070034344572732455, 0.006661701262047619, 0.006661701262047619]), ([0.01, 0.06, 0.55, 0.735, 1.0], [1.0, 0.09699348874276098, 0.024111194186836993, 0.012535631264950149, 0.012535631264950149]), ([0.01, 0.06, 0.55, 0.735, 1.0], [1.0, 0.07674092934528519, 0.0027802508208243895, -0.007750555775788648, -0.007750555775788648]), ([0.01, 0.06, 0.55, 0.735, 1.0], [1.0, 0.08699684032759426, 0.01401302634459953, 0.0025243153419099943, 0.0025243153419099943]), ([0.01, 0.06, 0.55, 0.735, 0.94, 1.0], [1.0, 0.08345362864008528, 0.010614091441706733, -0.001001711819383228, -0.0013727720261696554, -0.0013727720261696554]), ([0.01, 0.06, 0.55, 0.735, 1.0], [1.0, 0.08236456368614176, 0.008290967245515705, -0.0021821644722454328, -0.0021821644722454328])]"

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
