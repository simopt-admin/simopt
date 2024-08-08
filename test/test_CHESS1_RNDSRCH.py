import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class test_CHESS1_RNDSRCH(unittest.TestCase):
    def setUp(self):
        # Expected values
        self.expected_problem_name = "CHESS-1"
        self.expected_solver_name = "RNDSRCH"
        self.expected_all_recommended_xs = "[[(150,), (143.9959354336992,), (113.6269068626152,), (105.08828691067168,), (98.32992367837494,), (93.19677707732481,), (93.19677707732481,)], [(150,), (119.27048503018092,), (103.10156276095827,), (102.53925144519908,), (101.07902970687569,), (101.07902970687569,)], [(150,), (122.72928023719989,), (114.51330244302284,), (112.07126704757705,), (109.70428450521894,), (104.1369744076302,), (104.1369744076302,)], [(150,), (104.66583726274527,), (102.56893879857294,), (96.3539568789001,), (96.3539568789001,)], [(150,), (146.40270006499802,), (131.76472018707562,), (115.58445281553676,), (115.12938585410117,), (110.31212977724317,), (103.8139402247456,), (97.81325466504936,), (97.81325466504936,)], [(150,), (127.32339425713342,), (123.0554345330125,), (113.56452280248605,), (108.12599267323786,), (97.4876059250656,), (97.4876059250656,)], [(150,), (143.3698978636649,), (116.87967131671496,), (113.7524445719657,), (96.50746113024549,), (96.50746113024549,)], [(150,), (120.73420496683445,), (104.51921544298852,), (100.42258968518165,), (97.20843481870226,), (97.20843481870226,)], [(150,), (101.05580426232791,), (98.8905279005489,), (98.8905279005489,)], [(150,), (129.56816424339834,), (120.18826149831811,), (115.11223524428766,), (113.13627700861365,), (102.72878653310667,), (99.39214831242,), (98.35893958209186,), (98.35893958209186,)], [(150,), (124.1967182285037,), (107.11206420882289,), (100.5526441928546,), (100.5526441928546,)], [(150,), (110.32559072958807,), (102.35968700347232,), (102.4746358898518,), (102.4746358898518,)], [(150,), (114.34253108014934,), (110.13520546064949,), (101.98104519876895,), (100.34839530355569,), (96.95891901556797,), (96.95891901556797,)], [(150,), (143.19104172985053,), (101.07117608226869,), (97.89427259997454,), (97.89427259997454,)], [(150,), (130.06345817601976,), (109.65367455338614,), (105.70031895373583,), (100.16038297922574,), (100.16038297922574,)], [(150,), (133.5021012227031,), (101.67615214718572,), (99.26704413822343,), (98.59718129316231,), (98.32445055477578,)], [(150,), (128.94645084310133,), (128.73136527169376,), (128.22687045260113,), (107.33519684226395,), (107.33519684226395,)], [(150,), (104.29062984684742,), (104.29062984684742,)], [(150,), (100.21717820516838,), (98.04964031981808,), (98.04964031981808,)], [(150,), (146.4366686851828,), (114.93257276630024,), (100.15150852269403,), (100.15150852269403,)], [(150,), (139.04038767003587,), (100.04762556976482,), (100.04762556976482,)], [(150,), (129.32567482817157,), (119.85732212220412,), (103.41136672708156,), (102.42732370666914,), (101.09970147162579,), (101.09970147162579,)], [(150,), (143.2628676503081,), (98.52958180869902,), (98.20186183193037,), (98.09956880265253,), (98.09956880265253,)], [(150,), (142.4567373555971,), (136.27076924386228,), (114.77522526895017,), (107.70077142219725,), (106.47366661058012,), (106.47366661058012,)]]"
        self.expected_all_intermediate_budgets = "[[0, 20, 40, 120, 150, 370, 1000], [0, 40, 60, 850, 890, 1000], [0, 40, 180, 190, 290, 480, 1000], [0, 30, 250, 270, 1000], [0, 40, 60, 120, 260, 270, 560, 760, 1000], [0, 40, 70, 220, 450, 460, 1000], [0, 40, 60, 100, 150, 1000], [0, 30, 100, 370, 470, 1000], [0, 20, 800, 1000], [0, 20, 70, 180, 200, 210, 530, 720, 1000], [0, 20, 50, 300, 1000], [0, 20, 160, 410, 1000], [0, 20, 350, 480, 560, 780, 1000], [0, 50, 70, 600, 1000], [0, 150, 160, 640, 890, 1000], [0, 40, 50, 420, 870, 1000], [0, 90, 140, 160, 170, 1000], [0, 40, 1000], [0, 20, 190, 1000], [0, 20, 30, 60, 1000], [0, 50, 60, 1000], [0, 100, 210, 240, 440, 780, 1000], [0, 20, 30, 620, 800, 1000], [0, 20, 30, 40, 110, 750, 1000]]"
        self.expected_all_est_objectives = "[[72.4911062284239, 69.57034983565977, 55.01381796625596, 50.78505218148534, 47.60439421153544, 45.12652173261736, 45.12652173261736], [72.64117621579616, 57.80677970323161, 49.94824303532387, 49.63654910064693, 48.91841750415076, 48.91841750415076], [72.72774151770889, 59.52506493995998, 55.52388854241122, 54.399243765169494, 53.298820525804075, 50.64659913321464, 50.64659913321464], [72.50861580800658, 50.59981601207795, 49.58102619153274, 46.556667424060485, 46.556667424060485], [72.54634030480123, 70.82067379040747, 63.7243991504278, 55.807699350008114, 55.64499322523014, 53.44118985591325, 50.28403778001126, 47.464369973424255, 47.464369973424255], [72.59953486291201, 61.49289196191016, 59.46794435318925, 54.95796084036429, 52.3830749849172, 47.23272463374439, 47.23272463374439], [72.42808293921915, 69.3284406551505, 56.60606152637405, 55.01629576741032, 46.60491022097211, 46.60491022097211], [72.64581320272977, 58.40002311404187, 50.64106652788252, 48.63992806740216, 47.03614893289617, 47.03614893289617], [72.53636458538912, 48.94828258981628, 47.85997353070289, 47.85997353070289], [72.5003673807578, 62.627246645457326, 58.100956747449096, 55.736507771453255, 54.725401586995865, 49.54212521241625, 47.95268303308411, 47.41682786429558, 47.41682786429558], [72.52531689242514, 60.186076674469376, 51.751529635634796, 48.63123251960833, 48.63123251960833], [72.52730561873884, 53.358974865427015, 49.65384615237086, 49.692509456561154, 49.692509456561154], [72.6823185039676, 55.44582883559346, 53.396286904216495, 49.34077566623166, 48.507559874124865, 46.83515870816628, 46.83515870816628], [72.63913712576303, 69.24810684109376, 48.98208574765193, 47.42142672590316, 47.42142672590316], [72.5818418306719, 62.89003624526742, 53.0228363859162, 51.15533550090105, 48.408579519922625, 48.408579519922625], [72.49736763763313, 64.54528906194217, 49.030435355576216, 47.88683931784642, 47.57691355197318, 47.42887523872495], [72.53263990301983, 62.45217055197052, 62.35096162434098, 62.13659078799379, 51.98969757198311, 51.98969757198311], [72.69806912632215, 50.44681816118905, 50.44681816118905], [72.640166220944, 48.49906690725943, 47.47156586372659, 47.47156586372659], [72.75928505991975, 70.97755803464716, 55.58630791580454, 48.46213022312636, 48.46213022312636], [72.49878639007633, 67.11841785765066, 48.48228830738762, 48.48228830738762], [72.44178159726776, 62.44308751892562, 57.91811380867846, 50.0258960103505, 49.59597845864843, 48.90358140818853, 48.90358140818853], [72.58737882154891, 69.22757299634019, 47.68264599268375, 47.48144666618172, 47.41676565431444, 47.41676565431444], [72.53322191806315, 68.9220387349404, 65.8839487171269, 55.528455111179916, 52.08235749426336, 51.483656344237446, 51.483656344237446]]"
        self.expected_objective_curves = "[([0, 20, 40, 120, 150, 370, 1000], [72.52547100786995, 69.57034983565977, 55.01381796625596, 50.78505218148534, 47.60439421153544, 45.1104722136117, 45.1104722136117]), ([0, 40, 60, 850, 890, 1000], [72.52547100786995, 57.80677970323161, 49.94824303532387, 49.63654910064693, 48.91841750415076, 48.91841750415076]), ([0, 40, 180, 190, 290, 480, 1000], [72.52547100786995, 59.52506493995998, 55.52388854241122, 54.399243765169494, 53.298820525804075, 50.64659913321464, 50.64659913321464]), ([0, 30, 250, 270, 1000], [72.52547100786995, 50.59981601207795, 49.58102619153274, 46.556667424060485, 46.556667424060485]), ([0, 40, 60, 120, 260, 270, 560, 760, 1000], [72.52547100786995, 70.82067379040747, 63.7243991504278, 55.807699350008114, 55.64499322523014, 53.44118985591325, 50.28403778001126, 47.464369973424255, 47.464369973424255]), ([0, 40, 70, 220, 450, 460, 1000], [72.52547100786995, 61.49289196191016, 59.46794435318925, 54.95796084036429, 52.3830749849172, 47.23272463374439, 47.23272463374439]), ([0, 40, 60, 100, 150, 1000], [72.52547100786995, 69.3284406551505, 56.60606152637405, 55.01629576741032, 46.60491022097211, 46.60491022097211]), ([0, 30, 100, 370, 470, 1000], [72.52547100786995, 58.40002311404187, 50.64106652788252, 48.63992806740216, 47.03614893289617, 47.03614893289617]), ([0, 20, 800, 1000], [72.52547100786995, 48.94828258981628, 47.85997353070289, 47.85997353070289]), ([0, 20, 70, 180, 200, 210, 530, 720, 1000], [72.52547100786995, 62.627246645457326, 58.100956747449096, 55.736507771453255, 54.725401586995865, 49.54212521241625, 47.95268303308411, 47.41682786429558, 47.41682786429558]), ([0, 20, 50, 300, 1000], [72.52547100786995, 60.186076674469376, 51.751529635634796, 48.63123251960833, 48.63123251960833]), ([0, 20, 160, 410, 1000], [72.52547100786995, 53.358974865427015, 49.65384615237086, 49.692509456561154, 49.692509456561154]), ([0, 20, 350, 480, 560, 780, 1000], [72.52547100786995, 55.44582883559346, 53.396286904216495, 49.34077566623166, 48.507559874124865, 46.83515870816628, 46.83515870816628]), ([0, 50, 70, 600, 1000], [72.52547100786995, 69.24810684109376, 48.98208574765193, 47.42142672590316, 47.42142672590316]), ([0, 150, 160, 640, 890, 1000], [72.52547100786995, 62.89003624526742, 53.0228363859162, 51.15533550090105, 48.408579519922625, 48.408579519922625]), ([0, 40, 50, 420, 870, 1000], [72.52547100786995, 64.54528906194217, 49.030435355576216, 47.88683931784642, 47.57691355197318, 47.42887523872495]), ([0, 90, 140, 160, 170, 1000], [72.52547100786995, 62.45217055197052, 62.35096162434098, 62.13659078799379, 51.98969757198311, 51.98969757198311]), ([0, 40, 1000], [72.52547100786995, 50.44681816118905, 50.44681816118905]), ([0, 20, 190, 1000], [72.52547100786995, 48.49906690725943, 47.47156586372659, 47.47156586372659]), ([0, 20, 30, 60, 1000], [72.52547100786995, 70.97755803464716, 55.58630791580454, 48.46213022312636, 48.46213022312636]), ([0, 50, 60, 1000], [72.52547100786995, 67.11841785765066, 48.48228830738762, 48.48228830738762]), ([0, 100, 210, 240, 440, 780, 1000], [72.52547100786995, 62.44308751892562, 57.91811380867846, 50.0258960103505, 49.59597845864843, 48.90358140818853, 48.90358140818853]), ([0, 20, 30, 620, 800, 1000], [72.52547100786995, 69.22757299634019, 47.68264599268375, 47.48144666618172, 47.41676565431444, 47.41676565431444]), ([0, 20, 30, 40, 110, 750, 1000], [72.52547100786995, 68.9220387349404, 65.8839487171269, 55.528455111179916, 52.08235749426336, 51.483656344237446, 51.483656344237446])]"
        self.expected_progress_curves = "[([0.0, 0.02, 0.04, 0.12, 0.15, 0.37, 1.0], [1.0, 0.8922078678759928, 0.361238234112868, 0.20698815310771101, 0.09096925433555243, 0.0, 0.0]), ([0.0, 0.04, 0.06, 0.85, 0.89, 1.0], [1.0, 0.46311537654632334, 0.1764643820712253, 0.16509491468528403, 0.1389000714213633, 0.1389000714213633]), ([0.0, 0.04, 0.18, 0.19, 0.29, 0.48, 1.0], [1.0, 0.5257922071974429, 0.3798437638808318, 0.3388207900816402, 0.2986813303784397, 0.2019378866710882, 0.2019378866710882]), ([0.0, 0.03, 0.25, 0.27, 1.0], [1.0, 0.20023140761968333, 0.1630696397789865, 0.05275197060200759, 0.05275197060200759]), ([0.0, 0.04, 0.06, 0.12, 0.26, 0.27, 0.56, 0.76, 1.0], [1.0, 0.93781516350752, 0.6789687308217051, 0.39019615564005866, 0.38426122469225743, 0.30387444861191537, 0.18871295983726624, 0.08586167657631086, 0.08586167657631086]), ([0.0, 0.04, 0.07, 0.22, 0.45, 0.46, 1.0], [1.0, 0.5975714196175549, 0.5237086547887996, 0.3592007681873408, 0.26527824516368975, 0.07741209241187966, 0.07741209241187966]), ([0.0, 0.04, 0.06, 0.1, 0.15, 1.0], [1.0, 0.8833838959209062, 0.41931752027543273, 0.36132861533713717, 0.054511693346249585, 0.054511693346249585]), ([0.0, 0.03, 0.1, 0.37, 0.47, 1.0], [1.0, 0.4847547504986034, 0.20173607723919137, 0.12874178402406722, 0.07024172183030619, 0.07024172183030619]), ([0.0, 0.02, 0.8, 1.0], [1.0, 0.1399894417288234, 0.10029186350601037, 0.10029186350601037]), ([0.0, 0.02, 0.07, 0.18, 0.2, 0.21, 0.53, 0.72, 1.0], [1.0, 0.6389485756794674, 0.4738458911243177, 0.387599344343836, 0.3507178477570423, 0.16165067275993128, 0.10367357083625602, 0.08412751238810616, 0.08412751238810616]), ([0.0, 0.02, 0.05, 0.3, 1.0], [1.0, 0.5499035244902175, 0.24224175502841847, 0.12842460189106458, 0.12842460189106458]), ([0.0, 0.02, 0.16, 0.41, 1.0], [1.0, 0.3008755431184942, 0.1657258485712834, 0.16713614606866617, 0.16713614606866617]), ([0.0, 0.02, 0.35, 0.48, 0.56, 0.78, 1.0], [1.0, 0.37699642810658734, 0.3022365513413833, 0.1543061695667827, 0.12391347108957906, 0.06291032538421258, 0.06291032538421258]), ([0.0, 0.05, 0.07, 0.6, 1.0], [1.0, 0.8804536089396947, 0.14122245866562258, 0.0842952622261454, 0.0842952622261454]), ([0.0, 0.15, 0.16, 0.64, 0.89, 1.0], [1.0, 0.6485341898092455, 0.288614427149333, 0.22049474933974383, 0.12030302576565038, 0.12030302576565038]), ([0.0, 0.04, 0.05, 0.42, 0.87, 1.0], [1.0, 0.7089118257557926, 0.14298607748928685, 0.10127183025140968, 0.08996685926822087, 0.08456695703370998]), ([0.0, 0.09, 0.14, 0.16, 0.17, 1.0], [1.0, 0.6325624330135237, 0.6288706973913892, 0.6210512246291985, 0.2509292599280427, 0.2509292599280427]), ([0.0, 0.04, 1.0], [1.0, 0.19465059938995818, 0.19465059938995818]), ([0.0, 0.02, 0.19, 1.0], [1.0, 0.12360367837613889, 0.08612415662806418, 0.08612415662806418]), ([0.0, 0.02, 0.03, 0.06, 1.0], [1.0, 0.9435377333101694, 0.382120596860536, 0.12225636173351294, 0.12225636173351294]), ([0.0, 0.05, 0.06, 1.0], [1.0, 0.8027702575952058, 0.1229916557385407, 0.1229916557385407]), ([0.0, 0.1, 0.21, 0.24, 0.44, 0.78, 1.0], [1.0, 0.6322311168200392, 0.4671764420339559, 0.17929688174081837, 0.16361504440321797, 0.138358904300636, 0.138358904300636]), ([0.0, 0.02, 0.03, 0.62, 0.8, 1.0], [1.0, 0.8797046085509779, 0.09382359628667078, 0.08648457256421947, 0.08412524319299858, 0.08412524319299858]), ([0.0, 0.02, 0.03, 0.04, 0.11, 0.75, 1.0], [1.0, 0.8685598237675557, 0.7577412882420391, 0.3800103357929069, 0.2543091587555292, 0.2324707062165012, 0.2324707062165012])]"

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
