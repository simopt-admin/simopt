import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class test_CNTNEWS1_RNDSRCH(unittest.TestCase):
    def setUp(self):
        # Expected values
        self.expected_problem_name = "CNTNEWS-1"
        self.expected_solver_name = "RNDSRCH"
        self.expected_all_recommended_xs = "[[(0,), (0.08637653581125314,), (0.16201092394781372,), (0.220031799278404,), (0.18069759039261452,), (0.18069759039261452,)], [(0,), (0.3422671900811551,), (0.25411599712618166,), (0.19694301248184054,), (0.24037175375360612,), (0.23334625077738846,)], [(0,), (0.16067182294009807,), (0.27075075921979186,), (0.21701449798421404,), (0.21701449798421404,)], [(0,), (0.07150266276670732,), (0.16380950044050657,), (0.17407003882377992,), (0.1922372233210708,), (0.1922372233210708,)], [(0,), (0.23124098139956537,), (0.12808012265783883,), (0.15379949965266126,), (0.14781215577002094,), (0.14781215577002094,)], [(0,), (0.1422314185632828,), (0.14837586376590187,), (0.3022077181845406,), (0.20663001343178547,), (0.2576797005148086,), (0.2561104719278581,), (0.2561104719278581,)], [(0,), (0.09590415802387352,), (0.1800190952064003,), (0.1800190952064003,)], [(0,), (0.041719078922958744,), (0.054813625384861384,), (0.24350561539939186,), (0.22592668331808533,), (0.22592668331808533,)], [(0,), (0.05706434183349318,), (0.22615291294721201,), (0.2623222684966682,), (0.2623222684966682,)], [(0,), (0.15178070996172968,), (0.16594898609478428,), (0.16594898609478428,)], [(0,), (0.26801032534156105,), (0.1320835969561267,), (0.1320835969561267,)], [(0,), (0.4347332719449321,), (0.07749316528945176,), (0.1357764959384112,), (0.2681111900412129,), (0.18020954847198192,), (0.1811155946860015,), (0.19026299204625086,), (0.20923856633864776,), (0.20923856633864776,)], [(0,), (0.2854238437886837,), (0.2430806979666771,), (0.21587700288277115,), (0.21587700288277115,)], [(0,), (0.12231822646828096,), (0.18239194965952749,), (0.18239194965952749,)], [(0,), (0.11608216330903494,), (0.1520718655004388,), (0.18788823804974564,), (0.18742208209284925,), (0.18742208209284925,)], [(0,), (0.16048220116572243,), (0.28945645279948956,), (0.28627570825959686,), (0.21990832235287922,), (0.21990832235287922,)], [(0,), (0.1480636467703334,), (0.1480636467703334,)], [(0,), (0.39830288151022675,), (0.35570266212272883,), (0.17516569668062643,), (0.21326343004280876,), (0.2005754419413203,), (0.2005754419413203,)], [(0,), (0.09632424554331415,), (0.2662064497705015,), (0.23978763996728125,), (0.22088913370693503,), (0.19462560181805486,), (0.1755313728056026,), (0.1755313728056026,)], [(0,), (0.06555892378312193,), (0.17490344169619815,), (0.1376656261376925,), (0.13888739861470759,), (0.1535926496324463,), (0.15627721358949997,), (0.15627721358949997,)], [(0,), (0.055175769247415234,), (0.2137060816799074,), (0.2137060816799074,)], [(0,), (0.32147137162551165,), (0.2381538091903597,), (0.17307844565487004,), (0.17307844565487004,)], [(0,), (0.040387957898183315,), (0.2006257252064586,), (0.1463770375597288,), (0.1870836744291869,), (0.16333317436662395,), (0.16333317436662395,)], [(0,), (0.3977161084859951,), (0.3955960006182661,), (0.30188186074966916,), (0.1711392346384829,), (0.19956082700127895,), (0.19956082700127895,)]]"
        self.expected_all_intermediate_budgets = "[[0, 20, 80, 520, 940, 1000], [0, 30, 40, 370, 700, 1000], [0, 20, 120, 230, 1000], [0, 20, 60, 130, 290, 1000], [0, 30, 60, 620, 640, 1000], [0, 20, 50, 70, 160, 510, 940, 1000], [0, 60, 120, 1000], [0, 70, 80, 90, 130, 1000], [0, 50, 70, 840, 1000], [0, 30, 340, 1000], [0, 20, 40, 1000], [0, 40, 80, 90, 170, 180, 290, 480, 650, 1000], [0, 20, 90, 310, 1000], [0, 30, 180, 1000], [0, 40, 80, 350, 420, 1000], [0, 20, 130, 290, 780, 1000], [0, 40, 1000], [0, 80, 110, 120, 260, 490, 1000], [0, 100, 120, 280, 360, 510, 690, 1000], [0, 40, 50, 100, 270, 300, 520, 1000], [0, 70, 100, 1000], [0, 20, 40, 270, 1000], [0, 70, 160, 280, 830, 840, 1000], [0, 50, 80, 120, 130, 520, 1000]]"
        self.expected_all_est_objectives = "[[0.0, 0.3152152013202121, 0.4439426039369449, 0.42568320017429656, 0.44830230926972503, 0.44830230926972503], [0.0, 0.23176170058690984, 0.4460923882748864, 0.5055759457041912, 0.4655738923120514, 0.4742936269626158], [0.0, 0.43898245555866006, 0.33349582987942683, 0.42509600051400326, 0.42509600051400326], [0.0, 0.27316144251245833, 0.4856172195128214, 0.49446071587216717, 0.5000004738655908, 0.5000004738655908], [0.0, 0.3650021020471091, 0.38761217452342295, 0.41162594195133495, 0.40859229991965795, 0.40859229991965795], [0.0, 0.44649448526660557, 0.45489213464277095, 0.32355753197282283, 0.4837049698664353, 0.4280386926410918, 0.4308645786107479, 0.4308645786107479], [0.0, 0.34673483336643507, 0.4748257480816718, 0.4748257480816718], [0.0, 0.1631811912751526, 0.21181775198412514, 0.3929077012672134, 0.4169803668262306, 0.4169803668262306], [0.0, 0.21515166984373782, 0.4121027610273431, 0.34835283694400165, 0.34835283694400165], [0.0, 0.4149320032019911, 0.42616525220425006, 0.42616525220425006], [0.0, 0.43040647836752777, 0.43237804276441616, 0.43237804276441616], [0.0, -0.21027440964985755, 0.2796299716534581, 0.40824369033760627, 0.3286762857413491, 0.43603844106266615, 0.4359297155169838, 0.4337984794382499, 0.4224647210611315, 0.4224647210611315], [0.0, 0.3514791239786761, 0.4371599159571518, 0.46706606483980423, 0.46706606483980423], [0.0, 0.40107040773804664, 0.474990856710755, 0.474990856710755], [0.0, 0.3764910769685843, 0.4267911399654495, 0.44091228738132726, 0.44094913773582595, 0.44094913773582595], [0.0, 0.4419274210180157, 0.3319940881135118, 0.3396617952175178, 0.44642100326557627, 0.44642100326557627], [0.0, 0.42650752232790745, 0.42650752232790745], [0.0, -0.0640787263582251, 0.07967718670274458, 0.42890806762915984, 0.4122160540887609, 0.42205356485352424, 0.42205356485352424], [0.0, 0.3485682103926985, 0.41844098608389074, 0.4573314748853693, 0.4773975702684771, 0.4890880826018397, 0.48369036901317897, 0.48369036901317897], [0.0, 0.2557348732947487, 0.49250965636091126, 0.4426204154564126, 0.44481960591503994, 0.469180237251819, 0.47316385629159624, 0.47316385629159624], [0.0, 0.2127807874728603, 0.4381275900517878, 0.4381275900517878], [0.0, 0.2650834880792731, 0.4364748646815667, 0.47873408130527145, 0.47873408130527145], [0.0, 0.15902170899726037, 0.4631611346729024, 0.4381884892542297, 0.4654810547301422, 0.45447734387883676, 0.45447734387883676], [0.0, -0.01422578819747121, -0.0066945091718134545, 0.2876671798272609, 0.4671906789306053, 0.4655328930386689, 0.4655328930386689]]"
        self.expected_objective_curves = "[([0, 20, 80, 520, 940, 1000], [0.0, 0.3152152013202121, 0.4439426039369449, 0.42568320017429656, 0.44830230926972503, 0.44830230926972503]), ([0, 30, 40, 370, 700, 1000], [0.0, 0.23176170058690984, 0.4460923882748864, 0.5095842940352483, 0.4655738923120514, 0.4742936269626158]), ([0, 20, 120, 230, 1000], [0.0, 0.43898245555866006, 0.33349582987942683, 0.42509600051400326, 0.42509600051400326]), ([0, 20, 60, 130, 290, 1000], [0.0, 0.27316144251245833, 0.4856172195128214, 0.49446071587216717, 0.5000004738655908, 0.5000004738655908]), ([0, 30, 60, 620, 640, 1000], [0.0, 0.3650021020471091, 0.38761217452342295, 0.41162594195133495, 0.40859229991965795, 0.40859229991965795]), ([0, 20, 50, 70, 160, 510, 940, 1000], [0.0, 0.44649448526660557, 0.45489213464277095, 0.32355753197282283, 0.4837049698664353, 0.4280386926410918, 0.4308645786107479, 0.4308645786107479]), ([0, 60, 120, 1000], [0.0, 0.34673483336643507, 0.4748257480816718, 0.4748257480816718]), ([0, 70, 80, 90, 130, 1000], [0.0, 0.1631811912751526, 0.21181775198412514, 0.3929077012672134, 0.4169803668262306, 0.4169803668262306]), ([0, 50, 70, 840, 1000], [0.0, 0.21515166984373782, 0.4121027610273431, 0.34835283694400165, 0.34835283694400165]), ([0, 30, 340, 1000], [0.0, 0.4149320032019911, 0.42616525220425006, 0.42616525220425006]), ([0, 20, 40, 1000], [0.0, 0.43040647836752777, 0.43237804276441616, 0.43237804276441616]), ([0, 40, 80, 90, 170, 180, 290, 480, 650, 1000], [0.0, -0.21027440964985755, 0.2796299716534581, 0.40824369033760627, 0.3286762857413491, 0.43603844106266615, 0.4359297155169838, 0.4337984794382499, 0.4224647210611315, 0.4224647210611315]), ([0, 20, 90, 310, 1000], [0.0, 0.3514791239786761, 0.4371599159571518, 0.46706606483980423, 0.46706606483980423]), ([0, 30, 180, 1000], [0.0, 0.40107040773804664, 0.474990856710755, 0.474990856710755]), ([0, 40, 80, 350, 420, 1000], [0.0, 0.3764910769685843, 0.4267911399654495, 0.44091228738132726, 0.44094913773582595, 0.44094913773582595]), ([0, 20, 130, 290, 780, 1000], [0.0, 0.4419274210180157, 0.3319940881135118, 0.3396617952175178, 0.44642100326557627, 0.44642100326557627]), ([0, 40, 1000], [0.0, 0.42650752232790745, 0.42650752232790745]), ([0, 80, 110, 120, 260, 490, 1000], [0.0, -0.0640787263582251, 0.07967718670274458, 0.42890806762915984, 0.4122160540887609, 0.42205356485352424, 0.42205356485352424]), ([0, 100, 120, 280, 360, 510, 690, 1000], [0.0, 0.3485682103926985, 0.41844098608389074, 0.4573314748853693, 0.4773975702684771, 0.4890880826018397, 0.48369036901317897, 0.48369036901317897]), ([0, 40, 50, 100, 270, 300, 520, 1000], [0.0, 0.2557348732947487, 0.49250965636091126, 0.4426204154564126, 0.44481960591503994, 0.469180237251819, 0.47316385629159624, 0.47316385629159624]), ([0, 70, 100, 1000], [0.0, 0.2127807874728603, 0.4381275900517878, 0.4381275900517878]), ([0, 20, 40, 270, 1000], [0.0, 0.2650834880792731, 0.4364748646815667, 0.47873408130527145, 0.47873408130527145]), ([0, 70, 160, 280, 830, 840, 1000], [0.0, 0.15902170899726037, 0.4631611346729024, 0.4381884892542297, 0.4654810547301422, 0.45447734387883676, 0.45447734387883676]), ([0, 50, 80, 120, 130, 520, 1000], [0.0, -0.01422578819747121, -0.0066945091718134545, 0.2876671798272609, 0.4671906789306053, 0.4655328930386689, 0.4655328930386689])]"
        self.expected_progress_curves = "[([0.0, 0.02, 0.08, 0.52, 0.94, 1.0], [1.0, 0.38142677274427844, 0.12881419397467322, 0.16464615342942307, 0.12025877854328915, 0.12025877854328915]), ([0.0, 0.03, 0.04, 0.37, 0.7, 1.0], [1.0, 0.5451945766388187, 0.12459549186178434, -0.0, 0.08636530253845047, 0.0692538358927354]), ([0.0, 0.02, 0.12, 0.23, 1.0], [1.0, 0.13854790915456405, 0.34555316209106185, 0.1657984645723813, 0.1657984645723813]), ([0.0, 0.02, 0.06, 0.13, 0.29, 1.0], [1.0, 0.463952390782351, 0.04703260049998537, 0.029678265872996122, 0.018807134132345554, 0.018807134132345554]), ([0.0, 0.03, 0.06, 0.62, 0.64, 1.0], [1.0, 0.2837257617247881, 0.2393561201542614, 0.19223189024961884, 0.19818506044576928, 0.19818506044576928]), ([0.0, 0.02, 0.05, 0.07, 0.16, 0.51, 0.94, 1.0], [1.0, 0.12380642321028593, 0.10732701151243537, 0.36505591761734685, 0.05078516836514375, 0.16002377300214818, 0.15447829995140178, 0.15447829995140178]), ([0.0, 0.06, 0.12, 1.0], [1.0, 0.31957315516782553, 0.06820960999079036, 0.06820960999079036]), ([0.0, 0.07, 0.08, 0.09, 0.13, 1.0], [1.0, 0.6797758620404708, 0.5843322597193831, 0.22896426387891045, 0.18172445323170075, 0.18172445323170075]), ([0.0, 0.05, 0.07, 0.84, 1.0], [1.0, 0.5777898330813633, 0.19129618818503524, 0.31639801104249526, 0.31639801104249526]), ([0.0, 0.03, 0.34, 1.0], [1.0, 0.18574412897174192, 0.16370018230041464, 0.16370018230041464]), ([0.0, 0.02, 0.04, 1.0], [1.0, 0.1553772684804209, 0.15150830230550968, 0.15150830230550968]), ([0.0, 0.04, 0.08, 0.09, 0.17, 0.18, 0.29, 0.48, 0.65, 1.0], [1.0, 1.4126391101749944, 0.4512586535209896, 0.1988691662671853, 0.35501095777764624, 0.1443251957202098, 0.14453855697752288, 0.14872086028569054, 0.17096204493321904, 0.17096204493321904]), ([0.0, 0.02, 0.09, 0.31, 1.0], [1.0, 0.31026303578664843, 0.14212443147450468, 0.08343708723586181, 0.08343708723586181]), ([0.0, 0.03, 0.18, 1.0], [1.0, 0.21294590034930644, 0.06788560347996207, 0.06788560347996207]), ([0.0, 0.04, 0.08, 0.35, 0.42, 1.0], [1.0, 0.2611799826339582, 0.16247195025220293, 0.13476083831023833, 0.13468852376889545, 0.13468852376889545]), ([0.0, 0.02, 0.13, 0.29, 0.78, 1.0], [1.0, 0.1327687564337545, 0.34850015591228656, 0.33345317115675627, 0.12395062310398244, 0.12395062310398244]), ([0.0, 0.04, 1.0], [1.0, 0.1630285169299083, 0.1630285169299083]), ([0.0, 0.08, 0.11, 0.12, 0.26, 0.49, 1.0], [1.0, 1.1257470591387433, 0.8436427738543424, 0.15831772554691026, 0.1910738637085082, 0.17176889124386846, 0.17176889124386846]), ([0.0, 0.1, 0.12, 0.28, 0.36, 0.51, 0.69, 1.0], [1.0, 0.3159753656603322, 0.1788581575574484, 0.10254008956222783, 0.06316270760994999, 0.040221434752443205, 0.05081382084409028, 0.05081382084409028]), ([0.0, 0.04, 0.05, 0.1, 0.27, 0.3, 0.52, 1.0], [1.0, 0.49815000915813284, 0.03350699359104653, 0.13140883532451206, 0.12709317943721504, 0.07928826939205962, 0.0714708796365157, 0.0714708796365157]), ([0.0, 0.07, 0.1, 1.0], [1.0, 0.5824424144081997, 0.14022548343791338, 0.14022548343791338]), ([0.0, 0.02, 0.04, 0.27, 1.0], [1.0, 0.4798044382801619, 0.1434687650491531, 0.06053995990669782, 0.06053995990669782]), ([0.0, 0.07, 0.16, 0.28, 0.83, 0.84, 1.0], [1.0, 0.6879383629781558, 0.0911000592163753, 0.14010597582522843, 0.0865474855118974, 0.10814099021780242, 0.10814099021780242]), ([0.0, 0.05, 0.08, 0.12, 0.13, 0.52, 1.0], [1.0, 1.027916457324109, 1.0131371968292069, 0.4354865658254318, 0.08319254655385948, 0.0864457588512968, 0.0864457588512968])]"

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
