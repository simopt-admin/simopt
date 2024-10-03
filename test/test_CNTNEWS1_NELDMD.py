import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class test_CNTNEWS1_NELDMD(unittest.TestCase):
    def setUp(self):
        # Expected values
        self.expected_problem_name = "CNTNEWS-1"
        self.expected_solver_name = "NELDMD"
        self.expected_all_recommended_xs = "[[(0,), (0.2591296074337594,), (0.17275307162250628,), (0.21594133952813283,), (0.19434720557531954,), (0.19434720557531954,)], [(0,), (0.1336908966858528,), (0.16711362085731601,), (0.1712914613787489,), (0.1712914613787489,)], [(0,), (0.2410077344101471,), (0.22092375654263485,), (0.23096574547639098,), (0.23096574547639098,)], [(0,), (0.21450798830012197,), (0.14300532553341463,), (0.1787566569167683,), (0.16088099122509147,), (0.15976376211936166,), (0.15976376211936166,)], [(0,), (0.12036491544512594,), (0.1504561443064074,), (0.14669474069874722,), (0.14857544250257732,), (0.14834035477709856,), (0.14834035477709856,)], [(0,), (0.14667615039338538,), (0.1444537844783341,), (0.1444537844783341,)], [(0,), (0.13570434398518863,), (0.18659347297963438,), (0.18871385335440294,), (0.18871385335440294,)], [(0,), (0.19332816548160742,), (0.24166020685200926,), (0.21749418616680832,), (0.22957719650940878,), (0.22655644392375868,), (0.22806682021658373,), (0.22806682021658373,)], [(0,), (0.23109768282550885,), (0.23880093891969248,), (0.23880093891969248,)], [(0,), (0.11334357260860685,), (0.1558474123368344,), (0.1558474123368344,)], [(0,), (0.13400516267078053,), (0.16750645333847564,), (0.15075580800462807,), (0.15180272333799355,), (0.1515409945046522,), (0.15150827840048453,), (0.15150827840048453,)], [(0,), (0.18439096856582324,), (0.20743983963655116,), (0.20167762186886917,), (0.20167762186886917,)], [(0,), (0.21406788284151276,), (0.22298737795990914,), (0.21852763040071094,), (0.21852763040071094,)], [(0,), (0.15721039155220914,), (0.1670360410242222,), (0.1670360410242222,)], [(0,), (0.11615654731285296,), (0.1451956841410662,), (0.15245546834811952,), (0.14882557624459286,), (0.14973304927047454,), (0.14973304927047454,)], [(0,), (0.24072330174858364,), (0.20060275145715303,), (0.2106328890300107,), (0.20561782024358188,), (0.20561782024358188,)], [(0,), (0.15964364474241044,), (0.19955455592801305,), (0.17959910033521176,), (0.17928729634157425,), (0.17928729634157425,)], [(0,), (0.24133464968795887,), (0.24348942334588708,), (0.24295072993140504,), (0.24298439826981016,), (0.24298439826981016,)], [(0,), (0.17391053174240698,), (0.2173881646780087,), (0.20651875644410828,), (0.2119534605610585,), (0.2092361085025834,), (0.21059478453182096,), (0.20991544651720218,), (0.21025511552451157,), (0.21025511552451157,)], [(0,), (0.18609766814772227,), (0.18682461216392432,), (0.1866428761598738,), (0.1866428761598738,)], [(0,), (0.15034118281966968,), (0.1879264785245871,), (0.1691338306721284,), (0.17853015459835775,), (0.17383199263524307,), (0.17383199263524307,)], [(0,), (0.16073568581275582,), (0.20091960726594477,), (0.1808276465393503,), (0.1808276465393503,)], [(0,), (0.1800248871618857,), (0.17252385019680713,), (0.17627436867934643,), (0.1753367390587116,), (0.1753367390587116,)], [(0,), (0.1316041953442084,), (0.18095576859828652,), (0.18918103080729953,), (0.18918103080729953,)]]"
        self.expected_all_intermediate_budgets = "[[0, 120, 180, 240, 300, 1000], [0, 150, 240, 420, 1000], [0, 120, 240, 300, 1000], [0, 120, 180, 240, 300, 600, 1000], [0, 210, 300, 480, 540, 720, 1000], [0, 360, 420, 1000], [0, 210, 360, 540, 1000], [0, 180, 270, 330, 390, 510, 570, 1000], [0, 360, 420, 1000], [0, 180, 330, 1000], [0, 120, 210, 270, 510, 630, 960, 1000], [0, 150, 300, 420, 1000], [0, 180, 360, 420, 1000], [0, 150, 360, 1000], [0, 180, 270, 390, 450, 570, 1000], [0, 120, 180, 300, 360, 1000], [0, 150, 240, 300, 660, 1000], [0, 270, 510, 630, 870, 1000], [0, 180, 270, 390, 450, 510, 570, 630, 690, 1000], [0, 210, 720, 900, 1000], [0, 240, 330, 390, 450, 510, 1000], [0, 120, 210, 270, 1000], [0, 210, 390, 450, 630, 1000], [0, 210, 360, 420, 1000]]"
        self.expected_all_est_objectives = "[[0.0, 0.37576344253154204, 0.447496245826092, 0.4292835494593317, 0.44442091867189343, 0.44442091867189343], [0.0, 0.44776173256248425, 0.4952379274549976, 0.4982748554627797, 0.4982748554627797], [0.0, 0.39263560106052237, 0.42046578871462864, 0.40732629247790597, 0.40732629247790597], [0.0, 0.49279438759414335, 0.45952166948816253, 0.49680205193391586, 0.4826426505476509, 0.4814132709661594, 0.4814132709661594], [0.0, 0.3751008886366243, 0.4101673956500514, 0.40785311040869887, 0.40908080342849407, 0.40893034728418765, 0.40893034728418765], [0.0, 0.4526584724149607, 0.44968631833442196, 0.44968631833442196], [0.0, 0.4348815867197554, 0.4756846361605565, 0.47567363950421077, 0.47567363950421077], [0.0, 0.44902713797570704, 0.395712722259235, 0.42642476363566095, 0.41275720564514384, 0.41627503494787654, 0.41455647456682215, 0.41455647456682215], [0.0, 0.40512701785430333, 0.3931908125920347, 0.3931908125920347], [0.0, 0.35652152726201053, 0.4191837067727468, 0.4191837067727468], [0.0, 0.4363800698337299, 0.48209751373153287, 0.4652910704014944, 0.4666729986415368, 0.46632751658152627, 0.4662843313240251, 0.4662843313240251], [0.0, 0.43548053567603584, 0.4241234804911542, 0.42856770315812864, 0.42856770315812864], [0.0, 0.4683614855002915, 0.4611641473132849, 0.4650835938505466, 0.4650835938505466], [0.0, 0.4566227292843412, 0.4652747524643408, 0.4652747524643408], [0.0, 0.37663389425591487, 0.42001397490761005, 0.4271133663575014, 0.42377876651086877, 0.4246644314301174, 0.4246644314301174], [0.0, 0.4269514655817528, 0.45226412199207106, 0.4503444883549202, 0.45162354350774675, 0.45162354350774675], [0.0, 0.4329181650035738, 0.42850274075445915, 0.43578379198404915, 0.43580873630354033, 0.43580873630354033], [0.0, 0.37595446251132003, 0.372300068300626, 0.3732266209735351, 0.37316871143147834, 0.37316871143147834], [0.0, 0.48275162412160033, 0.4801407762347374, 0.48643917933872316, 0.4838381870853607, 0.4852415070953515, 0.48458702878095905, 0.48491542484833444, 0.4847523837248261, 0.4847523837248261], [0.0, 0.49630017703831764, 0.4963874103202619, 0.4963656019997758, 0.4963656019997758], [0.0, 0.42858296320987443, 0.4505366113242457, 0.4442698739732879, 0.4494779538291492, 0.44727039469866897, 0.44727039469866897], [0.0, 0.4752094397034982, 0.4699118887399473, 0.47835638833184163, 0.47835638833184163], [0.0, 0.46394062212980514, 0.4602248419563823, 0.46214878966613837, 0.4616875441645727, 0.4616875441645727], [0.0, 0.4273436774177678, 0.46979541517360046, 0.4693365950414783, 0.4693365950414783]]"
        self.expected_objective_curves = "[([0, 120, 180, 240, 300, 1000], [0.0, 0.37576344253154204, 0.447496245826092, 0.4292835494593317, 0.44442091867189343, 0.44442091867189343]), ([0, 150, 240, 420, 1000], [0.0, 0.44776173256248425, 0.4952379274549976, 0.5037389673405301, 0.5037389673405301]), ([0, 120, 240, 300, 1000], [0.0, 0.39263560106052237, 0.42046578871462864, 0.40732629247790597, 0.40732629247790597]), ([0, 120, 180, 240, 300, 600, 1000], [0.0, 0.49279438759414335, 0.45952166948816253, 0.49680205193391586, 0.4826426505476509, 0.4814132709661594, 0.4814132709661594]), ([0, 210, 300, 480, 540, 720, 1000], [0.0, 0.3751008886366243, 0.4101673956500514, 0.40785311040869887, 0.40908080342849407, 0.40893034728418765, 0.40893034728418765]), ([0, 360, 420, 1000], [0.0, 0.4526584724149607, 0.44968631833442196, 0.44968631833442196]), ([0, 210, 360, 540, 1000], [0.0, 0.4348815867197554, 0.4756846361605565, 0.47567363950421077, 0.47567363950421077]), ([0, 180, 270, 330, 390, 510, 570, 1000], [0.0, 0.44902713797570704, 0.395712722259235, 0.42642476363566095, 0.41275720564514384, 0.41627503494787654, 0.41455647456682215, 0.41455647456682215]), ([0, 360, 420, 1000], [0.0, 0.40512701785430333, 0.3931908125920347, 0.3931908125920347]), ([0, 180, 330, 1000], [0.0, 0.35652152726201053, 0.4191837067727468, 0.4191837067727468]), ([0, 120, 210, 270, 510, 630, 960, 1000], [0.0, 0.4363800698337299, 0.48209751373153287, 0.4652910704014944, 0.4666729986415368, 0.46632751658152627, 0.4662843313240251, 0.4662843313240251]), ([0, 150, 300, 420, 1000], [0.0, 0.43548053567603584, 0.4241234804911542, 0.42856770315812864, 0.42856770315812864]), ([0, 180, 360, 420, 1000], [0.0, 0.4683614855002915, 0.4611641473132849, 0.4650835938505466, 0.4650835938505466]), ([0, 150, 360, 1000], [0.0, 0.4566227292843412, 0.4652747524643408, 0.4652747524643408]), ([0, 180, 270, 390, 450, 570, 1000], [0.0, 0.37663389425591487, 0.42001397490761005, 0.4271133663575014, 0.42377876651086877, 0.4246644314301174, 0.4246644314301174]), ([0, 120, 180, 300, 360, 1000], [0.0, 0.4269514655817528, 0.45226412199207106, 0.4503444883549202, 0.45162354350774675, 0.45162354350774675]), ([0, 150, 240, 300, 660, 1000], [0.0, 0.4329181650035738, 0.42850274075445915, 0.43578379198404915, 0.43580873630354033, 0.43580873630354033]), ([0, 270, 510, 630, 870, 1000], [0.0, 0.37595446251132003, 0.372300068300626, 0.3732266209735351, 0.37316871143147834, 0.37316871143147834]), ([0, 180, 270, 390, 450, 510, 570, 630, 690, 1000], [0.0, 0.48275162412160033, 0.4801407762347374, 0.48643917933872316, 0.4838381870853607, 0.4852415070953515, 0.48458702878095905, 0.48491542484833444, 0.4847523837248261, 0.4847523837248261]), ([0, 210, 720, 900, 1000], [0.0, 0.49630017703831764, 0.4963874103202619, 0.4963656019997758, 0.4963656019997758]), ([0, 240, 330, 390, 450, 510, 1000], [0.0, 0.42858296320987443, 0.4505366113242457, 0.4442698739732879, 0.4494779538291492, 0.44727039469866897, 0.44727039469866897]), ([0, 120, 210, 270, 1000], [0.0, 0.4752094397034982, 0.4699118887399473, 0.47835638833184163, 0.47835638833184163]), ([0, 210, 390, 450, 630, 1000], [0.0, 0.46394062212980514, 0.4602248419563823, 0.46214878966613837, 0.4616875441645727, 0.4616875441645727]), ([0, 210, 360, 420, 1000], [0.0, 0.4273436774177678, 0.46979541517360046, 0.4693365950414783, 0.4693365950414783])]"
        self.expected_progress_curves = "[([0.0, 0.12, 0.18, 0.24, 0.3, 1.0], [1.0, 0.25405127080922446, 0.11165052767581068, 0.14780555547307136, 0.11775552918171885, 0.11775552918171885]), ([0.0, 0.15, 0.24, 0.42, 1.0], [1.0, 0.11112349531658326, 0.016875883020155115, -0.0, -0.0]), ([0.0, 0.12, 0.24, 0.3, 1.0], [1.0, 0.22055741859037348, 0.1653101785346068, 0.1913941170198348, 0.1913941170198348]), ([0.0, 0.12, 0.18, 0.24, 0.3, 0.6, 1.0], [1.0, 0.02172668873358809, 0.0877781960879681, 0.013770853271958402, 0.041879461706638324, 0.0443199709012753, 0.0443199709012753]), ([0.0, 0.21, 0.3, 0.48, 0.54, 0.72, 1.0], [1.0, 0.25536654307893925, 0.18575408645570174, 0.1903483016969222, 0.1879111405889048, 0.18820981937704925, 0.18820981937704925]), ([0.0, 0.36, 0.42, 1.0], [1.0, 0.1014027070314748, 0.10730289397994558, 0.10730289397994558]), ([0.0, 0.21, 0.36, 0.54, 1.0], [1.0, 0.13669258303423407, 0.05569219972813571, 0.05571402979699815, 0.05571402979699815]), ([0.0, 0.18, 0.27, 0.33, 0.39, 0.51, 0.57, 1.0], [1.0, 0.10861146925692891, 0.21444885562777766, 0.1534806888437606, 0.1806129118335254, 0.1736294749131995, 0.17704108388625048, 0.17704108388625048]), ([0.0, 0.36, 0.42, 1.0], [1.0, 0.19576001834212803, 0.21945523756506277, 0.21945523756506277]), ([0.0, 0.18, 0.33, 1.0], [1.0, 0.29224945780102785, 0.16785531009083826, 0.16785531009083826]), ([0.0, 0.12, 0.21, 0.27, 0.51, 0.63, 0.96, 1.0], [1.0, 0.1337178615790215, 0.04296164285890462, 0.0763250402128306, 0.07358169826464225, 0.07426753375168918, 0.0743532631875698, 0.0743532631875698]), ([0.0, 0.15, 0.3, 0.42, 1.0], [1.0, 0.13550357643535496, 0.1580490929056029, 0.14922662143702164, 0.14922662143702164]), ([0.0, 0.18, 0.36, 0.42, 1.0], [1.0, 0.07022978989894836, 0.08451762279185446, 0.07673691335427762, 0.07673691335427762]), ([0.0, 0.15, 0.36, 1.0], [1.0, 0.09353304213278799, 0.07635743384963771, 0.07635743384963771]), ([0.0, 0.18, 0.27, 0.39, 0.45, 0.57, 1.0], [1.0, 0.25232328909486884, 0.16620709903572253, 0.15211370561140136, 0.15873340363523608, 0.156975221368884, 0.156975221368884]), ([0.0, 0.12, 0.18, 0.3, 0.36, 1.0], [1.0, 0.1524351037684734, 0.10218555380025192, 0.1059963243810658, 0.10345720147068371, 0.10345720147068371]), ([0.0, 0.15, 0.24, 0.3, 0.66, 1.0], [1.0, 0.14059027974518623, 0.14935558188654266, 0.134901565616906, 0.13485204727286587, 0.13485204727286587]), ([0.0, 0.27, 0.51, 0.63, 0.87, 1.0], [1.0, 0.2536720665146144, 0.26092660596385975, 0.2590872551631846, 0.2592022145882267, 0.2592022145882267]), ([0.0, 0.18, 0.27, 0.39, 0.45, 0.51, 0.57, 0.63, 0.69, 1.0], [1.0, 0.04166313225623909, 0.0468460703573886, 0.03434276306464939, 0.03950613620430205, 0.03672032827405675, 0.03801956926358703, 0.03736765212263376, 0.03769131404692188, 0.03769131404692188]), ([0.0, 0.21, 0.72, 0.9, 1.0], [1.0, 0.014767152800358693, 0.014593981202368546, 0.014637274101866138, 0.014637274101866138]), ([0.0, 0.24, 0.33, 0.39, 0.45, 0.51, 1.0], [1.0, 0.14919632786686893, 0.10561493048108656, 0.11805537634145515, 0.10771652984848448, 0.11209887720218421, 0.11209887720218421]), ([0.0, 0.12, 0.21, 0.27, 1.0], [1.0, 0.0566355384171537, 0.06715199893939422, 0.05038835717374582, 0.05038835717374582]), ([0.0, 0.21, 0.39, 0.45, 0.63, 1.0], [1.0, 0.07900588954005044, 0.08638228964870226, 0.0825629549644838, 0.08347859884250422, 0.08347859884250422]), ([0.0, 0.21, 0.36, 0.42, 1.0], [1.0, 0.15165650242642184, 0.06738321703824764, 0.068294046181652, 0.068294046181652])]"

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
