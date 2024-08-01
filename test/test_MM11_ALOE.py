import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class test_MM11_ALOE(unittest.TestCase):
    def setUp(self):
        # Expected values
        self.expected_problem_name = "MM1-1"
        self.expected_solver_name = "ALOE"
        self.expected_all_recommended_xs = "[[(5,), (4.068363045037877,), (3.1954320291559246,), (2.54789745330792,), (2.7223159323474206,), (2.7223159323474206,)], [(5,), (4.070245430133759,), (3.203625828952615,), (2.6816920968594635,), (2.837137038961862,), (2.837137038961862,)], [(5,), (4.0698290513184405,), (3.20057733045431,), (2.5600315617711913,), (2.7043929379029126,), (2.899527251791042,), (2.899527251791042,)], [(5,), (4.070754629536534,), (3.2062270171567695,), (2.706137838725003,), (2.8410880882508316,), (2.8410880882508316,)], [(5,), (4.0740564892409665,), (3.2126838001722193,), (2.586211377512199,), (2.688745836222323,), (2.688745836222323,)], [(5,), (4.072039930037047,), (3.2059748709238622,), (2.5717788385211637,), (2.7368416976658168,), (2.764129255958074,), (2.764129255958074,)], [(5,), (4.067976139171115,), (3.1950912750175346,), (2.5419313584484087,), (2.6651435290553054,), (2.6651435290553054,)], [(5,), (4.066584889852279,), (3.1910395194526626,), (2.5353142333928718,), (2.6426848084176577,), (2.821478796038285,), (2.821478796038285,)], [(5,), (4.06990648820933,), (3.2017545501170557,), (2.565896832903403,), (2.6966927874536357,), (2.920604432561989,), (2.7790004170607814,), (2.7790004170607814,)], [(5,), (4.066667179797703,), (3.1922253533293192,), (2.538857510529519,), (2.679986733110487,), (2.8405555369814506,), (2.8405555369814506,)], [(5,), (4.069816177289685,), (3.2013513145276913,), (2.55911469283883,), (2.6730008528206466,), (2.6730008528206466,)], [(5,), (4.0722963563718615,), (3.2088176178930947,), (3.120853387483235,), (3.120853387483235,)], [(5,), (4.066018800984976,), (3.1891402237479904,), (2.5345533098291977,), (2.7879086436347125,), (2.7879086436347125,)], [(5,), (4.071382757226621,), (3.206565834672207,), (2.6978590327416763,), (2.6978590327416763,)], [(5,), (4.064887045134899,), (3.186236541645073,), (2.5255543728451606,), (2.6483764613308702,), (2.6483764613308702,)], [(5,), (4.071492014080726,), (3.207033088274213,), (2.7163985241951303,), (2.9972700499194795,), (2.9972700499194795,)], [(5,), (4.068335838174162,), (3.1967021298991236,), (2.5513291272917855,), (2.726632668404334,), (2.726632668404334,)], [(5,), (4.071990577721871,), (3.206565083698218,), (2.571444359266728,), (2.7153513100477187,), (2.767679514471602,), (2.767679514471602,)], [(5,), (4.073411947270552,), (3.2134223081104922,), (2.690369118245548,), (3.0461203124963783,), (3.0461203124963783,)], [(5,), (4.068686955461333,), (3.1990769613685295,), (2.5635217949516624,), (2.729272655436594,), (2.729272655436594,)], [(5,), (4.068045969049112,), (3.1967747028426086,), (2.5519381543498714,), (2.7675910651349334,), (2.7675910651349334,)], [(5,), (4.0735882710755265,), (3.211564265717699,), (2.6587670685408233,), (2.6587670685408233,)], [(5,), (4.069797690036461,), (3.201046713647302,), (2.696408089346639,), (2.794089812947188,), (2.8126693860521943,), (2.8126693860521943,)], [(5,), (4.0706972156245795,), (3.203892299804459,), (2.5695274166276008,), (2.685105033841334,), (2.752728386670604,), (2.752728386670604,)]]"
        self.expected_all_intermediate_budgets = "[[0, 60, 90, 120, 240, 1000], [0, 60, 90, 240, 480, 1000], [0, 60, 90, 120, 240, 480, 1000], [0, 60, 90, 240, 720, 1000], [0, 60, 90, 120, 240, 1000], [0, 60, 90, 120, 240, 960, 1000], [0, 60, 90, 120, 240, 1000], [0, 60, 90, 120, 240, 720, 1000], [0, 60, 90, 120, 240, 480, 720, 1000], [0, 60, 90, 120, 240, 480, 1000], [0, 60, 90, 120, 240, 1000], [0, 60, 90, 840, 1000], [0, 60, 90, 120, 240, 1000], [0, 60, 90, 240, 1000], [0, 60, 90, 120, 240, 1000], [0, 60, 90, 240, 960, 1000], [0, 60, 90, 120, 240, 1000], [0, 60, 90, 120, 240, 960, 1000], [0, 60, 90, 240, 960, 1000], [0, 60, 90, 120, 240, 1000], [0, 60, 90, 120, 240, 1000], [0, 60, 90, 240, 1000], [0, 60, 90, 240, 480, 720, 1000], [0, 60, 90, 120, 240, 720, 1000]]"
        self.expected_all_est_objectives = "[[2.7852321578182706, 2.0434955300959756, 1.6067949860630397, 1.5953185527727127, 1.551094392839577, 1.551094392839577], [2.7857037031168543, 2.0454590444287932, 1.6112733855379604, 1.5591639731106854, 1.5478056708472974, 1.5478056708472974], [2.7866293625352507, 2.04797163822759, 1.6205137429033964, 1.6163253621532803, 1.5759105545236445, 1.5664687724992692, 1.5664687724992692], [2.7889080044387127, 2.051164328901086, 1.6227059425210009, 1.5749298888776202, 1.5649659079370986, 1.5649659079370986], [2.7833651638972787, 2.0438598971642126, 1.608137408927407, 1.5763337097688268, 1.5521300465336114, 1.5521300465336114], [2.787955763524055, 2.050284006246868, 1.6215173623533616, 1.620189560285288, 1.5752097928232494, 1.5716967873961198, 1.5716967873961198], [2.7843462630059106, 2.042523510862577, 1.6072035528477624, 1.5934064648951312, 1.5584508725578103, 1.5584508725578103], [2.7907221687784363, 2.052665196346358, 1.628996218505635, 1.6431966479141764, 1.604964304544481, 1.5790413250028876, 1.5790413250028876], [2.789502875694011, 2.05027100272252, 1.6206037335904169, 1.609361043544037, 1.5736959250822544, 1.565914453421025, 1.56433775547408, 1.56433775547408], [2.7891645344327056, 2.0495789353187406, 1.6225178001355536, 1.637931354858589, 1.5896712431907034, 1.57215273101062, 1.57215273101062], [2.7863020842335002, 2.0462854168869495, 1.612986905052738, 1.5876082570434153, 1.5595124605927324, 1.5595124605927324], [2.781108319206661, 2.0401858799081696, 1.6043242541410145, 1.5797350249115747, 1.5797350249115747], [2.781564747274972, 2.036931781296998, 1.5993311729298625, 1.5825601479928275, 1.537272866398318, 1.537272866398318], [2.7819442310007103, 2.039725562437248, 1.5994678666963085, 1.5370194844154883, 1.5370194844154883], [2.784695397913865, 2.041025631877137, 1.6042400211713106, 1.6109631240563056, 1.5679869223098604, 1.5679869223098604], [2.782112928233372, 2.040521496398947, 1.6029140553101422, 1.5427593360165515, 1.5528079689590175, 1.5528079689590175], [2.784512429482461, 2.0437980957523743, 1.6093454050160851, 1.6058116849244488, 1.5580059210112633, 1.5580059210112633], [2.783456075233837, 2.0427195553294974, 1.605910359115762, 1.5819525560751315, 1.548106920588509, 1.5432967245164224, 1.5432967245164224], [2.7872953386099404, 2.050253708974358, 1.6189174807929039, 1.5672082530048066, 1.5767889956687597, 1.5767889956687597], [2.7844968268172887, 2.0415512366627646, 1.6060292308003838, 1.58042416156321, 1.546154338151436, 1.546154338151436], [2.781707203439503, 2.0373938517489285, 1.5970388690045578, 1.5715453243410613, 1.5328185251059057, 1.5328185251059057], [2.7902297278963424, 2.0549873031880153, 1.6260966450143388, 1.5854469954103962, 1.5854469954103962], [2.7850791792196157, 2.0447496148053887, 1.6146144585742979, 1.5730918184312797, 1.5610299964179097, 1.5600679872330065, 1.5600679872330065], [2.7868278653888137, 2.0470974464317795, 1.6135937528737265, 1.5989652499946052, 1.5671381788316951, 1.5578296072382614, 1.5578296072382614]]"
        self.expected_objective_curves = "[([0, 60, 90, 120, 240, 1000], [2.7854035060729516, 2.0434955300959756, 1.6067949860630397, 1.5953185527727127, 1.551094392839577, 1.551094392839577]), ([0, 60, 90, 240, 480, 1000], [2.7854035060729516, 2.0454590444287932, 1.6112733855379604, 1.5591639731106854, 1.5478056708472974, 1.5478056708472974]), ([0, 60, 90, 120, 240, 480, 1000], [2.7854035060729516, 2.04797163822759, 1.6205137429033964, 1.6163253621532803, 1.5759105545236445, 1.5664687724992692, 1.5664687724992692]), ([0, 60, 90, 240, 720, 1000], [2.7854035060729516, 2.051164328901086, 1.6227059425210009, 1.5749298888776202, 1.5649659079370986, 1.5649659079370986]), ([0, 60, 90, 120, 240, 1000], [2.7854035060729516, 2.0438598971642126, 1.608137408927407, 1.5763337097688268, 1.5521300465336114, 1.5521300465336114]), ([0, 60, 90, 120, 240, 960, 1000], [2.7854035060729516, 2.050284006246868, 1.6215173623533616, 1.620189560285288, 1.5752097928232494, 1.5716967873961198, 1.5716967873961198]), ([0, 60, 90, 120, 240, 1000], [2.7854035060729516, 2.042523510862577, 1.6072035528477624, 1.5934064648951312, 1.5584508725578103, 1.5584508725578103]), ([0, 60, 90, 120, 240, 720, 1000], [2.7854035060729516, 2.052665196346358, 1.628996218505635, 1.6431966479141764, 1.604964304544481, 1.5790413250028876, 1.5790413250028876]), ([0, 60, 90, 120, 240, 480, 720, 1000], [2.7854035060729516, 2.05027100272252, 1.6206037335904169, 1.609361043544037, 1.5736959250822544, 1.565914453421025, 1.56433775547408, 1.56433775547408]), ([0, 60, 90, 120, 240, 480, 1000], [2.7854035060729516, 2.0495789353187406, 1.6225178001355536, 1.637931354858589, 1.5896712431907034, 1.57215273101062, 1.57215273101062]), ([0, 60, 90, 120, 240, 1000], [2.7854035060729516, 2.0462854168869495, 1.612986905052738, 1.5876082570434153, 1.5595124605927324, 1.5595124605927324]), ([0, 60, 90, 840, 1000], [2.7854035060729516, 2.0401858799081696, 1.6043242541410145, 1.5797350249115747, 1.5797350249115747]), ([0, 60, 90, 120, 240, 1000], [2.7854035060729516, 2.036931781296998, 1.5993311729298625, 1.5825601479928275, 1.537272866398318, 1.537272866398318]), ([0, 60, 90, 240, 1000], [2.7854035060729516, 2.039725562437248, 1.5994678666963085, 1.5370194844154883, 1.5370194844154883]), ([0, 60, 90, 120, 240, 1000], [2.7854035060729516, 2.041025631877137, 1.6042400211713106, 1.6109631240563056, 1.5679869223098604, 1.5679869223098604]), ([0, 60, 90, 240, 960, 1000], [2.7854035060729516, 2.040521496398947, 1.6029140553101422, 1.5427593360165515, 1.5528079689590175, 1.5528079689590175]), ([0, 60, 90, 120, 240, 1000], [2.7854035060729516, 2.0437980957523743, 1.6093454050160851, 1.6058116849244488, 1.5580059210112633, 1.5580059210112633]), ([0, 60, 90, 120, 240, 960, 1000], [2.7854035060729516, 2.0427195553294974, 1.605910359115762, 1.5819525560751315, 1.548106920588509, 1.5432967245164224, 1.5432967245164224]), ([0, 60, 90, 240, 960, 1000], [2.7854035060729516, 2.050253708974358, 1.6189174807929039, 1.5672082530048066, 1.5767889956687597, 1.5767889956687597]), ([0, 60, 90, 120, 240, 1000], [2.7854035060729516, 2.0415512366627646, 1.6060292308003838, 1.58042416156321, 1.546154338151436, 1.546154338151436]), ([0, 60, 90, 120, 240, 1000], [2.7854035060729516, 2.0373938517489285, 1.5970388690045578, 1.5715453243410613, 1.5539167489128358, 1.5539167489128358]), ([0, 60, 90, 240, 1000], [2.7854035060729516, 2.0549873031880153, 1.6260966450143388, 1.5854469954103962, 1.5854469954103962]), ([0, 60, 90, 240, 480, 720, 1000], [2.7854035060729516, 2.0447496148053887, 1.6146144585742979, 1.5730918184312797, 1.5610299964179097, 1.5600679872330065, 1.5600679872330065]), ([0, 60, 90, 120, 240, 720, 1000], [2.7854035060729516, 2.0470974464317795, 1.6135937528737265, 1.5989652499946052, 1.5671381788316951, 1.5578296072382614, 1.5578296072382614])]"
        self.expected_progress_curves = "[([0.0, 0.06, 0.09, 0.12, 0.24, 1.0], [1.0, 0.3975509913822692, 0.042938534939786416, 0.0336193658755633, -0.0022918281961612683, -0.0022918281961612683]), ([0.0, 0.06, 0.09, 0.24, 0.48, 1.0], [1.0, 0.3991454172430438, 0.046575114422986175, 0.004260885606232618, -0.004962357922249182, -0.004962357922249182]), ([0.0, 0.06, 0.09, 0.12, 0.24, 0.48, 1.0], [1.0, 0.4011857102337625, 0.054078530364498105, 0.05067745379931047, 0.017859555113306884, 0.010192576991553777, 0.010192576991553777]), ([0.0, 0.06, 0.09, 0.24, 0.72, 1.0], [1.0, 0.40377825997490513, 0.05585865475873825, 0.017063228526502366, 0.008972210996196803, 0.008972210996196803]), ([0.0, 0.06, 0.09, 0.12, 0.24, 1.0], [1.0, 0.3978468671325511, 0.04402861800934614, 0.0182031684268257, -0.0014508498518852517, -0.0014508498518852517]), ([0.0, 0.06, 0.09, 0.12, 0.24, 0.96, 1.0], [1.0, 0.40306341456621547, 0.05489349605059252, 0.053815285456484646, 0.017290517974806813, 0.014437864134475863, 0.014437864134475863]), ([0.0, 0.06, 0.09, 0.12, 0.24, 1.0], [1.0, 0.3967616859124808, 0.043270302035410615, 0.03206669966420187, 0.0036818289913490318, 0.0036818289913490318]), ([0.0, 0.06, 0.09, 0.12, 0.24, 0.72, 1.0], [1.0, 0.40499700425822427, 0.060966526157323106, 0.07249765251818503, 0.041451972857072375, 0.020401824009858318, 0.020401824009858318]), ([0.0, 0.06, 0.09, 0.12, 0.24, 0.48, 0.72, 1.0], [1.0, 0.40305285535859725, 0.05415160519579223, 0.04502224186239665, 0.016061217105598925, 0.009742455157095315, 0.00846213448959509, 0.00846213448959509]), ([0.0, 0.06, 0.09, 0.12, 0.24, 0.48, 1.0], [1.0, 0.40249087822018664, 0.05570587813783398, 0.0682220945189017, 0.029033600296538835, 0.014808102475935272, 0.014808102475935272]), ([0.0, 0.06, 0.09, 0.12, 0.24, 1.0], [1.0, 0.39981645365764723, 0.047966537842535545, 0.027358400676816223, 0.004543866710187482, 0.004543866710187482]), ([0.0, 0.06, 0.09, 0.84, 1.0], [1.0, 0.394863467404798, 0.04093223490638383, 0.020965126785672795, 0.020965126785672795]), ([0.0, 0.06, 0.09, 0.12, 0.24, 1.0], [1.0, 0.392221052785029, 0.03687772016465295, 0.023259201865917855, -0.013515275270113085, -0.013515275270113085]), ([0.0, 0.06, 0.09, 0.24, 1.0], [1.0, 0.3944896773755954, 0.03698871913857712, -0.01372102817923406, -0.01372102817923406]), ([0.0, 0.06, 0.09, 0.12, 0.24, 1.0], [1.0, 0.3955453683380277, 0.04086383549468557, 0.046323173848025964, 0.011425355015161696, 0.011425355015161696]), ([0.0, 0.06, 0.09, 0.24, 0.96, 1.0], [1.0, 0.3951359969215191, 0.03978711594942131, -0.009060116019447938, -0.0009003588121200922, -0.0009003588121200922]), ([0.0, 0.06, 0.09, 0.12, 0.24, 1.0], [1.0, 0.3977966827424397, 0.04500954296177022, 0.042140068262923026, 0.003320516501417685, 0.003320516501417685]), ([0.0, 0.06, 0.09, 0.12, 0.24, 0.96, 1.0], [1.0, 0.3969208792337085, 0.042220194330653306, 0.022765821068955704, -0.004717735120209235, -0.008623742264921986, -0.008623742264921986]), ([0.0, 0.06, 0.09, 0.24, 0.96, 1.0], [1.0, 0.4030388123751373, 0.052782323075859734, 0.01079305482961246, 0.01857287268656362, 0.01857287268656362]), ([0.0, 0.06, 0.09, 0.12, 0.24, 1.0], [1.0, 0.3959721734031829, 0.04231672130013205, 0.021524724075394748, -0.006303283990889538, -0.006303283990889538]), ([0.0, 0.06, 0.09, 0.12, 0.24, 1.0], [1.0, 0.3925962662814342, 0.03501630841014017, 0.014314872105387531, 0.0, 0.0]), ([0.0, 0.06, 0.09, 0.24, 1.0], [1.0, 0.4068826167734673, 0.0586119953640056, 0.025603398748900157, 0.025603398748900157]), ([0.0, 0.06, 0.09, 0.24, 0.48, 0.72, 1.0], [1.0, 0.3985693415205241, 0.04928815458920134, 0.015570666437910158, 0.005776146161309523, 0.0049949691171311175, 0.0049949691171311175]), ([0.0, 0.06, 0.09, 0.12, 0.24, 0.72, 1.0], [1.0, 0.40047584324515895, 0.04845931441318098, 0.036580581008970006, 0.010736152737321154, 0.003177345028418195, 0.003177345028418195])]"

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
