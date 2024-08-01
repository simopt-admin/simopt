import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class test_IRONORECONT1_NELDMD(unittest.TestCase):
    def setUp(self):
        # Expected values
        self.expected_problem_name = "IRONORECONT-1"
        self.expected_solver_name = "NELDMD"
        self.expected_all_recommended_xs = "[[(80, 40, 100), (71.84677291324282, 59.313560684071284, 103.72927038686149), (56.13379149117459, 80.51840568135151, 101.48396323850142), (50.924717227140476, 88.038075807242, 102.50401828489335), (40.179801595873954, 99.84577986522817, 99.31842913731958), (43.12847863441446, 99.109866009233, 101.36142144891875), (43.12847863441446, 99.109866009233, 101.36142144891875)], [(80, 40, 100), (81.75200674676572, 40.59763606068053, 100.07218623080225), (81.75200674676572, 40.59763606068053, 100.07218623080225)], [(80, 40, 100), (72.99402336126965, 61.01444415352947, 100.27025191408055), (77.78692868226355, 62.860690909278254, 101.01622523655048), (67.02403415536372, 97.2866127479025, 98.17365677049997), (67.02403415536372, 97.2866127479025, 98.17365677049997)], [(80, 40, 100), (77.05992216766845, 102.0282154661873, 99.37299152796103), (79.28183844548062, 89.42792181444659, 99.98720277010761), (88.6060134858426, 97.82621392212657, 96.13571418562641), (88.6060134858426, 97.82621392212657, 96.13571418562641)], [(80, 40, 100), (80.94647133838481, 41.5986778194078, 102.09635916368613), (71.40647090729274, 43.5469072828665, 101.68315577157618), (96.25182809429387, 42.141913172792854, 101.95920876321753), (98.014269732472, 41.10309619221593, 101.24898242315336), (98.014269732472, 41.10309619221593, 101.24898242315336)], [(80, 40, 100), (100.37163734231885, 94.01676050768417, 100.33881897742998), (100.37163734231885, 94.01676050768417, 100.33881897742998)], [(80, 40, 100), (81.56825765245154, 39.40202263934076, 100.85877312834089), (81.56825765245154, 39.40202263934076, 100.85877312834089)], [(80, 40, 100), (80.75142988425276, 99.76906168314724, 91.24710742446698), (80.75142988425276, 99.76906168314724, 91.24710742446698)], [(80, 40, 100), (99.94506698217431, 106.1208741871661, 95.75645937152836), (101.7018336226699, 103.17509215814889, 95.30266657128709), (101.7018336226699, 103.17509215814889, 95.30266657128709)], [(80, 40, 100), (86.63835328843174, 38.52866321064976, 97.49557513910626), (84.74948605720766, 39.1933811584683, 97.47698081469643), (84.74948605720766, 39.1933811584683, 97.47698081469643)], [(80, 40, 100), (99.15612160800309, 92.23982092384719, 36.87429571837531), (98.66441513630755, 102.7719324139963, 45.88359519021209), (99.05413996403871, 97.1355956048056, 39.03534517329578), (101.60847787534527, 100.91008998219885, 40.58229785212498), (101.60847787534527, 100.91008998219885, 40.58229785212498)], [(80, 40, 100), (101.47022600789073, 110.44240567814148, 44.226100836298315), (105.56886089533526, 105.41497994272578, 93.53129171391933), (102.42038266953749, 99.25865855515582, 99.01306196108314), (102.42038266953749, 99.25865855515582, 99.01306196108314)], [(80, 40, 100), (97.94195678314816, 42.41737418972254, 99.44573924745399), (96.4401215965446, 50.63122277491476, 99.08742194988947), (96.4401215965446, 50.63122277491476, 99.08742194988947)], [(80, 40, 100), (44.088530399053695, 89.31622851514295, 98.33450763933305), (44.088530399053695, 89.31622851514295, 98.33450763933305)], [(80, 40, 100), (86.77675923112211, 51.450134622731866, 105.39535879137527), (77.02925960387057, 54.88107933922002, 102.62194791006402), (88.30726814503545, 75.61225431593664, 105.24736709475219), (79.78175880817706, 68.39944805002908, 98.00422480737741), (92.94144947348398, 117.56369113699813, 104.6933288968177), (79.4488318614137, 103.90593677506311, 101.00809058458869), (79.4488318614137, 103.90593677506311, 101.00809058458869)], [(80, 40, 100), (103.22332280557059, 110.0348112570916, 80.89724589969799), (93.57480667745352, 95.87295010906902, 83.94002001683371), (104.73804298646576, 94.40341740286917, 90.98693259597269), (103.27085170600634, 100.01082753193816, 89.66871582168353), (103.27085170600634, 100.01082753193816, 89.66871582168353)], [(80, 40, 100), (106.61103205613088, 98.4925210751569, 72.08539084220894), (101.5997079156551, 93.59935398644743, 70.60346419742598), (95.6335022766108, 98.8244986558901, 70.65968664984098), (102.2577792062608, 97.29309960857873, 71.58908965598891), (102.2577792062608, 97.29309960857873, 71.58908965598891)], [(80, 40, 100), (81.68582770068842, 93.97459433820497, 99.43674304748754), (82.11998196418878, 89.26680427152661, 101.59629487743564), (81.05101572685248, 97.38711135442935, 100.14008977080887), (81.05101572685248, 97.38711135442935, 100.14008977080887)], [(80, 40, 100), (89.7501009302194, 53.24708541765377, 95.44389042068761), (86.61970783662153, 55.70179010142961, 96.87763493883926), (104.94351023189353, 99.04555049180914, 96.31030018065157), (103.80700753095232, 97.50083539587065, 93.25245939612881), (103.80700753095232, 97.50083539587065, 93.25245939612881)], [(80, 40, 100), (95.79114745359803, 104.67520689269932, 86.7053430193585), (104.99137389246675, 103.73890739531424, 86.69173424110082), (102.69291616262828, 105.39185514639468, 86.38112998241229), (95.4293984640091, 101.30362082709648, 87.3844219032892), (95.4293984640091, 101.30362082709648, 87.3844219032892)], [(80, 40, 100), (80, 40, 100)], [(80, 40, 100), (78.60342021360222, 108.79496477650574, 101.2524010767217), (79.16221395521217, 86.90784299575, 98.74349721500417), (78.98343864369494, 99.23260767490774, 95.3309242800454), (78.98343864369494, 99.23260767490774, 95.3309242800454)], [(80, 40, 100), (120.85202365348692, 100.26883093246852, 103.85659178992047), (91.24935393478893, 98.4597605081984, 100.45488149770733), (97.34304195734344, 95.21543796613733, 99.51701908070109), (102.3574814679732, 99.1352918344142, 100.79219456171407), (102.3574814679732, 99.1352918344142, 100.79219456171407)], [(80, 40, 100), (103.71221498937601, 495.23060665822715, 107.44785650911345), (103.36829521093878, 484.3723251851716, 105.71482375503956), (110.97094305313897, 90.68375935549363, 108.96938742172173), (106.84987193712612, 94.71759873061598, 106.53847689414988), (106.84987193712612, 94.71759873061598, 106.53847689414988)]]"
        self.expected_all_intermediate_budgets = "[[0, 480, 630, 720, 840, 900, 1000], [0, 810, 1000], [0, 540, 720, 900, 1000], [0, 840, 900, 990, 1000], [0, 690, 750, 810, 870, 1000], [0, 870, 1000], [0, 990, 1000], [0, 780, 1000], [0, 660, 960, 1000], [0, 600, 900, 1000], [0, 300, 720, 870, 990, 1000], [0, 690, 750, 960, 1000], [0, 600, 810, 1000], [0, 270, 1000], [0, 450, 570, 750, 810, 900, 960, 1000], [0, 570, 630, 720, 840, 1000], [0, 690, 750, 840, 960, 1000], [0, 810, 930, 990, 1000], [0, 600, 660, 780, 870, 1000], [0, 780, 840, 900, 960, 1000], [0, 1000], [0, 450, 870, 930, 1000], [0, 360, 720, 780, 870, 1000], [0, 720, 780, 840, 960, 1000]]"
        self.expected_all_est_objectives = "[[149895.8055235529, 153706.6747016864, 183253.04288145647, 191502.65160461538, 200315.3982262673, 202507.49597840346, 202507.49597840346], [122793.09736468189, 123208.33385017878, 123208.33385017878], [99852.80349485856, 120481.85206985116, 127101.83947312846, 184780.73634381968, 184780.73634381968], [126011.12695446546, 207115.8568076835, 208585.30770905715, 229588.6733113368, 229588.6733113368], [136147.71179130895, 134563.3775276705, 134519.62742134838, 135591.0723866743, 137173.95510059112, 137173.95510059112], [132850.26196652921, 250128.8744846016, 250128.8744846016], [134982.68434045353, 134859.37231653478, 134859.37231653478], [161256.2908821113, 219326.3821098657, 219326.3821098657], [146337.47315675917, 254054.42514026337, 259610.41605255895, 259610.41605255895], [134867.2205665852, 118851.84356596734, 118639.63598780682, 118639.63598780682], [149243.01256369415, 251820.42345680014, 263335.67408948706, 263514.1733866956, 264892.29601196665, 264892.29601196665], [112822.77485929335, 217708.62758642106, 226133.0815366795, 238748.7418554251, 238748.7418554251], [132809.38556277155, 139830.6553619829, 144155.14966514474, 144155.14966514474], [118379.15455996453, 167597.47155556513, 167597.47155556513], [127606.7164810152, 108711.54289716977, 139989.50615754697, 170195.7519909911, 160806.84027998487, 185957.08640286513, 215583.4337724548, 215583.4337724548], [145498.2552215891, 236095.7719816307, 245794.3745587961, 249036.61014827024, 255747.54269069698, 255747.54269069698], [161264.15011124164, 253167.78729643312, 253857.4247186155, 258573.44630465284, 260664.35624648392, 260664.35624648392], [132500.94479520118, 226519.09020027026, 215948.13594054827, 226008.3789273902, 226008.3789273902], [112031.98326897933, 120942.54159967315, 123715.24668964263, 242994.97317268504, 244599.51373007803, 244599.51373007803], [130863.18264271188, 232395.7760272165, 234025.61066717148, 235408.74926413872, 237802.8573289337, 237802.8573289337], [147610.26102665017, 147610.26102665017], [132677.02997009846, 203008.66910349685, 212563.93435467652, 217473.70417449667, 217473.70417449667], [132803.08586581453, 173107.80170251933, 249154.92604200795, 253287.1448235666, 253394.28889702607, 253394.28889702607], [137521.1409071744, 116801.2256346231, 123066.59888886147, 187732.4558021342, 223405.36181624013, 223405.36181624013]]"
        self.expected_objective_curves = "[([0, 480, 630, 720, 840, 900, 1000], [121270.73497283501, 153706.6747016864, 183253.04288145647, 191502.65160461538, 200315.3982262673, 202507.49597840346, 202507.49597840346]), ([0, 810, 1000], [121270.73497283501, 123208.33385017878, 123208.33385017878]), ([0, 540, 720, 900, 1000], [121270.73497283501, 120481.85206985116, 127101.83947312846, 184780.73634381968, 184780.73634381968]), ([0, 840, 900, 990, 1000], [121270.73497283501, 207115.8568076835, 208585.30770905715, 229588.6733113368, 229588.6733113368]), ([0, 690, 750, 810, 870, 1000], [121270.73497283501, 134563.3775276705, 134519.62742134838, 135591.0723866743, 137173.95510059112, 137173.95510059112]), ([0, 870, 1000], [121270.73497283501, 250128.8744846016, 250128.8744846016]), ([0, 990, 1000], [121270.73497283501, 134859.37231653478, 134859.37231653478]), ([0, 780, 1000], [121270.73497283501, 219326.3821098657, 219326.3821098657]), ([0, 660, 960, 1000], [121270.73497283501, 254054.42514026337, 259610.41605255895, 259610.41605255895]), ([0, 600, 900, 1000], [121270.73497283501, 118851.84356596734, 118639.63598780682, 118639.63598780682]), ([0, 300, 720, 870, 990, 1000], [121270.73497283501, 251820.42345680014, 263335.67408948706, 263514.1733866956, 241687.56737160246, 241687.56737160246]), ([0, 690, 750, 960, 1000], [121270.73497283501, 217708.62758642106, 226133.0815366795, 238748.7418554251, 238748.7418554251]), ([0, 600, 810, 1000], [121270.73497283501, 139830.6553619829, 144155.14966514474, 144155.14966514474]), ([0, 270, 1000], [121270.73497283501, 167597.47155556513, 167597.47155556513]), ([0, 450, 570, 750, 810, 900, 960, 1000], [121270.73497283501, 108711.54289716977, 139989.50615754697, 170195.7519909911, 160806.84027998487, 185957.08640286513, 215583.4337724548, 215583.4337724548]), ([0, 570, 630, 720, 840, 1000], [121270.73497283501, 236095.7719816307, 245794.3745587961, 249036.61014827024, 255747.54269069698, 255747.54269069698]), ([0, 690, 750, 840, 960, 1000], [121270.73497283501, 253167.78729643312, 253857.4247186155, 258573.44630465284, 260664.35624648392, 260664.35624648392]), ([0, 810, 930, 990, 1000], [121270.73497283501, 226519.09020027026, 215948.13594054827, 226008.3789273902, 226008.3789273902]), ([0, 600, 660, 780, 870, 1000], [121270.73497283501, 120942.54159967315, 123715.24668964263, 242994.97317268504, 244599.51373007803, 244599.51373007803]), ([0, 780, 840, 900, 960, 1000], [121270.73497283501, 232395.7760272165, 234025.61066717148, 235408.74926413872, 237802.8573289337, 237802.8573289337]), ([0, 1000], [121270.73497283501, 121270.73497283501]), ([0, 450, 870, 930, 1000], [121270.73497283501, 203008.66910349685, 212563.93435467652, 217473.70417449667, 217473.70417449667]), ([0, 360, 720, 780, 870, 1000], [121270.73497283501, 173107.80170251933, 249154.92604200795, 253287.1448235666, 253394.28889702607, 253394.28889702607]), ([0, 720, 780, 840, 960, 1000], [121270.73497283501, 116801.2256346231, 123066.59888886147, 187732.4558021342, 223405.36181624013, 223405.36181624013])]"
        self.expected_progress_curves = "[([0.0, 0.48, 0.63, 0.72, 0.84, 0.9, 1.0], [1.0, 0.730636165370653, 0.485268739644609, 0.41675997256593467, 0.3435746342199799, 0.3253703872848264, 0.3253703872848264]), ([0.0, 0.81, 1.0], [1.0, 0.983909235621418, 0.983909235621418]), ([0.0, 0.54, 0.72, 0.9, 1.0], [1.0, 1.0065512676863266, 0.9515756694131979, 0.47258202939047966, 0.47258202939047966]), ([0.0, 0.84, 0.9, 0.99, 1.0], [1.0, 0.2871003154229526, 0.2748972797501036, 0.1004751065050395, 0.1004751065050395]), ([0.0, 0.69, 0.75, 0.81, 0.87, 1.0], [1.0, 0.8896114248312386, 0.88997474701345, 0.8810769463988503, 0.8679319177314708, 0.8679319177314708]), ([0.0, 0.87, 1.0], [1.0, -0.0701007238343993, -0.0701007238343993]), ([0.0, 0.99, 1.0], [1.0, 0.8871533400023329, 0.8871533400023329]), ([0.0, 0.78, 1.0], [1.0, 0.18569816873845663, 0.18569816873845663]), ([0.0, 0.66, 0.96, 1.0], [1.0, -0.10270040759507214, -0.14884006101077232, -0.14884006101077232]), ([0.0, 0.6, 0.9, 1.0], [1.0, 1.0200876518563233, 1.0218499268965584, 1.0218499268965584]), ([0.0, 0.3, 0.72, 0.87, 0.99, 1.0], [1.0, -0.08414817001365829, -0.17977641735497255, -0.18125876241963468, -0.0, -0.0]), ([0.0, 0.69, 0.75, 0.96, 1.0], [1.0, 0.1991327898891554, 0.12917202292295285, 0.024405437824882123, 0.024405437824882123]), ([0.0, 0.6, 0.81, 1.0], [1.0, 0.8458693853722574, 0.8099566793409194, 0.8099566793409194]), ([0.0, 0.27, 1.0], [1.0, 0.6152802256970488, 0.6152802256970488]), ([0.0, 0.45, 0.57, 0.75, 0.81, 0.9, 0.96, 1.0], [1.0, 1.104297645316518, 0.8445502110309325, 0.5937028400137782, 0.6716729337620864, 0.46281304580561095, 0.21678143395021623, 0.21678143395021623]), ([0.0, 0.57, 0.63, 0.72, 0.84, 1.0], [1.0, 0.04643699122938389, -0.03410492624148833, -0.06103002902726245, -0.11676088001164223, -0.11676088001164223]), ([0.0, 0.69, 0.75, 0.84, 0.96, 1.0], [1.0, -0.09533733528891734, -0.10106442018597407, -0.14022855938554996, -0.15759249348163137, -0.15759249348163137]), ([0.0, 0.81, 0.93, 0.99, 1.0], [1.0, 0.12596641905594136, 0.21375276959467385, 0.13020761410074047, 0.13020761410074047]), ([0.0, 0.6, 0.66, 0.78, 0.87, 1.0], [1.0, 1.0027254775484795, 0.9796995846169373, -0.010857334269955145, -0.024182220213470572, -0.024182220213470572]), ([0.0, 0.78, 0.84, 0.9, 0.96, 1.0], [1.0, 0.07716355894179026, 0.06362861862250253, 0.052142362345748, 0.03226052342752483, 0.03226052342752483]), ([0.0, 1.0], [1.0, 1.0]), ([0.0, 0.45, 0.87, 0.93, 1.0], [1.0, 0.321208401662802, 0.24185682712929463, 0.20108370827194783, 0.20108370827194783]), ([0.0, 0.36, 0.72, 0.78, 0.87, 1.0], [1.0, 0.5695197615062418, -0.06201258180979957, -0.096328538302282, -0.09721831485033676, -0.09721831485033676]), ([0.0, 0.72, 0.78, 0.84, 0.96, 1.0], [1.0, 1.0371169814815495, 0.9850862717424808, 0.44806951399280054, 0.15182433544522855, 0.15182433544522855])]"

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
