import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class test_PARAMESTI1_ASTRODF(unittest.TestCase):
    def setUp(self):
        # Expected values
        self.expected_problem_name = "PARAMESTI-1"
        self.expected_solver_name = "ASTRODF"
        self.expected_all_recommended_xs = "[[(1, 1), (1.5149285897956326, 4.120071881769887), (2.513063822133061, 4.430739007802012), (2.513063822133061, 4.430739007802012)], [(1, 1), (1.0, 4.16227766016838), (1.1034868749749513, 5.343607616915401), (1.4185931120402107, 6.1753068921120144), (2.0823865428708404, 6.109545427342727), (1.3408250070660603, 5.437819561378433), (2.08711054629488, 5.359123684531698), (2.08711054629488, 5.359123684531698)], [(1, 1), (2.988348405084185, 3.4589572220758904), (1.836484603166295, 3.7408433370784984), (2.183380222906105, 4.559793307470667), (2.3896338255504723, 5.877839060715303), (2.3586844131042524, 6.544163623471122), (1.6847672970322431, 5.804592889155728), (1.6847672970322431, 5.804592889155728)], [(1, 1), (2.827479762870058, 3.5807591356615207), (1.6882502677498659, 3.910010039150322), (1.9943299880287926, 4.74507340790384), (1.9943299880287926, 4.74507340790384)], [(1, 1), (3.5889705285536357, 2.815828076190229), (1.281015745990938, 6.475085612424144), (2.9511955867168447, 5.863071070413266), (2.291520534447577, 5.764201891818322), (2.1203659444727943, 5.294107865129809), (2.1203659444727943, 5.294107865129809)], [(1, 1), (2.564533422142073, 3.748133033715875), (1.6855383569167222, 4.029418827623008), (1.6855383569167222, 4.029418827623008)], [(1, 1), (3.5041045641201816, 2.9311810717673454), (2.7390078646882223, 5.176092437616412), (2.2829700315291777, 4.504726722333899), (2.2829700315291777, 4.504726722333899)], [(1, 1), (3.157720222638415, 3.3117619775438016), (1.4536762757777157, 4.438459893547172), (2.6711911979027776, 6.128695360062492), (2.024491949444858, 5.829380927701434), (2.024491949444858, 5.829380927701434)], [(1, 1), (3.4314935812327083, 3.0218404893621407), (1.5892925862415654, 7.3929153564919226), (3.2104334546814286, 3.212411850623414), (1.8631737585377213, 4.373856559228505), (1.8631737585377213, 3.7068136152867375), (2.3301157311394154, 4.591739795605449), (2.3301157311394154, 4.591739795605449)], [(1, 1), (1.0415628563574333, 4.162004511219333), (1.7688663872221673, 3.2253724735483535), (1.7688663872221673, 4.114763065470711), (1.7688663872221673, 4.114763065470711)], [(1, 1), (3.6382790543403294, 2.7434114922843365), (1.6218958157082302, 7.03691805859625), (2.3118384246039607, 8.676442897209606), (2.366340935990269, 7.343470792286706), (2.521366464936597, 6.867814011642756), (2.3157964904830006, 6.845748433309315)], [(1, 1), (3.2323257669432195, 3.2398039356696753), (1.5606879288231326, 4.701852652851575), (2.450056936202248, 4.708048918943869), (2.2015319174628862, 5.318565395726482), (1.8980722615769832, 4.920828642524989), (1.8980722615769832, 4.920828642524989)], [(1, 1), (1.843201017702759, 4.047788057550104), (2.0027519307694224, 5.222859760102589), (1.6629758183012864, 6.044789086989056), (1.8153327248309108, 6.694199328743236), (1.6865482932967122, 6.97331049052862), (1.6865482932967122, 6.97331049052862)], [(1, 1), (2.3144655652827177, 3.876139822346262), (2.3144655652827177, 3.876139822346262)], [(1, 1), (3.808087340481991, 2.454182068457998), (3.2468266719160863, 4.758522855303761), (2.933013203853923, 3.926334936464033), (2.291749804471177, 4.109985533181236), (2.1720865662353193, 4.595745759025803)], [(1, 1), (1.0, 4.16227766016838), (1.9394249853687682, 4.885970065037251), (1.9394249853687682, 4.885970065037251)], [(1, 1), (3.127854780655885, 3.3392806655978453), (2.1304297289458796, 3.6891968801109503), (1.9683899559373121, 4.563701732563658), (1.9683899559373121, 4.563701732563658)], [(1, 1), (1.0, 4.16227766016838), (2.7505419470220414, 5.657882966448427), (2.1176484530910846, 7.320263285551088), (2.703423031093147, 6.761898217262623), (2.0130645416461115, 6.0376508103366255), (2.0130645416461115, 6.0376508103366255)], [(1, 1), (3.100108553312285, 3.364221661415144), (1.234882024557603, 4.751426736445525), (2.19997581640053, 4.727511149757701), (2.2249985250181203, 6.061362348468951), (2.0854729924741617, 6.713649796478512), (2.048408522085547, 6.214742477798511), (2.048408522085547, 6.214742477798511)], [(1, 1), (1.4877912592082663, 4.1244294979147815), (1.4877912592082663, 4.1244294979147815)], [(1, 1), (2.855472797946561, 3.5607070695572265), (1.727314134998837, 3.926094577849455), (1.727314134998837, 3.926094577849455)], [(1, 1), (2.217393780062109, 3.9185531319929896), (2.217393780062109, 4.5114801932745605), (2.217393780062109, 4.5114801932745605)], [(1, 1), (2.4786068846054965, 3.795303504236567), (1.5839932356541897, 4.09051296682205), (1.5839932356541897, 4.09051296682205)], [(1, 1), (2.184879953674061, 3.931903732284086), (1.5543761182822535, 4.561934239045764), (1.5543761182822535, 4.561934239045764)]]"
        self.expected_all_intermediate_budgets = "[[4, 27, 184, 1000], [4, 24, 128, 277, 610, 663, 942, 1000], [4, 24, 189, 344, 368, 587, 627, 1000], [4, 24, 82, 202, 1000], [4, 25, 41, 103, 359, 731, 1000], [4, 25, 103, 1000], [4, 24, 64, 80, 1000], [4, 24, 40, 72, 104, 1000], [4, 24, 40, 56, 105, 447, 499, 1000], [4, 24, 296, 686, 1000], [4, 24, 40, 92, 127, 273, 1000], [4, 24, 61, 223, 497, 991, 1000], [4, 24, 97, 165, 350, 650, 1000], [4, 24, 1000], [4, 24, 61, 226, 568, 1000], [4, 24, 238, 1000], [4, 27, 62, 261, 1000], [4, 25, 60, 111, 153, 257, 1000], [4, 24, 57, 73, 124, 273, 684, 1000], [4, 25, 1000], [4, 25, 132, 1000], [4, 25, 618, 1000], [4, 24, 120, 1000], [4, 24, 86, 1000]]"
        self.expected_all_est_objectives = "[[-9.16944227603262, -5.087901713959044, -5.038434341062556, -5.038434341062556], [-8.940090362495347, -6.105529335023807, -5.672060760205896, -5.065096221668084, -4.696451969437038, -5.099026054774974, -4.602907076090707, -4.602907076090707], [-9.121210005202611, -6.1327216667434845, -4.834807231469014, -4.674149226964931, -4.856235025198865, -4.96295717979425, -4.736035238913933, -4.736035238913933], [-8.779886386724968, -5.633801495862974, -4.753957603975482, -4.5377413626394985, -4.5377413626394985], [-8.99288952739613, -7.333600595755891, -5.7523757908576885, -5.4806204040842665, -4.704794901253494, -4.580481029440518, -4.580481029440518], [-8.87740808504234, -5.054919789182981, -4.767843414834182, -4.767843414834182], [-9.024638576352391, -7.6104184714524274, -5.290415040394047, -4.760361694139695, -4.760361694139695], [-8.921050660074993, -6.596419499573125, -4.986486714056616, -5.349440327820792, -4.671136822330359, -4.671136822330359], [-8.550164686658025, -6.655817932151406, -5.5509882159508335, -6.064705878619034, -4.517347115958269, -4.60968959603932, -4.560859802350674, -4.560859802350674], [-8.983830735669818, -6.0981523912146915, -5.03604768870009, -4.730473300524734, -4.730473300524734], [-9.025843710458552, -8.182655212598858, -5.1238441056308766, -5.868686657276348, -5.25244340927715, -5.247457722078471, -5.023773396334706], [-9.203733926294058, -6.502665548216621, -4.867181941526699, -4.7462339608084925, -4.563707291646994, -4.556750802777809, -4.556750802777809], [-9.33623207280299, -4.777697380662419, -4.5933248385269465, -4.913043152511098, -4.9324046912626365, -5.139102405509359, -5.139102405509359], [-9.573886675373538, -5.309967580064792, -5.309967580064792], [-8.941889405024408, -9.267047779399803, -6.599804451492301, -5.961105631800999, -4.904408751412655, -4.736611415937758], [-9.587195496910567, -6.648764996126192, -4.668167463013499, -4.668167463013499], [-9.346621843523279, -6.570065646270418, -5.014973372360015, -4.79095038311002, -4.79095038311002], [-9.398264139884379, -6.6107792781241, -5.334146509812032, -5.216190557753698, -5.514136165520643, -4.818386869051045, -4.818386869051045], [-8.499265659696825, -6.094281344649115, -5.2130241953489715, -4.529729113930459, -4.727830752315895, -4.8726812753187305, -4.694395658805532, -4.694395658805532], [-9.254478807791063, -5.193041877275148, -5.193041877275148], [-9.605611629911163, -5.904600621404379, -5.053927490266147, -5.053927490266147], [-8.501695309087717, -4.692586573093009, -4.615940318722932, -4.615940318722932], [-9.152042163756049, -5.115903831139051, -4.933409432677312, -4.933409432677312], [-9.213076761398039, -4.788252775223845, -4.961391778662112, -4.961391778662112]]"
        self.expected_objective_curves = "[([4, 27, 184, 1000], [-9.265122221743944, -5.087901713959044, -5.038434341062556, -5.038434341062556]), ([4, 24, 128, 277, 610, 663, 942, 1000], [-9.265122221743944, -6.105529335023807, -5.672060760205896, -5.065096221668084, -4.696451969437038, -5.099026054774974, -4.602907076090707, -4.602907076090707]), ([4, 24, 189, 344, 368, 587, 627, 1000], [-9.265122221743944, -6.1327216667434845, -4.834807231469014, -4.674149226964931, -4.856235025198865, -4.96295717979425, -4.736035238913933, -4.736035238913933]), ([4, 24, 82, 202, 1000], [-9.265122221743944, -5.633801495862974, -4.753957603975482, -4.5377413626394985, -4.5377413626394985]), ([4, 25, 41, 103, 359, 731, 1000], [-9.265122221743944, -7.333600595755891, -5.7523757908576885, -5.4806204040842665, -4.704794901253494, -4.580481029440518, -4.580481029440518]), ([4, 25, 103, 1000], [-9.265122221743944, -5.054919789182981, -4.767843414834182, -4.767843414834182]), ([4, 24, 64, 80, 1000], [-9.265122221743944, -7.6104184714524274, -5.290415040394047, -4.760361694139695, -4.760361694139695]), ([4, 24, 40, 72, 104, 1000], [-9.265122221743944, -6.596419499573125, -4.986486714056616, -5.349440327820792, -4.671136822330359, -4.671136822330359]), ([4, 24, 40, 56, 105, 447, 499, 1000], [-9.265122221743944, -6.655817932151406, -5.5509882159508335, -6.064705878619034, -4.517347115958269, -4.60968959603932, -4.560859802350674, -4.560859802350674]), ([4, 24, 296, 686, 1000], [-9.265122221743944, -6.0981523912146915, -5.03604768870009, -4.730473300524734, -4.730473300524734]), ([4, 24, 40, 92, 127, 273, 1000], [-9.265122221743944, -8.182655212598858, -5.1238441056308766, -5.868686657276348, -5.25244340927715, -5.247457722078471, -5.023773396334706]), ([4, 24, 61, 223, 497, 991, 1000], [-9.265122221743944, -6.502665548216621, -4.867181941526699, -4.7462339608084925, -4.563707291646994, -4.556750802777809, -4.556750802777809]), ([4, 24, 97, 165, 350, 650, 1000], [-9.265122221743944, -4.777697380662419, -4.5933248385269465, -4.913043152511098, -4.9324046912626365, -5.139102405509359, -5.139102405509359]), ([4, 24, 1000], [-9.265122221743944, -5.309967580064792, -5.309967580064792]), ([4, 24, 61, 226, 568, 1000], [-9.265122221743944, -9.267047779399803, -6.599804451492301, -5.961105631800999, -4.904408751412655, -4.736611415937758]), ([4, 24, 238, 1000], [-9.265122221743944, -6.648764996126192, -4.668167463013499, -4.668167463013499]), ([4, 27, 62, 261, 1000], [-9.265122221743944, -6.570065646270418, -5.014973372360015, -4.79095038311002, -4.79095038311002]), ([4, 25, 60, 111, 153, 257, 1000], [-9.265122221743944, -6.6107792781241, -5.334146509812032, -5.216190557753698, -5.514136165520643, -4.818386869051045, -4.818386869051045]), ([4, 24, 57, 73, 124, 273, 684, 1000], [-9.265122221743944, -6.094281344649115, -5.2130241953489715, -4.529729113930459, -4.727830752315895, -4.8726812753187305, -4.694395658805532, -4.694395658805532]), ([4, 25, 1000], [-9.265122221743944, -5.193041877275148, -5.193041877275148]), ([4, 25, 132, 1000], [-9.265122221743944, -5.904600621404379, -5.053927490266147, -5.053927490266147]), ([4, 25, 618, 1000], [-9.265122221743944, -4.692586573093009, -4.615940318722932, -4.615940318722932]), ([4, 24, 120, 1000], [-9.265122221743944, -5.115903831139051, -4.933409432677312, -4.933409432677312]), ([4, 24, 86, 1000], [-9.265122221743944, -4.788252775223845, -4.961391778662112, -4.961391778662112])]"
        self.expected_progress_curves = "[([0.004, 0.027, 0.184, 1.0], [1.0, 0.1032120166568272, 0.09259209714373186, 0.09259209714373186]), ([0.004, 0.024, 0.128, 0.277, 0.61, 0.663, 0.942, 1.0], [1.0, 0.32168174321021686, 0.22862239709019858, 0.09831601190853982, 0.019173497187754653, 0.1056002494895799, -0.0009092195612812687, -0.0009092195612812687]), ([0.004, 0.024, 0.189, 0.344, 0.368, 0.587, 0.627, 1.0], [1.0, 0.32751953805007317, 0.04887634294160907, 0.014385425452659974, 0.05347657601301961, 0.07638825750535845, 0.02767144464878974, 0.02767144464878974]), ([0.004, 0.024, 0.082, 0.202, 1.0], [1.0, 0.22040869411465794, 0.031519113592847765, -0.014899342572531648, -0.014899342572531648]), ([0.004, 0.025, 0.041, 0.103, 0.359, 0.731, 1.0], [1.0, 0.5853306330069499, 0.2458648563369957, 0.18752296013759823, 0.020964602298175677, -0.0057237628951325855, -0.0057237628951325855]), ([0.004, 0.025, 0.103, 1.0], [1.0, 0.0961312810931721, 0.03450019354633626, 0.03450019354633626]), ([0.004, 0.024, 0.064, 0.08, 1.0], [1.0, 0.6447593713358437, 0.14668865794863883, 0.03289397783398828, 0.03289397783398828]), ([0.004, 0.024, 0.04, 0.072, 0.104, 1.0], [1.0, 0.4270686625477895, 0.08143970344670298, 0.1593605215428717, 0.013738706355064604, 0.013738706355064604]), ([0.004, 0.024, 0.04, 0.056, 0.105, 0.447, 0.499, 1.0], [1.0, 0.43982063493381723, 0.20262989738893183, 0.3129175458033275, -0.019277688249753663, 0.0005468880048300608, -0.009936152602553377, -0.009936152602553377]), ([0.004, 0.024, 0.296, 0.686, 1.0], [1.0, 0.3200980215585868, 0.09207971788218985, 0.026477378043439748, 0.026477378043439748]), ([0.004, 0.024, 0.04, 0.092, 0.127, 0.273, 1.0], [1.0, 0.7676102077068696, 0.11092832104723277, 0.2708350936376353, 0.1385367056085337, 0.13746635168296253, 0.08944460731601238]), ([0.004, 0.024, 0.061, 0.223, 0.497, 0.991, 1.0], [1.0, 0.4069410641098421, 0.055826718410139614, 0.02986096067348952, -0.00932483840956811, -0.01081829455164132, -0.01081829455164132]), ([0.004, 0.024, 0.097, 0.165, 0.35, 0.65, 1.0], [1.0, 0.03661569549950533, -0.002966385441010451, 0.06567244784571218, 0.0698290862752741, 0.114204054265474, 0.114204054265474]), ([0.004, 0.024, 1.0], [1.0, 0.1508863014744922, 0.1508863014744922]), ([0.004, 0.024, 0.061, 0.226, 0.568, 1.0], [1.0, 1.000413388990069, 0.4277953620839175, 0.2906760921274553, 0.0638187685548971, 0.027795141406066832]), ([0.004, 0.024, 0.238, 1.0], [1.0, 0.4383064730017593, 0.013101228455914804, 0.013101228455914804]), ([0.004, 0.027, 0.062, 0.261, 1.0], [1.0, 0.4214108767276225, 0.08755537122255577, 0.03946092076737817, 0.03946092076737817]), ([0.004, 0.025, 0.06, 0.111, 0.153, 0.257, 1.0], [1.0, 0.4301514964139892, 0.15607716310293948, 0.13075375004093404, 0.1947183026034135, 0.0453511319379928, 0.0453511319379928]), ([0.004, 0.024, 0.057, 0.073, 0.124, 0.273, 0.684, 1.0], [1.0, 0.31926696463054066, 0.1300739784679044, -0.016619454869316597, 0.025910061696439304, 0.05700734472658521, 0.01873203745132978, 0.01873203745132978]), ([0.004, 0.025, 1.0], [1.0, 0.12578406782168644, 0.12578406782168644]), ([0.004, 0.025, 0.132, 1.0], [1.0, 0.2785452950512816, 0.095918249067967, 0.095918249067967]), ([0.004, 0.025, 0.618, 1.0], [1.0, 0.018343653279413324, 0.0018888264970468617, 0.0018888264970468617]), ([0.004, 0.024, 0.12, 1.0], [1.0, 0.10922366055936898, 0.07004478952229419, 0.07004478952229419]), ([0.004, 0.024, 0.086, 1.0], [1.0, 0.038881783915113675, 0.07605218883917306, 0.07605218883917306])]"

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
