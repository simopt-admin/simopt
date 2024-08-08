import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class test_MM11_RNDSRCH(unittest.TestCase):
    def setUp(self):
        # Expected values
        self.expected_problem_name = "MM1-1"
        self.expected_solver_name = "RNDSRCH"
        self.expected_all_recommended_xs = "[[(5,), (4.296923990261382,), (2.377685478570012,), (2.9477762989744125,), (2.8725872351049095,), (2.772194393803471,), (2.772194393803471,)], [(5,), (3.985053300589926,), (2.5692082889849917,), (2.678966873705628,), (2.7655068232601367,), (2.7655068232601367,)], [(5,), (3.8484877158342767,), (3.0893992471426497,), (3.0474371736837815,), (2.699569241853814,), (2.8627075639581703,), (2.8627075639581703,)], [(5,), (3.773796218317722,), (2.638230326582391,), (2.9681054377092044,), (2.9600716575770796,), (2.9600716575770796,)], [(5,), (1.9121296882013312,), (2.865358591560582,), (2.865358591560582,)], [(5,), (3.4301202689575265,), (2.3471789635658644,), (2.971703077078345,), (2.7473614222541527,), (2.7473614222541527,)], [(5,), (3.061329038300266,), (2.6213500819625137,), (2.75793594287965,), (2.781142773961063,), (2.781142773961063,)], [(5,), (3.4557418828007354,), (2.9876013780902837,), (2.891632856275529,), (2.754896253073898,), (2.754896253073898,)], [(5,), (3.0560414166146552,), (2.798595332742974,), (2.798595332742974,)], [(5,), (2.0277986812387487,), (2.46722995472664,), (3.0783373963808143,), (2.8431466204775697,), (2.833368087907808,), (2.833368087907808,)], [(5,), (2.1730539546151535,), (2.5077210838663055,), (2.8324316442000788,), (2.8324316442000788,)], [(5,), (3.1713678739798383,), (3.105454922823737,), (3.105454922823737,)], [(5,), (3.7560892026039454,), (2.6424526119366494,), (2.6424526119366494,)], [(5,), (3.1882219968979504,), (2.6346954605019355,), (2.988890362219531,), (2.785538282320524,), (2.785538282320524,)], [(5,), (3.263437291106127,), (2.903370435537305,), (2.8190293269137374,), (2.765398257245309,), (2.765398257245309,)], [(5,), (2.4046001587316237,), (3.2517700113314674,), (3.082235460189414,), (2.669010127674985,), (2.7600798854453057,), (2.7600798854453057,)], [(5,), (2.3072447165712027,), (3.1500719309497094,), (2.9840940213561105,), (2.9840940213561105,)], [(5,), (3.9208238863700844,), (2.8666971699924857,), (2.860958781807017,), (2.860958781807017,)], [(5,), (2.7340183872964703,), (2.917164902536248,), (2.917164902536248,)], [(5,), (4.3013521065005635,), (2.1214675932257014,), (2.7277692939048337,), (2.9621093835909305,), (2.961643042704625,), (2.961643042704625,)], [(5,), (2.3218762530341057,), (3.1735813849084233,), (2.9823619348246106,), (2.7586733244223978,), (2.7586733244223978,)], [(5,), (2.349191960285218,), (2.795141998380235,), (2.87290614633717,), (2.87290614633717,)], [(5,), (2.7005266439516853,), (2.8171004160319555,), (2.8171004160319555,)], [(5,), (3.7919543550088965,), (2.6185359342178978,), (2.787074332418977,), (2.787074332418977,)]]"
        self.expected_all_intermediate_budgets = "[[0, 60, 130, 230, 620, 800, 1000], [0, 20, 240, 370, 990, 1000], [0, 80, 200, 370, 420, 690, 1000], [0, 40, 60, 120, 130, 1000], [0, 40, 70, 1000], [0, 20, 50, 150, 370, 1000], [0, 20, 200, 230, 420, 1000], [0, 80, 170, 320, 410, 1000], [0, 120, 340, 1000], [0, 40, 100, 240, 540, 890, 1000], [0, 30, 70, 80, 1000], [0, 30, 110, 1000], [0, 50, 80, 1000], [0, 20, 160, 430, 450, 1000], [0, 30, 60, 170, 820, 1000], [0, 30, 260, 480, 530, 650, 1000], [0, 30, 50, 120, 1000], [0, 50, 90, 120, 1000], [0, 60, 230, 1000], [0, 50, 70, 80, 670, 730, 1000], [0, 20, 110, 180, 440, 1000], [0, 30, 50, 160, 1000], [0, 60, 670, 1000], [0, 50, 120, 250, 1000]]"
        self.expected_all_est_objectives = "[[2.7852321578182706, 2.2031388012791577, 1.6933515906330974, 1.5532659579862353, 1.5466636819482629, 1.5466569750020915, 1.5466569750020915], [2.7857037031168543, 1.9900956095535671, 1.5875356719512517, 1.559610708067732, 1.549585693229223, 1.549585693229223], [2.7866293625352507, 1.9102060479378624, 1.5927288694029482, 1.5844274565222343, 1.5767432376944044, 1.5651124409141386, 1.5651124409141386], [2.7889080044387127, 1.8699426487564068, 1.5888074513529902, 1.572898711985816, 1.5719529323204375, 1.5719529323204375], [2.7833651638972787, 2.716122200172547, 1.5430157510654368, 1.5430157510654368], [2.787955763524055, 1.7005668937245308, 1.7731170121688715, 1.5731787324099513, 1.5737336931676196, 1.5737336931676196], [2.7843462630059106, 1.5730671354449515, 1.567980237243296, 1.5471789891695702, 1.5460352551469974, 1.5460352551469974], [2.7907221687784363, 1.720987076480683, 1.587321657640274, 1.579083169733054, 1.5840464809440606, 1.5840464809440606], [2.789502875694011, 1.5850413713046327, 1.56332141744088, 1.56332141744088], [2.7891645344327056, 2.3911416272269825, 1.677984111037892, 1.5955139999950716, 1.5721201794950088, 1.572282353214191, 1.572282353214191], [2.7863020842335002, 1.9233733152312782, 1.6081209366996734, 1.5494638903639646, 1.5494638903639646], [2.781108319206661, 1.5932821297188673, 1.5759151715576163, 1.5759151715576163], [2.781564747274972, 1.8476163827218754, 1.552314162541785, 1.552314162541785], [2.7819442310007103, 1.5938276568791048, 1.5476066178447347, 1.5468364952156264, 1.5309196613583334, 1.5309196613583334], [2.784695397913865, 1.6285699164483969, 1.5503383273538227, 1.5480612763304962, 1.5503794996945488, 1.5503794996945488], [2.782112928233372, 1.6559506451649093, 1.6171347222143693, 1.569575159283053, 1.5497866950656987, 1.5387258540205817, 1.5387258540205817], [2.784512429482461, 1.790295011847391, 1.5961508738611374, 1.5616567275168292, 1.5616567275168292], [2.783456075233837, 1.9456364237709358, 1.5427547182216943, 1.5425246367072534, 1.5425246367072534], [2.7872953386099404, 1.5608533325052445, 1.558800541492758, 1.558800541492758], [2.7844968268172887, 2.204560143343991, 2.013184752218728, 1.5462913745752047, 1.553580138889096, 1.5535160179571323, 1.5535160179571323], [2.781707203439503, 1.7105683559165206, 1.5901326564428728, 1.546681513222238, 1.533196273838619, 1.533196273838619], [2.7902297278963424, 1.754059242481837, 1.5672682138802623, 1.5666097329190607, 1.5666097329190607], [2.7850791792196157, 1.5723280848387682, 1.5598996422587066, 1.5598996422587066], [2.7868278653888137, 1.875224207850116, 1.5827668226933276, 1.5552262353075528, 1.5552262353075528]]"
        self.expected_objective_curves = "[([0, 60, 130, 230, 620, 800, 1000], [2.7854035060729516, 2.2031388012791577, 1.6933515906330974, 1.5532659579862353, 1.5466636819482629, 1.5466569750020915, 1.5466569750020915]), ([0, 20, 240, 370, 990, 1000], [2.7854035060729516, 1.9900956095535671, 1.5875356719512517, 1.559610708067732, 1.549585693229223, 1.549585693229223]), ([0, 80, 200, 370, 420, 690, 1000], [2.7854035060729516, 1.9102060479378624, 1.5927288694029482, 1.5844274565222343, 1.5767432376944044, 1.5651124409141386, 1.5651124409141386]), ([0, 40, 60, 120, 130, 1000], [2.7854035060729516, 1.8699426487564068, 1.5888074513529902, 1.572898711985816, 1.5719529323204375, 1.5719529323204375]), ([0, 40, 70, 1000], [2.7854035060729516, 2.716122200172547, 1.5430157510654368, 1.5430157510654368]), ([0, 20, 50, 150, 370, 1000], [2.7854035060729516, 1.7005668937245308, 1.7731170121688715, 1.5731787324099513, 1.5737336931676196, 1.5737336931676196]), ([0, 20, 200, 230, 420, 1000], [2.7854035060729516, 1.5730671354449515, 1.567980237243296, 1.5471789891695702, 1.5460352551469974, 1.5460352551469974]), ([0, 80, 170, 320, 410, 1000], [2.7854035060729516, 1.720987076480683, 1.587321657640274, 1.579083169733054, 1.5840464809440606, 1.5840464809440606]), ([0, 120, 340, 1000], [2.7854035060729516, 1.5850413713046327, 1.56332141744088, 1.56332141744088]), ([0, 40, 100, 240, 540, 890, 1000], [2.7854035060729516, 2.3911416272269825, 1.677984111037892, 1.5955139999950716, 1.5721201794950088, 1.572282353214191, 1.572282353214191]), ([0, 30, 70, 80, 1000], [2.7854035060729516, 1.9233733152312782, 1.6081209366996734, 1.5494638903639646, 1.5494638903639646]), ([0, 30, 110, 1000], [2.7854035060729516, 1.5932821297188673, 1.5759151715576163, 1.5759151715576163]), ([0, 50, 80, 1000], [2.7854035060729516, 1.8476163827218754, 1.552314162541785, 1.552314162541785]), ([0, 20, 160, 430, 450, 1000], [2.7854035060729516, 1.5938276568791048, 1.5476066178447347, 1.5468364952156264, 1.553045723303714, 1.553045723303714]), ([0, 30, 60, 170, 820, 1000], [2.7854035060729516, 1.6285699164483969, 1.5503383273538227, 1.5480612763304962, 1.5503794996945488, 1.5503794996945488]), ([0, 30, 260, 480, 530, 650, 1000], [2.7854035060729516, 1.6559506451649093, 1.6171347222143693, 1.569575159283053, 1.5497866950656987, 1.5387258540205817, 1.5387258540205817]), ([0, 30, 50, 120, 1000], [2.7854035060729516, 1.790295011847391, 1.5961508738611374, 1.5616567275168292, 1.5616567275168292]), ([0, 50, 90, 120, 1000], [2.7854035060729516, 1.9456364237709358, 1.5427547182216943, 1.5425246367072534, 1.5425246367072534]), ([0, 60, 230, 1000], [2.7854035060729516, 1.5608533325052445, 1.558800541492758, 1.558800541492758]), ([0, 50, 70, 80, 670, 730, 1000], [2.7854035060729516, 2.204560143343991, 2.013184752218728, 1.5462913745752047, 1.553580138889096, 1.5535160179571323, 1.5535160179571323]), ([0, 20, 110, 180, 440, 1000], [2.7854035060729516, 1.7105683559165206, 1.5901326564428728, 1.546681513222238, 1.533196273838619, 1.533196273838619]), ([0, 30, 50, 160, 1000], [2.7854035060729516, 1.754059242481837, 1.5672682138802623, 1.5666097329190607, 1.5666097329190607]), ([0, 60, 670, 1000], [2.7854035060729516, 1.5723280848387682, 1.5598996422587066, 1.5598996422587066]), ([0, 50, 120, 250, 1000], [2.7854035060729516, 1.875224207850116, 1.5827668226933276, 1.5552262353075528, 1.5552262353075528])]"
        self.expected_progress_curves = "[([0.0, 0.06, 0.13, 0.23, 0.62, 0.8, 1.0], [1.0, 0.5275197568960989, 0.11385156915559155, 0.0001787100187953636, -0.005178724429451044, -0.005184166798757351, -0.005184166798757351]), ([0.0, 0.02, 0.24, 0.37, 0.99, 1.0], [1.0, 0.3546452924310309, 0.027986960548125144, 0.005327174344828406, -0.0028076506050995083, -0.0028076506050995083]), ([0.0, 0.08, 0.2, 0.37, 0.42, 0.69, 1.0], [1.0, 0.28981869521006437, 0.03220099443041775, 0.02546479087266538, 0.019229411070411335, 0.009791570093637389, 0.009791570093637389]), ([0.0, 0.04, 0.06, 0.12, 0.13, 1.0], [1.0, 0.2571468528730285, 0.02901894932567048, 0.016109760460545956, 0.015342305035991256, 0.015342305035991256]), ([0.0, 0.04, 0.07, 1.0], [1.0, 0.9437815000894283, -0.008138847645152922, -0.008138847645152922]), ([0.0, 0.02, 0.05, 0.15, 0.37, 1.0], [1.0, 0.11970644603657325, 0.17857743257857644, 0.01633698377836044, 0.016787308160960763, 0.016787308160960763]), ([0.0, 0.02, 0.2, 0.23, 0.42, 1.0], [1.0, 0.01624642812434487, 0.012118651051176536, -0.004760577014380254, -0.0056886630284943935, -0.0056886630284943935]), ([0.0, 0.08, 0.17, 0.32, 0.41, 1.0], [1.0, 0.1362764576368293, 0.027813298066360504, 0.021128155145684353, 0.02515564722663952, 0.02515564722663952]), ([0.0, 0.12, 0.34, 1.0], [1.0, 0.025962953655407694, 0.00833823933344701, 0.00833823933344701]), ([0.0, 0.04, 0.1, 0.24, 0.54, 0.89, 1.0], [1.0, 0.680075149961709, 0.10138158697178686, 0.03446099605581008, 0.015478018200552493, 0.015609614496246529, 0.015609614496246529]), ([0.0, 0.03, 0.07, 0.08, 1.0], [1.0, 0.3005033092706236, 0.04469092837000593, -0.002906487864020055, -0.002906487864020055]), ([0.0, 0.03, 0.11, 1.0], [1.0, 0.032649938985038755, 0.018557474601663317, 0.018557474601663317]), ([0.0, 0.05, 0.08, 1.0], [1.0, 0.2390301449277418, -0.0005936269256848907, -0.0005936269256848907]), ([0.0, 0.02, 0.16, 0.43, 0.45, 1.0], [1.0, 0.03309260844991739, -0.004413576588737937, -0.005038494644091808, 0.0, 0.0]), ([0.0, 0.03, 0.06, 0.17, 0.82, 1.0], [1.0, 0.06128430736646304, -0.0021969236432357576, -0.0040446427514072205, -0.0021635142378652055, -0.0021635142378652055]), ([0.0, 0.03, 0.26, 0.48, 0.53, 0.65, 1.0], [1.0, 0.08350247249622353, 0.05200518859599402, 0.013412854781665344, -0.0026445471303731317, -0.011619896010194445, -0.011619896010194445]), ([0.0, 0.03, 0.05, 0.12, 1.0], [1.0, 0.1925165660986478, 0.03497778904804866, 0.0069874222677161835, 0.0069874222677161835]), ([0.0, 0.05, 0.09, 0.12, 1.0], [1.0, 0.31856876789874217, -0.008350663440364512, -0.008537363697106438, -0.008537363697106438]), ([0.0, 0.06, 0.23, 1.0], [1.0, 0.006335505249121674, 0.004669762523114193, 0.004669762523114193]), ([0.0, 0.05, 0.07, 0.08, 0.67, 0.73, 1.0], [1.0, 0.5286731086943401, 0.3733810386469368, -0.005480834237384859, 0.0004336529479134151, 0.00038162184715667566, 0.00038162184715667566]), ([0.0, 0.02, 0.11, 0.18, 0.44, 1.0], [1.0, 0.12782215912885023, 0.030094290519933703, -0.005164255194765744, -0.016106888553494056, -0.016106888553494056]), ([0.0, 0.03, 0.05, 0.16, 1.0], [1.0, 0.16311295468627993, 0.011540877799780512, 0.011006551672734921, 0.011006551672734921]), ([0.0, 0.06, 0.67, 1.0], [1.0, 0.015646723544622442, 0.005561630762448764, 0.005561630762448764]), ([0.0, 0.05, 0.12, 0.25, 1.0], [1.0, 0.2614325880447097, 0.024117265136125565, 0.0017693822640847778, 0.0017693822640847778])]"

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
