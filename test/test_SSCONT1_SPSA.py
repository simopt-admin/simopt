import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class test_SSCONT1_SPSA(unittest.TestCase):
    def setUp(self):
        # Expected values
        self.expected_problem_name = "SSCONT-1"
        self.expected_solver_name = "SPSA"
        self.expected_all_recommended_xs = "[[(600, 600), (599.2872982749515, 599.2872982749515), (598.7845545608287, 598.8881253296626), (598.4127241074534, 598.5579971331124), (597.8471457340399, 598.0221481731817), (597.6158505168473, 597.8286983148453), (597.4079067970073, 597.6441746009053), (597.402583500057, 597.6494978978557), (597.402583500057, 597.6494978978557)], [(600, 600), (599.0426358459758, 599.0426358459758), (598.7822142856272, 598.9953442839411), (598.6262024702149, 598.8791661765002), (598.2712656902249, 598.5521358017697), (598.2712656902249, 598.5521358017697)], [(600, 600), (599.4392346690771, 599.4392346690771), (599.0214634818176, 599.0214634818176), (598.366993350482, 598.366993350482), (598.366993350482, 598.366993350482)], [(600, 600), (600, 600)], [(600, 600), (599.492399063192, 599.492399063192), (599.1314948928496, 599.1314948928496), (598.3581971271973, 598.6335017585271), (598.1262783865739, 598.6258333918236), (598.1262783865739, 598.6258333918236)], [(600, 600), (598.7254532688579, 598.7254532688579), (597.7426423939182, 597.7426423939182), (596.9407092249073, 596.9407092249073), (596.2432541374054, 596.2432541374054), (595.6393761828489, 595.6393761828489), (595.0991611466105, 595.0991611466105), (594.0409370142887, 593.7796640514524), (593.7542061300892, 593.4000081014597), (593.7542061300892, 593.4000081014597)], [(600, 600), (599.9672777198275, 600.0327222801725), (599.9672777198275, 600.0327222801725)], [(600, 600), (599.9, 599.9), (599.68054058265, 599.6810261434726), (599.6495132230045, 599.6562054867165), (599.6220719009358, 599.6350357917096), (599.6193597647665, 599.6377479278789), (599.6193597647665, 599.6377479278789)], [(600, 600), (599.7736378818547, 599.7736378818547), (599.7736378818547, 599.7736378818547)], [(600, 600), (599.9, 599.9), (599.8071867387588, 599.9035526671404), (599.8071867387588, 599.9035526671404)], [(600, 600), (599.5674779959809, 599.5674779959809), (599.2487750114585, 599.2487750114585), (599.1799384817488, 599.3176115411683), (598.765948438997, 598.9885872625542), (598.7348274564398, 599.0197082451115), (598.5588153989044, 598.933472188258), (598.5588153989044, 598.933472188258)], [(600, 600), (597.6843600424194, 597.6843600424194), (597.1914526567381, 597.1914526567381), (596.7449139508727, 596.8798278604726), (596.0924496247968, 596.3364217449649), (595.8213730345334, 596.0653451547015), (595.8213730345334, 596.0653451547015)], [(600, 600), (599.9, 599.9), (599.9, 599.9)], [(600, 600), (599.9, 599.9), (599.7325376895627, 599.7325376895627), (599.6456846522889, 599.6865751043553), (599.6034489658774, 599.6753887941269), (599.499693339402, 599.6006731494092), (599.499693339402, 599.6006731494092)], [(600, 600), (599.9, 599.9), (599.9, 599.9)], [(600, 600), (599.9961315396787, 600.0038684603213), (599.9921064559443, 600.0078935440557), (599.9020281986134, 599.931275608204), (599.8940857575313, 599.9392180492861), (599.8940857575313, 599.9392180492861)], [(600, 600), (599.9, 599.9), (599.9082465343852, 599.8917534656148), (599.8270766111767, 599.7700243677203), (599.8287310489886, 599.7683699299084), (599.8287310489886, 599.7683699299084)], [(600, 600), (600, 600)], [(600, 600), (599.9252152354165, 599.9310132691492), (599.9180542585609, 599.9381742460048), (599.9125979482538, 599.9436305563119), (599.8708907859616, 599.9113067398862), (599.8708907859616, 599.9113067398862)], [(600, 600), (599.9325860182214, 599.9512960213176), (599.8134186706583, 599.841282219368), (599.7819724984737, 599.8186610854103), (599.775869712484, 599.8247638714), (599.7079010038821, 599.7679466915146), (599.7079010038821, 599.7679466915146)], [(600, 600), (599.9, 599.9), (599.9, 599.9)], [(600, 600), (599.918402116375, 599.9389217748234), (599.918402116375, 599.9389217748234)], [(600, 600), (599.890255973487, 599.9602350603311), (599.866025318666, 599.9844657151522), (599.8032800021269, 599.9632399743537), (599.6714027680456, 599.9279487791179), (599.6714027680456, 599.9279487791179)], [(600, 600), (599.1093010570777, 599.3093010570777), (598.7932182370974, 599.1102710573654), (598.7932182370974, 599.1102710573654)]]"
        self.expected_all_intermediate_budgets = "[[0, 210, 330, 450, 630, 810, 930, 990, 1000], [0, 330, 570, 690, 930, 1000], [0, 210, 270, 390, 1000], [0, 1000], [0, 210, 270, 630, 930, 1000], [0, 210, 270, 330, 390, 450, 510, 810, 930, 1000], [0, 210, 1000], [0, 210, 570, 690, 810, 870, 1000], [0, 330, 1000], [0, 210, 450, 1000], [0, 210, 270, 330, 510, 570, 750, 1000], [0, 330, 390, 510, 690, 750, 1000], [0, 210, 1000], [0, 210, 390, 570, 690, 990, 1000], [0, 210, 1000], [0, 210, 270, 450, 510, 1000], [0, 210, 270, 810, 870, 1000], [0, 1000], [0, 270, 330, 390, 510, 1000], [0, 330, 570, 690, 750, 990, 1000], [0, 210, 1000], [0, 270, 1000], [0, 270, 330, 450, 810, 1000], [0, 330, 450, 1000]]"
        self.expected_all_est_objectives = "[[618.5809976153716, 618.3692233528762, 617.9299406028401, 617.6746457035173, 617.1732892507664, 616.9857516330617, 616.8577169250245, 616.8577169250245, 616.8577169250245], [619.371245290233, 618.2272994305168, 618.0887271690285, 618.0249728242222, 617.7547609969663, 617.7547609969663], [620.2040298994102, 619.64783408728, 619.5306295828523, 619.4342585968005, 619.4342585968005], [620.3929887875448, 620.3929887875448], [617.140803174291, 616.977964795912, 616.7649833217964, 616.1071648472699, 615.9963528483212, 615.9963528483212], [617.6250759903628, 617.1869781633847, 616.9055776556821, 616.4670237125466, 615.9587677574567, 615.2456251770844, 614.9413427463967, 614.3346396838655, 613.890611420458, 613.890611420458], [622.8299886318688, 622.8155391889641, 622.8155391889641], [617.1638109984892, 617.0908993444739, 616.8045016641678, 616.774009802673, 616.7534117021602, 616.7534117021602, 616.7534117021602], [625.4509909440814, 625.4314168403478, 625.4314168403478], [616.3517529689802, 616.5177270489065, 616.4784096549038, 616.4784096549038], [620.885724515664, 620.4999828391695, 620.0931890014077, 620.0775643127557, 619.80616287101, 619.776503954566, 619.6990194679445, 619.6990194679445], [618.8018614763591, 616.4589615187879, 615.7428938133063, 615.2945592102758, 614.5642869534898, 613.7425048578112, 613.7425048578112], [619.9876951863847, 619.9832212351804, 619.9832212351804], [620.5752955225382, 620.6258336604656, 620.6885782766934, 620.6518414280144, 620.4415319131963, 620.34298109015, 620.34298109015], [624.383736629565, 624.2496678836155, 624.2496678836155], [621.6851306868389, 621.6851306868389, 621.6851306868389, 621.9100465448226, 621.8213685238633, 621.8213685238633], [619.7087176057688, 619.565964346056, 619.6555069506658, 619.6502682930953, 619.6502682930953, 619.6502682930953], [623.808754566687, 623.808754566687], [618.1205002744614, 618.055741470913, 618.055741470913, 618.055741470913, 617.9562109669075, 617.9562109669075], [621.6456203016469, 621.5938127018212, 621.5262192699106, 621.2863379412372, 621.5020810993941, 621.2305924709416, 621.2305924709416], [617.7541201292993, 617.6975681405688, 617.6975681405688], [626.0524700155847, 626.0251862549349, 626.0251862549349], [616.5518744333754, 616.2150656519393, 616.4871521963182, 616.1788591100899, 616.1068049586391, 616.1068049586391], [619.165760431194, 618.5416904863797, 618.4514832714677, 618.4514832714677]]"
        self.expected_objective_curves = "[([0, 210, 330, 450, 630, 810, 930, 990, 1000], [624.4131899421741, 618.3692233528762, 617.9299406028401, 617.6746457035173, 617.1732892507664, 616.9857516330617, 616.8577169250245, 616.8577169250245, 616.8577169250245]), ([0, 330, 570, 690, 930, 1000], [624.4131899421741, 618.2272994305168, 618.0887271690285, 618.0249728242222, 617.7547609969663, 617.7547609969663]), ([0, 210, 270, 390, 1000], [624.4131899421741, 619.64783408728, 619.5306295828523, 619.4342585968005, 619.4342585968005]), ([0, 1000], [624.4131899421741, 624.4131899421741]), ([0, 210, 270, 630, 930, 1000], [624.4131899421741, 616.977964795912, 616.7649833217964, 616.1071648472699, 615.9963528483212, 615.9963528483212]), ([0, 210, 270, 330, 390, 450, 510, 810, 930, 1000], [624.4131899421741, 617.1869781633847, 616.9055776556821, 616.4670237125466, 615.9587677574567, 615.2456251770844, 614.9413427463967, 614.3346396838655, 613.890611420458, 613.890611420458]), ([0, 210, 1000], [624.4131899421741, 622.8155391889641, 622.8155391889641]), ([0, 210, 570, 690, 810, 870, 1000], [624.4131899421741, 617.0908993444739, 616.8045016641678, 616.774009802673, 616.7534117021602, 616.7534117021602, 616.7534117021602]), ([0, 330, 1000], [624.4131899421741, 625.4314168403478, 625.4314168403478]), ([0, 210, 450, 1000], [624.4131899421741, 616.5177270489065, 616.4784096549038, 616.4784096549038]), ([0, 210, 270, 330, 510, 570, 750, 1000], [624.4131899421741, 620.4999828391695, 620.0931890014077, 620.0775643127557, 619.80616287101, 619.776503954566, 619.6990194679445, 619.6990194679445]), ([0, 330, 390, 510, 690, 750, 1000], [624.4131899421741, 616.4589615187879, 615.7428938133063, 615.2945592102758, 614.5642869534898, 622.4727011633026, 622.4727011633026]), ([0, 210, 1000], [624.4131899421741, 619.9832212351804, 619.9832212351804]), ([0, 210, 390, 570, 690, 990, 1000], [624.4131899421741, 620.6258336604656, 620.6885782766934, 620.6518414280144, 620.4415319131963, 620.34298109015, 620.34298109015]), ([0, 210, 1000], [624.4131899421741, 624.2496678836155, 624.2496678836155]), ([0, 210, 270, 450, 510, 1000], [624.4131899421741, 621.6851306868389, 621.6851306868389, 621.9100465448226, 621.8213685238633, 621.8213685238633]), ([0, 210, 270, 810, 870, 1000], [624.4131899421741, 619.565964346056, 619.6555069506658, 619.6502682930953, 619.6502682930953, 619.6502682930953]), ([0, 1000], [624.4131899421741, 624.4131899421741]), ([0, 270, 330, 390, 510, 1000], [624.4131899421741, 618.055741470913, 618.055741470913, 618.055741470913, 617.9562109669075, 617.9562109669075]), ([0, 330, 570, 690, 750, 990, 1000], [624.4131899421741, 621.5938127018212, 621.5262192699106, 621.2863379412372, 621.5020810993941, 621.2305924709416, 621.2305924709416]), ([0, 210, 1000], [624.4131899421741, 617.6975681405688, 617.6975681405688]), ([0, 270, 1000], [624.4131899421741, 626.0251862549349, 626.0251862549349]), ([0, 270, 330, 450, 810, 1000], [624.4131899421741, 616.2150656519393, 616.4871521963182, 616.1788591100899, 616.1068049586391, 616.1068049586391]), ([0, 330, 450, 1000], [624.4131899421741, 618.5416904863797, 618.4514832714677, 618.4514832714677])]"
        self.expected_progress_curves = "[([0.0, 0.21, 0.33, 0.45, 0.63, 0.81, 0.93, 0.99, 1.0], [1.0, -2.1146619630611414, -2.341039334999151, -2.472601497121688, -2.730967563552881, -2.8276120892758225, -2.893592737775945, -2.893592737775945, -2.893592737775945]), ([0.0, 0.33, 0.57, 0.69, 0.93, 1.0], [1.0, -2.187800197048741, -2.259211205964192, -2.2920659926037534, -2.4313153560621115, -2.4313153560621115]), ([0.0, 0.21, 0.27, 0.39, 1.0], [1.0, -1.455750276312114, -1.5161497517967817, -1.5658130052518138, -1.5658130052518138]), ([0.0, 1.0], [1.0, 1.0]), ([0.0, 0.21, 0.27, 0.63, 0.93, 1.0], [1.0, -2.831624911836006, -2.9413815238992247, -3.280377802408535, -3.3374830019620174, -3.3374830019620174]), ([0.0, 0.21, 0.27, 0.33, 0.39, 0.45, 0.51, 0.81, 0.93, 1.0], [1.0, -2.723913200359804, -2.8689284721647703, -3.094930265069009, -3.3568518801919276, -3.7243585559002153, -3.8811656624399995, -4.193820427124627, -4.42264332383097, -4.42264332383097]), ([0.0, 0.21, 1.0], [1.0, 0.17667611861214233, 0.17667611861214233]), ([0.0, 0.21, 0.57, 0.69, 0.81, 0.87, 1.0], [1.0, -2.77342589012998, -2.9210163752822105, -2.9367298706792164, -2.9473447738609035, -2.9473447738609035, -2.9473447738609035]), ([0.0, 0.33, 1.0], [1.0, 1.524727022006211, 1.524727022006211]), ([0.0, 0.21, 0.45, 1.0], [1.0, -3.0688011078628588, -3.0890627009371654, -3.0890627009371654]), ([0.0, 0.21, 0.27, 0.33, 0.51, 0.57, 0.75, 1.0], [1.0, -1.0166089830627114, -1.2262437112771203, -1.2342956458320233, -1.3741580581792852, -1.3894423086046925, -1.4293727052476417, -1.4293727052476417]), ([0.0, 0.33, 0.39, 0.51, 0.69, 0.75, 1.0], [1.0, -3.0990849882740745, -3.4680990806400365, -3.6991411809150234, -4.075475362662092, 0.0, 0.0]), ([0.0, 0.21, 1.0], [1.0, -1.2829138489376346, -1.2829138489376346]), ([0.0, 0.21, 0.39, 0.57, 0.69, 0.99, 1.0], [1.0, -0.9517537658275953, -0.9194193267362487, -0.9383510768597076, -1.0467307372360364, -1.097517334983625, -1.097517334983625]), ([0.0, 0.21, 1.0], [1.0, 0.915731510360159, 0.915731510360159]), ([0.0, 0.21, 0.27, 0.45, 0.51, 1.0], [1.0, -0.4058619070818898, -0.4058619070818898, -0.2899551002852116, -0.3356539066503542, -0.3356539066503542]), ([0.0, 0.21, 0.27, 0.81, 0.87, 1.0], [1.0, -1.4979405441020512, -1.4517961883166963, -1.4544958470972418, -1.4544958470972418, -1.4544958470972418]), ([0.0, 1.0], [1.0, 1.0]), ([0.0, 0.27, 0.33, 0.39, 0.51, 1.0], [1.0, -2.276209860362364, -2.276209860362364, -2.276209860362364, -2.32750132109594, -2.32750132109594]), ([0.0, 0.33, 0.57, 0.69, 0.75, 0.99, 1.0], [1.0, -0.45292117689677464, -0.48775437595805465, -0.6113733998273411, -0.500193597858906, -0.6401009404875041, -0.6401009404875041]), ([0.0, 0.21, 1.0], [1.0, -2.460788784107771, -2.460788784107771]), ([0.0, 0.27, 1.0], [1.0, 1.8307166371239103, 1.8307166371239103]), ([0.0, 0.27, 0.33, 0.45, 0.81, 1.0], [1.0, -3.224772840481294, -3.0845573713987315, -3.2434313054225243, -3.280563265285054, -3.280563265285054]), ([0.0, 0.33, 0.45, 1.0], [1.0, -2.0257837714522395, -2.0722706235763906, -2.0722706235763906])]"

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
