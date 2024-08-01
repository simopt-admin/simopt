import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class test_IRONORECONT1_STRONG(unittest.TestCase):
    def setUp(self):
        # Expected values
        self.expected_problem_name = "IRONORECONT-1"
        self.expected_solver_name = "STRONG"
        self.expected_all_recommended_xs = "[[(80, 40, 100), (80.01930400897865, 40.07446358899551, 101.99852008475052), (80.14767474106527, 40.269184088079335, 103.98487475660046), (80.15666435688847, 40.37311093294335, 101.98759700983514), (80.15666435688847, 40.37311093294335, 101.98759700983514)], [(80, 40, 100), (80.06393142612873, 40.0, 98.00102206796744), (80.06393142612873, 40.0, 98.00102206796744)], [(80, 40, 100), (80.0, 40.055414221101834, 101.99923216863367), (80.0, 40.39479867850451, 100.02823795502636), (80.01353109797068, 40.70761923101493, 102.00357595428491), (80.01353109797068, 40.70761923101493, 102.00357595428491)], [(80, 40, 100), (80, 40, 100)], [(80, 40, 100), (79.99888053118211, 40.13989122284082, 101.9951012988221), (79.99888053118211, 40.13989122284082, 101.9951012988221)], [(80, 40, 100), (80.00436602023422, 40.04296730554915, 101.99953363275569), (80.02053417576117, 40.16107893403187, 100.00308973573178), (80.02552329457308, 40.19167835244931, 102.00284941736504), (80.02552329457308, 40.19167835244931, 102.00284941736504)], [(80, 40, 100), (80, 40, 100)], [(80, 40, 100), (80.0, 40.05108013805388, 101.99934759846715), (80.0, 40.05407218808596, 99.99934983655925), (80.0, 40.074015964360434, 101.99925039553409), (80.0, 40.07745417300037, 99.99925335085594), (80.0, 40.4452531368485, 101.96514346024462), (80.0, 40.55662898298251, 99.96824701303011), (80.0, 40.55662898298251, 99.96824701303011)], [(80, 40, 100), (80, 40, 100)], [(80, 40, 100), (79.99636768431262, 40.35706723857862, 98.03213572789774), (80.04898217424699, 42.261261984477635, 98.6414583724298), (80.04898217424699, 42.261261984477635, 98.6414583724298)], [(80, 40, 100), (80.0, 40.62076063878095, 101.90122492865528), (80.0, 40.62076063878095, 101.90122492865528)], [(80, 40, 100), (80.00059296563177, 40.02815712981671, 101.99980169627698), (80.00503580343819, 40.239126457831524, 100.01096479903548), (80.00515307487123, 40.27786836367937, 102.0105895265728), (80.0086281816409, 41.4259080506897, 100.37291043692119), (80.01108641793805, 41.45451570176904, 102.37270431614326), (80.01108641793805, 41.45451570176904, 102.37270431614326)], [(80, 40, 100), (80.04349243409415, 40.731068377974985, 101.86108770209832), (80.04349243409415, 40.731068377974985, 101.86108770209832)], [(80, 40, 100), (80.0, 40.159953350527736, 101.99359347050871), (80.0, 40.159953350527736, 101.99359347050871)], [(80, 40, 100), (79.99580308462615, 41.22717993088093, 98.4207559393354), (79.9887836115072, 42.531958845019375, 96.90500440553546), (80.00080696622854, 42.69939438374892, 98.89794714914643), (79.99750628811078, 42.93662793452097, 100.88382462470027), (79.99709723171223, 43.0090155909061, 102.88251416034004), (79.99709723171223, 43.0090155909061, 102.88251416034004)], [(80, 40, 100), (80.00686198553603, 40.053059235012896, 101.99928427961967), (80.01774990219023, 40.12087543861286, 103.99810453562075), (80.01774990219023, 40.12087543861286, 103.99810453562075)], [(80, 40, 100), (79.99560929101102, 40.925537671749595, 98.22704745132677), (80.0274483276814, 41.79146180792874, 100.02959056208066), (80.0274483276814, 41.79146180792874, 100.02959056208066)], [(80, 40, 100), (79.9961753275655, 40.09869437352935, 101.99755970937402), (80.91703267615034, 40.34249088713411, 103.75613441734393), (80.91703267615034, 40.34249088713411, 103.75613441734393)], [(80, 40, 100), (80.0, 40.1650034711469, 101.99318184180709), (80.0, 40.66965820432217, 100.05789798658486), (80.0, 40.798373451719655, 102.05375178511005), (80.0, 40.798373451719655, 102.05375178511005)], [(80, 40, 100), (80.03082171255735, 40.03565248164706, 101.99944465354444), (80.0534890300827, 40.062114076242125, 99.99974818239497), (80.0534890300827, 40.062114076242125, 99.99974818239497)], [(80, 40, 100), (80.0, 40.48317629124692, 98.05924224294301), (80.0, 41.14092881688226, 96.1704961999239), (80.0, 41.65864107991922, 94.23866442342235), (80.0, 41.84655356528272, 96.22981707900767), (80.0, 42.68278690238036, 94.41303040855719), (80.0, 42.95570412879428, 96.39432195386962), (80.0, 42.95570412879428, 96.39432195386962)], [(80, 40, 100), (80, 40, 100)], [(80, 40, 100), (80.0, 40.73916411995068, 101.85839619128365), (79.99673243217131, 41.57128749982687, 103.67706603588543), (79.99501910390815, 41.98199205542282, 105.63444153193083), (79.99448117076302, 42.02763669621979, 103.63496253044208), (79.99448117076302, 42.02763669621979, 103.63496253044208)], [(80, 40, 100), (80.56862863202623, 41.51962860790261, 98.83064531775686), (81.11587741325205, 41.73073933108679, 96.91859117783207), (82.47638170531233, 42.714170483130374, 98.00573939241725), (82.47638170531233, 42.714170483130374, 98.00573939241725)]]"
        self.expected_all_intermediate_budgets = "[[10, 80, 430, 535, 1000], [10, 157, 1000], [10, 80, 332, 430, 1000], [10, 1000], [10, 80, 1000], [10, 80, 241, 332, 1000], [10, 1000], [10, 80, 157, 241, 332, 535, 892, 1000], [10, 1000], [10, 80, 332, 1000], [10, 157, 1000], [10, 80, 157, 241, 332, 430, 1000], [10, 80, 1000], [10, 80, 1000], [10, 332, 430, 647, 766, 892, 1000], [10, 80, 241, 1000], [10, 157, 241, 1000], [10, 80, 157, 1000], [10, 80, 157, 241, 1000], [10, 332, 430, 1000], [10, 80, 157, 430, 535, 647, 766, 1000], [10, 1000], [10, 80, 157, 241, 332, 1000], [10, 80, 157, 332, 1000]]"
        self.expected_all_est_objectives = "[[149895.8055235529, 148477.7274778582, 141044.12952090046, 148492.96399261276, 148492.96399261276], [122793.09736468189, 120197.12182066315, 120197.12182066315], [99852.80349485856, 84568.07900562166, 99958.3701362542, 84698.53949921897, 84698.53949921897], [126011.12695446546, 126011.12695446546], [136147.71179130895, 133702.03874348794, 133702.03874348794], [132850.26196652921, 131969.56278031142, 132947.82175185136, 132016.88135926626, 132016.88135926626], [134982.68434045353, 134982.68434045353], [161256.2908821113, 158053.61037127997, 161256.2908821113, 158053.61037127997, 161257.80298595421, 157786.45114511155, 161189.10153840613, 161189.10153840613], [146337.47315675917, 146337.47315675917], [134867.2205665852, 123320.73717829157, 129089.95839731517, 129089.95839731517], [149243.01256369415, 146124.09881893423, 146124.09881893423], [112822.77485929335, 101512.07685631915, 112659.17891335982, 101923.9969158206, 113527.79827962496, 101301.09778454759, 101301.09778454759], [132809.38556277155, 126115.61165684783, 126115.61165684783], [118379.15455996453, 115161.5222917321, 115161.5222917321], [127606.7164810152, 118533.75719570466, 114816.84011873225, 124224.15652539047, 128662.26059129316, 125170.60036673951, 125170.60036673951], [145498.2552215891, 143303.41278434847, 127809.43326583259, 127809.43326583259], [161264.15011124164, 154491.01787398063, 162423.69110736647, 162423.69110736647], [132500.94479520118, 123474.63744989334, 112019.73132980526, 112019.73132980526], [112031.98326897933, 107866.00241060051, 113001.83742953719, 108554.01038891386, 108554.01038891386], [130863.18264271188, 122795.6749530164, 130863.18264271188, 130863.18264271188], [147610.26102665017, 140781.48814155225, 132230.73585178048, 122449.32496627721, 132234.24352654073, 123648.36419800289, 133791.76990130323, 133791.76990130323], [132677.02997009846, 132677.02997009846], [132803.08586581453, 125694.76691102673, 114356.87632543869, 88010.17966383627, 114772.0492647192, 114772.0492647192], [137521.1409071744, 135582.38462594422, 124354.4123240913, 130297.89970464252, 130297.89970464252]]"
        self.expected_objective_curves = "[([10, 80, 430, 535, 1000], [121270.73497283501, 148477.7274778582, 141044.12952090046, 148492.96399261276, 148492.96399261276]), ([10, 157, 1000], [121270.73497283501, 120197.12182066315, 120197.12182066315]), ([10, 80, 332, 430, 1000], [121270.73497283501, 84568.07900562166, 99958.3701362542, 84698.53949921897, 84698.53949921897]), ([10, 1000], [121270.73497283501, 121270.73497283501]), ([10, 80, 1000], [121270.73497283501, 133702.03874348794, 133702.03874348794]), ([10, 80, 241, 332, 1000], [121270.73497283501, 131969.56278031142, 132947.82175185136, 132016.88135926626, 132016.88135926626]), ([10, 1000], [121270.73497283501, 121270.73497283501]), ([10, 80, 157, 241, 332, 535, 892, 1000], [121270.73497283501, 158053.61037127997, 161256.2908821113, 158053.61037127997, 161257.80298595421, 157786.45114511155, 161189.10153840613, 161189.10153840613]), ([10, 1000], [121270.73497283501, 121270.73497283501]), ([10, 80, 332, 1000], [121270.73497283501, 123320.73717829157, 129089.95839731517, 129089.95839731517]), ([10, 157, 1000], [121270.73497283501, 146124.09881893423, 146124.09881893423]), ([10, 80, 157, 241, 332, 430, 1000], [121270.73497283501, 101512.07685631915, 112659.17891335982, 101923.9969158206, 113527.79827962496, 101301.09778454759, 101301.09778454759]), ([10, 80, 1000], [121270.73497283501, 126115.61165684783, 126115.61165684783]), ([10, 80, 1000], [121270.73497283501, 115161.5222917321, 115161.5222917321]), ([10, 332, 430, 647, 766, 892, 1000], [121270.73497283501, 118533.75719570466, 114816.84011873225, 124224.15652539047, 128662.26059129316, 125170.60036673951, 125170.60036673951]), ([10, 80, 241, 1000], [121270.73497283501, 143303.41278434847, 127809.43326583259, 127809.43326583259]), ([10, 157, 241, 1000], [121270.73497283501, 154491.01787398063, 121655.54226105828, 121655.54226105828]), ([10, 80, 157, 1000], [121270.73497283501, 123474.63744989334, 112019.73132980526, 112019.73132980526]), ([10, 80, 157, 241, 1000], [121270.73497283501, 107866.00241060051, 113001.83742953719, 108554.01038891386, 108554.01038891386]), ([10, 332, 430, 1000], [121270.73497283501, 122795.6749530164, 130863.18264271188, 130863.18264271188]), ([10, 80, 157, 430, 535, 647, 766, 1000], [121270.73497283501, 140781.48814155225, 132230.73585178048, 122449.32496627721, 132234.24352654073, 123648.36419800289, 133791.76990130323, 133791.76990130323]), ([10, 1000], [121270.73497283501, 121270.73497283501]), ([10, 80, 157, 241, 332, 1000], [121270.73497283501, 125694.76691102673, 114356.87632543869, 88010.17966383627, 114772.0492647192, 114772.0492647192]), ([10, 80, 157, 332, 1000], [121270.73497283501, 135582.38462594422, 124354.4123240913, 130297.89970464252, 130297.89970464252])]"
        self.expected_progress_curves = "[([0.01, 0.08, 0.43, 0.535, 1.0], [1.0, -69.70290334323815, -50.38518721764144, -69.7424985256083, -69.7424985256083]), ([0.01, 0.157, 1.0], [1.0, 3.790002125814597, 3.790002125814597]), ([0.01, 0.08, 0.332, 0.43, 1.0], [1.0, 96.3793160640927, 56.384514505908484, 96.0402879386129, 96.0402879386129]), ([0.01, 1.0], [1.0, 1.0]), ([0.01, 0.08, 1.0], [1.0, -31.30527110869129, -31.30527110869129]), ([0.01, 0.08, 0.241, 0.332, 1.0], [1.0, -26.80307996991189, -29.345284864358558, -26.926046920910466, -26.926046920910466]), ([0.01, 1.0], [1.0, 1.0]), ([0.01, 0.08, 0.157, 0.241, 0.332, 0.535, 0.892, 1.0], [1.0, -94.5877825710604, -102.91059923500343, -94.5877825710604, -102.91452874436875, -93.8935149874033, -102.73599406051365, -102.73599406051365]), ([0.01, 1.0], [1.0, 1.0]), ([0.01, 0.08, 0.332, 1.0], [1.0, -4.32734765737375, -19.31984233090572, -19.31984233090572]), ([0.01, 0.157, 1.0], [1.0, -63.586520595418726, -63.586520595418726]), ([0.01, 0.08, 0.157, 0.241, 0.332, 0.43, 1.0], [1.0, 52.3468916031856, 23.378879826409985, 51.276433552862905, 21.12159574461484, 52.89516362980365, 52.89516362980365]), ([0.01, 0.08, 1.0], [1.0, -11.590397407446739, -11.590397407446739]), ([0.01, 0.08, 1.0], [1.0, 16.876031634718757, 16.876031634718757]), ([0.01, 0.332, 0.43, 0.647, 0.766, 0.892, 1.0], [1.0, 8.112593396470043, 17.771758362222585, -6.675066566935387, -18.20838259739396, -9.134593375065675, -9.134593375065675]), ([0.01, 0.08, 0.241, 1.0], [1.0, -56.25639426748601, -15.992137345391976, -15.992137345391976]), ([0.01, 0.157, 0.241, 1.0], [1.0, -85.32966141189901, -0.0, -0.0]), ([0.01, 0.08, 0.157, 1.0], [1.0, -4.7272888131464486, 25.040614422204726, 25.040614422204726]), ([0.01, 0.08, 0.157, 0.241, 1.0], [1.0, 35.834923798161086, 22.48841198272778, 34.04699514044253, 34.04699514044253]), ([0.01, 0.332, 0.43, 1.0], [1.0, -2.9628666785973405, -23.927926168361196, -23.927926168361196]), ([0.01, 0.08, 0.157, 0.43, 0.535, 0.647, 0.766, 1.0], [1.0, -49.70266017777981, -27.48179131312723, -2.062805797894305, -27.49090671937756, -5.178753100404831, -31.538455771667902, -31.538455771667902]), ([0.01, 1.0], [1.0, 1.0]), ([0.01, 0.08, 0.157, 0.241, 0.332, 1.0], [1.0, -10.496746744632562, 18.967067825869453, 87.43431745424962, 17.888156505874836, 17.888156505874836]), ([0.01, 0.08, 0.157, 0.332, 1.0], [1.0, -36.191732306290284, -7.013562751096167, -22.45892348735849, -22.45892348735849])]"

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
