import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class test_IRONORE1_RNDSRCH(unittest.TestCase):
    def setUp(self):
        # Expected values
        self.expected_problem_name = "IRONORE-1"
        self.expected_solver_name = "RNDSRCH"
        self.expected_all_recommended_xs = "[[(80, 7000, 40, 100), (106.04208455290791, 1717.1639026711428, 97.66810871178762, 68.74180720329448), (106.04208455290791, 1717.1639026711428, 97.66810871178762, 68.74180720329448)], [(80, 7000, 40, 100), (80, 7000, 40, 100)], [(80, 7000, 40, 100), (27.30183995910418, 3684.661514592841, 47.10014902971169, 100.49702956782365), (20.234251678135664, 1639.8984085157767, 19.023066684981703, 105.52339096424805), (20.234251678135664, 1639.8984085157767, 19.023066684981703, 105.52339096424805)], [(80, 7000, 40, 100), (80, 7000, 40, 100)], [(80, 7000, 40, 100), (19.43057951053071, 4523.015305913609, 95.33915164608602, 97.42837688353313), (19.43057951053071, 4523.015305913609, 95.33915164608602, 97.42837688353313)], [(80, 7000, 40, 100), (80, 7000, 40, 100)], [(80, 7000, 40, 100), (80, 7000, 40, 100)], [(80, 7000, 40, 100), (46.20031637628806, 1315.9533298551744, 104.36517401825128, 23.648951463720735), (111.97881993274422, 1743.441481887965, 113.58952843057426, 67.19165591948381), (96.41189511709807, 8689.944235572488, 101.38676788412684, 79.68718768258695), (96.41189511709807, 8689.944235572488, 101.38676788412684, 79.68718768258695)], [(80, 7000, 40, 100), (57.866334150734936, 2176.5511407069566, 100.1193670841009, 91.42973478979306), (17.20340129146427, 3415.010435064405, 96.39031491554285, 100.15485449202394), (17.20340129146427, 3415.010435064405, 96.39031491554285, 100.15485449202394)], [(80, 7000, 40, 100), (107.94498227259388, 3774.5559125009777, 99.13828483086061, 66.74270278080165), (107.94498227259388, 3774.5559125009777, 99.13828483086061, 66.74270278080165)], [(80, 7000, 40, 100), (50.88652198626231, 4076.444573809437, 92.71303955462609, 99.82640641536598), (50.88652198626231, 4076.444573809437, 92.71303955462609, 99.82640641536598)], [(80, 7000, 40, 100), (108.33335717014097, 2450.2398711342657, 105.61516441094503, 31.355077063004778), (108.33335717014097, 2450.2398711342657, 105.61516441094503, 31.355077063004778)], [(80, 7000, 40, 100), (91.17042205282135, 1681.8835725619715, 107.71266380618235, 17.59866402338089), (91.17042205282135, 1681.8835725619715, 107.71266380618235, 17.59866402338089)], [(80, 7000, 40, 100), (41.961891985033304, 1563.007556639911, 114.77388535487694, 92.56932461735758), (105.73152489761598, 10298.284329178769, 91.13200861935223, 52.57446245792126), (105.40640557749535, 5706.059635745616, 104.65197883791697, 26.571687407807318), (105.40640557749535, 5706.059635745616, 104.65197883791697, 26.571687407807318)], [(80, 7000, 40, 100), (47.891957368904514, 7785.5306658641675, 95.60230152600356, 24.12133409381844), (98.2223602897452, 2151.416972838893, 39.29407144009676, 95.65563043467921), (18.313001435478537, 2627.0394122303983, 90.78949130847202, 83.84166750182962), (73.49637746062, 3476.696336494598, 105.32338029480697, 74.22174250415378), (54.62119942528872, 1434.0305219079958, 89.29907586047986, 104.10013200440706), (54.62119942528872, 1434.0305219079958, 89.29907586047986, 104.10013200440706)], [(80, 7000, 40, 100), (62.734387919524515, 3213.1187742968086, 94.4371092866155, 57.067392335043095), (93.84337029583611, 1961.6841785774202, 77.99769758824796, 39.63410017709867), (93.84337029583611, 1961.6841785774202, 77.99769758824796, 39.63410017709867)], [(80, 7000, 40, 100), (38.60182442703274, 10576.656638757942, 50.920628085719045, 96.2697929830869), (88.93764359772668, 2130.2242213682835, 126.21402597294336, 22.948319357191384), (40.31170427773956, 867.8663392150008, 74.43472155081977, 98.01115669505288), (40.31170427773956, 867.8663392150008, 74.43472155081977, 98.01115669505288)], [(80, 7000, 40, 100), (45.70846638774951, 3356.2324250783267, 31.854579449812764, 105.79464327915638), (58.89726499359386, 3314.230166416349, 70.67847984402904, 106.81202133345157), (58.89726499359386, 3314.230166416349, 70.67847984402904, 106.81202133345157)], [(80, 7000, 40, 100), (16.835375215189018, 12207.23278738965, 88.13011650597284, 96.05763455959553), (16.835375215189018, 12207.23278738965, 88.13011650597284, 96.05763455959553)], [(80, 7000, 40, 100), (107.36619718339689, 4136.356839631631, 106.80549069761886, 23.57710825890863), (107.36619718339689, 4136.356839631631, 106.80549069761886, 23.57710825890863)], [(80, 7000, 40, 100), (280.4875469286801, 5426.764489756512, 41.31237022056034, 151.32424463911912), (27.73463916141148, 2989.8393560200684, 87.48538274703203, 64.77069319019671), (8.943556863529167, 3139.13992702152, 88.48034099506992, 32.07555525245516), (116.22898260989274, 762.8904117620966, 19.478781239659355, 101.72356063899113), (110.5401464797777, 1174.49955598665, 113.82350342004109, 103.67727947831365), (110.5401464797777, 1174.49955598665, 113.82350342004109, 103.67727947831365)], [(80, 7000, 40, 100), (76.06738332396772, 2651.1627238523556, 104.78950728212797, 42.348506268551134), (89.47359426854881, 2123.913044959137, 100.76469391986038, 96.98507352054978), (89.47359426854881, 2123.913044959137, 100.76469391986038, 96.98507352054978)], [(80, 7000, 40, 100), (23.437266986924882, 1376.9650626593195, 19.328309261685284, 103.65854897931389), (30.434018896701428, 4519.600878196835, 90.73652386511726, 104.60878609627176), (30.434018896701428, 4519.600878196835, 90.73652386511726, 104.60878609627176)], [(80, 7000, 40, 100), (81.6835193799152, 2485.2590317390354, 80.17998838518352, 40.898439608647294), (71.15996919982027, 968.5400067005638, 93.28103484598974, 25.204465482175852), (87.01808769697116, 513.3759520810161, 103.09258489935078, 36.113506310021066), (87.01808769697116, 513.3759520810161, 103.09258489935078, 36.113506310021066)]]"
        self.expected_all_intermediate_budgets = "[[0, 80, 1000], [0, 1000], [0, 330, 530, 1000], [0, 1000], [0, 700, 1000], [0, 1000], [0, 1000], [0, 40, 50, 290, 1000], [0, 140, 360, 1000], [0, 540, 1000], [0, 770, 1000], [0, 870, 1000], [0, 80, 1000], [0, 20, 90, 890, 1000], [0, 140, 360, 440, 470, 860, 1000], [0, 230, 510, 1000], [0, 60, 190, 250, 1000], [0, 30, 600, 1000], [0, 280, 1000], [0, 380, 1000], [0, 20, 130, 240, 260, 330, 1000], [0, 610, 670, 1000], [0, 80, 780, 1000], [0, 170, 260, 330, 1000]]"
        self.expected_all_est_objectives = "[[149895.8055235529, 246274.4870298372, 246274.4870298372], [122793.09736468189, 122793.09736468189], [99852.80349485856, 103434.7620045503, 116604.48760425061, 116604.48760425061], [126011.12695446546, 126011.12695446546], [136147.71179130895, 187329.9969989881, 187329.9969989881], [132850.26196652921, 132850.26196652921], [134982.68434045353, 134982.68434045353], [161256.2908821113, 138805.58809889155, 200516.08070662283, 257602.68249970078, 257602.68249970078], [146337.47315675917, 188483.98140272632, 204676.83651708864, 204676.83651708864], [134867.2205665852, 238266.75878051444, 238266.75878051444], [149243.01256369415, 205742.7756343193, 205742.7756343193], [112822.77485929335, 216603.76141448744, 216603.76141448744], [132809.38556277155, 214873.87310267077, 214873.87310267077], [118379.15455996453, 117274.38322909277, 222641.11835525592, 228881.56341311347, 228881.56341311347], [127606.7164810152, 132131.77263663532, 126278.5726191838, 139739.56752334637, 160648.4170564049, 181898.5230036875, 181898.5230036875], [145498.2552215891, 154602.32552858902, 166051.47160311777, 166051.47160311777], [161264.15011124164, 139557.81317299747, 150757.20881756715, 196013.36106823932, 196013.36106823932], [132500.94479520118, 93344.84334327355, 99450.02320723001, 99450.02320723001], [112031.98326897933, 158410.294748688, 158410.294748688], [130863.18264271188, 220442.84253325922, 220442.84253325922], [147610.26102665017, 0.0, 128532.45800218811, 126924.58155734639, 204844.3296916969, 210363.84251946135, 210363.84251946135], [132677.02997009846, 176414.01962810184, 242701.74788169746, 242701.74788169746], [132803.08586581453, 161115.30788531256, 166837.49552738637, 166837.49552738637], [137521.1409071744, 136800.72549981947, 155723.37180192597, 209670.817692181, 209670.817692181]]"
        self.expected_objective_curves = "[([0, 80, 1000], [121270.73497283501, 246274.4870298372, 246274.4870298372]), ([0, 1000], [121270.73497283501, 121270.73497283501]), ([0, 330, 530, 1000], [121270.73497283501, 103434.7620045503, 116604.48760425061, 116604.48760425061]), ([0, 1000], [121270.73497283501, 121270.73497283501]), ([0, 700, 1000], [121270.73497283501, 187329.9969989881, 187329.9969989881]), ([0, 1000], [121270.73497283501, 121270.73497283501]), ([0, 1000], [121270.73497283501, 121270.73497283501]), ([0, 40, 50, 290, 1000], [121270.73497283501, 138805.58809889155, 200516.08070662283, 238692.2902877062, 238692.2902877062]), ([0, 140, 360, 1000], [121270.73497283501, 188483.98140272632, 204676.83651708864, 204676.83651708864]), ([0, 540, 1000], [121270.73497283501, 238266.75878051444, 238266.75878051444]), ([0, 770, 1000], [121270.73497283501, 205742.7756343193, 205742.7756343193]), ([0, 870, 1000], [121270.73497283501, 216603.76141448744, 216603.76141448744]), ([0, 80, 1000], [121270.73497283501, 214873.87310267077, 214873.87310267077]), ([0, 20, 90, 890, 1000], [121270.73497283501, 117274.38322909277, 222641.11835525592, 228881.56341311347, 228881.56341311347]), ([0, 140, 360, 440, 470, 860, 1000], [121270.73497283501, 132131.77263663532, 126278.5726191838, 139739.56752334637, 160648.4170564049, 181898.5230036875, 181898.5230036875]), ([0, 230, 510, 1000], [121270.73497283501, 154602.32552858902, 166051.47160311777, 166051.47160311777]), ([0, 60, 190, 250, 1000], [121270.73497283501, 139557.81317299747, 150757.20881756715, 196013.36106823932, 196013.36106823932]), ([0, 30, 600, 1000], [121270.73497283501, 93344.84334327355, 99450.02320723001, 99450.02320723001]), ([0, 280, 1000], [121270.73497283501, 158410.294748688, 158410.294748688]), ([0, 380, 1000], [121270.73497283501, 220442.84253325922, 220442.84253325922]), ([0, 20, 130, 240, 260, 330, 1000], [121270.73497283501, 0.0, 128532.45800218811, 126924.58155734639, 204844.3296916969, 210363.84251946135, 210363.84251946135]), ([0, 610, 670, 1000], [121270.73497283501, 176414.01962810184, 242701.74788169746, 242701.74788169746]), ([0, 80, 780, 1000], [121270.73497283501, 161115.30788531256, 166837.49552738637, 166837.49552738637]), ([0, 170, 260, 330, 1000], [121270.73497283501, 136800.72549981947, 155723.37180192597, 209670.817692181, 209670.817692181])]"
        self.expected_progress_curves = "[([0.0, 0.08, 1.0], [1.0, -0.06457244346490748, -0.06457244346490748]), ([0.0, 1.0], [1.0, 1.0]), ([0.0, 0.33, 0.53, 1.0], [1.0, 1.151896923188053, 1.0397392740717124, 1.0397392740717124]), ([0.0, 1.0], [1.0, 1.0]), ([0.0, 0.7, 1.0], [1.0, 0.4374179268106933, 0.4374179268106933]), ([0.0, 1.0], [1.0, 1.0]), ([0.0, 1.0], [1.0, 1.0]), ([0.0, 0.04, 0.05, 0.29, 1.0], [1.0, 0.8506675109264561, 0.3251209667484998, -0.0, -0.0]), ([0.0, 0.14, 0.36, 1.0], [1.0, 0.42759022183230355, 0.2896866225234675, 0.2896866225234675]), ([0.0, 0.54, 1.0], [1.0, 0.0036239641524987566, 0.0036239641524987566]), ([0.0, 0.77, 1.0], [1.0, 0.2806087397244168, 0.2806087397244168]), ([0.0, 0.87, 1.0], [1.0, 0.18811306675326658, 0.18811306675326658]), ([0.0, 0.08, 1.0], [1.0, 0.20284535595841208, 0.20284535595841208]), ([0.0, 0.02, 0.09, 0.89, 1.0], [1.0, 1.0340342259394013, 0.1366969794379603, 0.08355132793365587, 0.08355132793365587]), ([0.0, 0.14, 0.36, 0.44, 0.47, 0.86, 1.0], [1.0, 0.9075038851710323, 0.9573516324756554, 0.8427134396151852, 0.6646469042419267, 0.48367411870608973, 0.48367411870608973]), ([0.0, 0.23, 0.51, 1.0], [1.0, 0.7161373781298174, 0.6186327415762702, 0.6186327415762702]), ([0.0, 0.06, 0.19, 0.25, 1.0], [1.0, 0.8442613185362361, 0.7488836375428444, 0.36346758570026944, 0.36346758570026944]), ([0.0, 0.03, 0.6, 1.0], [1.0, 1.2378259388123152, 1.1858322495140843, 1.1858322495140843]), ([0.0, 0.28, 1.0], [1.0, 0.6837074787822254, 0.6837074787822254]), ([0.0, 0.38, 1.0], [1.0, 0.1554182083988775, 0.1554182083988775]), ([0.0, 0.02, 0.13, 0.24, 0.26, 0.33, 1.0], [1.0, 2.032780860785246, 0.9381568144802676, 0.9518500111044319, 0.2882601963944733, 0.24125423728446474, 0.24125423728446474]), ([0.0, 0.61, 0.67, 1.0], [1.0, 0.5303819259811571, -0.03414583960534107, -0.03414583960534107]), ([0.0, 0.08, 0.78, 1.0], [1.0, 0.6606707107086723, 0.6119387072300136, 0.6119387072300136]), ([0.0, 0.17, 0.26, 0.33, 1.0], [1.0, 0.8677415702309504, 0.7065901849391735, 0.24715626119670103, 0.24715626119670103])]"

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
