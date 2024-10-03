import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class test_DYNAMNEWS1_RNDSRCH(unittest.TestCase):
    def setUp(self):
        # Expected values
        self.expected_problem_name = "DYNAMNEWS-1"
        self.expected_solver_name = "RNDSRCH"
        self.expected_all_recommended_xs = "[[(3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0), (0.2567518673381742, 3.4457871100687703, 1.448259339956088, 4.47075722504349, 7.039501835642472, 1.8020712455806367, 4.195236703988453, 0.7427899037721335, 1.5964700239863632, 2.407701427769358), (1.2799051814294138, 0.43643305561926116, 0.7094952877552761, 2.0759540008843023, 0.10045085355960243, 4.017729413622925, 1.4488345108361866, 6.52174595895297, 1.307398810037168, 6.724377169429896), (1.2799051814294138, 0.43643305561926116, 0.7094952877552761, 2.0759540008843023, 0.10045085355960243, 4.017729413622925, 1.4488345108361866, 6.52174595895297, 1.307398810037168, 6.724377169429896)], [(3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0), (4.297877972470276, 0.0786263533761449, 2.5284771448753887, 7.0485505941553335, 4.108524321246208, 2.201884001957223, 3.1867969927503204, 1.7329181126428228, 0.5196833769078699, 3.7340069740716024), (4.297877972470276, 0.0786263533761449, 2.5284771448753887, 7.0485505941553335, 4.108524321246208, 2.201884001957223, 3.1867969927503204, 1.7329181126428228, 0.5196833769078699, 3.7340069740716024)], [(3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0), (2.000604657485562, 1.0335920366894322, 0.2049180987810168, 4.147547223300166, 2.8474247856681134, 7.4514132737866525, 1.7436671635794385, 0.9091984362139542, 5.254270858803843, 1.8942621895127316), (0.294442831362623, 1.3908935639322415, 0.5622719663550539, 3.058029859343127, 2.289477117874483, 3.7969603575225355, 2.786325309322138, 7.281175391861351, 3.585029529800207, 1.8985743832084054), (1.2287495391396583, 0.5141311038609757, 4.426196376012836, 2.108124270683585, 0.14025330291425042, 2.694491604448821, 2.4811130706387385, 5.003943948731838, 0.8906083985340194, 3.531582167970271), (1.2287495391396583, 0.5141311038609757, 4.426196376012836, 2.108124270683585, 0.14025330291425042, 2.694491604448821, 2.4811130706387385, 5.003943948731838, 0.8906083985340194, 3.531582167970271)], [(3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0), (3.5157217903225995, 3.6425285757626282, 0.925871474338059, 3.5973821436649853, 3.4999801446674095, 7.090938162271665, 0.8502702850048011, 2.9865966833224777, 1.7338202615814784, 1.2906439459076946), (3.5157217903225995, 3.6425285757626282, 0.925871474338059, 3.5973821436649853, 3.4999801446674095, 7.090938162271665, 0.8502702850048011, 2.9865966833224777, 1.7338202615814784, 1.2906439459076946)], [(3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0), (0.8735068891405672, 3.0338923425994837, 5.216407509290791, 2.756497669814042, 6.394155309531909, 1.0891515520735464, 2.8009538288690146, 0.7588440011813196, 0.8869817607319463, 3.825433022736122), (0.8735068891405672, 3.0338923425994837, 5.216407509290791, 2.756497669814042, 6.394155309531909, 1.0891515520735464, 2.8009538288690146, 0.7588440011813196, 0.8869817607319463, 3.825433022736122)], [(3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0), (4.664814244089966, 8.136107314450276, 0.7257692448236055, 0.04750559336532918, 1.7329599034170762, 4.233118749803095, 1.269742458617881, 1.1772280826381039, 1.1240477542863072, 0.7148699971598014), (4.664814244089966, 8.136107314450276, 0.7257692448236055, 0.04750559336532918, 1.7329599034170762, 4.233118749803095, 1.269742458617881, 1.1772280826381039, 1.1240477542863072, 0.7148699971598014)], [(3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0), (3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0)], [(3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0), (5.293138185742484, 1.9949781393994237, 0.369144069678608, 7.909657884204957, 2.469639317990509, 7.284396659851659, 0.3046382296282686, 0.6887372241488992, 1.0576093406376297, 2.429705305345986), (5.293138185742484, 1.9949781393994237, 0.369144069678608, 7.909657884204957, 2.469639317990509, 7.284396659851659, 0.3046382296282686, 0.6887372241488992, 1.0576093406376297, 2.429705305345986)], [(3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0), (2.9216751381076014, 0.8091981169565601, 4.270505050258956, 0.8015779188666976, 1.8205797459652153, 1.1619459375936434, 5.6859643949849055, 2.386258588252074, 2.64488086340372, 2.0093509806192027), (2.9216751381076014, 0.8091981169565601, 4.270505050258956, 0.8015779188666976, 1.8205797459652153, 1.1619459375936434, 5.6859643949849055, 2.386258588252074, 2.64488086340372, 2.0093509806192027)], [(3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0), (3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0)], [(3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0), (8.292671717907238, 0.6935014352780532, 0.33682836221072343, 1.1672658060657066, 0.9028791118876207, 4.690330856383969, 4.10816244652909, 0.7186191527808048, 1.3382211766089325, 6.931557688341476), (8.292671717907238, 0.6935014352780532, 0.33682836221072343, 1.1672658060657066, 0.9028791118876207, 4.690330856383969, 4.10816244652909, 0.7186191527808048, 1.3382211766089325, 6.931557688341476)], [(3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0), (4.593175662537231, 1.5218992593127878, 1.4775427936876429, 2.8783529109082666, 4.679370930723183, 1.24607522254429, 1.1247337106486346, 3.828511851916664, 1.293521236873329, 3.3967773142572684), (4.593175662537231, 1.5218992593127878, 1.4775427936876429, 2.8783529109082666, 4.679370930723183, 1.24607522254429, 1.1247337106486346, 3.828511851916664, 1.293521236873329, 3.3967773142572684)], [(3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0), (1.821737773465332, 4.530189219461605, 4.793207525971152, 0.6046561398004362, 2.6852369840557904, 1.413870892507291, 2.5088673275533147, 8.48863293082352, 1.0157138344991203, 0.9446273759199526), (1.821737773465332, 4.530189219461605, 4.793207525971152, 0.6046561398004362, 2.6852369840557904, 1.413870892507291, 2.5088673275533147, 8.48863293082352, 1.0157138344991203, 0.9446273759199526)], [(3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0), (0.07545831280186052, 0.7194678531143148, 1.5286109498588085, 7.550246114947646, 3.8708224252637162, 9.036005851693735, 1.996571481061836, 0.6878799998851121, 3.845197344152501, 0.5087813771857244), (0.07545831280186052, 0.7194678531143148, 1.5286109498588085, 7.550246114947646, 3.8708224252637162, 9.036005851693735, 1.996571481061836, 0.6878799998851121, 3.845197344152501, 0.5087813771857244)], [(3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0), (3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0)], [(3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0), (0.320286722066728, 1.0522075763110015, 1.6259946437102943, 2.721687153938908, 1.5444063467989957, 4.864864408013364, 2.1237409444847417, 1.6605975468187337, 4.18835953603936, 4.357212166371786), (0.320286722066728, 1.0522075763110015, 1.6259946437102943, 2.721687153938908, 1.5444063467989957, 4.864864408013364, 2.1237409444847417, 1.6605975468187337, 4.18835953603936, 4.357212166371786)], [(3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0), (1.820384321883307, 3.5223057360014867, 1.666921532414779, 0.4694952321366892, 0.7404007515877851, 3.0132895910097837, 1.1048293904877533, 2.22600949998246, 6.403108833321984, 7.235023289659258), (1.820384321883307, 3.5223057360014867, 1.666921532414779, 0.4694952321366892, 0.7404007515877851, 3.0132895910097837, 1.1048293904877533, 2.22600949998246, 6.403108833321984, 7.235023289659258)], [(3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0), (3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0)], [(3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0), (3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0)], [(3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0), (6.348382895920342, 0.22559229911379475, 2.7739886164175425, 0.6722700735163352, 0.7188232754159816, 3.5867346627732766, 3.9029076979041104, 2.527734384818178, 5.3139972186906785, 3.4632741730569467), (6.348382895920342, 0.22559229911379475, 2.7739886164175425, 0.6722700735163352, 0.7188232754159816, 3.5867346627732766, 3.9029076979041104, 2.527734384818178, 5.3139972186906785, 3.4632741730569467)], [(3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0), (3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0)], [(3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0), (3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0)], [(3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0), (3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0)], [(3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0), (6.9754890657266895, 3.6294731835207026, 3.6038634948440844, 1.4415513141645755, 1.1475665095946366, 7.814195026958494, 0.5485689905710402, 0.10406491617800262, 0.8611991370863796, 0.047701778803479394), (6.9754890657266895, 3.6294731835207026, 3.6038634948440844, 1.4415513141645755, 1.1475665095946366, 7.814195026958494, 0.5485689905710402, 0.10406491617800262, 0.8611991370863796, 0.047701778803479394)]]"
        self.expected_all_intermediate_budgets = "[[0, 320, 610, 1000], [0, 390, 1000], [0, 100, 300, 740, 1000], [0, 310, 1000], [0, 340, 1000], [0, 340, 1000], [0, 1000], [0, 160, 1000], [0, 680, 1000], [0, 1000], [0, 530, 1000], [0, 70, 1000], [0, 540, 1000], [0, 190, 1000], [0, 1000], [0, 140, 1000], [0, 130, 1000], [0, 1000], [0, 1000], [0, 240, 1000], [0, 1000], [0, 1000], [0, 1000], [0, 400, 1000]]"
        self.expected_all_est_objectives = "[[120.0, 132.97336658427025, 146.88837878936502, 146.88837878936502], [120.0, 122.81327077773405, 122.81327077773405], [120.0, 132.5655063808954, 135.2840984470892, 145.90403108532502, 145.90403108532502], [120.0, 124.33123266578103, 124.33123266578103], [120.0, 131.8208805701563, 131.8208805701563], [120.0, 150.86918328674278, 150.86918328674278], [120.0, 120.0], [120.0, 120.99177821685792, 120.99177821685792], [120.0, 138.44031632495714, 138.44031632495714], [120.0, 120.0], [120.0, 124.09981123003192, 124.09981123003192], [120.0, 139.80019553295347, 139.80019553295347], [120.0, 125.96629997971242, 125.96629997971242], [120.0, 120.90479145017369, 120.90479145017369], [120.0, 120.0], [120.0, 147.70321477723041, 147.70321477723041], [120.0, 128.9911591075736, 128.9911591075736], [120.0, 120.0], [120.0, 120.0], [120.0, 122.33147351186406, 122.33147351186406], [120.0, 120.0], [120.0, 120.0], [120.0, 120.0], [120.0, 139.13163291275964, 139.13163291275964]]"
        self.expected_objective_curves = "[([0, 320, 610, 1000], [120.0, 132.97336658427025, 146.88837878936502, 146.88837878936502]), ([0, 390, 1000], [120.0, 122.81327077773405, 122.81327077773405]), ([0, 100, 300, 740, 1000], [120.0, 132.5655063808954, 135.2840984470892, 145.90403108532502, 145.90403108532502]), ([0, 310, 1000], [120.0, 124.33123266578103, 124.33123266578103]), ([0, 340, 1000], [120.0, 131.8208805701563, 131.8208805701563]), ([0, 340, 1000], [120.0, 150.86918328674278, 150.86918328674278]), ([0, 1000], [120.0, 120.0]), ([0, 160, 1000], [120.0, 120.99177821685792, 120.99177821685792]), ([0, 680, 1000], [120.0, 138.44031632495714, 138.44031632495714]), ([0, 1000], [120.0, 120.0]), ([0, 530, 1000], [120.0, 124.09981123003192, 124.09981123003192]), ([0, 70, 1000], [120.0, 139.80019553295347, 139.80019553295347]), ([0, 540, 1000], [120.0, 125.96629997971242, 125.96629997971242]), ([0, 190, 1000], [120.0, 120.90479145017369, 120.90479145017369]), ([0, 1000], [120.0, 120.0]), ([0, 140, 1000], [120.0, 147.70321477723041, 147.70321477723041]), ([0, 130, 1000], [120.0, 128.9911591075736, 128.9911591075736]), ([0, 1000], [120.0, 120.0]), ([0, 1000], [120.0, 120.0]), ([0, 240, 1000], [120.0, 122.33147351186406, 122.33147351186406]), ([0, 1000], [120.0, 120.0]), ([0, 1000], [120.0, 120.0]), ([0, 1000], [120.0, 120.0]), ([0, 400, 1000], [120.0, 139.13163291275964, 139.13163291275964])]"
        self.expected_progress_curves = "[([0.0, 0.32, 0.61, 1.0], [1.0, 0.5797308123198112, 0.12895723415809887, 0.12895723415809887]), ([0.0, 0.39, 1.0], [1.0, 0.9088647486523478, 0.9088647486523478]), ([0.0, 0.1, 0.3, 0.74, 1.0], [1.0, 0.5929433485759938, 0.5048751920283818, 0.16084494867572743, 0.16084494867572743]), ([0.0, 0.31, 1.0], [1.0, 0.8596907269768572, 0.8596907269768572]), ([0.0, 0.34, 1.0], [1.0, 0.6170653282157629, 0.6170653282157629]), ([0.0, 0.34, 1.0], [1.0, -0.0, -0.0]), ([0.0, 1.0], [1.0, 1.0]), ([0.0, 0.16, 1.0], [1.0, 0.9678715757509575, 0.9678715757509575]), ([0.0, 0.68, 1.0], [1.0, 0.4026302492791703, 0.4026302492791703]), ([0.0, 1.0], [1.0, 1.0]), ([0.0, 0.53, 1.0], [1.0, 0.8671875704663479, 0.8671875704663479]), ([0.0, 0.07, 1.0], [1.0, 0.35857727919037796, 0.35857727919037796]), ([0.0, 0.54, 1.0], [1.0, 0.8067231023156116, 0.8067231023156116]), ([0.0, 0.19, 1.0], [1.0, 0.9706894917896236, 0.9706894917896236]), ([0.0, 1.0], [1.0, 1.0]), ([0.0, 0.14, 1.0], [1.0, 0.10256081218941865, 0.10256081218941865]), ([0.0, 0.13, 1.0], [1.0, 0.708733495666049, 0.708733495666049]), ([0.0, 1.0], [1.0, 1.0]), ([0.0, 1.0], [1.0, 1.0]), ([0.0, 0.24, 1.0], [1.0, 0.9244724588205951, 0.9244724588205951]), ([0.0, 1.0], [1.0, 1.0]), ([0.0, 1.0], [1.0, 1.0]), ([0.0, 1.0], [1.0, 1.0]), ([0.0, 0.4, 1.0], [1.0, 0.38023520949528977, 0.38023520949528977])]"

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
