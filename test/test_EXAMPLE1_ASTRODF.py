import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class test_EXAMPLE1_ASTRODF(unittest.TestCase):
    def setUp(self):
        # Expected values
        self.expected_problem_name = "EXAMPLE-1"
        self.expected_solver_name = "ASTRODF"
        self.expected_all_recommended_xs = "[[(2.0, 2.0), (-0.2360679774997898, -0.23606797749978936), (0.18319476828142112, 0.18319476828142078), (0.18319476828142112, 0.18319476828142078)], [(2.0, 2.0), (-0.23606797749979025, -0.23606797749978936), (0.18319476828142112, 0.18319476828142028), (-0.131252291054487, -0.13125229105448755)], [(2.0, 2.0), (-0.23606797749979025, -0.23606797749978936), (0.23606797749978758, 0.2360679774997868), (-0.2360679774997851, -0.23606797749978536), (0.08170719434686402, 0.09099523864654202), (0.08170719434686402, 0.09099523864654202)], [(2.0, 2.0), (-0.23606797749979025, -0.23606797749978936), (0.23606797749978908, 0.23606797749978847), (-0.0783790818361188, -0.07837908183611941)], [(2.0, 2.0), (-0.23606797749979025, -0.23606797749978936), (0.2360679774997882, 0.23606797749978725), (-0.22525570372117154, -0.22525570372117065), (0.22525570372117143, 0.22525570372117049), (-0.11619624388108618, -0.1161962438810964), (-0.11619624388108618, -0.1161962438810964)], [(2.0, 2.0), (-0.2360679774997898, -0.23606797749978936), (0.18319476828142123, 0.18319476828142078), (0.18319476828142123, 0.18319476828142078)], [(2.0, 2.0), (-0.23606797749979025, -0.23606797749978936), (0.18319476828142128, 0.18319476828142034), (0.18319476828142128, 0.18319476828142034)], [(2.0, 2.0), (-0.23606797749979025, -0.23606797749978936), (0.18319476828142128, 0.18319476828142023), (-0.13125229105448777, -0.1312522910544866)], [(2.0, 2.0), (-0.23606797749979025, -0.23606797749978936), (0.18319476828142112, 0.18319476828142034), (0.18319476828142112, 0.18319476828142034)], [(2.0, 2.0), (-0.23606797749979025, -0.23606797749978936), (0.23606797749978753, 0.23606797749978675), (0.23606797749978753, 0.23606797749978675)], [(2.0, 2.0), (-0.23606797749979025, -0.23606797749978936), (0.23606797749978836, 0.2360679774997872), (-0.22880434742287015, -0.22880434742286915), (0.20658015198281976, 0.20658015198281937), (-0.11995822257145139, -0.11995822257144317), (-0.11995822257145139, -0.11995822257144317)], [(2.0, 2.0), (-0.23606797749979025, -0.23606797749978936), (0.23606797749978886, 0.23606797749978675), (-0.07837908183612041, -0.07837908183611986), (-0.07837908183612041, -0.07837908183611986)], [(2.0, 2.0), (-0.2360679774997898, -0.23606797749978936), (0.18319476828142112, 0.18319476828142084), (-0.13125229105448716, -0.13125229105448677)], [(2.0, 2.0), (-0.2360679774997898, -0.2360679774997898), (0.2360679774997898, 0.23606797749978958), (0.2360679774997898, 0.23606797749978958)], [(2.0, 2.0), (-0.23606797749979025, -0.23606797749978936), (0.22702322740401376, 0.22702322740401332), (-0.08742383193190573, -0.08742383193188302)], [(2.0, 2.0), (-0.23606797749979025, -0.23606797749978936), (0.18319476828142112, 0.1831947682814205), (0.18319476828142112, 0.1831947682814205)], [(2.0, 2.0), (-0.23606797749979025, -0.23606797749978936), (0.23423087261652753, 0.23423087261652775), (-0.08021618671938657, -0.08021618671937403)], [(2.0, 2.0), (-0.23606797749979025, -0.23606797749978936), (0.22880434742287398, 0.22880434742287287), (-0.08564271191303496, -0.08564271191303402), (-0.08564271191303496, -0.08564271191303402)], [(2.0, 2.0), (-0.23606797749979025, -0.23606797749978936), (0.18319476828142134, 0.18319476828142023), (0.18319476828142134, 0.18319476828142023)], [(2.0, 2.0), (-0.2360679774997898, -0.2360679774997898), (0.2360679774997892, 0.2360679774997892), (-0.2360679774997887, -0.23606797749978936), (-0.11211195529790562, 0.09278717274099013), (-0.11211195529790562, 0.09278717274099013)], [(2.0, 2.0), (-0.23606797749979025, -0.23606797749978936), (0.18319476828142112, 0.18319476828142034), (0.18319476828142112, 0.18319476828142034)], [(2.0, 2.0), (-0.2360679774997898, -0.23606797749978936), (0.23606797749978953, 0.23606797749978836), (0.23606797749978953, 0.23606797749978836)], [(2.0, 2.0), (-0.2360679774997898, -0.2360679774997898), (0.23606797749978903, 0.23606797749978936), (-0.04861901357873616, -0.10555641179444086)], [(2.0, 2.0), (-0.23606797749979025, -0.23606797749978936), (0.18319476828142112, 0.18319476828142045), (-0.13125229105448738, -0.13125229105448677), (-0.13125229105448738, -0.13125229105448677)]]"
        self.expected_all_intermediate_budgets = "[[4, 24, 475, 1000], [4, 24, 374, 1000], [4, 24, 40, 56, 871, 1000], [4, 24, 72, 1000], [4, 24, 40, 170, 344, 918, 1000], [4, 24, 434, 1000], [4, 24, 433, 1000], [4, 24, 320, 1000], [4, 24, 450, 1000], [4, 24, 40, 1000], [4, 24, 40, 88, 238, 697, 1000], [4, 24, 40, 851, 1000], [4, 24, 206, 1000], [4, 24, 72, 1000], [4, 24, 56, 1000], [4, 24, 458, 1000], [4, 24, 56, 1000], [4, 24, 72, 993, 1000], [4, 24, 296, 1000], [4, 24, 40, 56, 704, 1000], [4, 24, 403, 1000], [4, 24, 72, 1000], [4, 24, 72, 1000], [4, 24, 236, 978, 1000]]"
        self.expected_all_est_objectives = "[[7.984539704940337, 0.0959958849420176, 0.05166035119170231, 0.05166035119170231], [8.081590387702734, 0.19304656770441356, 0.14871103395409793, 0.11604471551683475], [7.9253347189439385, 0.03679089894561915, 0.036790898945616685, 0.03679089894561484, -0.059709081991685994, -0.059709081991685994], [8.073099810658121, 0.18455599065980263, 0.1845559906598017, 0.08538637159706623], [7.880122723414122, -0.008421096584198224, -0.008421096584200163, -0.018397012468040694, -0.018397012468040843, -0.09287414240173267, -0.09287414240173267], [8.025785950362149, 0.13724213036382893, 0.09290659661351369, 0.09290659661351369], [8.015084462897443, 0.12654064289912453, 0.08220510914880892, 0.08220510914880892], [7.994852045957048, 0.10630822595872841, 0.06197269220841279, 0.02930637377114952], [7.910902809206077, 0.02235898920775803, -0.021976544542557642, -0.021976544542557642], [7.943417039435916, 0.05487321943759722, 0.05487321943759472, 0.05487321943759472], [8.091417684005862, 0.20287386400754176, 0.20287386400753987, 0.19612054280506974, 0.17676840239234892, 0.12019763433086102, 0.12019763433086102], [7.909472416682599, 0.020928596684279075, 0.02092859668427719, -0.07824102237845705, -0.07824102237845705], [7.977013328578315, 0.088469508579996, 0.044133974829680726, 0.01146765639241722], [8.026689092895502, 0.1381452728971826, 0.13814527289718248, 0.13814527289718248], [7.9661065902554125, 0.07756277025709338, 0.06918568181727974, -0.01860755696527691], [7.989698471543214, 0.10115465154489581, 0.0568191177945802, 0.0568191177945802], [7.920874622531411, 0.032330802533091375, 0.030602825904808853, -0.06625610424499417], [7.936832268065668, 0.04828844806735008, 0.04153512686488155, -0.04849838372669442, -0.04849838372669442], [8.053894643913893, 0.16535082391557304, 0.1210152901652574, 0.1210152901652574], [8.098435148416002, 0.2098913284176821, 0.20989132841768154, 0.20989132841768135, 0.1196136983619855, 0.1196136983619855], [7.934581643563343, 0.04603782356502288, 0.0017022898147072251, 0.0017022898147072251], [8.017557178997036, 0.12901335899871555, 0.12901335899871502, 0.12901335899871502], [7.988890794564379, 0.10034697456605905, 0.10034697456605848, 0.0023967591166634073], [7.9800797052490156, 0.09153588525069647, 0.04720035150038083, 0.014534033063117536, 0.014534033063117536]]"
        self.expected_objective_curves = "[([4, 24, 475, 1000], [8.090508544469758, 0.0959958849420176, 0.05166035119170231, 0.05166035119170231]), ([4, 24, 374, 1000], [8.090508544469758, 0.19304656770441356, 0.14871103395409793, 0.11604471551683475]), ([4, 24, 40, 56, 871, 1000], [8.090508544469758, 0.03679089894561915, 0.036790898945616685, 0.03679089894561484, -0.059709081991685994, -0.059709081991685994]), ([4, 24, 72, 1000], [8.090508544469758, 0.18455599065980263, 0.1845559906598017, 0.08538637159706623]), ([4, 24, 40, 170, 344, 918, 1000], [8.090508544469758, -0.008421096584198224, -0.008421096584200163, -0.018397012468040694, -0.018397012468040843, -0.09287414240173267, -0.09287414240173267]), ([4, 24, 434, 1000], [8.090508544469758, 0.13724213036382893, 0.09290659661351369, 0.09290659661351369]), ([4, 24, 433, 1000], [8.090508544469758, 0.12654064289912453, 0.08220510914880892, 0.08220510914880892]), ([4, 24, 320, 1000], [8.090508544469758, 0.10630822595872841, 0.06197269220841279, 0.02930637377114952]), ([4, 24, 450, 1000], [8.090508544469758, 0.02235898920775803, -0.021976544542557642, -0.021976544542557642]), ([4, 24, 40, 1000], [8.090508544469758, 0.05487321943759722, 0.05487321943759472, 0.05487321943759472]), ([4, 24, 40, 88, 238, 697, 1000], [8.090508544469758, 0.20287386400754176, 0.20287386400753987, 0.19612054280506974, 0.17676840239234892, 0.12019763433086102, 0.12019763433086102]), ([4, 24, 40, 851, 1000], [8.090508544469758, 0.020928596684279075, 0.02092859668427719, -0.07824102237845705, -0.07824102237845705]), ([4, 24, 206, 1000], [8.090508544469758, 0.088469508579996, 0.044133974829680726, 0.01146765639241722]), ([4, 24, 72, 1000], [8.090508544469758, 0.1381452728971826, 0.13814527289718248, 0.13814527289718248]), ([4, 24, 56, 1000], [8.090508544469758, 0.07756277025709338, 0.06918568181727974, -0.01860755696527691]), ([4, 24, 458, 1000], [8.090508544469758, 0.10115465154489581, 0.0568191177945802, 0.0568191177945802]), ([4, 24, 56, 1000], [8.090508544469758, 0.032330802533091375, 0.030602825904808853, -0.06625610424499417]), ([4, 24, 72, 993, 1000], [8.090508544469758, 0.04828844806735008, 0.04153512686488155, -0.04849838372669442, -0.04849838372669442]), ([4, 24, 296, 1000], [8.090508544469758, 0.16535082391557304, 0.1210152901652574, 0.1210152901652574]), ([4, 24, 40, 56, 704, 1000], [8.090508544469758, 0.2098913284176821, 0.20989132841768154, 0.20989132841768135, 0.1196136983619855, 0.1196136983619855]), ([4, 24, 403, 1000], [8.090508544469758, 0.04603782356502288, 0.0017022898147072251, 0.0017022898147072251]), ([4, 24, 72, 1000], [8.090508544469758, 0.12901335899871555, 0.12901335899871502, 0.12901335899871502]), ([4, 24, 72, 1000], [8.090508544469758, 0.10034697456605905, 0.10034697456605848, 0.0023967591166634073]), ([4, 24, 236, 978, 1000], [8.090508544469758, 0.09153588525069647, 0.04720035150038083, 0.014534033063117536, 0.014534033063117536])]"
        self.expected_progress_curves = "[([0.004, 0.024, 0.475, 1.0], [1.0, 0.011865247334499795, 0.006385303335105502, 0.006385303335105502]), ([0.004, 0.024, 0.374, 1.0], [1.0, 0.023860869393231152, 0.01838092539383682, 0.014343315365034347]), ([0.004, 0.024, 0.04, 0.056, 0.871, 1.0], [1.0, 0.004547414880461062, 0.004547414880460758, 0.00454741488046053, -0.007380139538014573, -0.007380139538014573]), ([0.004, 0.024, 0.072, 1.0], [1.0, 0.022811420276659285, 0.022811420276659167, 0.010553894248765341]), ([0.004, 0.024, 0.04, 0.17, 0.344, 0.918, 1.0], [1.0, -0.0010408612187863566, -0.0010408612187865962, -0.002273900628980352, -0.00227390062898037, -0.011479394884913199, -0.011479394884913199]), ([0.004, 0.024, 0.434, 1.0], [1.0, 0.016963350277608983, 0.0114834062782147, 0.0114834062782147]), ([0.004, 0.024, 0.433, 1.0], [1.0, 0.015640629041251183, 0.010160685041856852, 0.010160685041856852]), ([0.004, 0.024, 0.32, 1.0], [1.0, 0.013139869437675221, 0.00765992543828089, 0.0036223154094784066]), ([0.004, 0.024, 0.45, 1.0], [1.0, 0.0027636073906678518, -0.002716336608726486, -0.002716336608726486]), ([0.004, 0.024, 0.04, 1.0], [1.0, 0.006782419069949026, 0.0067824190699487166, 0.0067824190699487166]), ([0.004, 0.024, 0.04, 0.088, 0.238, 0.697, 1.0], [1.0, 0.02507553918180033, 0.025075539181800097, 0.024240817709676277, 0.021848861715025123, 0.01485662287731242, 0.01485662287731242]), ([0.004, 0.024, 0.04, 0.851, 1.0], [1.0, 0.0025868085509389584, 0.0025868085509387255, -0.009670717476954952, -0.009670717476954952]), ([0.004, 0.024, 0.206, 1.0], [1.0, 0.010934974988743946, 0.005455030989349658, 0.0014174209605471463]), ([0.004, 0.024, 0.072, 1.0], [1.0, 0.01707498016198393, 0.017074980161983916, 0.017074980161983916]), ([0.004, 0.024, 0.056, 1.0], [1.0, 0.009586884412860694, 0.008551462672216247, -0.0022999242708909872]), ([0.004, 0.024, 0.458, 1.0], [1.0, 0.012502879267588161, 0.007022935268193831, 0.007022935268193831]), ([0.004, 0.024, 0.056, 1.0], [1.0, 0.003996139717965071, 0.003782558999424989, -0.008189362124867086]), ([0.004, 0.024, 0.072, 0.993, 1.0], [1.0, 0.005968530630915346, 0.005133809158791724, -0.0059944790194734225, -0.0059944790194734225]), ([0.004, 0.024, 0.296, 1.0], [1.0, 0.02043763046620822, 0.014957686466813886, 0.014957686466813886]), ([0.004, 0.024, 0.04, 0.056, 0.704, 1.0], [1.0, 0.02594290918352131, 0.02594290918352124, 0.025942909183521216, 0.014784447442892458, 0.014784447442892458]), ([0.004, 0.024, 0.403, 1.0], [1.0, 0.0056903497860455125, 0.0002104057866511766, 0.0002104057866511766]), ([0.004, 0.024, 0.072, 1.0], [1.0, 0.01594626076835457, 0.015946260768354505, 0.015946260768354505]), ([0.004, 0.024, 0.072, 1.0], [1.0, 0.012403049080845593, 0.012403049080845524, 0.000296243320613227]), ([0.004, 0.024, 0.236, 0.978, 1.0], [1.0, 0.011313984126902078, 0.005834040127507743, 0.0017964300987052574, 0.0017964300987052574])]"

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
