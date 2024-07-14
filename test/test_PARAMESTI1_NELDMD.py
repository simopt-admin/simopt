import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class test_PARAMESTI1_NELDMD(unittest.TestCase):
    def setUp(self):
        # Expected values
        self.expected_problem_name = "PARAMESTI-1"
        self.expected_solver_name = "NELDMD"
        self.expected_all_recommended_xs = "[[(1, 1), (2.4849999699999996, 2.4849999699999996), (2.17562497625, 4.52687492875), (1.893320294453125, 5.598085844609374), (2.023596170727539, 4.886765058198241), (1.9693346972143555, 5.418140779885253), (2.03178373159729, 5.371039645582885), (2.012077692566681, 5.1406776354661545), (1.9948174418701459, 5.230328688991021), (1.9948174418701459, 5.230328688991021)], [(1, 1), (2.4849999699999996, 2.4849999699999996), (1.2474999949999996, 4.712499924999999), (1.7192968604687495, 4.952265545156248), (1.8706005683496087, 5.069731363095701), (1.7775161586090082, 4.94685752524597), (1.8124643543141168, 4.910731045873259), (1.8327954124055856, 4.9992628243276584), (1.8204040790758984, 5.039459008627613), (1.8204040790758984, 5.039459008627613)], [(1, 1), (2.4849999699999996, 2.4849999699999996), (1.2474999949999996, 4.712499924999999), (1.4949999899999993, 7.434999869999996), (1.8005077963281244, 5.598085844609373), (1.8965228090368647, 5.575305693702391), (1.9114024168905634, 5.8907684864016705), (1.8671148124996182, 5.981024260029982), (1.870746909670505, 5.800728920273264), (1.870746909670505, 5.800728920273264)], [(1, 1), (2.4849999699999996, 2.4849999699999996), (2.17562497625, 4.52687492875), (1.8498144359570312, 4.895224530683592), (2.038521097183838, 4.775885543836669), (1.9353155328625489, 4.516542287357176), (1.9826810256800844, 4.709758149542693), (1.9386823273807146, 4.691934720795096), (1.9479986046964741, 4.608694361263035), (1.9479986046964741, 4.608694361263035)], [(1, 1), (2.4849999699999996, 2.4849999699999996), (2.17562497625, 4.52687492875), (2.1362707266139744, 4.516538274772572), (2.1522356795947872, 4.472089871954433), (2.1599390896771906, 4.510594501056751), (2.1599390896771906, 4.510594501056751)], [(1, 1), (2.4849999699999996, 2.4849999699999996), (1.2474999949999996, 4.712499924999999), (1.7424999849999994, 6.197499894999996), (2.1137499774999995, 5.331249912499998), (1.9590624806249997, 4.372187431874998), (1.8957372865917965, 4.870571210869139), (1.8912658511352536, 4.4220983195385735), (1.9262820247442622, 4.509261098539428), (1.9022556122657772, 4.668125459954069), (1.920147490986051, 4.58965960403344), (1.920147490986051, 4.58965960403344)], [(1, 1), (2.4849999699999996, 2.4849999699999996), (2.17562497625, 4.52687492875), (2.216230444179687, 5.174628821914063), (2.169945044724121, 4.813409346789552), (2.2334817255694572, 4.97839927265564), (2.208971914663238, 5.035266565818329), (2.1955859324202347, 4.910121133013268), (2.1955859324202347, 4.910121133013268)], [(1, 1), (2.4849999699999996, 2.4849999699999996), (2.17562497625, 4.52687492875), (2.19882810078125, 5.957734274843751), (2.0267382605078126, 5.905527244648438), (2.144204078447266, 5.229252844248046), (2.0424487094091797, 5.554338286899414), (2.0496959092749787, 5.361873958207015), (2.0684833024653817, 5.391211144427699), (2.05076915763968, 5.465440419108385), (2.05076915763968, 5.465440419108385)], [(1, 1), (2.4849999699999996, 2.4849999699999996), (2.17562497625, 4.52687492875), (2.071210915859375, 6.008007711328125), (2.2500683341210936, 5.817548730800782), (2.168132300620117, 5.219826574907227), (2.14015561661499, 5.763347682091065), (2.2021061463693234, 5.654567929649964), (2.1696315910561372, 5.464392190388871), (2.1956423446390962, 5.557768574668098), (2.1695927043906247, 5.537847449300931), (2.1695927043906247, 5.537847449300931)], [(1, 1), (2.4849999699999996, 2.4849999699999996), (2.17562497625, 4.52687492875), (1.893320294453125, 5.598085844609374), (1.8498144359570312, 4.895224530683592), (2.023596170727539, 4.886765058198241), (1.9204057126071166, 4.803665847230528), (1.9384056950773427, 4.929465567360628), (1.956302530519776, 4.8418488942306075), (1.956302530519776, 4.8418488942306075)]]"
        self.expected_all_intermediate_budgets = "[[0, 150, 270, 480, 600, 690, 750, 810, 960, 1000], [0, 150, 210, 480, 600, 720, 780, 840, 960, 1000], [0, 150, 210, 300, 540, 720, 780, 900, 960, 1000], [0, 150, 270, 540, 660, 720, 840, 900, 960, 1000], [0, 150, 270, 840, 900, 960, 1000], [0, 150, 210, 300, 360, 480, 660, 720, 780, 840, 960, 1000], [0, 150, 270, 570, 690, 750, 810, 870, 1000], [0, 150, 270, 480, 540, 600, 690, 870, 930, 990, 1000], [0, 150, 270, 480, 540, 600, 660, 720, 780, 900, 960, 1000], [0, 150, 270, 480, 540, 600, 720, 900, 960, 1000]]"
        self.expected_all_est_objectives = "[[-9.16944227603262, -5.871306999372082, -4.733473119939698, -4.70195670115451, -4.661971214987844, -4.672729472534877, -4.671188904222991, -4.659190785420632, -4.661201281768439, -4.661201281768439], [-8.940090362495347, -5.894566377996935, -5.301941613745327, -4.62320670963959, -4.564870041789244, -4.593826651779861, -4.582341790772201, -4.573621950873492, -4.576308083367474, -4.576308083367474], [-9.121210005202611, -5.871803178327859, -5.413893744735817, -5.399481873661548, -4.6369610393619825, -4.602612841400927, -4.637750661821271, -4.661680227473518, -4.6337265689999025, -4.6337265689999025], [-8.779886386724968, -5.7170727489201365, -4.613311046152679, -4.545486835343257, -4.543872842445989, -4.551699000775199, -4.538282447567526, -4.538214051214219, -4.543415053565002, -4.543415053565002], [-8.99288952739613, -5.504851643116329, -4.569017770707743, -4.562675792755738, -4.56823696325917, -4.566849233161828, -4.566849233161828], [-8.87740808504234, -5.612334938595852, -5.410165644686853, -4.82725975260652, -4.5819737841154575, -4.573603514166729, -4.559903955589788, -4.583069003788504, -4.565758860463795, -4.561881043921569, -4.561379280029752, -4.561379280029752], [-9.024638576352391, -5.792069128742862, -4.693739890477101, -4.697327572926619, -4.671428985876372, -4.702761415196267, -4.689038054552377, -4.681701073687395, -4.681701073687395], [-8.921050660074993, -5.831898214873072, -4.684856910801284, -4.7681902308602035, -4.684975666440125, -4.647780152462962, -4.636109243428385, -4.619441301205763, -4.627315464131804, -4.628795973244435, -4.628795973244435], [-8.550164686658025, -5.331603422732724, -4.490390709006505, -4.702064059848396, -4.6934255621860235, -4.538287972354088, -4.642613210147299, -4.633069365437234, -4.581384520336194, -4.608837002457795, -4.596566155447619, -4.596566155447619], [-8.983830735669818, -5.731522782616615, -4.640719309645201, -4.632407649870032, -4.604274728718362, -4.578404016462647, -4.586296642189949, -4.580505014460019, -4.5799203113391895, -4.5799203113391895]]"
        self.expected_objective_curves = "[([0, 150, 270, 480, 600, 690, 750, 810, 960, 1000], [-9.265122221743944, -5.871306999372082, -4.733473119939698, -4.70195670115451, -4.661971214987844, -4.672729472534877, -4.671188904222991, -4.659190785420632, -4.661201281768439, -4.661201281768439]), ([0, 150, 210, 480, 600, 720, 780, 840, 960, 1000], [-9.265122221743944, -5.894566377996935, -5.301941613745327, -4.62320670963959, -4.564870041789244, -4.593826651779861, -4.582341790772201, -4.573621950873492, -4.576308083367474, -4.576308083367474]), ([0, 150, 210, 300, 540, 720, 780, 900, 960, 1000], [-9.265122221743944, -5.871803178327859, -5.413893744735817, -5.399481873661548, -4.6369610393619825, -4.602612841400927, -4.637750661821271, -4.661680227473518, -4.6337265689999025, -4.6337265689999025]), ([0, 150, 270, 540, 660, 720, 840, 900, 960, 1000], [-9.265122221743944, -5.7170727489201365, -4.613311046152679, -4.545486835343257, -4.543872842445989, -4.551699000775199, -4.538282447567526, -4.538214051214219, -4.543415053565002, -4.543415053565002]), ([0, 150, 270, 840, 900, 960, 1000], [-9.265122221743944, -5.504851643116329, -4.569017770707743, -4.562675792755738, -4.56823696325917, -4.566849233161828, -4.566849233161828]), ([0, 150, 210, 300, 360, 480, 660, 720, 780, 840, 960, 1000], [-9.265122221743944, -5.612334938595852, -5.410165644686853, -4.82725975260652, -4.5819737841154575, -4.573603514166729, -4.559903955589788, -4.583069003788504, -4.565758860463795, -4.561881043921569, -4.561379280029752, -4.561379280029752]), ([0, 150, 270, 570, 690, 750, 810, 870, 1000], [-9.265122221743944, -5.792069128742862, -4.693739890477101, -4.697327572926619, -4.671428985876372, -4.702761415196267, -4.689038054552377, -4.681701073687395, -4.681701073687395]), ([0, 150, 270, 480, 540, 600, 690, 870, 930, 990, 1000], [-9.265122221743944, -5.831898214873072, -4.684856910801284, -4.7681902308602035, -4.684975666440125, -4.647780152462962, -4.636109243428385, -4.619441301205763, -4.627315464131804, -4.628795973244435, -4.628795973244435]), ([0, 150, 270, 480, 540, 600, 660, 720, 780, 900, 960, 1000], [-9.265122221743944, -5.331603422732724, -4.490390709006505, -4.702064059848396, -4.6934255621860235, -4.538287972354088, -4.642613210147299, -4.633069365437234, -4.581384520336194, -4.608837002457795, -4.596566155447619, -4.596566155447619]), ([0, 150, 270, 480, 540, 600, 720, 900, 960, 1000], [-9.265122221743944, -5.731522782616615, -4.640719309645201, -4.632407649870032, -4.604274728718362, -4.578404016462647, -4.586296642189949, -4.580505014460019, -4.5799203113391895, -4.5799203113391895])]"
        self.expected_progress_curves = "[([0.0, 0.15, 0.27, 0.48, 0.6, 0.69, 0.75, 0.81, 0.96, 1.0], [1.0, 0.2713976426578085, 0.0271213952789498, 0.020355282359630678, 0.011770984873880855, 0.014080624997477477, 0.013749887573622412, 0.011174067421292353, 0.01160569150288159, 0.01160569150288159]), ([0.0, 0.15, 0.21, 0.48, 0.6, 0.72, 0.78, 0.84, 0.96, 1.0], [1.0, 0.27639109014565943, 0.14916324420791993, 0.003448814063944291, -0.009075213005963622, -0.002858653494789866, -0.005324284725614635, -0.007196306473871647, -0.006619633220022319, -0.006619633220022319]), ([0.0, 0.15, 0.21, 0.3, 0.54, 0.72, 0.78, 0.9, 0.96, 1.0], [1.0, 0.2715041650030617, 0.17319772493375815, 0.1701037075667533, 0.00640166694566155, -0.0009723874341731879, 0.006571187307717682, 0.011708514121938476, 0.005707273593003455, 0.005707273593003455]), ([0.0, 0.15, 0.27, 0.54, 0.66, 0.72, 0.84, 0.9, 0.96, 1.0], [1.0, 0.23828581095836304, 0.001324360234961386, -0.013236503171763047, -0.01358300377731498, -0.011902842356030995, -0.014783179573590422, -0.014797863267600858, -0.013681284336507933, -0.013681284336507933]), ([0.0, 0.15, 0.27, 0.84, 0.9, 0.96, 1.0], [1.0, 0.19272505180237187, -0.008184756434340848, -0.009546286094406848, -0.00835238434286994, -0.008650309643461706, -0.008650309643461706]), ([0.0, 0.15, 0.21, 0.3, 0.36, 0.48, 0.66, 0.72, 0.78, 0.84, 0.96, 1.0], [1.0, 0.21580013908026952, 0.17239735652649288, 0.047256009914945205, -0.0054032903579347, -0.007200264564436105, -0.010141358884453567, -0.005168162755722059, -0.008884396671220726, -0.009716907013970554, -0.00982462836308741, -0.00982462836308741]), ([0.0, 0.15, 0.27, 0.57, 0.69, 0.75, 0.81, 0.87, 1.0], [1.0, 0.2543864338711155, 0.018591253608174232, 0.019361476416087448, 0.0138014295837635, 0.020528042663122185, 0.01758183838839221, 0.0160066961948039, 0.0160066961948039]), ([0.0, 0.15, 0.27, 0.48, 0.54, 0.6, 0.69, 0.87, 0.93, 0.99, 1.0], [1.0, 0.2629371545626724, 0.016684208142245888, 0.034574649861004635, 0.01670970323633256, 0.008724371864232972, 0.006218798850456395, 0.002640436093582768, 0.004330903397810639, 0.004648746992364111, 0.004648746992364111]), ([0.0, 0.15, 0.27, 0.48, 0.54, 0.6, 0.66, 0.72, 0.78, 0.9, 0.96, 1.0], [1.0, 0.1555311996018342, -0.02506483350181733, 0.02037833069677375, 0.01852377193375996, -0.014781993482937435, 0.0076151051231839306, 0.005566181626105141, -0.005529796649685026, 0.0003638486663125696, -0.0022705222326284426, -0.0022705222326284426]), ([0.0, 0.15, 0.27, 0.48, 0.54, 0.6, 0.72, 0.9, 0.96, 1.0], [1.0, 0.24138802128069473, 0.007208512459769507, 0.005424120998002817, -0.0006156045989916105, -0.006169667121718496, -0.004475236124820292, -0.00571861366319043, -0.0058441408484576456, -0.0058441408484576456])]"

        # Convert the expected values from string to their actual types
        self.expected_all_recommended_xs = eval(self.expected_all_recommended_xs, {'nan': float('nan'), 'inf': float('inf')})
        self.expected_all_intermediate_budgets = eval(self.expected_all_intermediate_budgets, {'nan': float('nan'), 'inf': float('inf')})
        self.expected_all_est_objectives = eval(self.expected_all_est_objectives, {'nan': float('nan'), 'inf': float('inf')})
        self.expected_objective_curves = eval(self.expected_objective_curves, {'nan': float('nan'), 'inf': float('inf')})
        self.expected_progress_curves = eval(self.expected_progress_curves, {'nan': float('nan'), 'inf': float('inf')})
        
        # Number of macro-replications and post-replications
        self.num_macroreps = 10
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
