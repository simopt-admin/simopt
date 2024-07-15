import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class test_SSCONT1_ASTRODF(unittest.TestCase):
    def setUp(self):
        # Expected values
        self.expected_problem_name = "SSCONT-1"
        self.expected_solver_name = "ASTRODF"
        self.expected_all_recommended_xs = "[[(600, 600), (578.8563914364373, 576.4851575189023), (540.9216512203627, 548.0078773689489), (494.1304461441873, 555.7913937670363), (461.91294010002355, 570.8797105815213), (411.0137466920665, 554.8507255725015), (349.34149090496754, 503.82367952209165), (347.12151762011837, 452.84320640953047), (347.5616997737789, 463.6778852798452), (355.01363470750596, 460.42063480151614), (355.01363470750596, 460.42063480151614)], [(600, 600), (579.4910239161784, 575.9296468660463), (552.2145775545941, 537.1224893576224), (555.8459996687305, 541.8096018406935), (547.381197427183, 539.0803738341885), (547.381197427183, 539.0803738341885)], [(600, 600), (569.1071442280892, 593.2448936162399), (552.2169393507468, 619.9791596282589), (526.3888359800587, 601.7333635022179), (526.3888359800587, 601.7333635022179)], [(600, 600), (573.3916503038607, 582.9121175551807), (541.9523903435036, 586.3140189566151), (510.4460738202959, 589.0254796587925), (488.1490633615256, 597.1090014532509), (481.5235240168533, 591.1757231948005), (478.2416510138875, 588.1749399274844), (475.2997304500309, 584.8401994546803)], [(600, 600), (569.2350701903849, 592.6843254713281), (523.4962019464077, 580.116273444061), (480.69759285292673, 559.6644016553124), (421.91076302120564, 519.5818788013769), (360.64577817910913, 483.4003692654595), (255.34827173925282, 465.99174675968914), (257.3120396856475, 492.60109996401775), (284.99153597087445, 463.693451736929), (284.99153597087445, 463.693451736929)], [(600, 600), (578.739235908756, 576.5910292781492), (558.9272434017596, 551.9438194380331), (527.3492831373261, 516.5484464983696), (469.3850793814855, 475.285253663212), (398.472827129699, 469.4583779498476), (345.1566122363684, 467.2139449594926), (328.6827953111553, 446.22519991805285), (327.3090574290336, 439.6977599039682), (327.3090574290336, 439.6977599039682)], [(600, 600), (570.4938105683991, 611.3747433037621), (539.8340121238758, 619.1192096728031), (513.0687238506332, 602.2782078127527), (482.77976224782844, 638.7827117132829), (465.99728833966253, 644.6780335468836), (461.650344923239, 639.6185165896374), (458.9855189578502, 649.2627696770077), (456.4257423493956, 642.2086178198188), (456.4257423493956, 642.2086178198188)], [(600, 600), (578.0824341615731, 577.2048183223247), (562.4342486317666, 549.7251114529865), (533.4559270514488, 512.1717244101686), (477.81462874883886, 467.8259701241045), (455.4490846516674, 400.28130018459353), (454.6023799547482, 390.67979047631394), (454.6023799547482, 390.67979047631394)], [(600, 600), (569.7414862412073, 609.188163315318), (526.9957218695218, 629.7502553363246), (494.88328372737016, 614.4395951099829), (459.8401525397978, 620.5718016109577), (456.86916758038274, 622.0873561621012), (456.86916758038274, 622.0873561621012)], [(600, 600), (571.6814358892015, 585.9266590070946), (526.8379885546527, 570.4649245910092), (466.47560155173954, 532.7968713454466), (388.78255968531744, 459.62299454817423), (386.6063278939221, 419.6596281257228), (387.68491970642674, 427.08594312116105), (383.99373973559307, 437.71987929358704), (378.9048506201282, 447.7602286733057), (377.72153707322815, 453.26260280396974), (373.8159615046633, 457.8005780530243), (373.8159615046633, 457.8005780530243)]]"
        self.expected_all_intermediate_budgets = "[[4, 24, 40, 56, 88, 104, 120, 164, 246, 290, 1000], [4, 24, 40, 115, 131, 1000], [4, 24, 40, 56, 1000], [4, 24, 40, 56, 88, 147, 657, 1000], [4, 24, 40, 56, 72, 88, 104, 156, 178, 1000], [4, 24, 40, 56, 72, 88, 120, 152, 224, 1000], [4, 24, 40, 56, 72, 120, 284, 306, 389, 1000], [4, 24, 40, 56, 72, 88, 197, 1000], [4, 24, 40, 72, 88, 919, 1000], [4, 24, 40, 56, 72, 124, 212, 232, 252, 395, 415, 1000]]"
        self.expected_all_est_objectives = "[[618.5809976153716, 604.3426729892019, 580.4415981209335, 566.2145313796427, 562.3946193648277, 546.8794931679699, 535.8028002448107, 532.1622303277846, 532.939014848137, 532.4544015179036, 532.4544015179036], [619.371245290233, 603.6485267460105, 581.5706091178515, 583.9128190245713, 580.7662291262072, 580.7662291262072], [620.2040298994102, 606.2252067443167, 608.2893641318437, 591.7243023004486, 591.7243023004486], [620.3929887875448, 604.7435781022047, 593.9041943029691, 582.7580082494529, 577.4033820040011, 573.9475104848436, 573.3099401704529, 572.2208733061591], [617.140803174291, 601.3803055218472, 580.0102213895614, 562.8270862613596, 541.7823055381924, 532.2321735262267, 533.0254671389854, 533.6428450167786, 530.826054741922, 530.826054741922], [617.6250759903628, 601.2518580630849, 583.183035506711, 565.1946279773447, 542.9742703531032, 533.536305729741, 531.0217017441998, 530.8466345548103, 530.2832795013347, 530.2832795013347], [622.8299886318688, 612.1712702905619, 602.432864073124, 586.4351652344512, 584.1934249957848, 580.5716097961747, 577.3762443066248, 579.1370807297736, 577.2258547314669, 577.2258547314669], [617.1638109984892, 605.340061520366, 594.394681335377, 569.9373131844344, 548.5933480550608, 537.0310437425059, 536.1165507948401, 536.1165507948401], [625.4509909440814, 613.5269435924067, 603.5957837922593, 587.7054301061617, 579.7613183977356, 578.8536290753593, 578.8536290753593], [616.3517529689802, 596.6505073213333, 577.3173730148943, 552.5051111367109, 532.052575373237, 529.7572945355536, 531.5723229366752, 532.2861880351599, 530.8344945508438, 532.0559951508028, 531.9302310318463, 531.9302310318463]]"
        self.expected_objective_curves = "[([4, 24, 40, 56, 88, 104, 120, 164, 246, 290, 1000], [624.4131899421741, 604.3426729892019, 580.4415981209335, 566.2145313796427, 562.3946193648277, 546.8794931679699, 535.8028002448107, 532.1622303277846, 532.939014848137, 532.4544015179036, 532.4544015179036]), ([4, 24, 40, 115, 131, 1000], [624.4131899421741, 603.6485267460105, 581.5706091178515, 583.9128190245713, 580.7662291262072, 580.7662291262072]), ([4, 24, 40, 56, 1000], [624.4131899421741, 606.2252067443167, 608.2893641318437, 591.7243023004486, 591.7243023004486]), ([4, 24, 40, 56, 88, 147, 657, 1000], [624.4131899421741, 604.7435781022047, 593.9041943029691, 582.7580082494529, 577.4033820040011, 573.9475104848436, 573.3099401704529, 572.2208733061591]), ([4, 24, 40, 56, 72, 88, 104, 156, 178, 1000], [624.4131899421741, 601.3803055218472, 580.0102213895614, 562.8270862613596, 541.7823055381924, 532.2321735262267, 533.0254671389854, 533.6428450167786, 530.826054741922, 530.826054741922]), ([4, 24, 40, 56, 72, 88, 120, 152, 224, 1000], [624.4131899421741, 601.2518580630849, 583.183035506711, 565.1946279773447, 542.9742703531032, 533.536305729741, 531.0217017441998, 530.8466345548103, 530.2832795013347, 530.2832795013347]), ([4, 24, 40, 56, 72, 120, 284, 306, 389, 1000], [624.4131899421741, 612.1712702905619, 602.432864073124, 586.4351652344512, 584.1934249957848, 580.5716097961747, 577.3762443066248, 579.1370807297736, 577.2258547314669, 577.2258547314669]), ([4, 24, 40, 56, 72, 88, 197, 1000], [624.4131899421741, 605.340061520366, 594.394681335377, 569.9373131844344, 548.5933480550608, 537.0310437425059, 536.1165507948401, 536.1165507948401]), ([4, 24, 40, 72, 88, 919, 1000], [624.4131899421741, 613.5269435924067, 603.5957837922593, 587.7054301061617, 579.7613183977356, 578.8536290753593, 578.8536290753593]), ([4, 24, 40, 56, 72, 124, 212, 232, 252, 395, 415, 1000], [624.4131899421741, 596.6505073213333, 577.3173730148943, 552.5051111367109, 532.052575373237, 533.804570358681, 531.5723229366752, 532.2861880351599, 530.8344945508438, 532.0559951508028, 531.9302310318463, 531.9302310318463])]"
        self.expected_progress_curves = "[([0.004, 0.024, 0.04, 0.056, 0.088, 0.104, 0.12, 0.164, 0.246, 0.29, 1.0], [1.0, 0.778492189316737, 0.5147085120227207, 0.3576918086815897, 0.31553343531298206, 0.14430109264870578, 0.022053419369096532, -0.0181256489553177, -0.009552683999852023, -0.01490110816149422, -0.01490110816149422]), ([0.004, 0.024, 0.04, 0.115, 0.131, 1.0], [1.0, 0.7708312598556962, 0.5271688165953746, 0.5530185637550415, 0.5182912948392566, 0.5182912948392566]), ([0.004, 0.024, 0.04, 0.056, 1.0], [1.0, 0.7992687309279914, 0.8220497576891929, 0.6392298239175395, 0.6392298239175395]), ([0.004, 0.024, 0.04, 0.056, 0.088, 0.147, 0.657, 1.0], [1.0, 0.782916769614347, 0.6632881531641498, 0.5402735205083092, 0.48117730791765456, 0.44303665932325736, 0.4360001288328751, 0.4239806667850034]), ([0.004, 0.024, 0.04, 0.056, 0.072, 0.088, 0.104, 0.156, 0.178, 1.0], [1.0, 0.7457980871333901, 0.5099476323916764, 0.3203063465273878, 0.08804609557217911, -0.017353722412748573, -0.008598555228817396, -0.001784878112544093, -0.032872320872456756, -0.032872320872456756]), ([0.004, 0.024, 0.04, 0.056, 0.072, 0.088, 0.12, 0.152, 0.224, 1.0], [1.0, 0.7443804796325508, 0.5449643243105505, 0.3464356676324676, 0.10120118854666596, -0.002960696566983477, -0.030713067115174263, -0.032645192228593876, -0.03886264765463584, -0.03886264765463584]), ([0.004, 0.024, 0.04, 0.056, 0.072, 0.12, 0.284, 0.306, 0.389, 1.0], [1.0, 0.8648923280380446, 0.7574146260025977, 0.5808563811886877, 0.5561154652695265, 0.516143382963685, 0.4808778033285659, 0.5003112350621358, 0.47921803215172665, 0.47921803215172665]), ([0.004, 0.024, 0.04, 0.056, 0.072, 0.088, 0.197, 1.0], [1.0, 0.7894998454950222, 0.6687014023082432, 0.3987782066634202, 0.1632160137121659, 0.035608901213331004, 0.025516120285098705, 0.025516120285098705]), ([0.004, 0.024, 0.04, 0.072, 0.088, 0.919, 1.0], [1.0, 0.8798541860607862, 0.7702491634282969, 0.5948756309857769, 0.507200620098918, 0.4971829272287611, 0.4971829272287611]), ([0.004, 0.024, 0.04, 0.056, 0.072, 0.124, 0.212, 0.232, 0.252, 0.395, 0.415, 1.0], [1.0, 0.6935977752617861, 0.48022807163635917, 0.2063880993220288, -0.019335853404427453, 0.0, -0.024636148660766754, -0.01675759249507124, -0.03277917511038007, -0.019298111105940666, -0.02068610398713191, -0.02068610398713191])]"

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
