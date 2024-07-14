import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class test_EXAMPLE1_SPSA(unittest.TestCase):
    def setUp(self):
        # Expected values
        self.expected_problem_name = "EXAMPLE-1"
        self.expected_solver_name = "SPSA"
        self.expected_all_recommended_xs = "[[(2.0, 2.0), (2.0, 2.0)], [(2.0, 2.0), (2.0, 2.0), (1.9262482642039207, 1.9262482642039207), (1.9262482642039207, 1.9262482642039207), (1.8771695727453401, 1.8771695727453401), (1.8351854585409866, 1.8351854585409866), (1.8351854585409866, 1.8351854585409866), (1.8351854585409866, 1.8351854585409866), (1.8044194630360328, 1.8044194630360328), (1.8044194630360328, 1.8044194630360328), (1.8044194630360328, 1.8044194630360328), (1.7797439569579159, 1.7797439569579159), (1.756744958669717, 1.756744958669717), (1.7352014442722137, 1.7352014442722137), (1.7352014442722137, 1.7352014442722137), (1.7352014442722137, 1.7352014442722137)], [(2.0, 2.0), (2.0, 2.0)], [(2.0, 2.0), (2.0, 2.0), (2.0, 2.0), (2.0, 2.0), (2.0, 2.0), (1.9552687036760859, 1.9552687036760859), (1.9552687036760859, 1.9552687036760859), (1.9552687036760859, 1.9552687036760859), (1.922489570717992, 1.922489570717992), (1.922489570717992, 1.922489570717992), (1.922489570717992, 1.922489570717992), (1.922489570717992, 1.922489570717992), (1.8976459216227595, 1.8976459216227595), (1.8976459216227595, 1.8976459216227595), (1.8976459216227595, 1.8976459216227595), (1.8976459216227595, 1.8976459216227595)], [(2.0, 2.0), (2.0, 2.0), (2.0, 2.0), (2.0, 2.0), (2.0, 2.0), (2.0, 2.0), (1.959920230685785, 1.959920230685785), (1.92419940145256, 1.92419940145256), (1.92419940145256, 1.92419940145256), (1.92419940145256, 1.92419940145256), (1.92419940145256, 1.92419940145256), (1.897885900075135, 1.897885900075135), (1.8733601954666845, 1.8733601954666845), (1.8503865918462092, 1.8503865918462092), (1.8503865918462092, 1.8503865918462092), (1.8503865918462092, 1.8503865918462092)], [(2.0, 2.0), (2.0, 2.0), (1.9262482642039207, 1.9262482642039207), (1.9262482642039207, 1.9262482642039207), (1.8771695727453401, 1.8771695727453401), (1.8351854585409866, 1.8351854585409866), (1.7984085536274244, 1.7984085536274244), (1.7984085536274244, 1.7984085536274244), (1.768259105123751, 1.768259105123751), (1.768259105123751, 1.768259105123751), (1.742542863977896, 1.742542863977896), (1.742542863977896, 1.742542863977896), (1.742542863977896, 1.742542863977896), (1.742542863977896, 1.742542863977896), (1.7221899575569148, 1.7221899575569148), (1.7221899575569148, 1.7221899575569148)], [(2.0, 2.0), (2.0, 2.0), (1.9262482642039207, 1.9262482642039207), (1.9262482642039207, 1.9262482642039207), (1.8771695727453401, 1.8771695727453401), (1.8771695727453401, 1.8771695727453401), (1.8771695727453401, 1.8771695727453401), (1.8771695727453401, 1.8771695727453401), (1.8456997338969907, 1.8456997338969907), (1.8456997338969907, 1.8456997338969907), (1.8188572540238763, 1.8188572540238763), (1.7939843105945374, 1.7939843105945374), (1.7939843105945374, 1.7939843105945374), (1.7939843105945374, 1.7939843105945374), (1.7939843105945374, 1.7939843105945374), (1.7939843105945374, 1.7939843105945374)], [(2.0, 2.0), (2.0, 2.0), (1.926248264203921, 1.926248264203921), (1.8686282777737477, 1.8686282777737477), (1.8210176803238192, 1.8210176803238192), (1.7802894395889934, 1.7802894395889934), (1.7802894395889934, 1.7802894395889934), (1.7478424991157986, 1.7478424991157986), (1.7478424991157986, 1.7478424991157986), (1.7478424991157986, 1.7478424991157986), (1.7478424991157986, 1.7478424991157986), (1.7478424991157986, 1.7478424991157986), (1.7252557520233032, 1.7252557520233032), (1.7252557520233032, 1.7252557520233032), (1.705104759127191, 1.705104759127191), (1.705104759127191, 1.705104759127191)], [(2.0, 2.0), (2.0, 2.0)], [(2.0, 2.0), (2.0, 2.0)]]"
        self.expected_all_intermediate_budgets = "[[0, 1000], [0, 210, 270, 330, 390, 450, 510, 570, 630, 690, 750, 810, 870, 930, 990, 1000], [0, 1000], [0, 210, 270, 330, 390, 450, 510, 570, 630, 690, 750, 810, 870, 930, 990, 1000], [0, 210, 270, 330, 390, 450, 510, 570, 630, 690, 750, 810, 870, 930, 990, 1000], [0, 210, 270, 330, 390, 450, 510, 570, 630, 690, 750, 810, 870, 930, 990, 1000], [0, 210, 270, 330, 390, 450, 510, 570, 630, 690, 750, 810, 870, 930, 990, 1000], [0, 210, 270, 330, 390, 450, 510, 570, 630, 690, 750, 810, 870, 930, 990, 1000], [0, 1000], [0, 1000]]"
        self.expected_all_est_objectives = "[[7.984539704940337, 7.984539704940337], [8.081590387702734, 8.081590387702734, 7.5024551383999665, 7.5024551383999665, 7.129121597384576, 6.817401722183314, 6.817401722183314, 6.817401722183314, 6.593449584869222, 6.593449584869222, 6.593449584869222, 6.416567492359173, 6.253896087325662, 6.103438492111484, 6.103438492111484, 6.103438492111484], [7.9253347189439385, 7.9253347189439385], [8.073099810658121, 8.073099810658121, 8.073099810658121, 8.073099810658121, 8.073099810658121, 7.719251217808443, 7.719251217808443, 7.719251217808443, 7.4650321096970185, 7.4650321096970185, 7.4650321096970185, 7.4650321096970185, 7.2752198983611045, 7.2752198983611045, 7.2752198983611045, 7.2752198983611045], [7.880122723414122, 7.880122723414122, 7.880122723414122, 7.880122723414122, 7.880122723414122, 7.880122723414122, 7.5626973447169625, 7.285209396514899, 7.285209396514899, 7.285209396514899, 7.285209396514899, 7.084064502822128, 6.899079567332069, 6.727983801982577, 6.727983801982577, 6.727983801982577], [8.025785950362149, 8.025785950362149, 7.446650701059383, 7.446650701059383, 7.073317160043992, 6.761597284842729, 6.494332601882715, 6.494332601882715, 6.279266476068244, 6.279266476068244, 6.098697215962723, 6.098697215962723, 6.098697215962723, 6.098697215962723, 5.957662450181923, 5.957662450181923], [8.015084462897443, 8.015084462897443, 7.435949213594679, 7.435949213594679, 7.0626156725792875, 7.0626156725792875, 7.0626156725792875, 7.0626156725792875, 6.828299478312288, 6.828299478312288, 6.631567883927994, 6.451843876216158, 6.451843876216158, 6.451843876216158, 6.451843876216158, 6.451843876216158], [7.994852045957048, 7.994852045957048, 7.415716796654282, 6.978395326948412, 6.627062830060933, 6.333713023381231, 6.333713023381231, 6.104758849387767, 6.104758849387767, 6.104758849387767, 6.104758849387767, 6.104758849387767, 5.9478668657360325, 5.9478668657360325, 5.809616525153437, 5.809616525153437], [7.910902809206077, 7.910902809206077], [7.943417039435916, 7.943417039435916]]"
        self.expected_objective_curves = "[([0, 1000], [8.090508544469758, 8.090508544469758]), ([0, 210, 270, 330, 390, 450, 510, 570, 630, 690, 750, 810, 870, 930, 990, 1000], [8.090508544469758, 8.090508544469758, 7.5024551383999665, 7.5024551383999665, 7.129121597384576, 6.817401722183314, 6.817401722183314, 6.817401722183314, 6.593449584869222, 6.593449584869222, 6.593449584869222, 6.416567492359173, 6.253896087325662, 6.103438492111484, 6.103438492111484, 6.103438492111484]), ([0, 1000], [8.090508544469758, 8.090508544469758]), ([0, 210, 270, 330, 390, 450, 510, 570, 630, 690, 750, 810, 870, 930, 990, 1000], [8.090508544469758, 8.090508544469758, 8.090508544469758, 8.090508544469758, 8.090508544469758, 7.719251217808443, 7.719251217808443, 7.719251217808443, 7.4650321096970185, 7.4650321096970185, 7.4650321096970185, 7.4650321096970185, 7.2752198983611045, 7.2752198983611045, 7.2752198983611045, 7.2752198983611045]), ([0, 210, 270, 330, 390, 450, 510, 570, 630, 690, 750, 810, 870, 930, 990, 1000], [8.090508544469758, 8.090508544469758, 8.090508544469758, 8.090508544469758, 8.090508544469758, 8.090508544469758, 7.5626973447169625, 7.285209396514899, 7.285209396514899, 7.285209396514899, 7.285209396514899, 7.084064502822128, 6.899079567332069, 6.727983801982577, 6.727983801982577, 6.727983801982577]), ([0, 210, 270, 330, 390, 450, 510, 570, 630, 690, 750, 810, 870, 930, 990, 1000], [8.090508544469758, 8.090508544469758, 7.446650701059383, 7.446650701059383, 7.073317160043992, 6.761597284842729, 6.494332601882715, 6.494332601882715, 6.279266476068244, 6.279266476068244, 6.098697215962723, 6.098697215962723, 6.098697215962723, 6.098697215962723, 5.957662450181923, 5.957662450181923]), ([0, 210, 270, 330, 390, 450, 510, 570, 630, 690, 750, 810, 870, 930, 990, 1000], [8.090508544469758, 8.090508544469758, 7.435949213594679, 7.435949213594679, 7.0626156725792875, 7.0626156725792875, 7.0626156725792875, 7.0626156725792875, 6.828299478312288, 6.828299478312288, 6.631567883927994, 6.451843876216158, 6.451843876216158, 6.451843876216158, 6.451843876216158, 6.451843876216158]), ([0, 210, 270, 330, 390, 450, 510, 570, 630, 690, 750, 810, 870, 930, 990, 1000], [8.090508544469758, 8.090508544469758, 7.415716796654282, 6.978395326948412, 6.627062830060933, 6.333713023381231, 6.333713023381231, 6.104758849387767, 6.104758849387767, 6.104758849387767, 6.104758849387767, 6.104758849387767, 5.9478668657360325, 5.9478668657360325, 5.809616525153437, 5.809616525153437]), ([0, 1000], [8.090508544469758, 8.090508544469758]), ([0, 1000], [8.090508544469758, 8.090508544469758])]"
        self.expected_progress_curves = "[([0.0, 1.0], [1.0, 1.0]), ([0.0, 0.21, 0.27, 0.33, 0.39, 0.45, 0.51, 0.57, 0.63, 0.69, 0.75, 0.81, 0.87, 0.93, 0.99, 1.0], [1.0, 1.0, 0.9273156436536052, 0.9273156436536052, 0.8811710114635087, 0.8426419284660824, 0.8426419284660824, 0.8426419284660824, 0.8149610804596645, 0.8149610804596645, 0.8149610804596645, 0.7930981664613898, 0.7729917165220095, 0.7543949133189496, 0.7543949133189496, 0.7543949133189496]), ([0.0, 1.0], [1.0, 1.0]), ([0.0, 0.21, 0.27, 0.33, 0.39, 0.45, 0.51, 0.57, 0.63, 0.69, 0.75, 0.81, 0.87, 0.93, 0.99, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 0.9541119912771013, 0.9541119912771013, 0.9541119912771013, 0.9226900965081754, 0.9226900965081754, 0.9226900965081754, 0.9226900965081754, 0.8992289988166514, 0.8992289988166514, 0.8992289988166514, 0.8992289988166514]), ([0.0, 0.21, 0.27, 0.33, 0.39, 0.45, 0.51, 0.57, 0.63, 0.69, 0.75, 0.81, 0.87, 0.93, 0.99, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9347616782243461, 0.9004637170174774, 0.9004637170174774, 0.9004637170174774, 0.9004637170174774, 0.8756018813754813, 0.8527374428209353, 0.8315897282602179, 0.8315897282602179, 0.8315897282602179]), ([0.0, 0.21, 0.27, 0.33, 0.39, 0.45, 0.51, 0.57, 0.63, 0.69, 0.75, 0.81, 0.87, 0.93, 0.99, 1.0], [1.0, 1.0, 0.9204181245379832, 0.9204181245379832, 0.8742734923478865, 0.8357444093504601, 0.8027100603363055, 0.8027100603363055, 0.7761275377875247, 0.7761275377875247, 0.7538088838842484, 0.7538088838842484, 0.7538088838842484, 0.7538088838842484, 0.7363767577075565, 0.7363767577075565]), ([0.0, 0.21, 0.27, 0.33, 0.39, 0.45, 0.51, 0.57, 0.63, 0.69, 0.75, 0.81, 0.87, 0.93, 0.99, 1.0], [1.0, 1.0, 0.9190954033016254, 0.9190954033016254, 0.8729507711115287, 0.8729507711115287, 0.8729507711115287, 0.8729507711115287, 0.8439889088282035, 0.8439889088282035, 0.819672564150616, 0.7974583848164026, 0.7974583848164026, 0.7974583848164026, 0.7974583848164026, 0.7974583848164026]), ([0.0, 0.21, 0.27, 0.33, 0.39, 0.45, 0.51, 0.57, 0.63, 0.69, 0.75, 0.81, 0.87, 0.93, 0.99, 1.0], [1.0, 1.0, 0.9165946436980493, 0.8625409995665195, 0.8191157321737015, 0.782857219489697, 0.782857219489697, 0.7545581116233608, 0.7545581116233608, 0.7545581116233608, 0.7545581116233608, 0.7545581116233608, 0.7351660075560613, 0.7351660075560613, 0.7180780408574666, 0.7180780408574666]), ([0.0, 1.0], [1.0, 1.0]), ([0.0, 1.0], [1.0, 1.0])]"

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
