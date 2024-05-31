import unittest
import os
import pickle
import math

from simopt.experiment_base import ProblemSolver, post_normalize

class run_template(unittest.TestCase):
    def __init__(self, solver_name, problem_name):
        super().__init__()
        self.problem_name = problem_name
        self.solver_name = solver_name
        self.num_macroreps = 10
        self.num_postreps = 200

    def setUp(self):
        # Setup the solver and experiment
        self.myexperiment = ProblemSolver(self.solver_name, self.problem_name)
        self.assertEqual(self.myexperiment.solver.name, self.solver_name, "Solver name does not match (expected: " + self.solver_name + ", actual: " + self.myexperiment.solver.name + ")")
        self.assertEqual(self.myexperiment.problem.name, self.problem_name, "Problem name does not match (expected: " + self.problem_name + ", actual: " + self.myexperiment.problem.name + ")")

        # Configure the filename
        problem_filename = ''.join(e for e in self.problem_name if e.isalnum())
        solver_filename = ''.join(e for e in self.solver_name if e.isalnum())
        cwd = os.getcwd()
        self.filename = cwd + r"\test\expected_data\results_" + problem_filename + "_" + solver_filename

    def runTest(self):
        # Load the expected results
        with open(self.filename, "rb") as f:
            expected = pickle.load(f)

        # Check actual run results against expected
        self.myexperiment.run(n_macroreps=self.num_macroreps)
        self.assertEqual(self.myexperiment.n_macroreps, self.num_macroreps, "Number of macro-replications for problem " + self.problem_name + " and solver " + self.solver_name + " does not match.")
        # For each macroreplication
        for mrep in range(self.num_macroreps):
            # For each list of recommended solutions
            for list in range(len(self.myexperiment.all_recommended_xs[mrep])):
                # For each tuple of recommended solutions
                for tuple in range(len(self.myexperiment.all_recommended_xs[mrep][list])):
                    self.assertAlmostEqual(self.myexperiment.all_recommended_xs[mrep][list][tuple], expected.all_recommended_xs[mrep][list][tuple], 5, "Recommended solutions for problem " + self.problem_name + " and solver " + self.solver_name + " do not match at mrep " + str(mrep) + " and index " + str(list) + " and tuple " + str(tuple) + ".")
            # For each list of intermediate budgets
            for list in range(len(self.myexperiment.all_intermediate_budgets[mrep])):
                self.assertAlmostEqual(self.myexperiment.all_intermediate_budgets[mrep][list], expected.all_intermediate_budgets[mrep][list], 5, "Intermediate budgets for problem " + self.problem_name + " and solver " + self.solver_name + " do not match at mrep " + str(mrep) + " and index " + str(list) + ".")
            
        # self.assertListEqual(self.myexperiment.all_recommended_xs, expected.all_recommended_xs, "Recommended solutions for problem " + self.problem_name + " and solver " + self.solver_name + " do not match.")
        # self.assertListEqual(self.myexperiment.all_intermediate_budgets, expected.all_intermediate_budgets, "Intermediate budgets for problem " + self.problem_name + " and solver " + self.solver_name + " do not match.")

        # Check actual post-replication results against expected
        self.myexperiment.post_replicate(n_postreps=self.num_postreps)
        self.assertEqual(self.myexperiment.n_postreps, self.num_postreps, "Number of post-replications for problem " + self.problem_name + " and solver " + self.solver_name + " does not match.")
        for mrep in range(self.num_macroreps):
            self.assertListEqual(self.myexperiment.all_est_objectives[mrep], expected.all_est_objectives[mrep], "Estimated objectives for problem " + self.problem_name + " and solver " + self.solver_name + " do not match.")

        # Check actual post-normalization results against expected
        post_normalize([self.myexperiment], n_postreps_init_opt=self.num_postreps)
        for mrep in range(self.num_macroreps):
            # If both values are NaN, then they don't get compared
            # This is because Nan != Nan
            # If they're both NaN, then they're considered equal for these tests
            # If at least one is not NaN, then they're compared
            if not (math.isnan(self.myexperiment.objective_curves[mrep].compute_area_under_curve()) and math.isnan(expected.objective_curves[mrep].compute_area_under_curve())):
                self.assertAlmostEqual(self.myexperiment.objective_curves[mrep].compute_area_under_curve(), expected.objective_curves[mrep].compute_area_under_curve(), 5, "Objective curves for problem " + self.problem_name + " and solver " + self.solver_name + " do not match.")
            if not (math.isnan(self.myexperiment.progress_curves[mrep].compute_area_under_curve()) and math.isnan(expected.progress_curves[mrep].compute_area_under_curve())):
                self.assertAlmostEqual(self.myexperiment.progress_curves[mrep].compute_area_under_curve(), expected.progress_curves[mrep].compute_area_under_curve(), 5, "Progress curves for problem " + self.problem_name + " and solver " + self.solver_name + " do not match.")
        return True
    
    def tearDown(self):
        # Clean up the experiment
        del self.myexperiment
        del self.filename
        return True