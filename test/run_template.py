import unittest
import os
import pickle
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class run_template(unittest.TestCase):
    def __init__(self, solver_name: str, problem_name: str):
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
            pickled_data = pickle.load(f)

        # Check actual run results against expected
        self.myexperiment.run(n_macroreps=self.num_macroreps)
        self.assertEqual(self.myexperiment.n_macroreps, self.num_macroreps, "Number of macro-replications for problem " + self.problem_name + " and solver " + self.solver_name + " does not match.")
        # For each macroreplication
        for mrep in range(self.num_macroreps):
            # Check to make sure the list lengths are the same
            self.assertEqual(len(self.myexperiment.all_recommended_xs[mrep]), len(pickled_data["all_recommended_xs"][mrep]), "Length of recommended solutions for problem " + self.problem_name + " and solver " + self.solver_name + " do not match.")
            # For each list of recommended solutions
            for list in range(len(self.myexperiment.all_recommended_xs[mrep])):
                # Check to make sure the tuples are the same length
                self.assertEqual(len(self.myexperiment.all_recommended_xs[mrep][list]), len(pickled_data["all_recommended_xs"][mrep][list]), "Recommended solutions for problem " + self.problem_name + " and solver " + self.solver_name + " do not match at mrep " + str(mrep) + " and index " + str(list) + ".")
                # For each tuple of recommended solutions
                for tuple in range(len(self.myexperiment.all_recommended_xs[mrep][list])):
                    self.assertAlmostEqual(self.myexperiment.all_recommended_xs[mrep][list][tuple], pickled_data["all_recommended_xs"][mrep][list][tuple], 5, "Recommended solutions for problem " + self.problem_name + " and solver " + self.solver_name + " do not match at mrep " + str(mrep) + " and index " + str(list) + " and tuple " + str(tuple) + ".")
            # Check to make sure the list lengths are the same
            self.assertEqual(len(self.myexperiment.all_intermediate_budgets[mrep]), len(pickled_data["all_intermediate_budgets"][mrep]), "Length of intermediate budgets for problem " + self.problem_name + " and solver " + self.solver_name + " do not match.")
            # For each list of intermediate budgets
            for list in range(len(self.myexperiment.all_intermediate_budgets[mrep])):
                # Check the values in the list
                self.assertAlmostEqual(self.myexperiment.all_intermediate_budgets[mrep][list], pickled_data["all_intermediate_budgets"][mrep][list], 5, "Intermediate budgets for problem " + self.problem_name + " and solver " + self.solver_name + " do not match at mrep " + str(mrep) + " and index " + str(list) + ".")
            
        # Check actual post-replication results against expected
        self.myexperiment.post_replicate(n_postreps=self.num_postreps)
        self.assertEqual(self.myexperiment.n_postreps, self.num_postreps, "Number of post-replications for problem " + self.problem_name + " and solver " + self.solver_name + " does not match.")
        # For each macroreplication
        for mrep in range(self.num_macroreps):
            # Check to make sure the list lengths are the same
            self.assertEqual(len(self.myexperiment.all_est_objectives[mrep]), len(pickled_data["all_est_objectives"][mrep]), "Estimated objectives for problem " + self.problem_name + " and solver " + self.solver_name + " do not match.")
            # For each list in the estimated objectives
            for list in range(len(self.myexperiment.all_est_objectives[mrep])):
                # Check the values in the list
                self.assertAlmostEqual(self.myexperiment.all_est_objectives[mrep][list], pickled_data["all_est_objectives"][mrep][list], 5, "Estimated objectives for problem " + self.problem_name + " and solver " + self.solver_name + " do not match at mrep " + str(mrep) + " and index " + str(list) + ".")

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
            self.assertEqual(len(self.myexperiment.objective_curves[mrep]), len(pickled_data["objective_curves"][mrep]), "Number of objective curves for problem " + self.problem_name + " and solver " + self.solver_name + " does not match.")
            # Make sure that curves are only checked if they exist
            if (len(self.myexperiment.objective_curves[mrep]) > 0):
                # Make sure the lengths of the X and Y values are the same
                self.assertEqual(len(self.myexperiment.objective_curves[mrep][0]), len(pickled_data["objective_curves"][mrep][0]), "Length of X values for problem " + self.problem_name + " and solver " + self.solver_name + " do not match.")
                self.assertEqual(len(self.myexperiment.objective_curves[mrep][1]), len(pickled_data["objective_curves"][mrep][1]), "Length of Y values for problem " + self.problem_name + " and solver " + self.solver_name + " do not match.")
                # Check X (0) and Y (1) values
                for x_index in range(len(self.myexperiment.objective_curves[mrep][0])):
                    # If the value is NaN, make sure we're expecting NaN
                    if (math.isnan(self.myexperiment.objective_curves[mrep][0][x_index])):
                        self.assertTrue(math.isnan(pickled_data["objective_curves"][mrep][0][x_index]), "X values for problem " + self.problem_name + " and solver " + self.solver_name + " do not match at mrep " + str(mrep) + " and index " + str(x_index) + ".")
                    # Otherwise, check the value normally
                    else:
                        self.assertAlmostEqual(self.myexperiment.objective_curves[mrep][0][x_index], pickled_data["objective_curves"][mrep][0][x_index], 5, "X values for problem " + self.problem_name + " and solver " + self.solver_name + " do not match at mrep " + str(mrep) + " and index " + str(x_index) + ".")
                for y_index in range(len(self.myexperiment.objective_curves[mrep][1])):
                    # If the value is NaN, make sure we're expecting NaN
                    if (math.isnan(self.myexperiment.objective_curves[mrep][1][y_index])):
                        self.assertTrue(math.isnan(pickled_data["objective_curves"][mrep][1][y_index]), "Y values for problem " + self.problem_name + " and solver " + self.solver_name + " do not match at mrep " + str(mrep) + " and index " + str(y_index) + ".")
                    # Otherwise, check the value normally
                    else:
                        self.assertAlmostEqual(self.myexperiment.objective_curves[mrep][1][y_index], pickled_data["objective_curves"][mrep][1][y_index], 5, "Y values for problem " + self.problem_name + " and solver " + self.solver_name + " do not match at mrep " + str(mrep) + " and index " + str(y_index) + ".")
            
            # Check to make sure the same number of progress curves are present
            # This should probably always be 2 (x and y)
            self.assertEqual(len(self.myexperiment.progress_curves[mrep]), len(pickled_data["progress_curves"][mrep]), "Number of progress curves for problem " + self.problem_name + " and solver " + self.solver_name + " does not match.")
            # Make sure that curves are only checked if they exist
            if (len(self.myexperiment.progress_curves[mrep]) > 0):
                # Make sure the lengths of the X and Y values are the same
                self.assertEqual(len(self.myexperiment.progress_curves[mrep][0]), len(pickled_data["progress_curves"][mrep][0]), "Length of X values for problem " + self.problem_name + " and solver " + self.solver_name + " do not match.")
                self.assertEqual(len(self.myexperiment.progress_curves[mrep][1]), len(pickled_data["progress_curves"][mrep][1]), "Length of Y values for problem " + self.problem_name + " and solver " + self.solver_name + " do not match.")
                # Check X (0) and Y (1) values
                for x_index in range(len(self.myexperiment.progress_curves[mrep][0])):
                    # If the value is NaN, make sure we're expecting NaN
                    if (math.isnan(self.myexperiment.progress_curves[mrep][0][x_index])):
                        self.assertTrue(math.isnan(pickled_data["progress_curves"][mrep][0][x_index]), "X values for problem " + self.problem_name + " and solver " + self.solver_name + " do not match at mrep " + str(mrep) + " and index " + str(x_index) + ".")
                    # Otherwise, check the value normally
                    else:
                        self.assertAlmostEqual(self.myexperiment.progress_curves[mrep][0][x_index], pickled_data["progress_curves"][mrep][0][x_index], 5, "X values for problem " + self.problem_name + " and solver " + self.solver_name + " do not match at mrep " + str(mrep) + " and index " + str(x_index) + ".")
                for y_index in range(len(self.myexperiment.progress_curves[mrep][1])):
                    # If the value is NaN, make sure we're expecting NaN
                    if (math.isnan(self.myexperiment.progress_curves[mrep][1][y_index])):
                        self.assertTrue(math.isnan(pickled_data["progress_curves"][mrep][1][y_index]), "Y values for problem " + self.problem_name + " and solver " + self.solver_name + " do not match at mrep " + str(mrep) + " and index " + str(y_index) + ".")
                    # Otherwise, check the value normally
                    else:
                        self.assertAlmostEqual(self.myexperiment.progress_curves[mrep][1][y_index], pickled_data["progress_curves"][mrep][1][y_index], 5, "Y values for problem " + self.problem_name + " and solver " + self.solver_name + " do not match at mrep " + str(mrep) + " and index " + str(y_index) + ".")
        
        print("\n\nPassed test for problem " + self.problem_name + " and solver " + self.solver_name + ".\n\n")
        
        return True
    
    def tearDown(self):
        # Clean up the experiment
        del self.myexperiment
        del self.filename
        return True