import numpy as np
import os
import csv
import pickle
from copy import deepcopy

from directory import oracle_directory
from rng.mrg32k3a import MRG32k3a
from wrapper_base import Experiment


class DesignPoint(object):
    """
    Base class for design points represented as dictionaries of factors.

    Attributes
    ----------
    oracle : Oracle object
        oracle to simulate
    oracle_factors : dict
        oracle factor names and values
    rng_list : list of rng.MRG32k3a objects
        rngs for oracle to use when running replications at the solution
    n_reps : int
        number of replications run at a design point
    responses : dict
        responses observed from replications
    gradients : dict of dict
        gradients of responses (w.r.t. oracle factors) observed from replications

    Arguments
    ---------
    oracle : Oracle object
        oracle with factors oracle_factors
    """
    def __init__(self, oracle):
        super().__init__()
        # Create separate copy of Oracle object for use at this design point.
        self.oracle = deepcopy(oracle)
        self.oracle_factors = self.oracle.factors
        self.n_reps = 0
        self.responses = {}
        self.gradients = {}

    def attach_rngs(self, rng_list, copy=True):
        """
        Attach a list of random-number generators to the design point.

        Arguments
        ---------
        rng_list : list of rng.MRG32k3a objects
            list of random-number generators used to run simulation replications
        """
        if copy:
            self.rng_list = [deepcopy(rng) for rng in rng_list]
        else:
            self.rng_list = rng_list

    def simulate(self, m=1):
        """
        Simulate m replications for the current oracle factors.
        Append results to the responses and gradients dictionaries.

        Arguments
        ---------
        m : int > 0
            number of macroreplications to run at the design point
        """
        for _ in range(m):
            # Generate a single replication of oracle, as described by design point.
            responses, gradients = self.oracle.replicate(rng_list=self.rng_list)
            # If first replication, set up recording responses and gradients.
            if self.n_reps == 0:
                self.responses = {response_key: [] for response_key in responses}
                self.gradients = {response_key: {factor_key: [] for factor_key in gradients[response_key]} for response_key in responses}
            # Append responses and gradients.
            for key in self.responses:
                self.responses[key].append(responses[key])
            for outerkey in self.gradients:
                for innerkey in self.gradients[outerkey]:
                    self.gradients[outerkey][innerkey].append(gradients[outerkey][innerkey])
            self.n_reps += 1
            # Advance rngs to start of next subsubstream.
            for rng in self.rng_list:
                rng.advance_subsubstream()


class DataFarmingExperiment(object):
    """
    Base class for data-farming experiments consisting of an oracle
    and design of associated factors.

    Attributes
    ----------
    oracle : Oracle object
        oracle on which the experiment is run
    design : list of DesignPoint objects
        list of design points forming the design
    n_design_pts : int
        number of design points in the design

    Arguments
    ---------
    oracle_name : string
        name of oracle on which the experiment is run
    factor_settings_filename : string
        name of .txt file containing factor ranges and # of digits
    factor_headers : list of strings
        ordered list of factor names appearing in factor settings/design file
    design_filename : string
        name of .txt file containing design matrix
    oracle_fixed_factors : dictionary
        non-default values of oracle factors that will not be varied
    """
    def __init__(self, oracle_name, factor_settings_filename, factor_headers, design_filename=None, oracle_fixed_factors={}):
        # Initialize oracle object with fixed factors.
        self.oracle = oracle_directory[oracle_name](fixed_factors=oracle_fixed_factors)
        if design_filename is None:
            # Create oracle factor design from .txt file of factor settings.
            # Hard-coded for a single-stack NOLHS.
            command = "stack_nolhs.rb -s 1 ./simopt/data_farming_experiments/" + factor_settings_filename + ".txt > ./simopt/data_farming_experiments/" + factor_settings_filename + "_design.txt"
            os.system(command)
            # Append design to base filename.
            design_filename = factor_settings_filename + "_design"
        # Read in design matrix from .txt file.
        design_table = np.loadtxt("./simopt/data_farming_experiments/" + design_filename + ".txt")
        # Count number of design_points.
        self.n_design_pts = len(design_table)
        # Create all design points.
        self.design = []
        design_pt_factors = {}
        for i in range(self.n_design_pts):
            for j in range(len(factor_headers)):
                # Parse oracle factors for next design point.
                design_pt_factors[factor_headers[j]] = design_table[i][j]
            # Update oracle factors according to next design point.
            self.oracle.factors.update(design_pt_factors)
            # Create new design point and add to design.
            self.design.append(DesignPoint(self.oracle))

    def run(self, n_reps=10, crn_across_design_pts=True):
        """
        Run a fixed number of macroreplications at each design point.

        Arguments
        ---------
        n_reps : int
            number of replications run at each design point
        crn_across_design_pts : Boolean
            use CRN across design points?
        """
        # Setup random number generators for oracle.
        # Use stream 0 for all runs; start with substreams 0, 1, ..., oracle.n_rngs-1.
        main_rng_list = [MRG32k3a(s_ss_sss_index=[0, ss, 0]) for ss in range(self.oracle.n_rngs)]
        # All design points will share the same random number generator objects.
        # Simulate n_reps replications from each design point.
        for design_pt in self.design:
            # Attach random number generators.
            design_pt.attach_rngs(rng_list=main_rng_list, copy=False)
            # Simulate n_reps replications from each design point.
            design_pt.simulate(n_reps)
            # Manage random number streams.
            if crn_across_design_pts:
                # Reset rngs to start of current substream.
                for rng in main_rng_list:
                    rng.reset_substream()
            else:  # If not using CRN...
                # ...advance rngs to starts of next set of substreams.
                for rng in main_rng_list:
                    for _ in range(len(main_rng_list)):
                        rng.advance_substream()

    def print_to_csv(self, csv_filename="raw_results"):
        """
        Extract observed responses from simulated design points.
        Publish to .csv output file.

        Argument
        --------
        csv_filename : string
            name of .csv file to print output to
        """
        with open("./simopt/data_farming_experiments/" + csv_filename + ".csv", mode="w", newline="") as output_file:
            csv_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            # Print headers.
            oracle_factor_names = list(self.oracle.specifications.keys())
            response_names = list(self.design[0].responses.keys())
            csv_writer.writerow(["DesignPt#"] + oracle_factor_names + ["MacroRep#"] + response_names)
            for designpt_index in range(self.n_design_pts):
                designpt = self.design[designpt_index]
                # Parse list of oracle factors.
                oracle_factor_list = [designpt.oracle_factors[oracle_factor_name] for oracle_factor_name in oracle_factor_names]
                for mrep in range(designpt.n_reps):
                    # Parse list of responses.
                    response_list = [designpt.responses[response_name][mrep] for response_name in response_names]
                    print_list = [designpt_index] + oracle_factor_list + [mrep] + response_list
                    csv_writer.writerow(print_list)


class DataFarmingMetaExperiment(object):
    """
    Base class for data-farming meta experiments consisting of problem-solver
    pairs and a design of associated factors.

    Attributes
    ----------
    design : list of Experiment objects
        list of design points forming the design
    n_design_pts : int
        number of design points in the design

    Arguments
    ---------
    solver_name : string
        name of solver
    problem_name : string
        name of problem
    solver_factor_settings_filename : string
        name of .txt file containing solver factor ranges and # of digits
    solver_factor_headers : list of strings
        ordered list of solver factor names appearing in factor settings/design file
    design_filename : string
        name of .txt file containing design matrix
    solver_fixed_factors : dict
        dictionary of user-specified solver factors that will not be varied
    problem_fixed_factors : dict
        dictionary of user-specified problem factors that will not be varied
    oracle_fixed_factors : dict
        dictionary of user-specified oracle factors that will not be varied
    """
    def __init__(self, solver_name, problem_name, solver_factor_headers, solver_factor_settings_filename=None, design_filename=None, solver_fixed_factors={}, problem_fixed_factors={}, oracle_fixed_factors={}):
        # TO DO: Extend to allow a design on problem/oracle factors too.
        # Currently supports designs on solver factors only.
        if design_filename is None:
            # Create solver factor design from .txt file of factor settings.
            # Hard-coded for a single-stack NOLHS.
            command = "stack_nolhs.rb -s 1 ./simopt/data_farming_experiments/" + solver_factor_settings_filename + ".txt > ./simopt/data_farming_experiments/" + solver_factor_settings_filename + "_design.txt"
            os.system(command)
            # Append design to base filename.
            design_filename = solver_factor_settings_filename + "_design"
        # Read in design matrix from .txt file.
        design_table = np.loadtxt("./simopt/data_farming_experiments/" + design_filename + ".txt")
        # Count number of design_points.
        self.n_design_pts = len(design_table)
        # Create all design points.
        self.design = []
        design_pt_solver_factors = {}
        for i in range(self.n_design_pts):
            # TO DO: Fix this issue with numpy 1D and 2D arrays handled differently
            if len(solver_factor_headers) == 1:
                # TO DO: Resolve type-casting issues:
                # E.g., sample_size must be an integer for RNDSRCH, but np.loadtxt will make it a float.
                # parse solver factors for next design point
                design_pt_solver_factors[solver_factor_headers[0]] = int(design_table[i])
            else:
                for j in range(len(solver_factor_headers)):
                    # Parse solver factors for next design point.
                    design_pt_solver_factors[solver_factor_headers[j]] = design_table[i, j]
            # Merge solver fixed factors and solver factors specified for design point.
            new_design_pt_solver_factors = {**solver_fixed_factors, **design_pt_solver_factors}
            # In Python 3.9, will be able to use: dict1 | dict2.
            # Create new design point and add to design0.
            file_name_path = "data_farming_experiments/outputs/" + solver_name + "_on_" + problem_name + "_designpt_" + str(i) + ".pickle"
            new_design_pt = Experiment(solver_name, problem_name, new_design_pt_solver_factors, problem_fixed_factors, oracle_fixed_factors, file_name_path=file_name_path)
            self.design.append(new_design_pt)

    # Largely taken from MetaExperiment class in wrapper_base.py.
    def run(self, n_macroreps=10):
        """
        Run n_macroreps of each problem-solver design point.

        Arguments
        ---------
        n_macroreps : int
            number of macroreplications for each design point
        """
        for design_pt_index in range(self.n_design_pts):
            # If the problem-solver pair has not been run in this way before,
            # run it now.
            experiment = self.design[design_pt_index]
            if (getattr(experiment, "n_macroreps", None) != n_macroreps):
                print("Running Design Point " + str(design_pt_index) + ".")
                experiment.clear_runs()
                experiment.run(n_macroreps)

    # Largely taken from MetaExperiment class in wrapper_base.py.
    def post_replicate(self, n_postreps, n_postreps_init_opt, crn_across_budget=True, crn_across_macroreps=False):
        """
        For each design point, run postreplications at solutions
        recommended by the solver on each macroreplication.

        Arguments
        ---------
        n_postreps : int
            number of postreplications to take at each recommended solution
        n_postreps_init_opt : int
            number of postreplications to take at initial x0 and optimal x*
        crn_across_budget : bool
            use CRN for post-replications at solutions recommended at different times?
        crn_across_macroreps : bool
            use CRN for post-replications at solutions recommended on different macroreplications?
        """
        for design_pt_index in range(self.n_design_pts):
            experiment = self.design[design_pt_index]
            # If the problem-solver pair has not been post-processed in this way before,
            # post-process it now.
            if (getattr(experiment, "n_postreps", None) != n_postreps
                    or getattr(experiment, "n_postreps_init_opt", None) != n_postreps_init_opt
                    or getattr(experiment, "crn_across_budget", None) != crn_across_budget
                    or getattr(experiment, "crn_across_macroreps", None) != crn_across_macroreps):
                print("Post-processing Design Point " + str(design_pt_index) + ".")
                experiment.clear_postreps()
                experiment.post_replicate(n_postreps, n_postreps_init_opt, crn_across_budget, crn_across_macroreps)

    def calculate_statistics(self, solve_tols=[0.05, 0.10, 0.20, 0.50], beta=0.50):
        """
        For each design point, calculate statistics from each macroreplication.
            - area under estimated progress curve
            - alpha-solve time

        Arguments
        ---------
        solve_tols : list of floats in (0,1]
            relative optimality gap(s) definining when a problem is solved
        beta : float in (0,1)
            quantile to compute, e.g., beta quantile
        """
        for design_pt_index in range(self.n_design_pts):
            experiment = self.design[design_pt_index]
            experiment.clear_stats()
            experiment.compute_area_stats(compute_CIs=False)
            experiment.compute_solvability(solve_tols=solve_tols)
            experiment.compute_solvability_quantiles(beta=0.50, compute_CIs=False)
            experiment.record_experiment_results()

    def print_to_csv(self, csv_filename="meta_raw_results"):
        """
        Extract observed statistics from simulated design points.
        Publish to .csv output file.

        Argument
        --------
        csv_filename : string
            name of .csv file to print output to
        """
        with open("./simopt/data_farming_experiments/" + csv_filename + ".csv", mode="w", newline="") as output_file:
            csv_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            # Print headers.
            base_experiment = self.design[0]
            solver_factor_names = list(base_experiment.solver.specifications.keys())
            problem_factor_names = []  # list(base_experiment.problem.specifications.keys())
            oracle_factor_names = list(base_experiment.problem.oracle.specifications.keys())
            csv_writer.writerow(["DesignPt#"]
                                + solver_factor_names
                                + problem_factor_names
                                + oracle_factor_names
                                + ["MacroRep#"]
                                + ["Final Relative Optimality Gap"]
                                + ["Area Under Progress Curve"]
                                + ["0.05-Solve Time", "0.05-Solved? (Y/N)"]
                                + ["0.10-Solve Time", "0.10-Solved? (Y/N)"]
                                + ["0.20-Solve Time", "0.20-Solved? (Y/N)"]
                                + ["0.50-Solve Time", "0.50-Solved? (Y/N)"])
            for designpt_index in range(self.n_design_pts):
                experiment = self.design[designpt_index]
                # Parse lists of factors.
                solver_factor_list = [experiment.solver.factors[solver_factor_name] for solver_factor_name in solver_factor_names]
                problem_factor_list = []
                oracle_factor_list = [experiment.problem.oracle.factors[oracle_factor_name] for oracle_factor_name in oracle_factor_names]
                for mrep in range(experiment.n_macroreps):
                    # Parse list of statistics.
                    statistics_list = [experiment.all_prog_curves[mrep][-1],
                                       experiment.areas[mrep],
                                       experiment.solve_times[0][mrep],
                                       int(experiment.solve_times[0][mrep] < np.infty),
                                       experiment.solve_times[1][mrep],
                                       int(experiment.solve_times[1][mrep] < np.infty),
                                       experiment.solve_times[2][mrep],
                                       int(experiment.solve_times[2][mrep] < np.infty),
                                       experiment.solve_times[3][mrep],
                                       int(experiment.solve_times[3][mrep] < np.infty)
                                       ]
                    print_list = [designpt_index] + solver_factor_list + problem_factor_list + oracle_factor_list + [mrep] + statistics_list
                    csv_writer.writerow(print_list)
