import numpy as np
import os
import csv
from copy import deepcopy
import itertools
import pandas as pd
from mrg32k3a.mrg32k3a import MRG32k3a


from .directory import model_directory, solver_directory
from .experiment_base import ProblemSolver, post_normalize


class DesignPoint(object):
    """Base class for design points represented as dictionaries of factors.

    Attributes
    ----------
    model : ``base.Model``
        Model to simulate.
    model_factors : dict
        Model factor names and values.
    rng_list : list [``mrg32k3a.mrg32k3a.MRG32k3a``]
        Rngs for model to use when running replications at the solution.
    n_reps : int
        Number of replications run at a design point.
    responses : dict
        Responses observed from replications.
    gradients : dict [dict]
        Gradients of responses (w.r.t. model factors) observed from replications.

    Parameters
    ----------
    model : ``base.Model``
        Model with factors model_factors.
    """
    def __init__(self, model):
        super().__init__()
        # Create separate copy of Model object for use at this design point.
        self.model = deepcopy(model)
        self.model_factors = self.model.factors
        self.n_reps = 0
        self.responses = {}
        self.gradients = {}

    def attach_rngs(self, rng_list, copy=True):
        """Attach a list of random-number generators to the design point.

        Parameters
        ----------
        rng_list : list [``mrg32k3a.mrg32k3a.MRG32k3a``]
            List of random-number generators used to run simulation replications.
        """
        if copy:
            self.rng_list = [deepcopy(rng) for rng in rng_list]
        else:
            self.rng_list = rng_list

    def simulate(self, m=1):
        """Simulate m replications for the current model factors and 
        append results to the responses and gradients dictionaries.

        Parameters
        ----------
        m : int, default=1
            Number of macroreplications to run at the design point; > 0.
        """
        for _ in range(m):
            # Generate a single replication of model, as described by design point.
            responses, gradients = self.model.replicate(rng_list=self.rng_list)
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
    """Base class for data-farming experiments consisting of an model
    and design of associated factors.

    Attributes
    ----------
    model : ``base.Model``
        Model on which the experiment is run.
    design : list [``data_farming_base.DesignPoint``]
        List of design points forming the design.
    n_design_pts : int
        Number of design points in the design.

    Parameters
    ----------
    model_name : str
        Name of model on which the experiment is run.
    factor_settings_filename : str
        Name of .txt file containing factor ranges and # of digits.
    factor_headers : list [str]
        Ordered list of factor names appearing in factor settings/design file.
    design_filename : str
        Name of .txt file containing design matrix.
    model_fixed_factors : dict
        Non-default values of model factors that will not be varied.
    """
    def __init__(self, model_name, factor_settings_filename, factor_headers, design_filename=None, model_fixed_factors={}):
        
        # Initialize model object with fixed factors.
        self.model = model_directory[model_name](fixed_factors=model_fixed_factors)
        if design_filename is None:
            # Create model factor design from .txt file of factor settings.
            # Hard-coded for a single-stack NOLHS.
            #command = "stack_nolhs.rb -s 1 model_factor_settings.txt > outputs.txt"
            command = f"stack_nolhs.rb -s 1 ./data_farming_experiments/{factor_settings_filename}.txt > ./data_farming_experiments/{factor_settings_filename}_design.txt"
            os.system(command)
            # Append design to base filename.
            design_filename = f"{factor_settings_filename}_design"
        # Read in design matrix from .txt file. Result is a pandas DataFrame.
        design_table = pd.read_csv(f"./data_farming_experiments/{design_filename}.txt", header=None, delimiter="\t", encoding="utf-8")
        # Count number of design_points.
        self.n_design_pts = len(design_table)
        # Create all design points.
        self.design = []
        design_pt_factors = {}
        for dp_index in range(self.n_design_pts):
            for factor_idx in range(len(factor_headers)):
                # Parse model factors for next design point.
                design_pt_factors[factor_headers[factor_idx]] = design_table[factor_idx][dp_index]
            # Update model factors according to next design point.
            self.model.factors.update(design_pt_factors)
            # Create new design point and add to design.
            self.design.append(DesignPoint(self.model))

    def run(self, n_reps=10, crn_across_design_pts=True):
        """Run a fixed number of macroreplications at each design point.

        Parameters
        ----------
        n_reps : int, default=10
            Number of replications run at each design point.
        crn_across_design_pts : bool, default=True
            True if CRN are to be used across design points, otherwise False.
        """
        # Setup random number generators for model.
        # Use stream 0 for all runs; start with substreams 0, 1, ..., model.n_rngs-1.
        main_rng_list = [MRG32k3a(s_ss_sss_index=[0, ss, 0]) for ss in range(self.model.n_rngs)]
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
        """Extract observed responses from simulated design points and
        publish to .csv output file.

        Parameters
        ----------
        csv_filename : str, default="raw_results"
            Name of .csv file to print output to.
        """
        # Create directory if they do no exist.
        if not os.path.exists("./data_farming_experiments"):
            os.makedirs("./data_farming_experiments")
        with open( csv_filename + ".csv", mode="w", newline="") as output_file:
            csv_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            # Print headers.
            model_factor_names = list(self.model.specifications.keys())
            response_names = list(self.design[0].responses.keys())
            csv_writer.writerow(["DesignPt#"] + model_factor_names + ["MacroRep#"] + response_names)
            for designpt_index in range(self.n_design_pts):
                designpt = self.design[designpt_index]
                # Parse list of model factors.
                model_factor_list = [designpt.model_factors[model_factor_name] for model_factor_name in model_factor_names]
                for mrep in range(designpt.n_reps):
                    # Parse list of responses.
                    response_list = [designpt.responses[response_name][mrep] for response_name in response_names]
                    print_list = [designpt_index] + model_factor_list + [mrep] + response_list
                    csv_writer.writerow(print_list)


class DataFarmingMetaExperiment(object):
    """Base class for data-farming meta experiments consisting of problem-solver
    pairs and a design of associated factors.

    Attributes
    ----------
    design : list [``experiment_base.ProblemSolver``]
        List of design points forming the design.
    n_design_pts : int
        Number of design points in the design.

    Parameters
    ----------
    solver_name : str
        Name of solver.
    problem_name : str
        Name of problem.
    solver_factor_headers : list [str]
        Ordered list of solver factor names appearing in factor settings/design file.
    solver_factor_settings_filename : str, default=None
        Name of .txt file containing solver factor ranges and # of digits.
    design_filename : str, default=None
        Name of .txt file containing design matrix.
    solver_fixed_factors : dict, default=None
        Dictionary of user-specified solver factors that will not be varied.
    problem_fixed_factors : dict, default=None
        Dictionary of user-specified problem factors that will not be varied.
    model_fixed_factors : dict, default=None
        Dictionary of user-specified model factors that will not be varied.
    """
    def __init__(self, solver_name = None, solver_factor_headers = None, n_stacks = 1, design_type = 'nolhs', solver_factor_settings_filename=None, design_filename=None, csv_filename = None, solver_fixed_factors=None, cross_design_factors = None):
        
        if solver_fixed_factors is None:
            solver_fixed_factors={}
        # if problem_fixed_factors is None:
        #     problem_fixed_factors={}
        # if model_fixed_factors is None:
        #     model_fixed_factors={}
        if cross_design_factors is None:
            cross_design_factors = {}
        if solver_name is not None:
            self.solver_object = solver_directory[solver_name]() #creates solver object
        # TO DO: Extend to allow a design on problem/model factors too.
        # Currently supports designs on solver factors only.
        if design_filename is None and csv_filename is None:
            # Create solver factor design from .txt file of factor settings.
            # Hard-coded for a single-stack NOLHS.
            command = f"stack_{design_type}.rb -s {n_stacks} ./data_farming_experiments/" + solver_factor_settings_filename + ".txt > ./data_farming_experiments/" + solver_factor_settings_filename + "_design.txt"
            os.system(command)
            # Append design to base filename.
            design_filename =  f"{solver_factor_settings_filename}_design"
        # # Read in design matrix from .txt file. Result is a pandas DataFrame.
        # design_table = pd.read_csv(f"./data_farming_experiments/{design_filename}.txt", header=None, delimiter="\t", encoding="utf-8")
        
        
        if csv_filename is None:
            
             # Read in design matrix from .txt file. Result is a pandas DataFrame.
             design_table = pd.read_csv(f"./data_farming_experiments/{design_filename}.txt", header=None, delimiter="\t", encoding="utf-8")
        
             # Create design csv file from design table 
   
             csv_filename = f"./data_farming_experiments/{design_filename}.csv"
             
             #self.solver_object = solver_directory[solver_name]() #creates solver object
           
             design_table.columns = solver_factor_headers #add factor headers names to dt
             
             solver_fixed_str = {}
             for factor in solver_fixed_factors: # make new dict containing strings of solver factors
                 solver_fixed_str [factor] = str(solver_fixed_factors[factor])
             
             for factor in self.solver_object.specifications: #add default values to str dict for unspecified factors
                 default = self.solver_object.specifications[factor].get("default")
                 if factor not in solver_fixed_str and factor not in solver_factor_headers:
                     print('default from df base', default)
                     solver_fixed_str[factor] = str(default)
                     
             all_solver_factor_names = solver_factor_headers + list(solver_fixed_str.keys()) # list of all solver factor names 
             
             #all_solver_factor_names = solver_factor_headers + list(cross_design_factors.keys()) + list(solver_fixed_factors.keys()) #creates list of all solver factors in order of design, cross-design, then fixed factors
             
             
             # Add fixed factors to dt
             for factor in solver_fixed_str:
                 design_table[factor] = solver_fixed_str[factor]
                 
            
             # Add cross design factors to design table
             if len(cross_design_factors) != 0:
                  #num_cross = 0 # number of times cross design is run
                 
                  # create combination of categorical factor options
                  cross_factor_names = list(cross_design_factors.keys())
                  combinations = itertools.product(*(cross_design_factors[opt] for opt in cross_factor_names))
                  
                  new_design_table = pd.DataFrame() #temp empty value
                  for combination in combinations:
                      combination_dict = dict(zip(cross_factor_names, combination)) # dictionary containing current combination of cross design factor values
                      working_design_table = design_table.copy()
                 
                      for factor in combination_dict:
                          str_factor_val = str(combination_dict[factor])
                          working_design_table[factor] = str_factor_val
                          
                      new_design_table = pd.concat([new_design_table, working_design_table], ignore_index=True)
                      print(new_design_table)
                      
                  design_table = new_design_table
                  
             # Add design information to table
             design_table.insert(0,'Design #', range(len(design_table)))
             design_table['Solver Name'] = solver_name
             design_table['Design Type'] = design_type 
             design_table['Number Stacks'] = str(n_stacks) 
                          
               
                 
             design_table.to_csv(csv_filename, mode = 'w', header = True, index = False)
       
       
        self.csv_filename = csv_filename     
       
        

                        
                        
                   

    # Largely taken from MetaExperiment class in wrapper_base.py.
    def run(self, problem_name, problem_fixed_factors = None, model_fixed_factors = None, n_macroreps=10):
        """Run n_macroreps of each problem-solver design point.

        Parameters
        ----------
        n_macroreps : int
            Number of macroreplications for each design point.
        """
        if problem_fixed_factors is None:
            problem_fixed_factors={}
        if model_fixed_factors is None:
            model_fixed_factors={}
        solver_factors = {}#holds solver factors for individual dp
        solver_factors_across_design = []#holds solver factors across all dps
        self.design = []
        num_extra_col = 3 #change if extra columns in design table changes
            
        # Read design points from csv file
        solver_factors_str = {} # holds design points as string for all factors
        with open(self.csv_filename, 'r') as file:
            reader = csv.reader(file)
            first_row = next(reader)
            #next(reader)
            #next(reader)
            #solver_factor_headers = next(reader)[1:]
            all_solver_factor_names = first_row[1:-1*num_extra_col]
            #second_row = next(reader)
            #solver_name = second_row[-1*num_extra_col]
            row_index = 0
            
            for row in reader:
                solver_name = row[-1*num_extra_col]
                dp = row[1:-1*num_extra_col]
                dp_index = 0
                self.solver_object = solver_directory[solver_name]() # make this less bulky later
                for factor in all_solver_factor_names:
                    solver_factors_str[factor] = dp[dp_index]
                    dp_index += 1
                # Convert str to proper data type
                for factor in solver_factors_str:
                    datatype = self.solver_object.specifications[factor].get('datatype')
                    val = solver_factors_str[factor]
                    
                    if datatype == int:
                        solver_factors[factor] = int(val)
                    elif datatype == float:
                        solver_factors[factor] = float(val)
                    elif datatype == bool:
                        if val == 'True':
                            solver_factors[factor] = True
                        else:
                            solver_factors[factor] = False
                
                solver_factors_insert = solver_factors.copy()
                solver_factors_across_design.append(solver_factors_insert)
                
                row_index += 1
            
                
                
            self.n_design_pts = len(solver_factors_across_design)
            for i in range(self.n_design_pts):
                # Create design point on problem solver
                
                file_name_path = "./data_farming_experiments/outputs/" + solver_name + "_on_" + problem_name + "_designpt_" + str(i) + ".pickle"
                current_solver_factors = solver_factors_across_design[i]
                new_design_pt = ProblemSolver(solver_name=solver_name,
                                            problem_name=problem_name,
                                            solver_fixed_factors=current_solver_factors,
                                            problem_fixed_factors=problem_fixed_factors,
                                            model_fixed_factors=model_fixed_factors,
                                            file_name_path=file_name_path)
                
                self.design.append(new_design_pt)
                
                
            #self.n_design_pts = len(self.design)
            
                            
                            
                            
        for design_pt_index in range(self.n_design_pts):
            # If the problem-solver pair has not been run in this way before,
            # run it now.
            experiment = self.design[design_pt_index]
             
            
            if (getattr(experiment, "n_macroreps", None) != n_macroreps):
                print("Running Design Point " + str(design_pt_index) + ".")
                experiment.clear_run()
                print(experiment.solver.name)
                print(experiment.problem.name)
                experiment.run(n_macroreps)

    # Largely taken from MetaExperiment class in wrapper_base.py.
    def post_replicate(self, n_postreps, crn_across_budget=True, crn_across_macroreps=False):
        """For each design point, run postreplications at solutions
        recommended by the solver on each macroreplication.

        Parameters
        ----------
        n_postreps : int
            Number of postreplications to take at each recommended solution.
        crn_across_budget : bool, default=True
            True if CRN are to be used for post-replications at solutions recommended at
            different times, otherwise False.
        crn_across_macroreps : bool, default=False
            True if CRN are to be used for post-replications at solutions recommended on
            different macroreplications, otherwise False.
        """
        for design_pt_index in range(self.n_design_pts):
            experiment = self.design[design_pt_index]
            # If the problem-solver pair has not been post-processed in this way before,
            # post-process it now.
            if (getattr(experiment, "n_postreps", None) != n_postreps
                    or getattr(experiment, "crn_across_budget", None) != crn_across_budget
                    or getattr(experiment, "crn_across_macroreps", None) != crn_across_macroreps):
                print("Post-processing Design Point " + str(design_pt_index) + ".")
                experiment.clear_postreplicate()
                experiment.post_replicate(n_postreps, crn_across_budget=crn_across_budget, crn_across_macroreps=crn_across_macroreps)

    # Largely taken from MetaExperiment class in wrapper_base.py.
    def post_normalize(self, n_postreps_init_opt, crn_across_init_opt=True):
        """Post-normalize problem-solver pairs.
        
        Parameters
        ----------
        n_postreps_init_opt : int
            Number of postreplications to take at initial x0 and optimal x*.
        crn_across_init_opt : bool, default=True
            True if CRN are to be used for post-replications at solutions x0 and x*, otherwise False.
        """
        post_normalize(experiments=self.design,
                       n_postreps_init_opt=n_postreps_init_opt,
                       crn_across_init_opt=crn_across_init_opt
                       )

    def report_statistics(self, solve_tols=[0.05, 0.10, 0.20, 0.50], csv_filename="df_solver_results"):
        """For each design point, calculate statistics from each macoreplication and print to csv.

        Parameters
        ----------
        solve_tols : list [float], default = [0.05, 0.10, 0.20, 0.50]
            Relative optimality gap(s) definining when a problem is solved; in (0,1].
        csv_filename : str, default="df_solver_results"
            Name of .csv file to print output to.
        """
        # Create directory if they do no exist.
        if not os.path.exists("./data_farming_experiments"):
            os.makedirs("./data_farming_experiments")
        if csv_filename == 'df_solver_results':
            file_path = "./data_farming_experiments/"
        else:
            file_path = ""
        with open(file_path + csv_filename + ".csv", mode="w", newline="") as output_file:
            csv_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            base_experiment = self.design[0]
            solver_factor_names = list(base_experiment.solver.specifications.keys())
            problem_factor_names = list(base_experiment.problem.specifications.keys())
            model_factor_names = list(set(base_experiment.problem.model.specifications.keys()) - base_experiment.problem.model_decision_factors)
            # Concatenate solve time headers.
            solve_time_headers = [[f"{solve_tol}-Solve Time"] + [f"{solve_tol}-Solved? (Y/N)"] for solve_tol in solve_tols]
            solve_time_headers = list(itertools.chain.from_iterable(solve_time_headers))
            # Print headers.
            csv_writer.writerow(["DesignPt#"]
                                + solver_factor_names
                                + problem_factor_names
                                + model_factor_names
                                + ["MacroRep#"]
                                + ["Final Relative Optimality Gap"]
                                + ["Area Under Progress Curve"]
                                + solve_time_headers)
            # Compute performance metrics.
            for designpt_index in range(self.n_design_pts):
                experiment = self.design[designpt_index]
                # Parse lists of factors.
                solver_factor_list = [experiment.solver.factors[solver_factor_name] for solver_factor_name in solver_factor_names]
                problem_factor_list = [experiment.problem.factors[problem_factor_name] for problem_factor_name in problem_factor_names]
                model_factor_list = [experiment.problem.model.factors[model_factor_name] for model_factor_name in model_factor_names]
                for mrep in range(experiment.n_macroreps):
                    progress_curve = experiment.progress_curves[mrep]
                    # Parse list of statistics.
                    solve_time_values = [[progress_curve.compute_crossing_time(threshold=solve_tol)] + [int(progress_curve.compute_crossing_time(threshold=solve_tol) < np.infty)] for solve_tol in solve_tols]
                    solve_time_values = list(itertools.chain.from_iterable(solve_time_values))
                    statistics_list = [progress_curve.y_vals[-1],
                                       progress_curve.compute_area_under_curve()] + solve_time_values
                    print_list = [designpt_index] + solver_factor_list + problem_factor_list + model_factor_list + [mrep] + statistics_list
                    csv_writer.writerow(print_list)
