import sys
import os.path as o
import os
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))
# os.chdir('../')

from wrapper_base import read_experiment_results
from data_farming_base import DesignPoint, DataFarmingExperiment, DataFarmingMetaExperiment
from csv import DictReader


solver_factor_headers = ["sample_size"]
myMetaExperiment = DataFarmingMetaExperiment(solver_name="RNDSRCH",
                                             problem_name="CNTNEWS-1",
                                             solver_factor_headers=solver_factor_headers,
                                             solver_factor_settings_filename="",  # "solver_factor_settings",
                                             design_filename="random_search_design",
                                             solver_fixed_factors={},
                                             problem_fixed_factors={},
                                             model_fixed_factors={})
myMetaExperiment.run(n_macroreps=20, crn_across_solns=True)
myMetaExperiment.post_replicate(n_postreps=100, n_postreps_init_opt=100, crn_across_budget=True, crn_across_macroreps=False)

file_name_path = "data_farming_experiments/outputs/" + "RNDSRCH_on_CNTNEWS-1_designpt_0" + ".pickle"
myexperiment = read_experiment_results(file_name_path=file_name_path)
myexperiment.plot_progress_curves(plot_type="all")



# myMetaExperiment.calculate_statistics() # solve_tols=[0.10], beta=0.50)
# myMetaExperiment.print_to_csv(csv_filename="meta_raw_results")

print("I ran this.")


# SCRATCH
# --------------------------------
# from csv import DictReader
# # open file in read mode
# with open('example_design_matrix.csv', 'r') as read_obj:
#     # pass the file object to DictReader() to get the DictReader object
#     csv_dict_reader = DictReader(read_obj)
#     # iterate over each line as a ordered dictionary
#     for row in csv_dict_reader:
#         # row variable is a dictionary that represents a row in csv
#         print(row)