# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: simopt
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Demo for Data Farming Models
#
# This script is intended to help with running a data-farming experiment on a simulation model.
#
# It creates a design of model factors and runs multiple replications at each configuration of the model.
#
# Outputs are printed to a file.

# %% [markdown]
# ## Append SimOpt Path
#
# Since the notebook is stored in simopt/notebooks, we need to append the parent simopt directory to the system path to import the necessary modules later on.

# %%
import sys
from pathlib import Path

# Take the current directory, find the parent, and add it to the system path
sys.path.append(str(Path.cwd().parent))

# %% [markdown]
# ## Configuration Parameters
#
# This section defines the core parameters for the data farming experiment.
#
# To query model/problem/solver names, run `python scripts/list_directories.py`

# %%
model_abbr_name = "CNTNEWS"

# Specify the names of the model factors (in order) that will be varied.
factor_headers = [
    "purchase_price",
    "sales_price",
    "salvage_price",
    "order_quantity",
]

# %% [markdown]
# ## Design Settings
#
# You can either create a new design by providing a factor_settings_filename with factor bounds and discretization, OR use an existing design by providing a design_filename. Only one of these should be active at a time.
#
# ### Factor Settings File
#
# If creating the design, provide the name of a .txt file containing the following:
# - one row corresponding to each model factor being varied
# - 3 columns:
#   1. lower bound for factor value
#   2. upper bound for factor value
#   3. (integer) number of digits for discretizing values (e.g., 0 corresponds to integral values for the factor)
#
# ### Design File
#
# If using an existing design, provide the name of a .txt file containing the design. The file should contain the following:
# - one row corresponding to each design point
# - the number of columns equal to the number of factors being varied
# - each value in the table gives the value of the factor (col index) for the design point (row index)
#

# %%
# factor_settings_filename = "model_factor_settings"
factor_settings_filename = None

# design_filename = None
design_filename = "model_factor_settings_design"

# %% [markdown]
# ## Replication and Output
# These settings control the number of simulation runs per design point and whether common random numbers are used, as well as the name for the output file.

# %%
# Specify a common number of replications to run of the model at each design point
n_reps = 10

# Specify whether to use common random numbers across different versions of the model
crn_across_design_pts = True

# Specify filename for outputs
output_filename = "cntnews_data_farming_output"

# %% [markdown]
# ## Run Data Farming Experiment
# This block initializes the DataFarmingExperiment object using the parameters defined above, executes the specified number of replications at each design point, and saves the results to a CSV file.

# %%
# Import experiment_base module, which contains functions for experimentation.
from simopt.data_farming_base import DataFarmingExperiment

# Create DataFarmingExperiment object.
myexperiment = DataFarmingExperiment(
    model_name=model_abbr_name,
    factor_settings_file_name=factor_settings_filename,
    factor_headers=factor_headers,
    design_path=design_filename,
    model_fixed_factors={},
)

# Run replications and print results to file.
myexperiment.run(n_reps=n_reps, crn_across_design_pts=crn_across_design_pts)
myexperiment.print_to_csv(csv_file_name=output_filename)
