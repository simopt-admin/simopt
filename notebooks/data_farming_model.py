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
# ## Configure Design
#
# To configure a design, the name of each factor being varied must be provided via the `factor_headers` parameter, along with **either** the `factor_settings` **or** `design_filename` parameter. Please note that the ordering of factors in `factor_headers` must match the ordering in either `factor_settings` or `design_filename`.
#
# ### `factor_settings`
#
# Use this parameter to define the factors for a new design of experiments. Each factor requires a lower bound, an upper bound, and a number of digits for discretization.
#
# You can provide these settings in one of two ways:
#
# #### 1. List of Tuples (Recommended)
# Pass a list where each element is a 3-tuple representing a single factor with the following structure:
# - lower_bound: The minimum value for the factor.
# - upper_bound: The maximum value for the factor.
# - num_digits: An integer for the number of decimal places (e.g., 0 for integers).
#
# **Example:**
# ```python
# # Defines two factors:
# # - An integer from 0 to 10.
# # - A float from 0.00 to 1.00 (in steps of 0.01).
# factor_settings = [(0, 10, 0), (0.0, 1.0, 2)]
# ```
#
# #### 2. File Path
# Pass a string containing the path to a settings file. The file must contain three columns, with each row corresponding to a different factor.
#
# The columns should be structured as follows:
# - Column 1: Lower bound for the factor's value.
# - Column 2: Upper bound for the factor's value.
# - Column 3: An integer for the number of digits for discretization.
#
# ### `design_filename`
#
# Use this parameter to specify the path to a previously generated design file.
#
# The file must be structured as a simple table with the following format:
# - Rows: Each row corresponds to a single design point (an experimental run).
# - Columns: The number of columns must equal the number of factors being varied.
# - Values: Each cell contains the specific value for a given factor (column) at a specific design point (row).

# %%
# Name of the model being run
model_abbr_name = "CNTNEWS"

# A list of names of the factors (in order) that will be varied.
factor_headers = [
    "purchase_price",
    "sales_price",
    "salvage_price",
    "order_quantity",
]

factor_settings = [(4.0, 6.0, 1), (8.0, 12.0, 1), (0.0, 2.0, 1), (0.4, 0.6, 2)]
# OR
design_filename = None

# %% [markdown]
# ## Create Experiment using Specified Configuration

# %%
from simopt.data_farming_base import DataFarmingExperiment

myexperiment = DataFarmingExperiment(
    model_name=model_abbr_name,
    factor_settings=factor_settings,
    factor_headers=factor_headers,
    design_path=design_filename,
)

# %% [markdown]
# ## Run Data Farming Experiment
# Executes the specified number of replications at each design point and saves the results to a CSV file.

# %%
# Number of replications to run of the model at each design point
n_reps = 10
# Specify whether to use common random numbers across different versions of the model
crn_across_design_pts = True
output_filename = "cntnews_data_farming_output"

myexperiment.run(n_reps=n_reps, crn_across_design_pts=crn_across_design_pts)
myexperiment.print_to_csv(csv_file_name=output_filename, overwrite=True)
