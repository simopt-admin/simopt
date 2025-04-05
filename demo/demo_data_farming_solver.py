"""Demo for Data Farming over Solvers.

This script is intended to help with running a data-farming experiment on
a solver. It creates a design of solver factors and runs multiple
macroreplications at each version of the solver. Outputs are printed to a file.
"""

import sys
from pathlib import Path

# Append the parent directory (simopt package) to the system path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from simopt.experiment_base import ProblemsSolvers, create_design


def main() -> None:
    """Main function to run the data farming experiment."""
    # Specify the name of the solver as it appears in directory.py
    solver_name = "ASTRODF"
    # list of problem names for solver design to be run on
    # (if more than one version of same problem, repeat name)
    # Specify the name of the problem as it appears in directory.py
    problem_names = ["SSCONT-1", "SAN-1"]

    # Specify the names of the sovler factors (in order) that will be varied.
    solver_factor_headers = ["eta_1", "eta_2", "lambda_min"]

    # OPTIONAL: factors chosen for cross design
    # factor name followed by list containing factor values to cross design over
    solver_cross_design_factors = {"crn_across_solns": [True, False]}

    # OPTIONAL: Provide additional overrides for solver default factors.
    # If empty, default factor settings are used.
    solver_fixed_factors = {}
    # OPTIONAL: Provide additional overrides for problem default factors.
    # If empty, default factor settings are used.
    # list of dictionaries that provide fixed factors for problems when you don't want
    # to use the default values. if you want to use all default values use empty
    # dictionary, order must match problem names
    problem_fixed_factors = [
        {"budget": 2000, "demand_mean": 90.0, "fixed_cost": 25},
        {"budget": 500},
    ]

    # Provide the name of a file  .txt locatated in the datafarming_experiments
    # folder containing
    # the following:
    #    - one row corresponding to each solver factor being varied
    #    - three columns:
    #         - first column: lower bound for factor value
    #         - second column: upper bound for factor value
    #         - third column: (integer) number of digits for discretizing values
    #                         (e.g., 0 corresponds to integral values for the factor)
    solver_factor_settings_filename = "astrodf_testing"

    # Specify the number stacks to use for ruby design creation
    solver_n_stacks = 1

    # Specify a common number of macroreplications of each unique solver/problem
    # combination (i.e., the number of runs at each design point.)
    n_macroreps = 3

    # Specify the number of postreplications to take at each recommended solution
    # from each macroreplication at each design point.
    n_postreps = 100

    # Specify the number of postreplications to take at x0 and x*.
    n_postreps_init_opt = 200

    # Specify the CRN control for postreplications.
    crn_across_budget = True  # Default
    crn_across_macroreps = False  # Default
    crn_across_init_opt = True  # Default

    # Create DataFarmingExperiment object for sovler design
    solver_design_list = create_design(
        name=solver_name,
        factor_headers=solver_factor_headers,
        factor_settings_filename=solver_factor_settings_filename,
        n_stacks=solver_n_stacks,
        fixed_factors=solver_fixed_factors,  # optional
        cross_design_factors=solver_cross_design_factors,  # optional
    )

    # create solver name list for ProblemsSolvers (do not edit)
    solver_names = []
    for _ in range(len(solver_design_list)):
        solver_names.append(solver_name)

    # Create ProblemsSovlers experiment with solver and model design
    experiment = ProblemsSolvers(
        solver_factors=solver_design_list,
        problem_factors=problem_fixed_factors,
        solver_names=solver_names,
        problem_names=problem_names,
    )

    # check compatibility of selected solvers and problems
    experiment.check_compatibility()

    # Run macroreplications at each design point.
    experiment.run(n_macroreps)

    # Postprocess the experimental results from each design point.
    experiment.post_replicate(
        n_postreps=n_postreps,
        crn_across_budget=crn_across_budget,
        crn_across_macroreps=crn_across_macroreps,
    )
    experiment.post_normalize(
        n_postreps_init_opt=n_postreps_init_opt,
        crn_across_init_opt=crn_across_init_opt,
    )

    # Record and log results
    experiment.record_group_experiment_results()
    experiment.log_group_experiment_results()
    experiment.report_group_statistics()


if __name__ == "__main__":
    main()
