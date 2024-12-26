"""
This script is intended to help with running a data-farming experiment on
a problem. It creates a design of problem factors and runs multiple
macroreplications at each version of the problem. Outputs are printed to a file.
"""

import sys
import os.path as o

sys.path.append(
    o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), ".."))
)

from simopt.experiment_base import create_design, ProblemsSolvers  # type:ignore


def main() -> None:
    # Specify the name of the problem as it appears in directory.py
    problem_name = "CNTNEWS-1"
    # Specify the name of the model as it appears in directory.py
    model_name = "CNTNEWS"
    # Specify the name of the solver as it appears in directory.py
    solver_names = ["ASTRODF", "RNDSRCH"]

    # Specify the names of the model factors (in order) that will be varied.
    model_factor_headers = ["purchase_price", "sales_price", "order_quantity"]

    # OPTIONAL: factors chosen for cross design
    # factor name followed by list containing factor values to cross design over
    # model_cross_design_factors = {}

    # OPTIONAL: Provide additional overrides for model default factors.
    # If empty, default factor settings are used.
    model_fixed_factors = {"salvage_price": 5, "Burr_c": 1}

    # OPTIONAL: Provide additional overrides for solver default factors.
    # If empty, default factor settings are used.
    # list of dictionaries that provide fixed factors for problems when you don't want to use the default values
    # if you want to use all default values use empty dictionary, order must match problem names
    solver_fixed_factors = [{"eta_1": 0.5, "eta_2": 0.4}, {"sample_size": 15}]

    # uncomment this version to run w/ only default solver factors
    # sp;ver_fixed_factors = [{},{}]

    # Provide the name of a file  .txt locatated in the datafarming_experiments folder containing
    # the following:
    #    - one row corresponding to each solver factor being varied
    #    - three columns:
    #         - first column: lower bound for factor value
    #         - second column: upper bound for factor value
    #         - third column: (integer) number of digits for discretizing values
    #                         (e.g., 0 corresponds to integral values for the factor)
    model_factor_settings_filename = "testing_model_cntnews_1"

    # Specify the number stacks to use for ruby design creation
    problem_n_stacks = 1

    # Specify a common number of macroreplications of each unique solver/problem combination
    # i.e., the number of runs at each design point.
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

    # Create DataFarmingExperiment object for model design
    model_design_list = create_design(
        name=model_name,
        factor_headers=model_factor_headers,
        factor_settings_filename=model_factor_settings_filename,
        n_stacks=problem_n_stacks,
        fixed_factors=model_fixed_factors,  # optional
        # cross_design_factors=model_cross_design_factors, #optional
    )

    # create proble name list for ProblemsSolvers (do not edit)
    problem_names = []
    for _ in range(len(model_design_list)):
        problem_names.append(problem_name)

    # Create ProblemsSovlers experiment with solver and model design
    experiment = ProblemsSolvers(
        solver_factors=solver_fixed_factors,
        problem_factors=model_design_list,
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
