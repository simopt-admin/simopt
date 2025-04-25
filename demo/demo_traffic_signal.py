"""Demo script for the Traffic Light problem.

This script is intended to help with running a data-farming experiment on
a problem. It creates a design of problem factors and runs multiple
macroreplications at each version of the problem. Outputs are printed to a file.
"""

import pickle
import sys
from pathlib import Path

# Append the parent directory (simopt package) to the system path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from simopt.experiment_base import ProblemsSolvers, create_design


def main() -> None:
    """Main function to run the Traffic Light problem experiment."""
    ####################################################################################
    # USER OPTIONS
    ####################################################################################

    run_design = False  # False = run w/ fixed factors, True = run with design
    run_model = True  # change to True to run model
    post_replicate = True  # change to True to post replicate
    post_normalize = True  # change to True to post process
    log = True  # change to True to log results
    load_pickle = False  # change to True to load pickle of previously run traffic light

    # specify name of loaded pickle file (can leave alone if not loading a file)
    pickle_file = (
        "../experiments/outputs/10_min_traffic_group_ASTRODF_on_TRAFFICCONTROL-1.pickle"
    )

    # Specify the name of the problem
    problem_name = "TRAFFICCONTROL-1"
    # Specify the name of the solver(s)
    # NOTE: can change this to any solver name, make sure to keep formatted as list
    solver_names = ["ASTRODF"]
    # OPTIONAL: change experiment name to prevent override of previous runs
    experiment_name = "Traffic_Light_10_min"

    # OPTIONAL: Provide additional overrides for problem default factors.
    # If creating a design, any factors used in design will be ignored here.
    # Otherwise all model values will be overwritten
    problem_fixed_factors = {
        "lambdas": [2, 2, 0, 1, 2, 2, 0, 1],
        "runtime": 60,
        "numintersections": 4,
        "decision_vector": [1, 2, 3],
        "speed": 5,
        "carlength": 4.5,
        "reaction": 0.1,
        "transition_probs": [0.7, 0.3, 0.3, 0.2, 0.25, 0.1, 0.15],
        "pause": 0.1,
        "car_distance": 0.5,
        "length_arteries": 100,
        "length_veins": 100,
        "redlight_arteries": [10, 10, 10, 10],  # length must equal numintersectsions
        "redlight_veins": [20, 20, 20, 20],  # length must equal numintersectsions,
        "initial_solution": (1, 1, 1),  # problem factor
        "budget": 100,  # problem factor
    }

    # solver factors, if solver is changed update factor dictionary to match new solver
    solver_fixed_factors = [
        {
            "crn_across_solns": True,
            "eta_1": 0.1,
            "eta_2": 0.8,
            "gamma_1": 1.5,
            "gamma_2": 0.5,
            "lambda_min": 4,
            "easy_solve": True,
            "reuse_points": True,
            "ps_sufficient_reduction": 0.1,
        }
    ]

    # Experiment options

    # Specify a common number of macroreplications of each unique solver/problem
    # combination (ie the number of runs at each design point.)
    n_macroreps = 5
    # Specify the number of postreplications to take at each recommended solution
    # from each macroreplication at each design point.
    n_postreps = 100
    # Specify the number of postreplications to take at x0 and x*.
    n_postreps_init_opt = 200

    if run_design:  # this part will only run if create_design is True
        # Specify the names and min/max/decimal values of the problem factors that will
        # be varied. To vary more factors, add another nested dictonary
        # to design_factor_settings with the min, max, and decimal value that you want
        # to vary. Factor names and data types must exactly match those used in model
        # for any integer factors, dec must be 0
        design_factor_settings = {"runtime": {"min": 60, "max": 7200, "dec": 0}}
        # Specify the number stacks to use for ruby design creation. More stacks mean
        # more design points.
        n_stacks = 1

        # !! Don't change anthing below this point !!

        # create the factor setting file
        factor_settings = Path(
            f"./data_farming_experiments/{experiment_name}_factor_settings.txt"
        )
        with factor_settings.open("w") as file:
            file.write("")
            for factor in design_factor_settings:
                min_val = design_factor_settings[factor]["min"]
                max_val = design_factor_settings[factor]["max"]
                dec_val = design_factor_settings[factor]["dec"]
                data = f"{min_val} {max_val} {dec_val}\n"
                file.write(data)
        design_factors = list(design_factor_settings.keys())

        # remove design factors from fixed factors if present
        for factor in design_factors:
            problem_fixed_factors.pop(factor, None)

        # Create DataFarmingExperiment object for model design
        design_list = create_design(
            name=problem_name,
            factor_headers=design_factors,
            factor_settings_filename=f"{experiment_name}_factor_settings",
            n_stacks=n_stacks,
            fixed_factors=problem_fixed_factors,
            class_type="problem",
        )
        # create problem name list for ProblemsSolvers (do not edit)
        exp_problem_names = []
        for _ in range(len(design_list)):
            exp_problem_names.append(problem_name)
        exp_problem_factors = design_list

    else:  # not creating a design
        exp_problem_names = [problem_name]
        exp_problem_factors = [problem_fixed_factors]

    if load_pickle:  # load previously run experiment
        pickle_file = Path(pickle_file)
        with pickle_file.open("rb") as file:
            experiment = pickle.load(file)

    else:  # create a new experiment
        print(exp_problem_names)

        # Create ProblemsSovlers experiment with solver and model design
        experiment = ProblemsSolvers(
            solver_factors=solver_fixed_factors,
            problem_factors=exp_problem_factors,
            solver_names=solver_names,
            problem_names=exp_problem_names,
            experiment_name=experiment_name,
        )
        # check compatibility of selected solvers and problems
        experiment.check_compatibility()

    if run_model:
        # Run macroreplications at each design point.
        experiment.run(n_macroreps)

    if post_replicate:
        # Postprocess the experimental results from each design point.
        experiment.post_replicate(n_postreps=n_postreps)

    if post_normalize:
        experiment.post_normalize(n_postreps_init_opt=n_postreps_init_opt)

    if log:
        # Record and log results
        experiment.record_group_experiment_results()
        experiment.log_group_experiment_results()
        experiment.report_group_statistics()


if __name__ == "__main__":
    main()
