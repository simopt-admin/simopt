"""Demo Plotting Script.

This script is intended to help with debugging problems and solvers.
It create a problem-solver pairing (using the directory) and runs multiple
macroreplications of the solver on the problem.
"""

import sys
from pathlib import Path

# Append the parent directory (simopt package) to the system path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import the ProblemSolver class and other useful functions
from simopt.experiment_base import (
    PlotType,
    ProblemSolver,
    plot_terminal_progress,
    post_normalize,
    read_experiment_results,
)


def main() -> None:
    """Main function to run the data farming experiment."""
    solver_names = {"RNDSRCH", "ASTRODF", "NELDMD"}
    problem_names = {"SAN-1"}  # CNTNEWS-1"} #, "SAN-1"}
    # solver_name = "RNDSRCH"  # Random search solver
    # problem_name = "CNTNEWS-1"  # Continuous newsvendor problem
    # solver_name = <solver_name>
    # problem_name = <problem_name>

    for problem_name in problem_names:
        problem_experiments = []
        for solver_name in solver_names:
            print(f"Testing solver {solver_name} on problem {problem_name}.")
            # Initialize an instance of the experiment class.
            myexperiment = ProblemSolver(solver_name, problem_name)

            file_name_path = (
                "experiments/outputs/" + solver_name + "_on_" + problem_name + ".pickle"
            )

            # Run a fixed number of macroreplications of the solver on the problem.
            # myexperiment.run(n_macroreps=10)

            # If the solver runs have already been performed, uncomment the
            # following pair of lines (and uncommmen the myexperiment.run(...)
            # line above) to read in results from a .pickle file.
            myexperiment = read_experiment_results(file_name_path)

            print("Post-processing results.")
            # Run a fixed number of postreplications at all recommended solutions.
            myexperiment.post_replicate(n_postreps=200)

            problem_experiments.append(myexperiment)

        # Find an optimal solution x* for normalization.
        post_normalize(problem_experiments, n_postreps_init_opt=200)

    # Re-compile problem-solver results.
    myexperiments = []
    for solver_name in solver_names:
        # solver_experiments = []
        for problem_name in problem_names:
            file_name_path = (
                "experiments/outputs/" + solver_name + "_on_" + problem_name + ".pickle"
            )
            myexperiment = read_experiment_results(file_name_path)
            myexperiments.append(myexperiment)
    #    solver_experiments.append(myexperiment)
    # myexperiments.append(solver_experiments)

    print("Plotting results.")
    # Produce basic plots.
    plot_terminal_progress(
        experiments=myexperiments, plot_type=PlotType.BOX, normalize=False
    )
    plot_terminal_progress(
        experiments=myexperiments, plot_type=PlotType.BOX, normalize=True
    )
    plot_terminal_progress(
        experiments=myexperiments,
        plot_type=PlotType.VIOLIN,
        normalize=False,
        all_in_one=False,
    )
    plot_terminal_progress(
        experiments=myexperiments, plot_type=PlotType.VIOLIN, normalize=True
    )
    # plot_terminal_scatterplots(experiments = myexperiments, all_in_one=False)

    # Plots will be saved in the folder experiments/plots.
    print("Finished. Plots can be found in experiments/plots folder.")


if __name__ == "__main__":
    main()
