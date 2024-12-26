"""
This script is intended to help with running a data-farming experiment on
a solver. It creates a design of solver factors and runs multiple
macroreplications at each version of the solver. Outputs are printed to a file.
"""

import sys
import os.path as o

sys.path.append(
    o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), ".."))
)  # type:ignore

from simopt.experiment_base import ProblemsSolvers


def main() -> None:
    # Specify the name of the solver as it appears in directory.py

    solver_names = ["RNDSRCH", "RNDSRCH", "ASTRODF"]

    solver_renames = ["RND_test1", "RND_test2", "AST_test"]

    problem_names = ["EXAMPLE-1", "CNTNEWS-1"]

    problem_renames = ["EX_test", "NEWS_test"]

    experiment_name = "test_exp"

    solver_factors = [{}, {"sample_size": 2}, {}]

    problem_factors = [{}, {}]

    # Create ProblemsSovlers experiment with solver and model design
    experiment = ProblemsSolvers(
        solver_factors=solver_factors,
        problem_factors=problem_factors,
        solver_names=solver_names,
        problem_names=problem_names,
        solver_renames=solver_renames,
        problem_renames=problem_renames,
        experiment_name=experiment_name,
        create_pair_pickles=True,
    )

    # check compatibility of selected solvers and problems
    experiment.check_compatibility()

    # Run macroreplications at each design point.
    experiment.run(2)

    # Postprocess the experimental results from each design point.
    experiment.post_replicate(10)
    experiment.post_normalize(10)

    # Record and log results
    experiment.record_group_experiment_results()
    experiment.log_group_experiment_results()
    experiment.report_group_statistics()


if __name__ == "__main__":
    main()
