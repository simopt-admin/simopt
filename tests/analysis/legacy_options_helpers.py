"""ProblemSolver-based option fixtures for test_saved_plots.py and test_web_plots.py.

Remove this module in Phase 4, once save_plots takes Experiment and AnalysisInput.
"""

from copy import deepcopy

import numpy as np
import pytest

from simopt.compat import convert
from simopt.curve import Curve
from simopt.experiment.single import ProblemSolver


@pytest.fixture
def option_experiment():
    experiment = ProblemSolver(solver_name="RNDSRCH", problem_name="SAN-2")
    experiment.problem.factors["budget"] = 10
    experiment.n_macroreps = 4
    experiment.n_postreps = 4
    experiment.n_postreps_init_opt = 4
    experiment.x0 = (0.0, 0.0)
    experiment.xstar = (9.0, 9.0)
    experiment.x0_postreps = [9.0, 10.0, 11.0, 12.0]
    experiment.xstar_postreps = [0.5, 1.0, 1.5, 2.0]
    experiment.all_intermediate_budgets = [[0, 2 + i, 10] for i in range(4)]
    experiment.all_recommended_xs = [
        [experiment.x0, (float(i + 1), 0.0), (float(i + 1), 1.0)] for i in range(4)
    ]
    experiment.all_post_replicates = [
        [
            experiment.x0_postreps,
            [3.0 + i + j / 2 for j in range(4)],
            [0.25 + i / 4 + j / 8 for j in range(4)],
        ]
        for i in range(4)
    ]
    experiment.all_stoch_constraints = [
        [
            np.array([[i - step - 1 + j / 4, i / 2 + step - 3 + j / 4] for j in range(4)])
            for step in range(3)
        ]
        for i in range(4)
    ]
    experiment.all_est_lhs = [
        [sample.mean(axis=0) for sample in macrorep]
        for macrorep in experiment.all_stoch_constraints
    ]
    experiment.has_run = experiment.has_postreplicated = experiment.has_postnormalized = True
    experiment.crn_across_budget = True
    experiment.crn_across_macroreps = False
    experiment.crn_across_init_opt = True
    refresh_curves(experiment)
    return experiment


def refresh_curves(experiment):
    df = convert(experiment).mean_df
    experiment.objective_curves = [
        Curve(group["budget"].tolist(), group["objective"].tolist())
        for _, group in df.groupby("mrep")
    ]
    experiment.progress_curves = [
        Curve(group["normalized_budget"].tolist(), group["normalized_objective"].tolist())
        for _, group in df.groupby("mrep")
    ]


@pytest.fixture
def option_pair(option_experiment):
    second = deepcopy(option_experiment)
    second.solver.name = "comparison"
    for macrorep in second.all_post_replicates:
        macrorep[1] = [value + 1.5 for value in macrorep[1]]
    refresh_curves(second)
    return [option_experiment, second]
