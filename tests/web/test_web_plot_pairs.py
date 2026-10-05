"""Check the server builds (Experiment, AnalysisInput) pairs from real ProblemSolvers."""

import numpy as np

from simopt.compat import convert
from simopt.curve import Curve
from simopt.experiment import Experiment
from simopt.experiment.api import AnalysisInput
from simopt.experiment.plot import save_plots
from simopt.experiment.single import ProblemSolver
from simopt.web.server import generate_plots


def fabricated_problem_solver():
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
    df = convert(experiment).mean_df
    experiment.objective_curves = [
        Curve(group["budget"].tolist(), group["objective"].tolist())
        for _, group in df.groupby("mrep")
    ]
    experiment.progress_curves = [
        Curve(group["normalized_budget"].tolist(), group["normalized_objective"].tolist())
        for _, group in df.groupby("mrep")
    ]
    return experiment


def test_generate_plots_builds_pairs_from_problem_solvers(tmp_path, monkeypatch):
    problem_solver = fabricated_problem_solver()
    records = []

    def spy_save(pairs, config, **kwargs):
        records.append(pairs)
        return save_plots(pairs, config, **kwargs)

    monkeypatch.setattr("simopt.experiment.plot.save_plots", spy_save)
    files = generate_plots(
        [{"plot_type": "MEAN", "params": {"plot_conf_ints": False, "print_max_hw": False}}],
        [[problem_solver]],
        [0],
        [0],
        {0: 0},
        {0: 0},
        [{"name": "RNDSRCH"}],
        [{"name": "SAN-2"}],
        tmp_path,
    )
    assert files and (tmp_path / files[0]).exists()
    (pairs,) = records
    ((experiment, analysis_input),) = pairs
    assert isinstance(experiment, Experiment) and isinstance(analysis_input, AnalysisInput)
    assert experiment.solver is problem_solver.solver
    assert experiment.problem is problem_solver.problem
