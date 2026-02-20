import numpy as np
import pandas.testing as pdt

from simopt.compat import convert
from simopt.experiment.api import SimulationConfig, run_experiment
from simopt.experiment.data import ManyPostReplicateSchema, ManySolverHistorySchema
from simopt.experiment.single import ProblemSolver
from simopt.options import CrnOptions
from test.utils import load_problem_solver


def _solver_history_df(result):
    df = result.full_df.reset_index()
    df["experiment"] = 0
    return df


def _post_replicate_df(result):
    df = result.full_df.reset_index()
    df["experiment"] = 0
    return df


def test_convert_matches_new_api_result():
    path = "test/expected_results/CNTNEWS1_ADAM.pickle.zst"
    experiment1 = load_problem_solver(path)

    result1 = convert(experiment1)
    solver_history_df1 = _solver_history_df(result1)
    post_replicate_df1 = _post_replicate_df(result1)
    ManySolverHistorySchema.validate(solver_history_df1)
    ManyPostReplicateSchema.validate(post_replicate_df1)

    # Run new experiment to compare
    n_mreps = experiment1.n_macroreps
    n_preps = experiment1.n_postreps
    n_preps_x0_xstar = experiment1.n_postreps_init_opt
    experiment2 = ProblemSolver(
        solver_name=experiment1.solver.name,
        problem_name=experiment1.problem.name,
        create_pickle=False,
    )
    result2 = run_experiment(
        [experiment2],
        simulation_config=SimulationConfig(
            n_mreps=n_mreps,
            n_preps=n_preps,
            n_preps_x0_xstar=n_preps_x0_xstar,
        ),
        # TODO: we are currently hardcoding the CRN options to True, False, True
        # because the values are not stored in the pickle file
        crn_options=CrnOptions(across_budget=True, across_macroreps=False, across_x0_xstar=True),
        n_jobs=1,
    )[0]

    solver_history_df2 = _solver_history_df(result2)
    post_replicate_df2 = _post_replicate_df(result2)
    ManySolverHistorySchema.validate(solver_history_df2)
    ManyPostReplicateSchema.validate(post_replicate_df2)

    columns = ["experiment", "mrep", "step", "budget", "solution"]
    pdt.assert_frame_equal(
        solver_history_df1[columns],
        solver_history_df2[columns],
        check_exact=False,
        rtol=1e-8,
        atol=1e-8,
    )

    columns = [
        "experiment",
        "mrep",
        "step",
        "rep",
        "objective",
        "stochastic_constraints",
    ]
    pdt.assert_frame_equal(
        post_replicate_df1[columns],
        post_replicate_df2[columns],
        check_exact=False,
        rtol=1e-8,
        atol=1e-8,
    )

    columns = [
        "experiment",
        "mrep",
        "step",
        "rep",
        "budget",
        "solution",
        "objective",
        "stochastic_constraints",
    ]
    pdt.assert_frame_equal(
        result1.full_df,
        result2.full_df,
        check_exact=False,
        rtol=1e-8,
        atol=1e-8,
    )

    assert np.allclose(result1.x0, result2.x0)
    assert np.allclose(result1.xstar, result2.xstar)
    assert np.allclose(result1.x0_sample, result2.x0_sample)
    assert np.allclose(result1.xstar_sample, result2.xstar_sample)
    assert np.allclose(np.mean(result1.x0_sample), np.mean(result2.x0_sample))
    assert np.allclose(np.mean(result1.xstar_sample), np.mean(result2.xstar_sample))
