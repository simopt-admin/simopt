"""Focused tests for scheduling complete experiment plans."""

from copy import deepcopy
from itertools import product

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

import simopt.experiment.api as api
from simopt.experiment.api import SimulationConfig, plan_experiment, run
from simopt.experiment.post_normalize import normalize
from simopt.experiment.post_replicate import _post_replicate_mrep
from simopt.experiment.run_solver import _run_mrep
from simopt.models.chessmm import ChessAvgDifference
from simopt.models.cntnv import CntNVMaxProfit
from simopt.models.example import ExampleProblem
from simopt.options import CrnOptions
from simopt.solvers.randomsearch import RandomSearch


class FixedNoise:
    def random(self, rng):
        return 100.0


def _problem(initial_solution=(2.0, 2.0)):
    return ExampleProblem(fixed_factors={"budget": 4, "initial_solution": initial_solution})


def _solver(sample_size=2):
    return RandomSearch(fixed_factors={"sample_size": sample_size})


def _config(n_mreps=2):
    return SimulationConfig(n_mreps=n_mreps, n_preps=2, n_preps_x0_xstar=2)


def test_plan_snapshots_inputs_without_simulation(monkeypatch):
    problem = _problem()
    solver = _solver()
    simulation_config = _config()
    _ = problem.optimization_problem

    def fail_if_simulated(*args, **kwargs):
        raise AssertionError("planning simulated the problem")

    with monkeypatch.context() as patcher:
        patcher.setattr(ExampleProblem, "simulate", fail_if_simulated)
        plan = plan_experiment(problem, [solver], simulation_config)

    baseline = api._run_plans([plan], n_jobs=1)[0]
    problem.factors["initial_solution"] = (9.0, 9.0)
    problem.model.noise_model = FixedNoise()
    solver.factors["sample_size"] = 1
    simulation_config.n_mreps = 9
    repeated = api._run_plans([plan], n_jobs=1)[0]

    pdt.assert_frame_equal(baseline.full_df, repeated.full_df)
    assert tuple(repeated.x0) == (2.0, 2.0)
    assert repeated.full_df.index.get_level_values("mrep").nunique() == 2


def test_all_plans_share_one_macroreplication_pool(monkeypatch):
    experiments = api.create_matrix([_solver(1), _solver(2)], [_problem()])
    experiments.extend(api.create_matrix([_solver(2)], [_problem((3.0, 3.0))]))
    calls = []

    class RecordingParallel:
        def __init__(self, n_jobs):
            assert n_jobs == 2

        def __call__(self, jobs):
            jobs = list(jobs)
            calls.append(jobs)
            return [func(*args, **kwargs) for func, args, kwargs in jobs]

    monkeypatch.setattr(api, "Parallel", RecordingParallel)
    pooled = run(experiments, _config(2), n_jobs=2)
    sequential = run(experiments, _config(2), n_jobs=1)

    assert len(calls) == 1
    assert len(calls[0]) == 6
    assert len(pooled) == 3
    assert [tuple(result.x0) for result in pooled] == [
        (2.0, 2.0),
        (2.0, 2.0),
        (3.0, 3.0),
    ]
    for actual, expected in zip(pooled, sequential, strict=True):
        pdt.assert_frame_equal(actual.full_df, expected.full_df)
        np.testing.assert_array_equal(actual.xstar_sample, expected.xstar_sample)


def test_process_pool_matches_sequential_run():
    experiments = api.create_matrix([_solver(1), _solver(2)], [_problem()])
    experiments.extend(api.create_matrix([_solver(2)], [_problem((3.0, 3.0))]))

    parallel = run(experiments, _config(2), n_jobs=2)
    sequential = run(experiments, _config(2), n_jobs=1)

    for actual, expected in zip(parallel, sequential, strict=True):
        pdt.assert_frame_equal(actual.full_df, expected.full_df)
        np.testing.assert_array_equal(actual.x0_sample, expected.x0_sample)
        np.testing.assert_array_equal(actual.xstar_sample, expected.xstar_sample)


@pytest.mark.parametrize(
    ("across_budget", "across_macroreps", "across_x0_xstar"),
    list(product((False, True), repeat=3)),
)
def test_planned_rngs_match_direct_macroreplications(
    across_budget, across_macroreps, across_x0_xstar
):
    problem = CntNVMaxProfit(fixed_factors={"budget": 4})
    solvers = [_solver(1), _solver(2)]
    crn_options = CrnOptions(
        across_budget=across_budget,
        across_macroreps=across_macroreps,
        across_x0_xstar=across_x0_xstar,
    )
    experiments = api.create_matrix(solvers, [problem])
    actual = run(experiments, _config(), crn_options, n_jobs=1)

    solver_history_dfs = []
    post_replicate_dfs = []
    for i, solver in enumerate(solvers):
        histories = []
        postreps = []
        for mrep in range(2):
            solver_history_df, _ = _run_mrep(deepcopy(solver), deepcopy(problem), mrep)
            histories.append(solver_history_df)
            postreps.append(
                _post_replicate_mrep(
                    deepcopy(problem),
                    solver_history_df,
                    2,
                    across_macroreps,
                    across_budget,
                )
            )
        solver_history_df = pd.concat(histories, ignore_index=True)
        solver_history_df["experiment"] = i
        solver_history_dfs.append(solver_history_df)
        post_replicate_df = pd.concat(postreps, ignore_index=True)
        post_replicate_df["experiment"] = i
        post_replicate_dfs.append(post_replicate_df)

    full_df = pd.concat(solver_history_dfs, ignore_index=True).merge(
        pd.concat(post_replicate_dfs, ignore_index=True),
        on=["experiment", "mrep", "step"],
    )
    full_df = full_df.set_index(["experiment", "mrep", "step", "rep"])
    normalization_result = normalize(
        problem,
        full_df,
        2,
        across_x0_xstar,
    )
    for i, result in enumerate(actual):
        expected_df = full_df.loc[i]
        pdt.assert_frame_equal(
            result.full_df.drop(columns="stochastic_constraints"),
            expected_df.drop(columns="stochastic_constraints"),
        )
        for observed, expected in zip(
            result.full_df["stochastic_constraints"],
            expected_df["stochastic_constraints"],
            strict=True,
        ):
            np.testing.assert_array_equal(observed, expected)
        np.testing.assert_array_equal(result.x0_sample, normalization_result.x0_sample)
        np.testing.assert_array_equal(result.xstar_sample, normalization_result.xstar_sample)
        assert tuple(result.xstar) == normalization_result.xstar


def test_legacy_run_preserves_interleaved_input_order():
    first_problem = _problem()
    second_problem = _problem((3.0, 3.0))
    experiments = [
        api.ProblemSolver(problem=first_problem, solver=_solver(1), create_pickle=False),
        api.ProblemSolver(problem=second_problem, solver=_solver(2), create_pickle=False),
        api.ProblemSolver(problem=first_problem, solver=_solver(2), create_pickle=False),
    ]

    results = run(experiments, _config(1), n_jobs=1)

    assert [tuple(result.x0) for result in results] == [
        (2.0, 2.0),
        (3.0, 3.0),
        (2.0, 2.0),
    ]
    assert len(results) == 3


def test_equivalent_problems_keep_their_own_model_state_and_normalization():
    first_problem = _problem()
    second_problem = _problem()
    second_problem.model.noise_model = FixedNoise()
    assert first_problem == second_problem
    assert first_problem is not second_problem
    solver = _solver(2)
    experiments = [
        api.ProblemSolver(problem=first_problem, solver=solver, create_pickle=False),
        api.ProblemSolver(problem=second_problem, solver=solver, create_pickle=False),
    ]

    results = run(experiments, _config(1), n_jobs=1)
    solver_history_df, _ = _run_mrep(deepcopy(solver), deepcopy(second_problem), 0)
    expected_postreps = _post_replicate_mrep(
        deepcopy(second_problem), solver_history_df, 2, False, True
    )

    np.testing.assert_array_equal(
        results[1].full_df["objective"].to_numpy(),
        expected_postreps["objective"].to_numpy(),
    )
    assert np.mean(results[1].full_df["objective"]) > np.mean(results[0].full_df["objective"])

    for experiment, result in zip(experiments, results, strict=True):
        expected = run([experiment], _config(1), n_jobs=1)[0]
        pdt.assert_frame_equal(result.full_df, expected.full_df)
        np.testing.assert_array_equal(result.x0, expected.x0)
        np.testing.assert_array_equal(result.xstar, expected.xstar)
        np.testing.assert_array_equal(result.x0_sample, expected.x0_sample)
        np.testing.assert_array_equal(result.xstar_sample, expected.xstar_sample)
    assert np.mean(results[1].x0_sample) > np.mean(results[0].x0_sample)


def test_stochastic_constraint_results_match_direct_macroreplication():
    problem = ChessAvgDifference(fixed_factors={"budget": 1, "upper_time": 100})
    solver = _solver(1)
    config = SimulationConfig(n_mreps=1, n_preps=2, n_preps_x0_xstar=2)

    experiments = api.create_matrix([solver], [problem])
    result = run(experiments, config, n_jobs=1)[0]
    solver_history_df, _ = _run_mrep(deepcopy(solver), deepcopy(problem), 0)
    expected = _post_replicate_mrep(deepcopy(problem), solver_history_df, 2, False, True)

    assert len(result.full_df) == len(expected)
    for actual, reference in zip(
        result.full_df["stochastic_constraints"], expected["stochastic_constraints"], strict=True
    ):
        assert len(actual) == 1
        np.testing.assert_array_equal(actual, reference)


def test_invalid_plan_configuration_is_rejected():
    with pytest.raises(ValueError, match="solvers must not be empty"):
        plan_experiment(_problem(), [], _config())
    with pytest.raises(ValueError, match="post-replications must be positive"):
        run(
            api.create_matrix([_solver()], [_problem()]),
            SimulationConfig(n_mreps=1, n_preps=0, n_preps_x0_xstar=1),
            n_jobs=1,
        )
    with pytest.raises(ValueError, match="macroreplications must be positive"):
        run(api.create_matrix([_solver()], [_problem()]), _config(0), n_jobs=1)


def test_run_requires_simulation_config():
    with pytest.raises(TypeError, match="simulation_config"):
        run(api.create_matrix([_solver()], [_problem()]))


def test_run_empty_experiments():
    assert run([], _config(), n_jobs=1) == []
