"""C-ASTRO-DF on SAN-3 reproduces values recorded from the pre-port nicole_astrodf branch."""

import json
from pathlib import Path

import numpy as np
import pytest

from simopt.analysis.merit import merit_progress
from simopt.experiment import Experiment
from simopt.experiment.api import SimulationConfig, post_normalize_merit, run
from simopt.models.san import SANLongestPathCostConst
from simopt.options import MeritOptions
from simopt_solvers import CASTRODF

FIXTURE = Path(__file__).parents[1] / "expected_results" / "astro_castrodf_san3.json"
MERIT_OPTIONS = MeritOptions(obj_const=1000, feas_tol_upper=10000, feas_tol_lower=1e-5)


@pytest.fixture(scope="module")
def analysis():
    problem = SANLongestPathCostConst(fixed_factors={"budget": 600, "total_cost": 5})
    solver = CASTRODF(
        fixed_factors={
            "feas_tol": 1e-5,
            "easy_solve_tangent": True,
            "a_normal": 0.2,
            "a_tangent": 0.8,
            "mu": 0.1,
            "sigma_min": 10,
            "reuse_interpolation_set": True,
            "lambda_min": 10,
            "delta_0": 1,
            "dogleg": True,
        }
    )
    experiments = [Experiment(solver=solver, problem=problem)]
    results = run(experiments, SimulationConfig(n_mreps=1, n_preps=5, n_preps_x0_xstar=5), n_jobs=1)
    (result,) = post_normalize_merit(experiments, results, MERIT_OPTIONS)
    return result


@pytest.fixture(scope="module")
def expected():
    return json.loads(FIXTURE.read_text())


def test_solver_history_matches_reference(analysis, expected):
    raw = analysis.full_df.groupby(["mrep", "step"]).agg(
        {"budget": "first", "objective": "mean", "deterministic_violation": "first"}
    )
    np.testing.assert_array_equal(raw["budget"], expected["budgets"][0])
    solutions = analysis.mean_df.sort_values(["mrep", "step"])["solution"].tolist()
    np.testing.assert_allclose(solutions, expected["xs"][0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(raw["objective"], expected["est_obj"][0], rtol=1e-10)
    np.testing.assert_allclose(
        raw["deterministic_violation"], expected["viol"][0], rtol=1e-9, atol=1e-12
    )


def test_normalization_matches_reference(analysis, expected):
    assert np.mean(analysis.x0_sample) == pytest.approx(expected["x0_mean"], rel=1e-10)
    assert np.mean(analysis.xstar_sample) == pytest.approx(expected["xstar_mean"], rel=1e-10)
    np.testing.assert_allclose(analysis.xstar, expected["xstar"], atol=1e-12)


def test_progress_and_merit_curves_match_reference(analysis, expected):
    progress = analysis.mean_df.sort_values(["mrep", "step"])
    np.testing.assert_allclose(
        progress["normalized_objective"], expected["progress_y"][0], atol=1e-10
    )
    np.testing.assert_allclose(progress["normalized_budget"], expected["progress_x"][0], atol=1e-12)
    merit = merit_progress(analysis, MERIT_OPTIONS)
    np.testing.assert_allclose(merit["objective"], expected["merit_y"][0], atol=1e-10)
