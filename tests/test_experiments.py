"""Regression test for every model/solver pair that has an expected-results pickle.

Each pickle in `expected_results` is the reference. The test runs the new pipeline
(`simopt.experiment.api.run`) once per pickle and compares the result with the stored
values, checking the stages in order: solver run, post-replication, normalization.
"""

import pickle
from pathlib import Path

import numpy as np
import pytest
import zstandard as zstd

from simopt.experiment import from_names
from simopt.experiment.api import SimulationConfig, run
from simopt.options import CrnOptions
from tools.common import current_machine_result_file, is_result_file_for_current_machine

EXPECTED_RESULTS_DIR = Path(__file__).parent / "expected_results"
RESULTS_SUFFIX = ".pickle.zst"

# The pickles do not store the CRN options, so they are fixed to the defaults of the old pipeline.
CRN_OPTIONS = CrnOptions(across_budget=True, across_macroreps=False, across_x0_xstar=True)

# Absolute tolerance 5e-6 is the exact equivalent of the old assertAlmostEqual(places=5).
SOLUTION_ATOL = 5e-6
CURVE_ATOL = 5e-6
PROGRESS_TOL = 1e-8
X0_XSTAR_TOL = 1e-8


def _collect_result_files() -> list[Path]:
    paths = {}
    for path in sorted(EXPECTED_RESULTS_DIR.glob(f"*{RESULTS_SUFFIX}")):
        # A missing platform-specific pickle becomes its own failing test.
        machine_path = current_machine_result_file(path)
        if machine_path is not None and not machine_path.exists():
            paths[machine_path.name] = machine_path
        elif is_result_file_for_current_machine(path):
            paths[path.name] = path
    return list(paths.values())


def _assert_close(actual, expected, stage, where, what, atol, rtol=0.0):
    """Assert two arrays match, treating NaN in the same position as equal."""
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    assert actual.shape == expected.shape, (
        f"[{stage}] [{where}] Shape of {what} does not match: {actual.shape} != {expected.shape}"
    )
    close = np.isclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True)
    if not close.all():
        differences = np.abs(actual - expected)[~close]
        finite = differences[np.isfinite(differences)]
        max_difference = (
            f"max abs difference {finite.max():.3e}"
            if finite.size > 0
            else "no finite difference (NaN or infinite values do not match)"
        )
        pytest.fail(
            f"[{stage}] [{where}] {what} do not match: "
            f"{int((~close).sum())} of {close.size} values differ, {max_difference}"
        )


@pytest.mark.parametrize(
    "path", _collect_result_files(), ids=lambda path: path.name[: -len(RESULTS_SUFFIX)]
)
def test_experiment(path: Path) -> None:
    if not path.exists():
        pytest.fail(f"Missing platform-specific expected results file: {path.name}")
    with zstd.open(path, "rb") as f:
        expected = pickle.load(f)

    n_mreps = expected["num_macroreps"]
    ps_names = f"{expected['problem_name']} | {expected['solver_name']}"

    experiment = from_names(expected["solver_name"], expected["problem_name"])
    assert experiment.solver.name == expected["solver_name"]
    assert experiment.problem.name == expected["problem_name"]
    [result] = run(
        [experiment],
        SimulationConfig(
            n_mreps=n_mreps,
            n_preps=expected["num_postreps"],
            n_preps_x0_xstar=expected["n_postreps_init_opt"],
        ),
        CRN_OPTIONS,
        n_jobs=1,
    )
    mean_df = result.mean_df
    est_objectives = result.full_df.groupby(["mrep", "step"])["objective"].mean()
    expected_est_lhs = expected.get("all_est_lhs", [])
    problem_budget = experiment.problem.factors["budget"]

    # Each macroreplication's rows, computed once.
    assert set(mean_df["mrep"]) == set(range(n_mreps)), (
        f"[solver run] [{ps_names}] Macroreplications {sorted(set(mean_df['mrep']))} "
        f"do not match range({n_mreps})"
    )
    mrep_dfs = [mean_df[mean_df["mrep"] == mrep].sort_values("step") for mrep in range(n_mreps)]
    assert len(expected_est_lhs) in (0, n_mreps), (
        f"[post-replication] [{ps_names}] Expected estimated LHS has {len(expected_est_lhs)} "
        f"entries, expected 0 or {n_mreps}"
    )
    if len(expected_est_lhs) == 0:
        assert experiment.problem.n_stochastic_constraints == 0, (
            f"[post-replication] [{ps_names}] Expected estimated LHS is empty but the problem "
            "has stochastic constraints"
        )

    # Stage 1: solver run
    for mrep in range(n_mreps):
        where = f"{ps_names} | {mrep}"
        mrep_df = mrep_dfs[mrep]
        rec_xs = list(mrep_df["solution"])
        expected_rec_xs = expected["all_recommended_xs"][mrep]
        assert len(rec_xs) == len(expected_rec_xs), (
            f"[solver run] [{where}] Length of recommended solutions do not match"
        )
        for sol_list_idx, (sol, expected_sol) in enumerate(
            zip(rec_xs, expected_rec_xs, strict=True)
        ):
            _assert_close(
                sol,
                expected_sol,
                "solver run",
                f"{where} | {sol_list_idx}",
                "Recommended solutions",
                SOLUTION_ATOL,
            )
        _assert_close(
            mrep_df["budget"],
            expected["all_intermediate_budgets"][mrep],
            "solver run",
            where,
            "Intermediate budgets",
            SOLUTION_ATOL,
        )

    # Stage 2: post-replication
    rep_counts = result.full_df.groupby(level=["mrep", "step"]).size()
    wrong_counts = rep_counts[rep_counts != expected["num_postreps"]]
    assert wrong_counts.empty, (
        f"[post-replication] [{ps_names}] Expected {expected['num_postreps']} replications for "
        f"every (mrep, step), got {dict(wrong_counts.head(5))}"
    )
    for mrep in range(n_mreps):
        where = f"{ps_names} | {mrep}"
        mrep_df = mrep_dfs[mrep]
        _assert_close(
            est_objectives.loc[mrep].sort_index(),
            expected["all_est_objectives"][mrep],
            "post-replication",
            where,
            "Estimated objectives",
            CURVE_ATOL,
        )
        if len(expected_est_lhs) > 0:
            _assert_close(
                np.stack(
                    [np.asarray(lhs, dtype=float) for lhs in mrep_df["stochastic_constraints"]]
                ),
                expected_est_lhs[mrep],
                "post-replication",
                where,
                "Estimated stochastic constraint LHS",
                SOLUTION_ATOL,
            )

    # Stage 3: normalization
    _assert_close(result.x0, expected["x0"], "normalization", ps_names, "x0", X0_XSTAR_TOL)
    _assert_close(
        result.x0_sample,
        expected["x0_postreps"],
        "normalization",
        ps_names,
        "x0 samples",
        X0_XSTAR_TOL,
        X0_XSTAR_TOL,
    )
    _assert_close(
        result.xstar_sample,
        expected["xstar_postreps"],
        "normalization",
        ps_names,
        "xstar samples",
        X0_XSTAR_TOL,
        X0_XSTAR_TOL,
    )
    for mrep in range(n_mreps):
        where = f"{ps_names} | {mrep}"
        mrep_df = mrep_dfs[mrep]
        objective_x, objective_y = expected["objective_curves"][mrep]
        _assert_close(
            mrep_df["budget"],
            objective_x,
            "normalization",
            where,
            "Objective curve X values",
            CURVE_ATOL,
        )
        _assert_close(
            mrep_df["objective"],
            objective_y,
            "normalization",
            where,
            "Objective curve Y values",
            CURVE_ATOL,
        )
        progress_x, progress_y = expected["progress_curves"][mrep]
        _assert_close(
            mrep_df["budget"] / problem_budget,
            progress_x,
            "normalization",
            where,
            "Progress curve X values",
            PROGRESS_TOL,
            PROGRESS_TOL,
        )
        _assert_close(
            mrep_df["normalized_objective"],
            progress_y,
            "normalization",
            where,
            "Progress curve Y values",
            PROGRESS_TOL,
            PROGRESS_TOL,
        )

    # Checked last so that a stale stored x* does not hide differences in the curves.
    _assert_close(result.xstar, expected["xstar"], "normalization", ps_names, "xstar", X0_XSTAR_TOL)
