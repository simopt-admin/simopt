"""Deterministic feasibility and merit analyses on synthetic data with known answers."""

from types import SimpleNamespace
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from simopt.analysis import (
    deterministic_feasibility_progress,
    deterministic_terminal_feasibility,
    solvability_profile,
)
from simopt.analysis.bootstrap import bootstrap
from simopt.analysis.merit import merit_progress
from simopt.deterministic_feasibility import (
    deterministic_violation,
    merit_from_objective_and_violation,
)
from simopt.experiment import api
from simopt.experiment import post_normalize_merit as pnm
from simopt.experiment.api import AnalysisInput
from simopt.experiment.experiment import from_names
from simopt.experiment.plot import save_plots
from simopt.options import ConfidenceIntervalOptions, CrnOptions, MeritOptions
from simopt.problem import Problem

OPTIONS = MeritOptions(obj_const=10.0, feas_tol_upper=0.5, feas_tol_lower=1e-3)
MERIT_KWARGS: dict[str, Any] = {
    "curve_source": "merit_progress_curves",
    "feas_obj_const": 10.0,
    "feas_tol_upper": 0.5,
    "feas_tol_lower": 1e-3,
}

# Solutions are 1-d points whose deterministic violation is their coordinate (inequality
# constraint x <= 0), so the violation of each step is known from the data.
SOLUTIONS = [[0.2, 0.0005, 0.0], [0.2, 0.7, 0.0002]]
OBJECTIVES = [[10.0, 6.0, 2.5], [10.0, 4.0, 2.0]]
BUDGETS = [0, 5, 10]


class StubProblem:
    model = SimpleNamespace(n_rngs=1)

    def get_deterministic_equality_constraints(self, x):
        return None

    def get_deterministic_inequality_constraints(self, x):
        return np.array([x[0]])


def stub_problem() -> Problem:
    return cast(Problem, StubProblem())


def make_full_df(n_preps=2, with_violation=False):
    rows = []
    for mrep in range(2):
        for step in range(3):
            for rep in range(n_preps):
                offset = (rep - (n_preps - 1) / 2) * 0.5
                rows.append(
                    {
                        "mrep": mrep,
                        "step": step,
                        "rep": rep,
                        "budget": BUDGETS[step],
                        "solution": (SOLUTIONS[mrep][step],),
                        "objective": OBJECTIVES[mrep][step] + offset,
                        "stochastic_constraints": np.array([]),
                    }
                )
    df = pd.DataFrame(rows).set_index(["mrep", "step", "rep"])
    if with_violation:
        df = pnm.attach_deterministic_violation(stub_problem(), df)
    return df


def make_input(**kwargs):
    return AnalysisInput(
        make_full_df(with_violation=True),
        np.array([0.2]),
        np.array([10.0, 10.0]),
        np.array([9.0]),
        np.array([2.0, 2.0]),
        **kwargs,
    )


def make_merit_input():
    normalization = pnm.MeritNormalization(
        OPTIONS,
        x0_violation=0.2,
        best_merit=3.0,
        xstar_merit=(0.0002,),
        xstar_merit_violation=0.0002,
    )
    return make_input(xstar_merit_sample=np.array([3.0, 3.0]), merit_normalization=normalization)


@pytest.fixture
def experiment():
    return from_names("RNDSRCH", "SAN-2", problem_fixed_factors={"budget": 10})


def test_violation_is_recorded_per_solution():
    df = make_full_df(with_violation=True)
    np.testing.assert_allclose(
        df.groupby(["mrep", "step"])["deterministic_violation"].first(),
        [0.2, 0.0005, 0.0, 0.2, 0.7, 0.0002],
    )
    assert deterministic_violation(stub_problem(), (0.7,)) == pytest.approx(0.7)
    # Problems without deterministic constraints are left untouched.
    plain = make_full_df()
    assert pnm.attach_deterministic_violation(cast(Problem, SimpleNamespace()), plain) is plain
    with pytest.raises(ValueError, match="deterministic constraints"):
        pnm.attach_deterministic_violation(cast(Problem, SimpleNamespace()), plain, required=True)
    assert "deterministic_violation" in make_input().mean_df


def test_merit_function():
    assert merit_from_objective_and_violation(5.0, 0.0, 10.0, 1e-3, 0.5) == 5.0
    assert merit_from_objective_and_violation(5.0, 0.2, 10.0, 1e-3, 0.5) == pytest.approx(7.0)
    assert merit_from_objective_and_violation(5.0, 0.7, 10.0, 1e-3, 0.5) == np.inf


def test_merit_progress_known_values():
    result = merit_progress(make_merit_input(), OPTIONS)
    # initial merit 10 + 10 * 0.2 = 12, best merit 3, gap 9; running best per macrorep.
    np.testing.assert_allclose(result["budget"], [0, 0.5, 1.0] * 2)
    np.testing.assert_allclose(result["objective"], [1.0, 1 / 3, 0.0, 1.0, 1.0, 0.0])


def test_merit_progress_infinite_initial_merit():
    data = make_merit_input()
    normalization = data.merit_normalization
    assert normalization is not None
    data.merit_normalization = pnm.MeritNormalization(
        OPTIONS,
        x0_violation=0.7,
        best_merit=normalization.best_merit,
        xstar_merit=normalization.xstar_merit,
        xstar_merit_violation=normalization.xstar_merit_violation,
    )
    result = merit_progress(data, OPTIONS)
    # Infinite initial gap: every finite merit normalizes to 0.
    np.testing.assert_allclose(result["objective"], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def test_merit_progress_requires_normalization_and_matching_options():
    with pytest.raises(ValueError, match="post_normalize_merit"):
        merit_progress(make_input(), OPTIONS)
    with pytest.raises(ValueError, match="differ"):
        merit_progress(make_merit_input(), MeritOptions(obj_const=1.0))
    no_violation = AnalysisInput(
        make_full_df(),
        np.array([0.2]),
        np.array([10.0]),
        np.array([9.0]),
        np.array([2.0]),
    )
    with pytest.raises(ValueError, match="deterministic violations"):
        deterministic_feasibility_progress.analyze(no_violation)


def test_normalize_merit_matches_hand_computation(monkeypatch):
    calls = []

    def simulate(problem, x, rngs, n_reps):
        calls.append((x, n_reps))
        return np.array([3.0, 3.0])

    monkeypatch.setattr(pnm, "_simulate", simulate)
    experiments: list[Any] = [SimpleNamespace(problem=StubProblem())]
    plain = AnalysisInput(
        make_full_df(),
        np.array([0.2]),
        np.array([10.0, 10.0]),
        np.array([9.0]),
        np.array([2.0, 2.0]),
    )
    (data,) = api.post_normalize_merit(experiments, [plain], OPTIONS)
    # Best empirical merit is 2.0 at (0.0002,) (mrep 1, last step); re-simulated at 3.0.
    assert calls == [((0.0002,), 2)]
    normalization = data.merit_normalization
    assert normalization is not None
    assert normalization.xstar_merit == (0.0002,)
    assert normalization.best_merit == pytest.approx(3.0)
    assert normalization.x0_violation == pytest.approx(0.2)
    assert data.xstar_merit_sample is not None
    np.testing.assert_allclose(data.xstar_merit_sample, [3.0, 3.0])
    assert "deterministic_violation" not in plain.full_df
    result = merit_progress(data, OPTIONS)
    np.testing.assert_allclose(result["objective"], [1.0, 1 / 3, 0.0, 1.0, 1.0, 0.0])

    # A supplied m* skips the search and re-simulation.
    calls.clear()
    (data,) = api.post_normalize_merit(experiments, [plain], OPTIONS, proxy_opt_merit=2.0)
    assert calls == [] and data.xstar_merit_sample is None
    result = merit_progress(data, OPTIONS)
    # gap = 12 - 2 = 10: mrep 0 -> [1, 0.4, 0.05], mrep 1 -> [1, 1, 0].
    np.testing.assert_allclose(result["objective"], [1.0, 0.4, 0.05, 1.0, 1.0, 0.0])


def test_bootstrap_carries_violation_and_merit_sample():
    data = make_merit_input()
    samples = bootstrap(data, 3, lambda d: d, True, False, True)
    for sample in samples:
        assert len(sample.xstar_merit_sample) == 2
        assert set(sample.xstar_merit_sample) <= {3.0}
        assert sample.merit_normalization is data.merit_normalization
        expected = sample.full_df["solution"].map(lambda s: s[0])
        np.testing.assert_allclose(sample.full_df["deterministic_violation"], expected)


def test_solvability_profile_merit_cdf():
    data = make_merit_input()
    result = solvability_profile.analyze(
        data, "cdf", None, CrnOptions(), solve_tolerance=0.5, **MERIT_KWARGS
    )
    # Merit progress solves at 0.5 for mrep 0 (1/3 < 0.5) and at 1.0 for mrep 1.
    np.testing.assert_allclose(result.df["budget"], [0, 0.5, 1.0, 1.0])
    np.testing.assert_allclose(result.df["value"], [0, 0.5, 1.0, 1.0])

    with pytest.raises(ValueError, match="normalize"):
        solvability_profile.analyze(
            data, "cdf", None, CrnOptions(), normalize=False, **MERIT_KWARGS
        )

    ci = ConfidenceIntervalOptions(n_bootstraps=5, bias_correction=False)
    with_ci = solvability_profile.analyze(
        data, "cdf", ci, CrnOptions(), solve_tolerance=0.5, **MERIT_KWARGS
    )
    assert with_ci.ci is not None
    assert (with_ci.ci["lb"] <= with_ci.ci["ub"]).all()


def test_solvability_profile_config_merit(experiment):
    config = solvability_profile.SolvabilityProfile(
        curve_source="merit_progress_curves",
        feas_obj_const=10.0,
        feas_tol_upper=0.5,
        feas_tol_lower=1e-3,
        solve_tolerance=0.5,
        ci_options=None,
    )
    fig, ax = config.plot([make_merit_input()], [experiment])
    np.testing.assert_allclose(np.asarray(ax.lines[0].get_ydata()), [0, 0.5, 1.0])
    plt.close(fig)
    with pytest.raises(ValueError, match="curve_source"):
        solvability_profile.analyze(
            make_merit_input(),
            "cdf",
            None,
            CrnOptions(),
            curve_source="x",  # ty: ignore[invalid-argument-type]
        )


def test_deterministic_progress_norm_and_objective():
    data = make_input()
    result = deterministic_feasibility_progress.analyze(data)
    np.testing.assert_allclose(result.df["value"], [0.2, 0.0005, 0.0, 0.2, 0.7, 0.0002])

    result = deterministic_feasibility_progress.analyze(data, floor_feas=True, feas_tol_lower=1e-3)
    np.testing.assert_allclose(result.df["value"], [0.2, 1e-3, 1e-3, 0.2, 0.7, 1e-3])

    result = deterministic_feasibility_progress.analyze(
        data, score_type="objective", obj_const=10.0, feas_tol_upper=0.5, feas_tol_lower=1e-3
    )
    np.testing.assert_allclose(result.df["value"], [12.0, 6.0, 2.5, 12.0, np.inf, 2.0])

    with pytest.raises(ValueError, match="score_type"):
        deterministic_feasibility_progress.analyze(
            data,
            score_type="value",  # ty: ignore[invalid-argument-type]
        )


def test_deterministic_progress_plot(experiment):
    data = make_input()
    results = deterministic_feasibility_progress.analyze_many(
        [data], score_type="objective", obj_const=10.0, feas_tol_upper=0.5, feas_tol_lower=1e-3
    )
    fig, ax = deterministic_feasibility_progress.plot_many(
        results, [experiment], log_scale=True, log_base=2, plot_zero=True
    )
    curves = [line for line in ax.lines if line.get_linestyle() != "--"]
    assert len(curves) == 2
    np.testing.assert_allclose(np.asarray(curves[0].get_xdata()), BUDGETS)
    np.testing.assert_allclose(np.asarray(curves[0].get_ydata()), [12.0, 6.0, 2.5])
    assert ax.get_yscale() == "log"
    assert ax.get_ylabel() == "Merit"
    legend = ax.get_legend()
    assert legend is not None
    assert legend.get_texts()[0].get_text() == experiment.solver.name
    plt.close(fig)
    with pytest.raises(ValueError, match="same length"):
        deterministic_feasibility_progress.plot_many(results, [])


def test_deterministic_terminal_scores_and_ci(experiment):
    data = make_input()
    result = deterministic_terminal_feasibility.analyze(data, None, CrnOptions())
    np.testing.assert_allclose(result.df["objective"], [2.5, 2.0])
    np.testing.assert_allclose(result.df["feasibility"], [0.0, 0.0002])
    assert "lb_objective" not in result.df

    floored = deterministic_terminal_feasibility.analyze(
        data, None, CrnOptions(), feas_tol=1e-3, floor_feas=True
    )
    np.testing.assert_allclose(floored.df["feasibility"], [1e-3, 1e-3])
    raw = deterministic_terminal_feasibility.analyze(
        data, None, CrnOptions(), score_type="value", feas_tol=1e-3, floor_feas=True
    )
    np.testing.assert_allclose(raw.df["feasibility"], [0.0, 0.0002])

    ci = ConfidenceIntervalOptions(n_bootstraps=20, bias_correction=False)
    with_ci = deterministic_terminal_feasibility.analyze(data, ci, CrnOptions())
    assert (with_ci.df["lb_objective"] <= with_ci.df["ub_objective"]).all()
    # Objective means of 2 post-replications (offsets -0.25 and +0.25) bound the CI.
    assert (with_ci.df["lb_objective"] >= with_ci.df["objective"] - 0.25 - 1e-9).all()
    assert (with_ci.df["ub_objective"] <= with_ci.df["objective"] + 0.25 + 1e-9).all()
    np.testing.assert_allclose(with_ci.df["feasibility"], [0.0, 0.0002])

    fig, ax = deterministic_terminal_feasibility.plot_many(
        [with_ci], [experiment], log_scale=True, plot_zero=True
    )
    assert ax.get_yscale() == "log"
    assert any(line.get_linestyle() == "--" for line in ax.lines)
    plt.close(fig)

    fig, ax = deterministic_terminal_feasibility.plot_many([result], [experiment])
    assert len(ax.collections) == 1
    np.testing.assert_allclose(
        np.asarray(ax.collections[0].get_offsets()), [[2.5, 0.0], [2.0, 0.0002]]
    )
    plt.close(fig)

    with pytest.raises(ValueError, match="score_type"):
        deterministic_terminal_feasibility.analyze(
            data,
            None,
            CrnOptions(),
            score_type="inf_norm",  # ty: ignore[invalid-argument-type]
        )


def test_save_plots_registration(experiment, tmp_path):
    pairs = [(experiment, make_merit_input())]
    configs = [
        deterministic_feasibility_progress.DeterministicFeasibilityProgress(score_type="objective"),
        deterministic_terminal_feasibility.DeterministicTerminalFeasibility(
            ci_options=ConfidenceIntervalOptions(n_bootstraps=5)
        ),
        solvability_profile.SolvabilityProfile(
            curve_source="merit_progress_curves",
            feas_obj_const=10.0,
            feas_tol_upper=0.5,
            feas_tol_lower=1e-3,
            ci_options=ConfidenceIntervalOptions(n_bootstraps=5),
        ),
    ]
    names = []
    for config in configs:
        (path,) = save_plots(pairs, config, output_dir=tmp_path, plot_optimal=True)
        assert path.exists()
        names.append(path.name)
    assert names[0].startswith("DeterministicFeasibilityProgress_objective_")
    assert names[1].startswith("DeterministicTerminalFeasibility_norm_")
    assert "_merit_" in names[2]
    (path,) = save_plots(
        pairs, configs[1], output_dir=tmp_path, plot_conf_ints=False, filename="no_ci.png"
    )
    assert path.exists()
