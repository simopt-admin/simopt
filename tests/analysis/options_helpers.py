from copy import deepcopy
from statistics import quantiles

import numpy as np
import pytest
from scipy.stats import norm

from simopt.analysis import (
    auc,
    diff,
    feasibility_progress,
    progress_curve,
    solvability_cdf,
    solvability_profile,
    terminal_feasibility,
    terminal_progress,
    terminal_scatterplot,
)
from simopt.compat import convert
from simopt.curve import Curve
from simopt.curve_utils import (
    cdf_of_curves_crossing_times,
    mean_of_curves,
    quantile_cross_jump,
    quantile_of_curves,
)
from simopt.experiment.single import ProblemSolver
from simopt.options import CrnOptions


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


def interval_reference(observations, estimator, confidence_level, bias_correction):
    observations = np.asarray(observations)
    if not bias_correction:
        return np.quantile(
            observations, [(1 - confidence_level) / 2, (1 + confidence_level) / 2], axis=0
        )
    estimator = np.asarray(estimator)
    z0 = norm.ppf(np.mean(observations < estimator, axis=0))
    probabilities = norm.cdf(2 * z0 + np.array([-1, 1])[:, None] * norm.ppf(confidence_level))
    return np.array(
        [np.quantile(values, probabilities[:, i]) for i, values in enumerate(observations.T)]
    ).T


def analyze_case(
    name, mode, experiments, options, crn_options, normalize, beta=0.73, solve_tolerance=0.6
):
    inputs = [convert(experiment) for experiment in experiments]
    if name == "progress":
        return progress_curve.analyze(
            inputs[0], mode, options, crn_options, normalize=normalize, beta=beta
        )
    if name == "feasibility":
        return feasibility_progress.analyze(
            inputs[0],
            mode,
            options,
            crn_options,
            score_type="norm",
            norm_degree=2,
            two_sided=False,
            beta=beta,
        )
    if name == "cdf":
        return solvability_cdf.analyze(
            inputs[0], options, crn_options, normalize=normalize, solve_tolerance=solve_tolerance
        )
    if name == "profile":
        return solvability_profile.analyze_many(
            inputs[:1],
            experiments[:1],
            mode,
            options,
            crn_options,
            normalize=normalize,
            beta=beta,
            solve_tolerance=solve_tolerance,
        )[0]
    if name == "difference":
        return diff.analyze_many(
            inputs,
            experiments,
            mode,
            experiments[0].solver.name,
            options,
            crn_options,
            normalize=normalize,
            beta=beta,
            solve_tolerance=solve_tolerance,
        )[0]
    if name == "auc":
        return auc.analyze(inputs[0], options, crn_options, normalize=normalize)
    if name == "terminal":
        return terminal_feasibility.analyze(
            inputs[0], options, crn_options, score_type="norm", norm_degree=2, two_sided=False
        )
    raise AssertionError(name)


def curve_reference(name, mode, curves, feasibility_curves, beta, solve_tolerance):
    if name in ("progress", "feasibility"):
        selected = feasibility_curves if name == "feasibility" else curves
        return mean_of_curves(selected) if mode == "mean" else quantile_of_curves(selected, beta)
    budget = max(curve.x_vals[-1] for curve in curves)
    if budget != 1:
        crossings = [curve.compute_crossing_time(solve_tolerance) for curve in curves]
        if mode == "cdf":
            times = sorted({0, budget, *[time for time in crossings if np.isfinite(time)]})
            return Curve(
                times, [sum(time <= point for time in crossings) / len(curves) for point in times]
            )
        crossing = quantiles(crossings, n=100)[int(beta * 99)]
        if not np.isfinite(crossing):
            return Curve([0, budget], [0, 0])
        if crossing == budget:
            return Curve([0, budget], [0, 1])
        return Curve([0, crossing, budget], [0, 1, 1])
    if mode == "cdf":
        return cdf_of_curves_crossing_times(curves, solve_tolerance)
    return quantile_cross_jump(curves, solve_tolerance, beta)


CI_CASES = [
    (name, mode, normalize)
    for name, modes in [
        ("progress", ["mean", "quantile"]),
        ("feasibility", ["mean", "quantile"]),
        ("cdf", ["cdf"]),
        ("profile", ["cdf", "quantile"]),
        ("difference", ["cdf", "quantile"]),
        ("auc", ["summary"]),
        ("terminal", ["scatter"]),
    ]
    for mode in modes
    for normalize in ([False] if name in ("feasibility", "terminal") else [False, True])
]


def forbid_bootstrap(*args, **kwargs):
    raise AssertionError("CI-disabled analysis must not compute bootstrap samples")


def feasibility_reference(lhs, score_type, norm_degree, two_sided):
    if np.all(lhs <= 0):
        return -max(lhs) if two_sided else 0.0
    if score_type == "inf_norm":
        return -max(lhs)
    return -(sum(max(value, 0) ** norm_degree for value in lhs) ** (1 / norm_degree))


def distribution_records(monkeypatch):
    import seaborn as sns

    records = []
    for name in ["boxplot", "violinplot"]:
        original = getattr(sns, name)

        def record(*args, _original=original, **kwargs):
            records.append(np.asarray(kwargs["data"][kwargs["y"]], dtype=float))
            return _original(*args, **kwargs)

        monkeypatch.setattr(sns, name, record)
    return records


MODULES = [
    progress_curve,
    feasibility_progress,
    solvability_cdf,
    solvability_profile,
    diff,
    auc,
    terminal_feasibility,
    terminal_progress,
    terminal_scatterplot,
]


def module_results(module, inputs, experiments, options=None):
    if module == progress_curve:
        return module.analyze_many(inputs, "mean", options, CrnOptions())
    if module == feasibility_progress:
        return module.analyze_many(inputs, "mean", options, CrnOptions(), two_sided=False)
    if module in (solvability_cdf, auc, terminal_feasibility):
        return module.analyze_many(inputs, options, CrnOptions())
    if module == solvability_profile:
        return module.analyze_many(inputs, experiments, "cdf", options, CrnOptions())
    if module == diff:
        return module.analyze_many(
            inputs, experiments, "cdf", experiments[0].solver.name, options, CrnOptions()
        )
    return module.analyze_many(inputs)


def plot_module_many(module, results, experiments, **kwargs):
    if module == solvability_profile:
        return module.plot_many(results, experiments, "cdf", **kwargs)
    return module.plot_many(results, experiments, **kwargs)


def mean_reference(curves):
    # Continue each step curve through the longest raw problem budget.
    budget = max(curve.x_vals[-1] for curve in curves)
    curves = [
        Curve([*curve.x_vals, budget], [*curve.y_vals, curve.y_vals[-1]])
        if curve.x_vals[-1] < budget
        else curve
        for curve in curves
    ]
    return mean_of_curves(curves)
