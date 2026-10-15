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
from simopt.experiment.api import AnalysisInput
from simopt.experiment.experiment import Experiment, from_names
from simopt.feasibility import feasibility_score_history
from simopt.options import CrnOptions
from tests.utils import build_analysis_input, split_pairs


def curve_points(x_vals, y_vals):
    # Reference arrays store budgets in row 0 and values in row 1.
    return np.array([x_vals, y_vals], dtype=float)


def lookup(curve, x_val):
    if x_val < curve[0][0] or x_val > curve[0][-1]:
        return np.nan
    return curve[1][np.searchsorted(curve[0], x_val, side="right") - 1]


def compute_crossing_time(curve, threshold):
    crossings = curve[0][curve[1] < threshold]
    return crossings[0] if len(crossings) else np.inf


def mean_of_curves(curves):
    unique_x_vals = np.unique(np.concatenate([curve[0] for curve in curves]))
    mean_y_vals = [np.mean([lookup(curve, x_val) for curve in curves]) for x_val in unique_x_vals]
    return curve_points(unique_x_vals, mean_y_vals)


def quantile_of_curves(curves, beta):
    unique_x_vals = np.unique(np.concatenate([curve[0] for curve in curves]))
    quantile_idx = int(beta * 99)
    quantile_y_vals = [
        quantiles([lookup(curve, x_val) for curve in curves], n=100)[quantile_idx]
        for x_val in unique_x_vals
    ]
    return curve_points(unique_x_vals, quantile_y_vals)


def difference_of_curves(curve_1, curve_2):
    unique_x_vals = np.union1d(curve_1[0], curve_2[0])
    difference_y_vals = [lookup(curve_1, x_val) - lookup(curve_2, x_val) for x_val in unique_x_vals]
    return curve_points(unique_x_vals, difference_y_vals)


def option_input(has_stochastic_constraints):
    x0 = (0.0, 0.0)
    x0_postreps = [9.0, 10.0, 11.0, 12.0]
    all_intermediate_budgets = [[0, 2 + i, 10] for i in range(4)]
    all_recommended_xs = [[x0, (float(i + 1), 0.0), (float(i + 1), 1.0)] for i in range(4)]
    all_post_replicates = [
        [
            x0_postreps,
            [3.0 + i + j / 2 for j in range(4)],
            [0.25 + i / 4 + j / 8 for j in range(4)],
        ]
        for i in range(4)
    ]
    all_stoch_constraints = [
        [
            np.array([[i - step - 1 + j / 4, i / 2 + step - 3 + j / 4] for j in range(4)])
            for step in range(3)
        ]
        for i in range(4)
    ]
    return build_analysis_input(
        all_intermediate_budgets,
        all_recommended_xs,
        all_post_replicates,
        all_stoch_constraints,
        has_stochastic_constraints,
        x0,
        (9.0, 9.0),
        x0_postreps,
        [0.5, 1.0, 1.5, 2.0],
    )


def with_full_df(data, full_df):
    return AnalysisInput(full_df, data.x0, data.x0_sample, data.xstar, data.xstar_sample)


@pytest.fixture
def option_experiment():
    experiment = from_names("RNDSRCH", "SAN-2", problem_fixed_factors={"budget": 10})
    return experiment, option_input(experiment.problem.n_stochastic_constraints > 0)


@pytest.fixture
def option_pair(option_experiment):
    _, data = option_experiment
    second = from_names("RNDSRCH", "SAN-2", problem_fixed_factors={"budget": 10})
    second.solver.name = "comparison"
    full_df = data.full_df.copy()
    step1 = full_df.index.get_level_values("step") == 1
    full_df.loc[step1, "objective"] += 1.5
    return [option_experiment, (second, with_full_df(data, full_df))]


def another_problem_pair(experiment, data):
    """Copy a pair onto a differently named problem with doubled budgets and lower objectives."""
    problem = type(experiment.problem)(name="another-problem", fixed_factors={"budget": 20})
    full_df = data.full_df.copy()
    full_df["budget"] = full_df["budget"] * 2
    step1 = full_df.index.get_level_values("step") == 1
    full_df.loc[step1, "objective"] -= 1
    return Experiment(solver=experiment.solver, problem=problem), with_full_df(data, full_df)


def objective_curves(data, normalize=False):
    """Per-macrorep objective (or progress) curves of an analysis input."""
    df = data.mean_df
    if normalize:
        budget, objective = "normalized_budget", "normalized_objective"
    else:
        budget, objective = "budget", "objective"
    return [
        curve_points(group[budget].tolist(), group[objective].tolist())
        for _, group in df.groupby("mrep")
    ]


def _steps_by_mrep(data):
    """Raw (step, rows) groups of `full_df` for each macrorep."""
    return [
        list(mrep_df.groupby(level="step")) for _, mrep_df in data.full_df.groupby(level="mrep")
    ]


def estimated_lhs(data):
    """Per-macrorep, per-step mean of the stochastic constraint left-hand sides."""
    return [
        [
            np.vstack(step_df["stochastic_constraints"].to_numpy()).mean(axis=0)
            for _, step_df in steps
        ]
        for steps in _steps_by_mrep(data)
    ]


def intermediate_budgets(data):
    return [[step_df["budget"].iloc[0] for _, step_df in steps] for steps in _steps_by_mrep(data)]


def feasibility_curves(data, method, norm_degree, two_sided):
    return [
        curve_points(x_vals, feasibility_score_history(lhs, method, norm_degree, two_sided))
        for x_vals, lhs in zip(intermediate_budgets(data), estimated_lhs(data), strict=True)
    ]


def _raw_runs(data):
    """Recover per-macrorep, per-step lists of budgets, solutions and post-replicates."""
    all_intermediate_budgets = []
    all_recommended_xs = []
    all_post_replicates = []
    all_stoch_constraints = []
    for _, mrep_df in data.full_df.groupby(level="mrep"):
        steps = [step_df for _, step_df in mrep_df.groupby(level="step")]
        all_intermediate_budgets.append([step_df["budget"].iloc[0] for step_df in steps])
        all_recommended_xs.append([tuple(step_df["solution"].iloc[0]) for step_df in steps])
        all_post_replicates.append([step_df["objective"].tolist() for step_df in steps])
        all_stoch_constraints.append(
            [np.vstack(step_df["stochastic_constraints"].to_numpy()) for step_df in steps]
        )
    return all_intermediate_budgets, all_recommended_xs, all_post_replicates, all_stoch_constraints


def reference_bootstrap_sample(
    experiment,
    data,
    bootstrap_rng,
    crn_options,
    normalize=True,
    feasibility_score_method="inf_norm",
    feasibility_norm_degree=1,
    feasibility_two_sided=False,
    disable_macrorep_bootstrap=False,
):
    """Independent reference bootstrap of objective/progress and feasibility curves."""
    (
        all_intermediate_budgets,
        all_recommended_xs,
        all_post_replicates,
        all_stoch_constraints,
    ) = _raw_runs(data)
    n_macroreps = len(all_intermediate_budgets)
    n_postreps = len(all_post_replicates[0][0])
    n_postreps_init_opt = len(data.x0_sample)
    x0 = tuple(data.x0)
    xstar = tuple(data.xstar)
    x0_postreps = data.x0_sample
    xstar_postreps = data.xstar_sample
    bootstrap_curves = []
    bootstrap_feasibility_curves = []
    has_stochastic_constraints = experiment.problem.n_stochastic_constraints >= 1

    # Uniformly resample M macroreplications (with replacement) from 0, 1, ..., M-1.
    # Subsubstream 0: reserved for this outer-level bootstrapping.
    if disable_macrorep_bootstrap:
        bs_mrep_idxs = list(range(n_macroreps))
    else:
        bs_mrep_idxs = bootstrap_rng.choices(range(n_macroreps), k=n_macroreps)
    # Advance RNG subsubstream to prepare for inner-level bootstrapping.
    bootstrap_rng.advance_subsubstream()
    # Subsubstream 1: reserved for bootstrapping at x0 and x*.
    bs_postrep_idxs = bootstrap_rng.choices(range(n_postreps_init_opt), k=n_postreps_init_opt)
    bs_initial_obj_val = np.mean([x0_postreps[postrep] for postrep in bs_postrep_idxs])
    # Reset subsubstream if using CRN across x0 and x*.
    if crn_options.across_x0_xstar:
        bootstrap_rng.reset_subsubstream()
    bs_postrep_idxs = bootstrap_rng.choices(range(n_postreps_init_opt), k=n_postreps_init_opt)
    bs_optimal_obj_val = np.mean([xstar_postreps[postrep] for postrep in bs_postrep_idxs])
    bs_initial_opt_gap = bs_initial_obj_val - bs_optimal_obj_val
    # Advance RNG subsubstream to prepare for inner-level bootstrapping.
    bootstrap_rng.advance_subsubstream()
    use_special_indices = crn_options.across_budget and not crn_options.across_macroreps
    for idx in range(n_macroreps):
        mrep = bs_mrep_idxs[idx]
        est_objectives = []
        est_lhs = []
        if use_special_indices:
            # Same postreplication indices for all intermediate budgets on a macroreplication.
            bs_postrep_idxs = bootstrap_rng.choices(range(n_postreps), k=n_postreps)
        for budget in range(len(all_intermediate_budgets[mrep])):
            if not use_special_indices:
                bs_postrep_idxs = bootstrap_rng.choices(range(n_postreps), k=n_postreps)
            if has_stochastic_constraints:
                est_lhs.append(all_stoch_constraints[mrep][budget][bs_postrep_idxs].mean(axis=0))
            else:
                est_lhs.append(np.array([]))
            if all_recommended_xs[mrep][budget] == x0:
                est_objectives.append(bs_initial_obj_val)
            elif all_recommended_xs[mrep][budget] == xstar:
                est_objectives.append(bs_optimal_obj_val)
            else:
                est_objectives.append(
                    np.mean(
                        [all_post_replicates[mrep][budget][postrep] for postrep in bs_postrep_idxs]
                    )
                )
            if not use_special_indices and crn_options.across_budget:
                bootstrap_rng.reset_subsubstream()
        if not use_special_indices:
            if crn_options.across_macroreps:
                bootstrap_rng.reset_subsubstream()
            else:
                bootstrap_rng.advance_subsubstream()
        if normalize:
            bootstrap_curves.append(
                curve_points(
                    x_vals=[
                        budget / experiment.problem.factors["budget"]
                        for budget in all_intermediate_budgets[mrep]
                    ],
                    y_vals=[
                        (est_objective - bs_optimal_obj_val) / bs_initial_opt_gap
                        for est_objective in est_objectives
                    ],
                )
            )
        else:
            bootstrap_curves.append(curve_points(all_intermediate_budgets[mrep], est_objectives))
        bootstrap_feasibility_curves.append(
            curve_points(
                all_intermediate_budgets[mrep],
                feasibility_score_history(
                    est_lhs,
                    feasibility_score_method,
                    feasibility_norm_degree,
                    feasibility_two_sided,
                ),
            )
        )
    return bootstrap_curves, bootstrap_feasibility_curves


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
    name, mode, pairs, options, crn_options, normalize, beta=0.73, solve_tolerance=0.6
):
    experiments, inputs = split_pairs(pairs)
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
    budget = max(curve[0][-1] for curve in curves)
    crossings = [compute_crossing_time(curve, solve_tolerance) for curve in curves]
    if mode == "cdf":
        finite_crossings = sorted({time for time in crossings if np.isfinite(time)})
        times = [0, *finite_crossings, 1] if budget == 1 else sorted({0, budget, *finite_crossings})
        return curve_points(
            times, [sum(time <= point for time in crossings) / len(curves) for point in times]
        )
    crossing = quantiles(crossings, n=100)[int(beta * 99)]
    if not np.isfinite(crossing):
        return curve_points([0, budget], [0, 0])
    if crossing == budget and budget != 1:
        return curve_points([0, budget], [0, 1])
    return curve_points([0, crossing, budget], [0, 1, 1])


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
    budget = max(curve[0][-1] for curve in curves)
    curves = [
        curve_points([*curve[0], budget], [*curve[1], curve[1][-1]])
        if curve[0][-1] < budget
        else curve
        for curve in curves
    ]
    return mean_of_curves(curves)
