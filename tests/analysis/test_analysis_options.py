from itertools import product

import numpy as np
import pytest

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.analysis import (
    diff,
    solvability_profile,
)
from simopt.options import ConfidenceIntervalOptions, CrnOptions
from tests.analysis.options_helpers import (
    CI_CASES,
    analyze_case,
    another_problem_pair,
    curve_reference,
    difference_of_curves,
    feasibility_curves,
    interval_reference,
    lookup,
    mean_reference,
    objective_curves,
    reference_bootstrap_sample,
)
from tests.utils import split_pairs


@pytest.mark.parametrize(("name", "mode", "normalize"), CI_CASES)
@pytest.mark.parametrize("crn_flags", list(product([False, True], repeat=3)))
@pytest.mark.parametrize("bias_correction", [False, True])
@pytest.mark.parametrize("confidence_level", [0.8, 0.95])
def test_analysis_ci_options(
    option_pair, name, mode, normalize, crn_flags, bias_correction, confidence_level
):
    options = ConfidenceIntervalOptions(
        n_bootstraps=8, confidence_level=confidence_level, bias_correction=bias_correction
    )
    crn_options = CrnOptions(*crn_flags)
    beta, solve_tolerance = 0.73, 0.6
    result = analyze_case(
        name, mode, option_pair, options, crn_options, normalize, beta, solve_tolerance
    )
    if name in ("auc", "terminal"):
        if name == "auc":
            curves = objective_curves(option_pair[0][1], normalize)
            areas = [np.sum(np.diff(curve[0]) * curve[1][:-1]) for curve in curves]
            estimator = [np.mean(areas), np.std(areas, ddof=1)]
            actual = np.array([[result.lb_mean, result.lb_std], [result.ub_mean, result.ub_std]])
            np.testing.assert_allclose([result.mean, result.std], estimator)
        else:
            estimator = [curve[1][-1] for curve in objective_curves(option_pair[0][1])] + [
                curve[1][-1] for curve in feasibility_curves(option_pair[0][1], "norm", 2, False)
            ]
            actual = np.concatenate(
                [
                    result.df[["lb_objective", "ub_objective"]].to_numpy().T,
                    result.df[["lb_feasibility", "ub_feasibility"]].to_numpy().T,
                ],
                axis=1,
            )
            np.testing.assert_allclose(
                result.df[["objective", "feasibility"]].to_numpy().T.flatten(), estimator
            )
        observations = []
        rng = MRG32k3a(s_ss_sss_index=[1, 0, 0])
        for i in range(options.n_bootstraps):
            if name == "auc":
                rng = MRG32k3a(s_ss_sss_index=[1, i, 0])
            experiment, data = option_pair[0]
            curves, bootstrap_feasibility_curves = reference_bootstrap_sample(
                experiment,
                data,
                rng,
                crn_options,
                normalize=normalize,
                feasibility_score_method="norm",
                feasibility_norm_degree=2,
                feasibility_two_sided=False,
                disable_macrorep_bootstrap=name == "terminal",
            )
            if name == "auc":
                areas = [np.sum(np.diff(curve[0]) * curve[1][:-1]) for curve in curves]
                observations.append([np.mean(areas), np.std(areas, ddof=1)])
            else:
                observations.append(
                    [curve[1][-1] for curve in curves]
                    + [curve[1][-1] for curve in bootstrap_feasibility_curves]
                )
    else:
        ci = result.ci if name in ("cdf", "profile", "difference") else result.df
        budgets = ci["budget"].to_numpy()
        originals = []
        for _, data in option_pair[: 2 if name == "difference" else 1]:
            curves = objective_curves(data, normalize)
            originals.append(
                curve_reference(
                    name,
                    mode,
                    curves,
                    feasibility_curves(data, "norm", 2, False),
                    beta,
                    solve_tolerance,
                )
            )
        original = (
            difference_of_curves(originals[1], originals[0])
            if name == "difference"
            else originals[0]
        )
        estimator = [lookup(original, budget) for budget in budgets]
        observations = []
        for i in range(options.n_bootstraps):
            samples = []
            for experiment, data in option_pair[: 2 if name == "difference" else 1]:
                rng = MRG32k3a(s_ss_sss_index=[1, i, 0])
                curves, bootstrap_feasibility_curves = reference_bootstrap_sample(
                    experiment,
                    data,
                    rng,
                    crn_options,
                    normalize=normalize,
                    feasibility_score_method="norm",
                    feasibility_norm_degree=2,
                    feasibility_two_sided=False,
                )
                samples.append(
                    curve_reference(
                        name, mode, curves, bootstrap_feasibility_curves, beta, solve_tolerance
                    )
                )
            sample = (
                difference_of_curves(samples[1], samples[0]) if name == "difference" else samples[0]
            )
            observations.append([lookup(sample, budget) for budget in budgets])
        actual = ci[["lb", "ub"]].to_numpy().T
        value_column = "cdf" if name == "cdf" else "value"
        np.testing.assert_allclose(
            result.df[value_column], [lookup(original, budget) for budget in result.df["budget"]]
        )
    expected = interval_reference(observations, estimator, confidence_level, bias_correction)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("module", [solvability_profile, diff])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("mode", ["cdf", "quantile"])
@pytest.mark.parametrize("bias_correction", [None, False, True])
def test_multiple_problem_aggregation(option_pair, module, normalize, mode, bias_correction):
    pairs = []
    for pair in option_pair:
        pairs.append(pair)
        pairs.append(another_problem_pair(*pair))
    experiments, inputs = split_pairs(pairs)
    options = (
        None
        if bias_correction is None
        else ConfidenceIntervalOptions(n_bootstraps=5, bias_correction=bias_correction)
    )
    crn_options = CrnOptions(False, True, False)
    beta, solve_tolerance = 0.3, 0.6
    args = [inputs, experiments, mode]
    if module == diff:
        args.append(experiments[0].solver.name)
    results = module.analyze_many(
        *args, options, crn_options, normalize=normalize, beta=beta, solve_tolerance=solve_tolerance
    )
    solver_curves = []
    for group in [pairs[:2], pairs[2:]]:
        curves = [
            curve_reference(
                "profile",
                mode,
                objective_curves(data, normalize),
                [],
                beta,
                solve_tolerance,
            )
            for _, data in group
        ]
        solver_curves.append(mean_reference(curves))
    expected = (
        [difference_of_curves(solver_curves[1], solver_curves[0])]
        if module == diff
        else solver_curves
    )
    assert len(results) == len(expected)
    for i, (result, curve) in enumerate(zip(results, expected, strict=True)):
        np.testing.assert_allclose(
            result.df["value"], [lookup(curve, budget) for budget in result.df["budget"]]
        )
        if options is None:
            assert result.ci is None
            continue
        budgets = result.ci["budget"]
        observations = []
        for bootstrap_index in range(options.n_bootstraps):
            samples = []
            for group in [pairs[:2], pairs[2:]]:
                curves = []
                for experiment, data in group:
                    rng = MRG32k3a(s_ss_sss_index=[1, bootstrap_index, 0])
                    bootstrap_curves, _ = reference_bootstrap_sample(
                        experiment, data, rng, crn_options, normalize=normalize
                    )
                    curves.append(
                        curve_reference(
                            "profile", mode, bootstrap_curves, [], beta, solve_tolerance
                        )
                    )
                samples.append(mean_reference(curves))
            sample = difference_of_curves(samples[1], samples[0]) if module == diff else samples[i]
            observations.append([lookup(sample, budget) for budget in budgets])
        bounds = interval_reference(
            observations,
            [lookup(curve, budget) for budget in budgets],
            options.confidence_level,
            bias_correction,
        )
        np.testing.assert_allclose(result.ci[["lb", "ub"]].to_numpy().T, bounds)
