"""Bootstrapping procedures."""

from typing import Literal

import numpy as np

import simopt.curve_utils as curve_utils
from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.curve import Curve
from simopt.experiment import ProblemSolver
from simopt.plot_type import PlotType


def bootstrap_sample_all(
    experiments: list[list[ProblemSolver]],
    bootstrap_rng: MRG32k3a,
    normalize: bool = True,
    feasibility_score_method: Literal["inf_norm", "norm"] = "inf_norm",
    feasibility_norm_degree: int = 1,
    feasibility_two_sided: bool = False,
) -> tuple[list[list[list[Curve]]], list[list[list[Curve]]]]:
    """Generates bootstrap samples of progress and feasibility curves.

    Args:
        experiments (list[list[ProblemSolver]]): Grid of problem-solver pairs, where
            each inner list corresponds to different problems for a given solver.
        bootstrap_rng (MRG32k3a): Random number generator used for bootstrapping.
        normalize (bool, optional): If True, normalize progress curves by optimality
            gaps. Defaults to True.
        feasibility_score_method (Literal["inf_norm", "norm"], optional): Feasibility
            scoring rule.
        feasibility_norm_degree (int, optional): Degree of the norm when
            ``feasibility_score_method == "norm"``.
        feasibility_two_sided (bool, optional): Whether to give feasible solutions a
            non-zero score based on the best violation.

    Returns:
        tuple[list[list[list[Curve]]], list[list[list[Curve]]]]:
            Bootstrapped progress/objective curves and feasibility curves for all
            solutions from all macroreplications, grouped by solver and problem.
    """
    n_solvers = len(experiments)
    n_problems = len(experiments[0])
    bootstrap_curves = [[[] for _ in range(n_problems)] for _ in range(n_solvers)]
    bootstrap_feasibility_curves = [
        [[] for _ in range(n_problems)] for _ in range(n_solvers)
    ]
    # Obtain a bootstrap sample from each experiment.
    for solver_idx in range(n_solvers):
        for problem_idx in range(n_problems):
            experiment = experiments[solver_idx][problem_idx]
            objective_curves, feasibility_curves = experiment.bootstrap_sample(
                bootstrap_rng,
                normalize,
                feasibility_score_method,
                feasibility_norm_degree,
                feasibility_two_sided,
            )
            bootstrap_curves[solver_idx][problem_idx] = objective_curves
            bootstrap_feasibility_curves[solver_idx][problem_idx] = feasibility_curves
            # Reset substream for next solver-problem pair.
            bootstrap_rng.reset_substream()
    # Advance substream of random number generator to prepare for next bootstrap sample.
    bootstrap_rng.advance_substream()
    return bootstrap_curves, bootstrap_feasibility_curves


def bootstrap_procedure(
    experiments: list[list[ProblemSolver]],
    n_bootstraps: int,
    conf_level: float,
    plot_type: PlotType,
    beta: float | None = None,
    solve_tol: float | None = None,
    estimator: float | Curve | None = None,
    normalize: bool = True,
    feasibility_score_method: Literal["inf_norm", "norm"] = "inf_norm",
    feasibility_norm_degree: int = 1,
    feasibility_two_sided: bool = False,
) -> tuple[float, float] | tuple[Curve, Curve]:
    """Performs bootstrapping and computes confidence intervals for progress curves.

    Args:
        experiments (list[list[ProblemSolver]]): Grid of problem-solver pairs.
        n_bootstraps (int): Number of bootstrap samples to generate.
        conf_level (float): Confidence level for the interval (0 < conf_level < 1).
        plot_type (PlotType): Type of plot/metric for which to compute the interval.
        beta (float, optional): Quantile level (0 < beta < 1), used with some plot
            types.
        solve_tol (float, optional): Relative optimality gap that defines a "solved"
            instance.
        estimator (float or Curve, optional): Reference estimator for difference plot
            types.
        normalize (bool, optional): Whether to normalize progress curves. Defaults to
            True.
        feasibility_score_method (Literal["inf_norm", "norm"], optional): Feasibility
            scoring rule used when `plot_type` corresponds to feasibility metrics.
        feasibility_norm_degree (int, optional): Degree of the norm when
            ``feasibility_score_method == "norm"``.
        feasibility_two_sided (bool, optional): Whether to assign a non-zero score to
            feasible solutions based on the best violation.

    Returns:
        tuple[float, float] or tuple[Curve, Curve]: Lower and upper bounds of the CI.
    """
    # Value checking
    if n_bootstraps < 1:
        error_msg = "Number of bootstraps must be a positive integer."
        raise ValueError(error_msg)
    if not 0 < conf_level < 1:
        error_msg = "Confidence level must be in (0, 1)."
        raise ValueError(error_msg)
    if beta is not None and not 0 < beta < 1:
        error_msg = "Beta quantile must be in (0, 1)."
        raise ValueError(error_msg)
    if solve_tol is not None and not 0 < solve_tol <= 1:
        error_msg = "Solve tolerance must be in (0, 1]."
        raise ValueError(error_msg)
    acceptable_plot_types = [
        PlotType.MEAN,
        PlotType.QUANTILE,
        PlotType.AREA_MEAN,
        PlotType.AREA_STD_DEV,
        PlotType.SOLVE_TIME_QUANTILE,
        PlotType.SOLVE_TIME_CDF,
        PlotType.CDF_SOLVABILITY,
        PlotType.QUANTILE_SOLVABILITY,
        PlotType.DIFFERENCE_OF_CDF_SOLVABILITY,
        PlotType.DIFFERENCE_OF_QUANTILE_SOLVABILITY,
        PlotType.MEAN_FEASIBILITY_PROGRESS,
        PlotType.QUANTILE_FEASIBILITY_PROGRESS,
    ]
    if plot_type not in acceptable_plot_types:
        error_msg = (
            f"Plot type must be one of {acceptable_plot_types}.\nReceived: {plot_type}"
        )
        raise ValueError(error_msg)

    # Create random number generator for bootstrap sampling.
    # Stream 1 dedicated for bootstrapping.
    bootstrap_rng = MRG32k3a(s_ss_sss_index=[1, 0, 0])
    # Obtain n_bootstrap replications.
    bootstrap_replications = []
    for _ in range(n_bootstraps):
        # Generate bootstrap sample of estimated objective/progress curves.
        bootstrap_curves, bootstrap_feasibility_curves = bootstrap_sample_all(
            experiments,
            bootstrap_rng,
            normalize,
            feasibility_score_method,
            feasibility_norm_degree,
            feasibility_two_sided,
        )
        if plot_type in (
            PlotType.MEAN_FEASIBILITY_PROGRESS,
            PlotType.QUANTILE_FEASIBILITY_PROGRESS,
        ):
            curves = bootstrap_feasibility_curves
        else:
            curves = bootstrap_curves
        # Apply the functional of the bootstrap sample.
        bootstrap_replications.append(
            functional_of_curves(curves, plot_type, beta=beta, solve_tol=solve_tol)
        )
    # Distinguish cases where functional returns a scalar vs a curve.
    if plot_type in [
        PlotType.AREA_MEAN,
        PlotType.AREA_STD_DEV,
        PlotType.SOLVE_TIME_QUANTILE,
    ]:
        if estimator is None:
            error_msg = (
                "Estimator must be provided for functional that returns a scalar."
            )
            raise ValueError(error_msg)
        if isinstance(estimator, Curve):
            error_msg = (
                "Estimator must be a scalar for functional that returns a scalar."
            )
            raise ValueError(error_msg)
        # Functional returns a scalar.
        computed_bootstrap = compute_bootstrap_conf_int(
            bootstrap_replications,
            conf_level=conf_level,
            bias_correction=True,
            overall_estimator=estimator,
        )
        # Get the first and second float values from the computed bootstrap.
        float_1 = computed_bootstrap[0]
        float_2 = computed_bootstrap[1]
        # Keep indexing into them until they are floats.
        while not isinstance(float_1, (int, float)):
            float_1 = float_1[0]
        while not isinstance(float_2, (int, float)):
            float_2 = float_2[0]
        return float_1, float_2
    # Functional returns a curve.
    unique_budget_list = list(
        np.unique(
            [budget for curve in bootstrap_replications for budget in curve.x_vals]
        )
    )
    bs_conf_int_lower_bound_list: list[np.ndarray] = []
    bs_conf_int_upper_bound_list: list[np.ndarray] = []
    for budget in unique_budget_list:
        budget_float = float(budget)
        bootstrap_subreplications = [
            curve.lookup(budget_float) for curve in bootstrap_replications
        ]
        if estimator is None:
            error_msg = (
                "Estimator must be provided for functional that returns a curve."
            )
            raise ValueError(error_msg)
        if isinstance(estimator, (int, float)):
            error_msg = (
                "Estimator must be a Curve object for functional that returns a curve."
            )
            raise ValueError(error_msg)
        sub_estimator = estimator.lookup(budget_float)
        bs_conf_int_lower_bound, bs_conf_int_upper_bound = compute_bootstrap_conf_int(
            bootstrap_subreplications,
            conf_level=conf_level,
            bias_correction=True,
            overall_estimator=sub_estimator,
        )
        bs_conf_int_lower_bound_list.append(bs_conf_int_lower_bound)
        bs_conf_int_upper_bound_list.append(bs_conf_int_upper_bound)
    # Create the curves for the lower and upper bounds of the bootstrap
    # confidence intervals.
    unique_budget_list_floats = [float(val) for val in unique_budget_list]
    lower_bound_list = [float(val) for val in bs_conf_int_lower_bound_list]
    bs_conf_int_lower_bounds = Curve(
        x_vals=unique_budget_list_floats, y_vals=lower_bound_list
    )
    upper_bound_list = [float(val) for val in bs_conf_int_upper_bound_list]
    bs_conf_int_upper_bounds = Curve(
        x_vals=unique_budget_list_floats, y_vals=upper_bound_list
    )
    return bs_conf_int_lower_bounds, bs_conf_int_upper_bounds


def functional_of_curves(
    bootstrap_curves: list[list[list[Curve]]],
    plot_type: PlotType,
    beta: float | None = 0.5,
    solve_tol: float | None = 0.1,
) -> float | Curve:
    """Computes a functional of bootstrapped objective or progress curves.

    Args:
        bootstrap_curves (list[list[list[Curve]]]): Bootstrapped curves for all
            solutions across all macroreplications.
        plot_type (PlotType): Type of functional to compute:
            - PlotType.MEAN
            - PlotType.QUANTILE
            - PlotType.AREA_MEAN
            - PlotType.AREA_STD_DEV
            - PlotType.SOLVE_TIME_QUANTILE
            - PlotType.SOLVE_TIME_CDF
            - PlotType.CDF_SOLVABILITY
            - PlotType.QUANTILE_SOLVABILITY
            - PlotType.DIFFERENCE_OF_CDF_SOLVABILITY
            - PlotType.DIFFERENCE_OF_QUANTILE_SOLVABILITY
        beta (float, optional): Quantile level (0 < beta < 1). Defaults to 0.5.
        solve_tol (float, optional): Optimality gap for defining a solved instance
            (0 < solve_tol ≤ 1). Defaults to 0.1.

    Returns:
        Curve or float: The computed functional of the curves.

    Raises:
        ValueError: If input values are invalid or unsupported for the given plot_type.
    """
    # Set default arguments
    if beta is None:
        beta = 0.5
    if solve_tol is None:
        solve_tol = 0.1
    # Value checking
    if not 0 < beta < 1:
        error_msg = "Beta quantile must be in (0, 1)."
        raise ValueError(error_msg)
    if not 0 < solve_tol <= 1:
        error_msg = "Solve tolerance must be in (0, 1]."
        raise ValueError(error_msg)

    single_curves = bootstrap_curves[0][0]
    solver_1_curves = bootstrap_curves[0]
    solver_2_curves = bootstrap_curves[1] if len(bootstrap_curves) > 1 else None

    dispatch = {
        PlotType.MEAN: lambda: curve_utils.mean_of_curves(single_curves),
        PlotType.QUANTILE: lambda: curve_utils.quantile_of_curves(
            single_curves, beta=beta
        ),
        PlotType.AREA_MEAN: lambda: float(
            np.mean([c.compute_area_under_curve() for c in single_curves])
        ),
        PlotType.AREA_STD_DEV: lambda: float(
            np.std([c.compute_area_under_curve() for c in single_curves], ddof=1)
        ),
        PlotType.SOLVE_TIME_QUANTILE: lambda: float(
            np.quantile(
                [c.compute_crossing_time(threshold=solve_tol) for c in single_curves],
                q=beta,
            )
        ),
        PlotType.SOLVE_TIME_CDF: lambda: curve_utils.cdf_of_curves_crossing_times(
            single_curves, threshold=solve_tol
        ),
        PlotType.CDF_SOLVABILITY: lambda: curve_utils.mean_of_curves(
            [
                curve_utils.cdf_of_curves_crossing_times(curves, threshold=solve_tol)
                for curves in solver_1_curves
            ]
        ),
        PlotType.QUANTILE_SOLVABILITY: lambda: curve_utils.mean_of_curves(
            [
                curve_utils.quantile_cross_jump(curves, threshold=solve_tol, beta=beta)
                for curves in solver_1_curves
            ]
        ),
        PlotType.DIFFERENCE_OF_CDF_SOLVABILITY: lambda: curve_utils.difference_of_curves(  # noqa: E501
            curve_utils.mean_of_curves(
                [
                    curve_utils.cdf_of_curves_crossing_times(
                        curves, threshold=solve_tol
                    )
                    for curves in solver_1_curves
                ]
            ),
            curve_utils.mean_of_curves(
                [
                    curve_utils.cdf_of_curves_crossing_times(
                        curves, threshold=solve_tol
                    )
                    for curves in solver_2_curves  # type: ignore
                ]
            ),
        ),
        PlotType.DIFFERENCE_OF_QUANTILE_SOLVABILITY: lambda: curve_utils.difference_of_curves(  # noqa: E501
            curve_utils.mean_of_curves(
                [
                    curve_utils.quantile_cross_jump(
                        curves, threshold=solve_tol, beta=beta
                    )
                    for curves in solver_1_curves
                ]
            ),
            curve_utils.mean_of_curves(
                [
                    curve_utils.quantile_cross_jump(
                        curves, threshold=solve_tol, beta=beta
                    )
                    for curves in solver_2_curves  # type: ignore
                ]
            ),
        ),
        PlotType.MEAN_FEASIBILITY_PROGRESS: lambda: curve_utils.mean_of_curves(
            single_curves
        ),
        PlotType.QUANTILE_FEASIBILITY_PROGRESS: lambda: curve_utils.quantile_of_curves(
            single_curves, beta=beta
        ),
    }

    try:
        return dispatch[plot_type]()
    except KeyError as e:
        raise NotImplementedError(f"'{plot_type.value}' is not implemented.") from e


# TODO: double check observations type and return type
def compute_bootstrap_conf_int(
    observations: list[float | int],
    conf_level: float,
    bias_correction: bool = True,
    overall_estimator: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct a bootstrap confidence interval for an estimator.

    Args:
        observations (list[float | int]): Estimators from all bootstrap instances.
        conf_level (float): Confidence level for confidence intervals, i.e., 1 - gamma;
            must be in (0, 1).
        bias_correction (bool, optional): Whether to use bias-corrected bootstrap CIs
            (via the percentile method). Defaults to True.
        overall_estimator (float | None, optional): The estimator to compute the CI
            around. Required if`bias_correction` is True.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the lower and upper bounds of
            the bootstrap confidence interval.

    Raises:
        ValueError: If `conf_level` is not in (0, 1), or if `overall_estimator` is None
            when `bias_correction` is True.
    """
    # Value checking
    if not 0 < conf_level < 1:
        error_msg = "Confidence level must be in (0, 1)."
        raise ValueError(error_msg)
    if bias_correction and overall_estimator is None:
        error_msg = "Overall estimator must be provided for bias correction."
        raise ValueError(error_msg)

    # Compute bootstrapping confidence interval via percentile method.
    # See Efron (1981) "Nonparameteric Standard Errors and Confidence Intervals."
    if bias_correction:
        if overall_estimator is None:
            error_msg = "Overall estimator must be provided for bias correction."
            raise ValueError(error_msg)
        # Lazy imports
        from scipy.stats import norm

        # For biased-corrected CIs, see equation (4.4) on page 146.
        z0 = norm.ppf(np.mean([obs < overall_estimator for obs in observations]))
        zconflvl = norm.ppf(conf_level)
        q_lower = norm.cdf(2 * z0 - zconflvl)
        q_upper = norm.cdf(2 * z0 + zconflvl)
    else:
        # For uncorrected CIs, see equation (4.3) on page 146.
        q_lower = (1 - conf_level) / 2
        q_upper = 1 - (1 - conf_level) / 2
    bs_conf_int_lower_bound = np.quantile(observations, q=q_lower)
    bs_conf_int_upper_bound = np.quantile(observations, q=q_upper)
    # Sometimes quantile returns a scalar, so convert to array.
    if not isinstance(bs_conf_int_lower_bound, np.ndarray):
        bs_conf_int_lower_bound = np.array([bs_conf_int_lower_bound])
    if not isinstance(bs_conf_int_upper_bound, np.ndarray):
        bs_conf_int_upper_bound = np.array([bs_conf_int_upper_bound])
    return bs_conf_int_lower_bound, bs_conf_int_upper_bound
