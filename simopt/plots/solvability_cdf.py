"""Solvability CDF plot."""

from pathlib import Path

import matplotlib.pyplot as plt

import simopt.curve_utils as curve_utils
from simopt.bootstrap import bootstrap_procedure
from simopt.experiment import ProblemSolver
from simopt.plot_type import PlotType

from .utils import (
    check_common_problem_and_reference,
    plot_bootstrap_conf_ints,
    report_max_halfwidth,
    save_plot,
    setup_plot,
)


def plot_solvability_cdfs(
    experiments: list[ProblemSolver],
    solve_tol: float = 0.1,
    all_in_one: bool = True,
    n_bootstraps: int = 100,
    conf_level: float = 0.95,
    plot_conf_ints: bool = True,
    print_max_hw: bool = True,
    plot_title: str | None = None,
    legend_loc: str | None = None,
    ext: str = ".png",
    save_as_pickle: bool = False,
    solver_set_name: str = "SOLVER_SET",
) -> list[Path]:
    """Plots solvability CDFs for one or more solvers on a single problem.

    Args:
        experiments (list[ProblemSolver]): Problem-solver pairs for different solvers
            on a common problem.
        solve_tol (float, optional): Optimality gap that defines when a problem is
            considered solved (0 < solve_tol ≤ 1). Defaults to 0.1.
        all_in_one (bool, optional): If True, plot all curves together.
            Defaults to True.
        n_bootstraps (int, optional): Number of bootstrap samples. Defaults to 100.
        conf_level (float, optional): Confidence level for intervals
            (0 < conf_level < 1). Defaults to 0.95.
        plot_conf_ints (bool, optional): If True, include bootstrapped confidence
            intervals. Defaults to True.
        print_max_hw (bool, optional): If True, print the max half-width in the caption.
            Defaults to True.
        plot_title (str, optional): Custom title to override the generated one
            (used only if all_in_one is True).
        legend_loc (str, optional): Location of the plot legend (e.g., "best").
        ext (str, optional): File extension for saved plots. Defaults to ".png".
        save_as_pickle (bool, optional): If True, save plots as pickle files.
            Defaults to False.
        solver_set_name (str, optional): Label for solver group in plot titles.
            Defaults to "SOLVER_SET".

    Returns:
        list[Path]: List of file paths for the generated plots.

    Raises:
        ValueError: If any input parameter is out of bounds or invalid.
    """
    # Value checking
    if not 0 < solve_tol <= 1:
        error_msg = "Solve tolerance must be in (0, 1]."
        raise ValueError(error_msg)
    if n_bootstraps < 1:
        error_msg = "Number of bootstraps must be a positive integer."
        raise ValueError(error_msg)
    if not 0 < conf_level < 1:
        error_msg = "Confidence level must be in (0, 1)."
        raise ValueError(error_msg)

    if legend_loc is None:
        legend_loc = "best"

    # Check if problems are the same with the same x0 and x*.
    check_common_problem_and_reference(experiments)
    file_list = []
    # Set up plot.
    n_experiments = len(experiments)
    if all_in_one:
        ref_experiment = experiments[0]
        setup_plot(
            plot_type=PlotType.SOLVE_TIME_CDF,
            solver_name=solver_set_name,
            problem_name=ref_experiment.problem.name,
            solve_tol=solve_tol,
            plot_title=plot_title,
        )
        solver_curve_handles = []
        curve_pairs = []
        for exp_idx in range(n_experiments):
            experiment = experiments[exp_idx]
            color_str = "C" + str(exp_idx)
            # Plot cdf of solve times.
            estimator = curve_utils.cdf_of_curves_crossing_times(
                experiment.progress_curves, threshold=solve_tol
            )
            handle = estimator.plot(color_str=color_str)
            solver_curve_handles.append(handle)
            if plot_conf_ints or print_max_hw:
                bs_conf_int_lb_curve, bs_conf_int_ub_curve = bootstrap_procedure(
                    experiments=[[experiment]],
                    n_bootstraps=n_bootstraps,
                    conf_level=conf_level,
                    plot_type=PlotType.SOLVE_TIME_CDF,
                    solve_tol=solve_tol,
                    estimator=estimator,
                    normalize=True,
                )
                if plot_conf_ints:
                    if isinstance(bs_conf_int_lb_curve, (int, float)) or isinstance(
                        bs_conf_int_ub_curve, (int, float)
                    ):
                        error_msg = (
                            "Bootstrap confidence intervals are not available "
                            "for scalar estimators."
                        )
                        raise ValueError(error_msg)
                    plot_bootstrap_conf_ints(
                        bs_conf_int_lb_curve,
                        bs_conf_int_ub_curve,
                        color_str=color_str,
                    )
                if print_max_hw:
                    curve_pairs.append([bs_conf_int_lb_curve, bs_conf_int_ub_curve])
        plt.legend(
            handles=solver_curve_handles,
            labels=[experiment.solver.name for experiment in experiments],
            loc=legend_loc,
        )
        if print_max_hw:
            report_max_halfwidth(
                curve_pairs=curve_pairs, normalize=True, conf_level=conf_level
            )
        file_list.append(
            save_plot(
                solver_name="SOLVER SET",
                problem_name=ref_experiment.problem.name,
                plot_type=PlotType.SOLVE_TIME_CDF,
                normalize=True,
                extra=solve_tol,
                plot_title=plot_title,
                ext=ext,
                save_as_pickle=save_as_pickle,
            )
        )
    else:  # Plot separately.
        for experiment in experiments:
            setup_plot(
                plot_type=PlotType.SOLVE_TIME_CDF,
                solver_name=experiment.solver.name,
                problem_name=experiment.problem.name,
                solve_tol=solve_tol,
            )
            estimator = curve_utils.cdf_of_curves_crossing_times(
                experiment.progress_curves, threshold=solve_tol
            )
            estimator.plot()
            if plot_conf_ints or print_max_hw:
                bs_conf_int_lb_curve, bs_conf_int_ub_curve = bootstrap_procedure(
                    experiments=[[experiment]],
                    n_bootstraps=n_bootstraps,
                    conf_level=conf_level,
                    plot_type=PlotType.SOLVE_TIME_CDF,
                    solve_tol=solve_tol,
                    estimator=estimator,
                    normalize=True,
                )
                if plot_conf_ints:
                    if isinstance(bs_conf_int_lb_curve, (int, float)) or isinstance(
                        bs_conf_int_ub_curve, (int, float)
                    ):
                        error_msg = (
                            "Bootstrap confidence intervals are not available "
                            "for scalar estimators."
                        )
                        raise ValueError(error_msg)
                    plot_bootstrap_conf_ints(bs_conf_int_lb_curve, bs_conf_int_ub_curve)
                if print_max_hw:
                    if isinstance(bs_conf_int_lb_curve, (int, float)) or isinstance(
                        bs_conf_int_ub_curve, (int, float)
                    ):
                        error_msg = (
                            "Max halfwidth is not available for scalar estimators."
                        )
                        raise ValueError(error_msg)
                    report_max_halfwidth(
                        curve_pairs=[[bs_conf_int_lb_curve, bs_conf_int_ub_curve]],
                        normalize=True,
                        conf_level=conf_level,
                    )
            file_list.append(
                save_plot(
                    solver_name=experiment.solver.name,
                    problem_name=experiment.problem.name,
                    plot_type=PlotType.SOLVE_TIME_CDF,
                    normalize=True,
                    extra=solve_tol,
                    ext=ext,
                    save_as_pickle=save_as_pickle,
                )
            )
    return file_list
