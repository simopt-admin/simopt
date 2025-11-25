"""Area scatter plot."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from simopt.bootstrap import bootstrap_procedure
from simopt.curve import Curve
from simopt.experiment import ProblemSolver
from simopt.plot_type import PlotType

from .utils import save_plot, setup_plot


# TODO: Add the capability to compute and print the max halfwidth
# of the bootstrapped CI intervals.
def plot_area_scatterplots(
    experiments: list[list[ProblemSolver]],
    all_in_one: bool = True,
    n_bootstraps: int = 100,
    conf_level: float = 0.95,
    plot_conf_ints: bool = True,
    print_max_hw: bool = True,  # noqa: ARG001
    plot_title: str | None = None,
    legend_loc: str = "best",
    ext: str = ".png",
    save_as_pickle: bool = False,
    solver_set_name: str = "SOLVER_SET",
    problem_set_name: str = "PROBLEM_SET",
) -> list[Path]:
    """Plots scatterplots of mean vs. standard deviation of area under progress curves.

    Can generate either one plot per solver or a combined plot for all solvers.

    Note:
        The `print_max_hw` flag is currently not implemented.

    Args:
        experiments (list[list[ProblemSolver]]): Problem-solver pairs used for plotting.
        all_in_one (bool, optional): If True, plot all solvers together.
            Defaults to True.
        n_bootstraps (int, optional): Number of bootstrap samples. Defaults to 100.
        conf_level (float, optional): Confidence level for CIs (0 < conf_level < 1).
            Defaults to 0.95.
        plot_conf_ints (bool, optional): If True, show bootstrapped confidence
            intervals. Defaults to True.
        print_max_hw (bool, optional): Placeholder for printing max half-widths.
            Currently unused.
        plot_title (str, optional): Custom title for the plot
            (applies only if `all_in_one=True`).
        legend_loc (str, optional): Location of the legend
            (e.g., "best", "lower right").
        ext (str, optional): File extension for saved plots. Defaults to ".png".
        save_as_pickle (bool, optional): If True, save plot as a pickle file.
            Defaults to False.
        solver_set_name (str, optional): Label for solver group in plot titles.
            Defaults to "SOLVER_SET".
        problem_set_name (str, optional): Label for problem group in plot titles.
            Defaults to "PROBLEM_SET".

    Returns:
        list[Path]: List of file paths for the plots produced.

    Raises:
        ValueError: If `n_bootstraps` is not positive or `conf_level` is outside (0, 1).
    """
    # Value checking
    if n_bootstraps < 1:
        error_msg = "Number of bootstraps must be a positive integer."
        raise ValueError(error_msg)
    if not 0 < conf_level < 1:
        error_msg = "Confidence level must be in (0, 1)."
        raise ValueError(error_msg)

    file_list = []
    # Set up plot.
    n_solvers = len(experiments)
    n_problems = len(experiments[0])
    if all_in_one:
        marker_list = ["o", "v", "s", "*", "P", "X", "D", "V", ">", "<"]
        setup_plot(
            plot_type=PlotType.AREA,
            solver_name=solver_set_name,
            problem_name=problem_set_name,
            plot_title=plot_title,
        )
        solver_names = [
            solver_experiments[0].solver.name for solver_experiments in experiments
        ]
        solver_curve_handles = []
        # TODO: Build up capability to print max half-width.
        # if print_max_hw:
        #     curve_pairs = []
        handle = None
        for solver_idx in range(n_solvers):
            for problem_idx in range(n_problems):
                experiment = experiments[solver_idx][problem_idx]
                color_str = "C" + str(solver_idx)
                marker_str = marker_list[
                    solver_idx % len(marker_list)
                ]  # Cycle through list of marker types.
                # Plot mean and standard deviation of area under progress curve.
                areas = [
                    curve.compute_area_under_curve()
                    for curve in experiment.progress_curves
                ]
                mean_estimator = float(np.mean(areas))
                std_dev_estimator = float(np.std(areas, ddof=1))
                if plot_conf_ints:
                    mean_bs_conf_int_lb, mean_bs_conf_int_ub = bootstrap_procedure(
                        experiments=[[experiment]],
                        n_bootstraps=n_bootstraps,
                        conf_level=conf_level,
                        plot_type=PlotType.AREA_MEAN,
                        estimator=mean_estimator,
                        normalize=True,
                    )
                    std_dev_bs_conf_int_lb, std_dev_bs_conf_int_ub = (
                        bootstrap_procedure(
                            experiments=[[experiment]],
                            n_bootstraps=n_bootstraps,
                            conf_level=conf_level,
                            plot_type=PlotType.AREA_STD_DEV,
                            estimator=std_dev_estimator,
                            normalize=True,
                        )
                    )
                    # if print_max_hw:
                    #     curve_pairs.append([bs_CI_lb_curve, bs_CI_ub_curve])
                    if isinstance(mean_bs_conf_int_lb, (Curve)) or isinstance(
                        mean_bs_conf_int_ub, (Curve)
                    ):
                        error_msg = "Mean confidence intervals should be scalar values."
                        raise ValueError(error_msg)
                    if isinstance(std_dev_bs_conf_int_lb, (Curve)) or isinstance(
                        std_dev_bs_conf_int_ub, (Curve)
                    ):
                        error_msg = (
                            "Standard deviation confidence intervals should "
                            "be scalar values."
                        )
                        raise ValueError(error_msg)
                    x_err = [
                        [mean_estimator - mean_bs_conf_int_lb],
                        [mean_bs_conf_int_ub - mean_estimator],
                    ]
                    y_err_x = std_dev_estimator - std_dev_bs_conf_int_lb
                    y_err_y = std_dev_bs_conf_int_ub - std_dev_estimator
                    # If y_err_x or y_err_y is negative, set it to zero and warn.
                    if y_err_x < 0 or y_err_y < 0:
                        old_coords = (y_err_x, y_err_y)
                        y_err_x = max(0, y_err_x)
                        y_err_y = max(0, y_err_y)
                        new_coords = (y_err_x, y_err_y)
                        logging.warning(
                            "Warning: Negative error values detected in "
                            "area scatterplot. "
                            f"Old coordinates: {old_coords}, "
                            "Negative error values detected in area scatterplot "
                            "error bars. "
                            "This can occur due to statistical fluctuations in "
                            "bootstrap confidence interval estimation, especially with "
                            "small sample sizes or high variance. "
                            "Negative error bars are set to zero, which may affect the "
                            "visual interpretation of uncertainty. "
                            f"Old coordinates: {old_coords}, "
                            f"new coordinates: {new_coords}. "
                            "If this occurs frequently, consider reviewing your data "
                            "or increasing the number of replications."
                        )
                    y_err = [[y_err_x], [y_err_y]]
                    handle = plt.errorbar(
                        x=mean_estimator,
                        y=std_dev_estimator,
                        xerr=x_err,
                        yerr=y_err,
                        color=color_str,
                        marker=marker_str,
                        elinewidth=1,
                    )
                else:
                    handle = plt.scatter(
                        x=mean_estimator,
                        y=std_dev_estimator,
                        color=color_str,
                        marker=marker_str,
                    )
            solver_curve_handles.append(handle)
        plt.legend(handles=solver_curve_handles, labels=solver_names, loc=legend_loc)
        file_list.append(
            save_plot(
                solver_name=solver_set_name,
                problem_name=problem_set_name,
                plot_type=PlotType.AREA,
                normalize=True,
                plot_title=plot_title,
                ext=ext,
                save_as_pickle=save_as_pickle,
            )
        )
    else:
        for solver_idx in range(n_solvers):
            ref_experiment = experiments[solver_idx][0]
            setup_plot(
                plot_type=PlotType.AREA,
                solver_name=ref_experiment.solver.name,
                problem_name=problem_set_name,
            )
            # if print_max_hw:
            #     curve_pairs = []
            experiment = None
            for problem_idx in range(n_problems):
                experiment = experiments[solver_idx][problem_idx]
                # Plot mean and standard deviation of area under progress curve.
                areas = [
                    curve.compute_area_under_curve()
                    for curve in experiment.progress_curves
                ]
                mean_estimator = float(np.mean(areas))
                std_dev_estimator = float(np.std(areas, ddof=1))
                if plot_conf_ints:
                    mean_bs_conf_int_lb, mean_bs_conf_int_ub = bootstrap_procedure(
                        experiments=[[experiment]],
                        n_bootstraps=n_bootstraps,
                        conf_level=conf_level,
                        plot_type=PlotType.AREA_MEAN,
                        estimator=mean_estimator,
                        normalize=True,
                    )
                    std_dev_bs_conf_int_lb, std_dev_bs_conf_int_ub = (
                        bootstrap_procedure(
                            experiments=[[experiment]],
                            n_bootstraps=n_bootstraps,
                            conf_level=conf_level,
                            plot_type=PlotType.AREA_STD_DEV,
                            estimator=std_dev_estimator,
                            normalize=True,
                        )
                    )
                    # if print_max_hw:
                    #     curve_pairs.append([bs_CI_lb_curve, bs_CI_ub_curve])
                    if isinstance(mean_bs_conf_int_lb, (Curve)) or isinstance(
                        mean_bs_conf_int_ub, (Curve)
                    ):
                        error_msg = "Mean confidence intervals should be scalar values."
                        raise ValueError(error_msg)
                    if isinstance(std_dev_bs_conf_int_lb, (Curve)) or isinstance(
                        std_dev_bs_conf_int_ub, (Curve)
                    ):
                        error_msg = (
                            "Standard deviation confidence intervals should "
                            "be scalar values."
                        )
                        raise ValueError(error_msg)
                    x_err = [
                        [mean_estimator - mean_bs_conf_int_lb],
                        [mean_bs_conf_int_ub - mean_estimator],
                    ]
                    y_err = [
                        [std_dev_estimator - std_dev_bs_conf_int_lb],
                        [std_dev_bs_conf_int_ub - std_dev_estimator],
                    ]
                    handle = plt.errorbar(
                        x=mean_estimator,
                        y=std_dev_estimator,
                        xerr=x_err,
                        yerr=y_err,
                        marker="o",
                        color="C0",
                        elinewidth=1,
                    )
                else:
                    handle = plt.scatter(
                        x=mean_estimator,
                        y=std_dev_estimator,
                        color="C0",
                        marker="o",
                    )
            if experiment is not None:
                file_list.append(
                    save_plot(
                        solver_name=experiment.solver.name,
                        problem_name=problem_set_name,
                        plot_type=PlotType.AREA,
                        normalize=True,
                        ext=ext,
                        save_as_pickle=save_as_pickle,
                    )
                )
    return file_list
