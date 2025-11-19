"""Plotting utilities."""

import pickle
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt

import simopt.curve_utils as curve_utils
from simopt.curve import Curve, CurveType
from simopt.experiment import EXPERIMENT_DIR, ProblemSolver
from simopt.plot_type import PlotType


def setup_plot(
    plot_type: PlotType,
    solver_name: str = "SOLVER SET",
    problem_name: str = "PROBLEM SET",
    normalize: bool = True,
    budget: int | None = None,
    beta: float | None = None,
    feasibility_score_method: Literal["inf_norm", "norm"] = "inf_norm",
    feasibility_norm_degree: int = 1,
    solve_tol: float | None = None,
    plot_title: str | None = None,
) -> None:
    """Create a new figure, add labels to the plot, and reformat axes.

    Args:
        plot_type (PlotType): Type of plot to produce. Valid options include:
            - ALL: All estimated progress curves.
            - MEAN: Estimated mean progress curve.
            - QUANTILE: Estimated beta quantile progress curve.
            - SOLVE_TIME_CDF: CDF of solve time.
            - CDF_SOLVABILITY: CDF solvability profile.
            - QUANTILE_SOLVABILITY: Quantile solvability profile.
            - DIFFERENCE_OF_CDF_SOLVABILITY: Difference of CDF solvability profiles.
            - DIFFERENCE_OF_QUANTILE_SOLVABILITY: Difference of quantile solvability
            profiles.
            - AREA: Area scatterplot.
            - BOX: Box plot of terminal progress.
            - VIOLIN: Violin plot of terminal progress.
            - TERMINAL_SCATTER: Scatterplot of mean and std dev of terminal progress.
        solver_name (str, optional): Name of the solver. Defaults to "SOLVER SET".
        problem_name (str, optional): Name of the problem. Defaults to "PROBLEM SET".
        normalize (bool, optional): Whether to normalize with respect to optimality
            gaps. Defaults to True.
        budget (int, optional): Function evaluation budget.
        beta (float, optional): Quantile to compute (must be in (0, 1)).
        feasibility_score_method (Literal["inf_norm", "norm"], optional): Method to
            compute the feasibility score. Defaults to "inf_norm".
        feasibility_norm_degree (int, optional): Degree of the norm to use for the
            feasibility score. Defaults to 1.
        solve_tol (float, optional): Relative optimality gap for declaring a solve
            (must be in (0, 1]).
        plot_title (str, optional): Title to override the automatically generated one.

    Raises:
        ValueError: If any inputs are invalid.
    """
    # Value checking
    if isinstance(beta, float) and not 0 < beta < 1:
        error_msg = "Beta must be in (0, 1)."
        raise ValueError(error_msg)
    if isinstance(solve_tol, float) and not 0 < solve_tol <= 1:
        error_msg = "Solve tolerance must be in (0, 1]."
        raise ValueError(error_msg)

    plt.figure()
    feasibility_plots = {
        PlotType.FEASIBILITY_SCATTER,
        PlotType.FEASIBILITY_VIOLIN,
        PlotType.ALL_FEASIBILITY_PROGRESS,
        PlotType.MEAN_FEASIBILITY_PROGRESS,
        PlotType.QUANTILE_FEASIBILITY_PROGRESS,
    }
    # Set up axes and axis labels.
    if plot_type not in feasibility_plots:
        if normalize:
            plt.ylabel("Fraction of Initial Optimality Gap", size=14)
            if plot_type != PlotType.BOX and plot_type != PlotType.VIOLIN:
                plt.xlabel("Fraction of Budget", size=14)
                plt.xlim((0, 1))
                plt.ylim((-0.1, 1.1))
                plt.tick_params(axis="both", which="major", labelsize=12)
        else:
            plt.ylabel("Objective Function Value", size=14)
            if plot_type != PlotType.BOX and plot_type != PlotType.VIOLIN:
                plt.xlabel("Budget", size=14)
                plt.xlim((0, budget))
                plt.tick_params(axis="both", which="major", labelsize=12)
    # Specify title (plus alternative y-axis label and alternative axes).
    if plot_type == PlotType.ALL:
        title = f"{solver_name} on {problem_name}\n"
        title += "Progress Curves" if normalize else "Objective Curves"
    elif plot_type == PlotType.MEAN:
        title = f"{solver_name} on {problem_name}\n"
        title += "Mean Progress Curve" if normalize else "Mean Objective Curve"
    elif plot_type == PlotType.QUANTILE:
        if beta is None:
            error_msg = "Beta must be specified for quantile plot."
            raise ValueError(error_msg)
        beta_rounded = round(beta, 2)
        title = f"{solver_name} on {problem_name}\n{beta_rounded}-Quantile "
        title += "Progress Curve" if normalize else "Objective Curve"
    elif plot_type == PlotType.SOLVE_TIME_CDF:
        if solve_tol is None:
            error_msg = "Solve tolerance must be specified for cdf plot."
            raise ValueError(error_msg)
        plt.ylabel("Fraction of Macroreplications Solved", size=14)
        solve_tol_rounded = round(solve_tol, 2)
        title = f"{solver_name} on {problem_name}\n"
        title += f"CDF of {solve_tol_rounded}-Solve Times"
    elif plot_type == PlotType.CDF_SOLVABILITY:
        if solve_tol is None:
            error_msg = "Solve tolerance must be specified for cdf solvability plot."
            raise ValueError(error_msg)
        plt.ylabel("Problem Averaged Solve Fraction", size=14)
        title = (
            f"CDF-Solvability Profile for {solver_name}\n"
            f"Profile of CDFs of {round(solve_tol, 2)}-Solve Times"
        )
    elif plot_type == PlotType.QUANTILE_SOLVABILITY:
        if beta is None:
            error_msg = "Beta must be specified for quantile solvability plot."
            raise ValueError(error_msg)
        if solve_tol is None:
            error_msg = (
                "Solve tolerance must be specified for quantile solvability plot."
            )
            raise ValueError(error_msg)
        plt.ylabel("Fraction of Problems Solved", size=14)
        title = (
            f"Quantile Solvability Profile for {solver_name}\n"
            f"Profile of {round(beta, 2)}-Quantiles "
            f"of {round(solve_tol, 2)}-Solve Times"
        )
    elif plot_type == PlotType.DIFFERENCE_OF_CDF_SOLVABILITY:
        if solve_tol is None:
            error_msg = "Solve tolerance must be specified for cdf solvability plot."
            raise ValueError(error_msg)
        plt.ylabel("Difference in Problem Averaged Solve Fraction", size=14)
        title = (
            f"Difference of CDF-Solvability Profile for {solver_name}\n"
            f"Difference of Profiles of CDFs of {round(solve_tol, 2)}-Solve Times"
        )
        plt.plot([0, 1], [0, 0], color="black", linestyle="--")
        plt.ylim((-1, 1))
    elif plot_type == PlotType.DIFFERENCE_OF_QUANTILE_SOLVABILITY:
        if beta is None:
            error_msg = "Beta must be specified for quantile solvability plot."
            raise ValueError(error_msg)
        if solve_tol is None:
            error_msg = (
                "Solve tolerance must be specified for quantile solvability plot."
            )
            raise ValueError(error_msg)
        plt.ylabel("Difference in Fraction of Problems Solved", size=14)
        title = (
            f"Difference of Quantile Solvability Profile for {solver_name}\n"
            f"Difference of Profiles of {round(beta, 2)}-Quantiles "
            f"of {round(solve_tol, 2)}-Solve Times"
        )
        plt.plot([0, 1], [0, 0], color="black", linestyle="--")
        plt.ylim((-1, 1))
    elif plot_type == PlotType.AREA:
        plt.xlabel("Mean Area", size=14)
        plt.ylabel("Std Dev of Area")
        # plt.xlim((0, 1))
        # plt.ylim((0, 0.5))
        title = f"{solver_name}\nAreas Under Progress Curves"
    elif plot_type == PlotType.BOX or plot_type == PlotType.VIOLIN:
        plt.xlabel("Solvers")
        if normalize:
            plt.ylabel("Terminal Progress")
            title = f"{solver_name} on {problem_name}"
        else:
            plt.ylabel("Terminal Objective")
            title = f"{solver_name} on {problem_name}"
    elif plot_type == PlotType.TERMINAL_SCATTER:
        plt.xlabel("Mean Terminal Progress", size=14)
        plt.ylabel("Std Dev of Terminal Progress")
        # plt.xlim((0, 1))
        # plt.ylim((0, 0.5))
        title = f"{solver_name}\nTerminal Progress"
    elif plot_type == PlotType.FEASIBILITY_SCATTER:
        plt.xlabel("Terminal Objective Value", size=14)
        if feasibility_score_method == "inf_norm":
            ylabel = "Terminal $L^\\infty$ Feasibility Score"
        elif feasibility_score_method == "norm":
            ylabel = f"Terminal $L^{feasibility_norm_degree}$ Feasibility Score"
        else:
            ylabel = "Terminal Feasibility Score"
        plt.ylabel(ylabel, size=14)
        plt.tick_params(axis="both", which="major", labelsize=12)
        title = f"{solver_name} on {problem_name} \n Terminal Objective vs Feasibility"
    elif plot_type == PlotType.FEASIBILITY_VIOLIN:
        plt.xlabel("Solvers")
        if feasibility_score_method == "inf_norm":
            ylabel = "Terminal $L^\\infty$ Feasibility Score"
        elif feasibility_score_method == "norm":
            ylabel = f"Terminal $L^{feasibility_norm_degree}$ Feasibility Score"
        else:
            ylabel = "Terminal Feasibility Score"
        plt.ylabel(ylabel, size=14)
        title = f"{solver_name} on {problem_name} \n Terminal Feasibility"
    elif plot_type == PlotType.ALL_FEASIBILITY_PROGRESS:
        plt.xlabel("Budget", size=14)
        if budget is not None:
            plt.xlim((0, budget))
        if feasibility_score_method == "inf_norm":
            ylabel = "$L^\\infty$ Feasibility Score"
        elif feasibility_score_method == "norm":
            ylabel = f"$L^{feasibility_norm_degree}$ Feasibility Score"
        else:
            ylabel = "Feasibility Score"
        plt.ylabel(ylabel, size=14)
        plt.tick_params(axis="both", which="major", labelsize=12)
        title = f"{solver_name} on {problem_name} \n Feasibility Progress Curves"
    elif plot_type == PlotType.MEAN_FEASIBILITY_PROGRESS:
        plt.xlabel("Budget", size=14)
        if budget is not None:
            plt.xlim((0, budget))
        if feasibility_score_method == "inf_norm":
            ylabel = "$L^\\infty$ Feasibility Score"
        elif feasibility_score_method == "norm":
            ylabel = f"$L^{feasibility_norm_degree}$ Feasibility Score"
        else:
            ylabel = "Feasibility Score"
        plt.ylabel(ylabel, size=14)
        plt.tick_params(axis="both", which="major", labelsize=12)
        title = f"{solver_name} on {problem_name} \n Mean Feasibility Progress Curves"
    elif plot_type == PlotType.QUANTILE_FEASIBILITY_PROGRESS:
        if beta is None:
            error_msg = "Beta must be specified for quantile feasibility plot."
            raise ValueError(error_msg)
        plt.xlabel("Budget", size=14)
        if budget is not None:
            plt.xlim((0, budget))
        if feasibility_score_method == "inf_norm":
            ylabel = "$L^\\infty$ Feasibility Score"
        elif feasibility_score_method == "norm":
            ylabel = f"$L^{feasibility_norm_degree}$ Feasibility Score"
        else:
            ylabel = "Feasibility Score"
        plt.ylabel(ylabel, size=14)
        plt.tick_params(axis="both", which="major", labelsize=12)
        title = (
            f"{solver_name} on {problem_name} \n "
            f"{round(beta, 2)} Quantile Feasibility Progress Curves"
        )
    else:
        error_msg = f"'{plot_type}' is not implemented."
        raise NotImplementedError(error_msg)
    # if title argument provided, overide prevous title assignment
    if plot_title is not None:
        title = plot_title
    plt.title(title, size=14)


def save_plot(
    solver_name: str,
    problem_name: str,
    plot_type: PlotType,
    normalize: bool,
    extra: float | list[float] | None = None,
    plot_title: str | None = None,
    ext: str = ".png",
    save_as_pickle: bool = False,
) -> Path:
    """Create and save a plot with appropriate labels and formatting.

    Args:
        solver_name (str): Name of the solver.
        problem_name (str): Name of the problem.
        plot_type (PlotType): Type of plot to produce. Valid options include:
            - ALL: All estimated progress curves.
            - MEAN: Estimated mean progress curve.
            - QUANTILE: Estimated beta quantile progress curve.
            - SOLVE_TIME_CDF: CDF of solve time.
            - CDF_SOLVABILITY: CDF solvability profile.
            - QUANTILE_SOLVABILITY: Quantile solvability profile.
            - DIFFERENCE_OF_CDF_SOLVABILITY: Difference of CDF solvability profiles.
            - DIFFERENCE_OF_QUANTILE_SOLVABILITY: Difference of quantile solvability
            profiles.
            - AREA: Area scatterplot.
            - TERMINAL_SCATTER: Scatterplot of mean and std dev of terminal progress.
            - FEASIBILITY_SCATTER: Scatterplot of terminal objective vs feasibility.
            - FEASIBILITY_VIOLIN: Violin plot of terminal feasibility.
            - ALL_FEASIBILITY_PROGRESS: Feasibility progress curves for all macroreps.
            - MEAN_FEASIBILITY_PROGRESS: Mean feasibility progress curve.
            - QUANTILE_FEASIBILITY_PROGRESS: Quantile feasibility progress curve.
        normalize (bool): Whether to normalize with respect to optimality gaps.
        extra (float | list[float], optional): Extra number(s) specifying quantile
            (e.g., beta) and/or solve tolerance.
        plot_title (str | None, optional): If provided, overrides the default title
            and filename.
        ext (str, optional): File extension for the saved plot. Defaults to ".png".
        save_as_pickle (bool, optional): Whether to save the plot as a pickle file.
            Defaults to False.

    Returns:
        Path: Path pointing to the location where the plot will be saved.
    """
    # Form string name for plot filename.
    if plot_type == PlotType.ALL:
        plot_name = "all_prog_curves"
    elif plot_type == PlotType.MEAN:
        plot_name = "mean_prog_curve"
    elif plot_type == PlotType.QUANTILE:
        plot_name = f"{extra}_quantile_prog_curve"
    elif plot_type == PlotType.SOLVE_TIME_CDF:
        plot_name = f"cdf_{extra}_solve_times"
    elif plot_type == PlotType.CDF_SOLVABILITY:
        plot_name = f"profile_cdf_{extra}_solve_times"
    elif plot_type == PlotType.QUANTILE_SOLVABILITY:
        if not (isinstance(extra, list) and len(extra) >= 2):
            error_msg = (
                "Extra must be a list of two floats for "
                "'quantile_solvability' plot type."
            )
            raise ValueError(error_msg)
        extra_0 = float(extra[0])
        extra_1 = float(extra[1])
        plot_name = f"profile_{extra_1}_quantile_{extra_0}_solve_times"
    elif plot_type == PlotType.DIFFERENCE_OF_CDF_SOLVABILITY:
        plot_name = f"diff_profile_cdf_{extra}_solve_times"
    elif plot_type == PlotType.DIFFERENCE_OF_QUANTILE_SOLVABILITY:
        if not (isinstance(extra, list) and len(extra) == 2):
            error_msg = (
                "Extra must be a list of two floats for "
                "'diff_quantile_solvability' plot type."
            )
            raise ValueError(error_msg)
        extra_0 = float(extra[0])
        extra_1 = float(extra[1])
        plot_name = f"diff_profile_{extra_1}_quantile_{extra_0}_solve_times"
    elif plot_type == PlotType.AREA:
        plot_name = "area_scatterplot"
    elif plot_type == PlotType.BOX:
        plot_name = "terminal_box"
    elif plot_type == PlotType.VIOLIN:
        plot_name = "terminal_violin"
    elif plot_type == PlotType.TERMINAL_SCATTER:
        plot_name = "terminal_scatter"
    elif plot_type == PlotType.FEASIBILITY_SCATTER:
        plot_name = f"feasibility_scatter_{extra}"
    elif plot_type == PlotType.FEASIBILITY_VIOLIN:
        plot_name = f"feasibility_violin_{extra}"
    elif plot_type == PlotType.ALL_FEASIBILITY_PROGRESS:
        plot_name = f"all_feasibility_progress_{extra}"
    elif plot_type == PlotType.MEAN_FEASIBILITY_PROGRESS:
        plot_name = f"mean_feasibility_progress_{extra}"
    elif plot_type == PlotType.QUANTILE_FEASIBILITY_PROGRESS:
        plot_name = f"{extra[1]}_quantile_feasibility_progress_{extra[0]}"
    else:
        raise NotImplementedError(f"'{plot_type}' is not implemented.")

    plot_dir = EXPERIMENT_DIR / "plots"
    # Create the directory if it does not exist
    plot_dir.mkdir(parents=True, exist_ok=True)

    if not normalize:
        plot_name = plot_name + "_unnorm"

    # Reformat plot_name to be suitable as a string literal.
    plot_name = plot_name.replace("\\", "").replace("$", "").replace(" ", "_")

    # If the plot title is not provided, use the default title.
    if plot_title is None:
        plot_title = f"{solver_name}_{problem_name}_{plot_name}"

    # Read in the contents of the plot directory
    existing_plots = [path.name for path in list(plot_dir.glob("*"))]

    counter = 0
    while (plot_title + ext) in existing_plots:
        # If the plot title already exists, append a counter to the filename
        counter += 1
        plot_title = f"{plot_title} ({counter})"
    extended_path_name = plot_dir / (plot_title + ext)

    plt.savefig(extended_path_name, bbox_inches="tight")

    # save plot as pickle
    if save_as_pickle:
        fig = plt.gcf()
        pickle_path = extended_path_name.with_suffix(".pkl")
        with pickle_path.open("wb") as pickle_file:
            pickle.dump(fig, pickle_file)
    # Return path_name for use in GUI.
    return extended_path_name


def check_common_problem_and_reference(
    experiments: list[ProblemSolver],
) -> None:
    """Check if a collection of experiments share the same problem, x0, and x*.

    Args:
        experiments (list[ProblemSolver]): Problem-solver pairs of different solvers
            on a common problem.

    Raises:
        ValueError: If any experiments have different problem instances,
            starting solutions (`x0`), or optimal solutions (`x*`).
    """
    problem_list = [experiment.problem for experiment in experiments]
    if not all(prob == problem_list[0] for prob in problem_list[1:]):
        error_msg = "All experiments must have the same problem."
        raise ValueError(error_msg)

    x0_list = [experiment.x0 for experiment in experiments]
    if not all(start_sol == x0_list[0] for start_sol in x0_list[1:]):
        error_msg = "All experiments must have the same starting solution."
        raise ValueError(error_msg)

    xstar_list = [experiment.xstar for experiment in experiments]
    if not all(opt_sol == xstar_list[0] for opt_sol in xstar_list[1:]):
        error_msg = "All experiments must have the same optimal solution."
        raise ValueError(error_msg)


def plot_bootstrap_conf_ints(
    bs_conf_int_lower_bounds: Curve,
    bs_conf_int_upper_bounds: Curve,
    color_str: str = "C0",
) -> None:
    """Plot bootstrap confidence intervals.

    Args:
        bs_conf_int_lower_bounds (Curve): Lower bounds of bootstrap confidence
            intervals, as curves.
        bs_conf_int_upper_bounds (Curve): Upper bounds of bootstrap confidence
            intervals, as curves.
        color_str (str, optional): String indicating line color, e.g., "C0", "C1", etc.
            Defaults to "C0".
    """
    bs_conf_int_lower_bounds.plot(color_str=color_str, curve_type=CurveType.CONF_BOUND)
    bs_conf_int_upper_bounds.plot(color_str=color_str, curve_type=CurveType.CONF_BOUND)
    # Shade space between curves.
    # Convert to full curves to get piecewise-constant shaded areas.
    plt.fill_between(
        x=bs_conf_int_lower_bounds.curve_to_full_curve().x_vals,
        y1=bs_conf_int_lower_bounds.curve_to_full_curve().y_vals,
        y2=bs_conf_int_upper_bounds.curve_to_full_curve().y_vals,
        color=color_str,
        alpha=0.2,
    )


def report_max_halfwidth(
    curve_pairs: list[list[Curve]],
    normalize: bool,
    conf_level: float,
    difference: bool = False,
) -> None:
    """Print caption for the max halfwidth of bootstrap confidence interval curves.

    Args:
        curve_pairs (list[list[Curve]]): A list of paired bootstrap CI curves.
        normalize (bool): Whether to normalize progress curves with respect to
            optimality gaps.
        conf_level (float): Confidence level for confidence intervals
            (must be in (0, 1)).
        difference (bool, optional): Whether the plot is for difference profiles.
            Defaults to False.

    Raises:
        ValueError: If `conf_level` is not in (0, 1) or if `curve_pairs` is empty.
    """
    # Value checking
    if not 0 < conf_level < 1:
        error_msg = "Confidence level must be in (0, 1)."
        raise ValueError(error_msg)
    # Make sure there's something in the list
    if len(curve_pairs) == 0:
        error_msg = "No curve pairs to report on."
        raise ValueError(error_msg)

    # Compute max halfwidth of bootstrap confidence intervals.
    min_lower_bound = float("inf")
    max_upper_bound = -float("inf")
    max_halfwidths = []
    for curve_pair in curve_pairs:
        min_lower_bound = min(min_lower_bound, min(curve_pair[0].y_vals))
        max_upper_bound = max(max_upper_bound, max(curve_pair[1].y_vals))
        max_halfwidths.append(
            0.5 * curve_utils.max_difference_of_curves(curve_pair[1], curve_pair[0])
        )
    max_halfwidth = max(max_halfwidths)
    # Print caption about max halfwidth.
    if normalize:
        if difference:
            xloc = 0.05
            yloc = -1.35
        else:
            xloc = 0.05
            yloc = -0.35
    else:
        # xloc = 0.05 * budget of the problem
        xloc = 0.05 * curve_pairs[0][0].x_vals[-1]
        yloc = min_lower_bound - 0.25 * (max_upper_bound - min_lower_bound)
    boot_cis = round(conf_level * 100)
    max_hw_round = round(max_halfwidth, 2)
    txt = f"The max halfwidth of the bootstrap {boot_cis}% CIs is {max_hw_round}."
    plt.text(x=xloc, y=yloc, s=txt)
