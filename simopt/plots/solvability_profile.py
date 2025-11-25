"""Solvability profile plot."""

from pathlib import Path

import matplotlib.pyplot as plt

import simopt.curve_utils as curve_utils
from simopt.bootstrap import bootstrap_procedure
from simopt.experiment import ProblemSolver
from simopt.plot_type import PlotType

from .utils import plot_bootstrap_conf_ints, report_max_halfwidth, save_plot, setup_plot


def plot_solvability_profiles(
    experiments: list[list[ProblemSolver]],
    plot_type: PlotType,
    all_in_one: bool = True,
    n_bootstraps: int = 100,
    conf_level: float = 0.95,
    plot_conf_ints: bool = True,
    print_max_hw: bool = True,
    solve_tol: float = 0.1,
    beta: float = 0.5,
    ref_solver: str | None = None,
    plot_title: str | None = None,
    legend_loc: str | None = None,
    ext: str = ".png",
    save_as_pickle: bool = False,
    solver_set_name: str = "SOLVER_SET",
    problem_set_name: str = "PROBLEM_SET",
) -> list[Path]:
    """Plots solvability or difference profiles for solvers on multiple problems.

    Args:
        experiments (list[list[ProblemSolver]]): Problem-solver pairs used for plotting.
        plot_type (PlotType): Type of solvability plot to produce:
            - PlotType.CDF_SOLVABILITY
            - PlotType.QUANTILE_SOLVABILITY
            - PlotType.DIFFERENCE_OF_CDF_SOLVABILITY
            - PlotType.DIFFERENCE_OF_QUANTILE_SOLVABILITY
        all_in_one (bool, optional): If True, plot all curves together.
            Defaults to True.
        n_bootstraps (int, optional): Number of bootstrap samples. Defaults to 100.
        conf_level (float, optional): Confidence level for intervals
            (0 < conf_level < 1). Defaults to 0.95.
        plot_conf_ints (bool, optional): If True, show bootstrapped confidence
            intervals. Defaults to True.
        print_max_hw (bool, optional): If True, print max half-width in caption.
            Defaults to True.
        solve_tol (float, optional): Optimality gap defining when a problem is
            considered solved (0 < solve_tol ≤ 1). Defaults to 0.1.
        beta (float, optional): Quantile level to compute (0 < beta < 1).
            Defaults to 0.5.
        ref_solver (str, optional): Name of the reference solver for difference plots.
        plot_title (str, optional): Custom title for the plot
            (used only if `all_in_one=True`).
        legend_loc (str, optional): Location of the legend
            (e.g., "best", "upper right").
        ext (str, optional): File extension for saved plots (e.g., ".png").
            Defaults to ".png".
        save_as_pickle (bool, optional): If True, save plots as pickle files.
            Defaults to False.
        solver_set_name (str, optional): Name of solver group for plot titles.
            Defaults to "SOLVER_SET".
        problem_set_name (str, optional): Name of problem group for plot titles.
            Defaults to "PROBLEM_SET".

    Returns:
        list[Path]: List of file paths for the plots produced.

    Raises:
        ValueError: If any input parameter is out of bounds or invalid.
    """
    # Value checking
    if n_bootstraps < 1:
        error_msg = "Number of bootstraps must be a positive integer."
        raise ValueError(error_msg)
    if not 0 < conf_level < 1:
        error_msg = "Confidence level must be in (0, 1)."
        raise ValueError(error_msg)
    if not 0 < solve_tol <= 1:
        error_msg = "Solve tolerance must be in (0, 1]."
        raise ValueError(error_msg)
    if not 0 < beta < 1:
        error_msg = "Beta quantile must be in (0, 1)."
        raise ValueError(error_msg)

    if legend_loc is None:
        legend_loc = "best"

    file_list = []
    # Set up plot.
    n_solvers = len(experiments)
    n_problems = len(experiments[0])
    if all_in_one:
        if plot_type == PlotType.CDF_SOLVABILITY:
            setup_plot(
                plot_type=plot_type,
                solver_name=solver_set_name,
                problem_name=problem_set_name,
                solve_tol=solve_tol,
                plot_title=plot_title,
            )
        elif plot_type == PlotType.QUANTILE_SOLVABILITY:
            setup_plot(
                plot_type=plot_type,
                solver_name=solver_set_name,
                problem_name=problem_set_name,
                beta=beta,
                solve_tol=solve_tol,
                plot_title=plot_title,
            )
        elif plot_type == PlotType.DIFFERENCE_OF_CDF_SOLVABILITY:
            setup_plot(
                plot_type=plot_type,
                solver_name=solver_set_name,
                problem_name=problem_set_name,
                solve_tol=solve_tol,
                plot_title=plot_title,
            )
        elif plot_type == PlotType.DIFFERENCE_OF_QUANTILE_SOLVABILITY:
            setup_plot(
                plot_type=plot_type,
                solver_name=solver_set_name,
                problem_name=problem_set_name,
                beta=beta,
                solve_tol=solve_tol,
                plot_title=plot_title,
            )
        else:
            error_msg = f"Plot type {plot_type} is not supported."
            raise ValueError(error_msg)
        curve_pairs = []
        solver_names = [
            solver_experiments[0].solver.name for solver_experiments in experiments
        ]
        solver_curves = []
        solver_curve_handles = []
        for solver_idx in range(n_solvers):
            solver_sub_curves = []
            color_str = "C" + str(solver_idx)
            # For each problem compute the cdf or quantile of solve times.
            for problem_idx in range(n_problems):
                experiment = experiments[solver_idx][problem_idx]
                sub_curve = None
                if plot_type in [
                    PlotType.CDF_SOLVABILITY,
                    PlotType.DIFFERENCE_OF_CDF_SOLVABILITY,
                ]:
                    sub_curve = curve_utils.cdf_of_curves_crossing_times(
                        curves=experiment.progress_curves, threshold=solve_tol
                    )
                elif plot_type in [
                    PlotType.QUANTILE_SOLVABILITY,
                    PlotType.DIFFERENCE_OF_QUANTILE_SOLVABILITY,
                ]:
                    sub_curve = curve_utils.quantile_cross_jump(
                        curves=experiment.progress_curves,
                        threshold=solve_tol,
                        beta=beta,
                    )
                else:
                    error_msg = f"Plot type {plot_type} is not supported."
                    raise ValueError(error_msg)
                if sub_curve is not None:
                    solver_sub_curves.append(sub_curve)
            # Plot solvability profile for the solver.
            # Exploit the fact that each solvability profile is an average of more
            # basic curves.
            solver_curve = curve_utils.mean_of_curves(solver_sub_curves)
            # CAUTION: Using mean above requires an equal number of macro-replications
            # per problem.
            solver_curves.append(solver_curve)
            if plot_type in [PlotType.CDF_SOLVABILITY, PlotType.QUANTILE_SOLVABILITY]:
                handle = solver_curve.plot(color_str=color_str)
                solver_curve_handles.append(handle)
                if plot_conf_ints or print_max_hw:
                    bs_conf_int_lb_curve, bs_conf_int_ub_curve = bootstrap_procedure(
                        experiments=[experiments[solver_idx]],
                        n_bootstraps=n_bootstraps,
                        conf_level=conf_level,
                        plot_type=plot_type,  # type: ignore
                        solve_tol=solve_tol,
                        beta=beta,
                        estimator=solver_curve,
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
        if plot_type == PlotType.CDF_SOLVABILITY:
            plt.legend(
                handles=solver_curve_handles,
                labels=solver_names,
                loc=legend_loc,
            )
            if print_max_hw:
                report_max_halfwidth(
                    curve_pairs=curve_pairs,
                    normalize=True,
                    conf_level=conf_level,
                )
            file_list.append(
                save_plot(
                    solver_name=solver_set_name,
                    problem_name=problem_set_name,
                    plot_type=plot_type,
                    normalize=True,
                    extra=solve_tol,
                    plot_title=plot_title,
                    ext=ext,
                    save_as_pickle=save_as_pickle,
                )
            )
        elif plot_type == PlotType.QUANTILE_SOLVABILITY:
            plt.legend(
                handles=solver_curve_handles,
                labels=solver_names,
                loc=legend_loc,
            )
            if print_max_hw:
                report_max_halfwidth(
                    curve_pairs=curve_pairs,
                    normalize=True,
                    conf_level=conf_level,
                )
            file_list.append(
                save_plot(
                    solver_name=solver_set_name,
                    problem_name=problem_set_name,
                    plot_type=plot_type,
                    normalize=True,
                    extra=[solve_tol, beta],
                    plot_title=plot_title,
                    ext=ext,
                    save_as_pickle=save_as_pickle,
                )
            )
        elif plot_type in [
            PlotType.DIFFERENCE_OF_CDF_SOLVABILITY,
            PlotType.DIFFERENCE_OF_QUANTILE_SOLVABILITY,
        ]:
            if ref_solver is None:
                error_msg = (
                    "Reference solver must be specified for difference profiles."
                )
                raise ValueError(error_msg)
            non_ref_solvers = [
                solver_name for solver_name in solver_names if solver_name != ref_solver
            ]
            ref_solver_idx = solver_names.index(ref_solver)
            for solver_idx in range(n_solvers):
                if solver_idx is not ref_solver_idx:
                    diff_solver_curve = curve_utils.difference_of_curves(
                        solver_curves[solver_idx], solver_curves[ref_solver_idx]
                    )
                    color_str = "C" + str(solver_idx)
                    handle = diff_solver_curve.plot(color_str=color_str)
                    solver_curve_handles.append(handle)
                    if plot_conf_ints or print_max_hw:
                        bs_conf_int_lb_curve, bs_conf_int_ub_curve = (
                            bootstrap_procedure(
                                experiments=[
                                    experiments[solver_idx],
                                    experiments[ref_solver_idx],
                                ],
                                n_bootstraps=n_bootstraps,
                                conf_level=conf_level,
                                plot_type=plot_type,  # type: ignore
                                solve_tol=solve_tol,
                                beta=beta,
                                estimator=diff_solver_curve,
                                normalize=True,
                            )
                        )
                        if plot_conf_ints:
                            if isinstance(
                                bs_conf_int_lb_curve, (int, float)
                            ) or isinstance(bs_conf_int_ub_curve, (int, float)):
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
                            curve_pairs.append(
                                [bs_conf_int_lb_curve, bs_conf_int_ub_curve]
                            )
            offset_labels = [
                f"{non_ref_solver} - {ref_solver}" for non_ref_solver in non_ref_solvers
            ]
            plt.legend(
                handles=solver_curve_handles,
                labels=offset_labels,
                loc=legend_loc,
            )
            if print_max_hw:
                report_max_halfwidth(
                    curve_pairs=curve_pairs,
                    normalize=True,
                    conf_level=conf_level,
                    difference=True,
                )
            if plot_type == PlotType.DIFFERENCE_OF_CDF_SOLVABILITY:
                file_list.append(
                    save_plot(
                        solver_name=solver_set_name,
                        problem_name=problem_set_name,
                        plot_type=plot_type,
                        normalize=True,
                        extra=solve_tol,
                        plot_title=plot_title,
                        ext=ext,
                        save_as_pickle=save_as_pickle,
                    )
                )
            elif plot_type == PlotType.DIFFERENCE_OF_QUANTILE_SOLVABILITY:
                file_list.append(
                    save_plot(
                        solver_name=solver_set_name,
                        problem_name=problem_set_name,
                        plot_type=plot_type,
                        normalize=True,
                        extra=[solve_tol, beta],
                        plot_title=plot_title,
                        ext=ext,
                        save_as_pickle=save_as_pickle,
                    )
                )
        else:
            error_msg = f"Plot type {plot_type} is not supported."
            raise ValueError(error_msg)
    else:
        solver_names = [
            solver_experiments[0].solver.name for solver_experiments in experiments
        ]
        solver_curves = []
        for solver_idx in range(n_solvers):
            solver_sub_curves = []
            # For each problem compute the cdf or quantile of solve times.
            for problem_idx in range(n_problems):
                experiment = experiments[solver_idx][problem_idx]
                sub_curve = None
                if plot_type in [
                    PlotType.CDF_SOLVABILITY,
                    PlotType.DIFFERENCE_OF_CDF_SOLVABILITY,
                ]:
                    sub_curve = curve_utils.cdf_of_curves_crossing_times(
                        curves=experiment.progress_curves, threshold=solve_tol
                    )
                elif plot_type in [
                    PlotType.QUANTILE_SOLVABILITY,
                    PlotType.DIFFERENCE_OF_QUANTILE_SOLVABILITY,
                ]:
                    sub_curve = curve_utils.quantile_cross_jump(
                        curves=experiment.progress_curves,
                        threshold=solve_tol,
                        beta=beta,
                    )
                else:
                    error_msg = f"Plot type {plot_type} is not supported."
                    raise ValueError(error_msg)
                if sub_curve is not None:
                    solver_sub_curves.append(sub_curve)
            # Plot solvability profile for the solver.
            # Exploit the fact that each solvability profile is an average of more
            # basic curves.
            solver_curve = curve_utils.mean_of_curves(solver_sub_curves)
            solver_curves.append(solver_curve)
            if plot_type in {PlotType.CDF_SOLVABILITY, PlotType.QUANTILE_SOLVABILITY}:
                # Set up plot.
                if plot_type == PlotType.CDF_SOLVABILITY:
                    file_list.append(
                        setup_plot(
                            plot_type=plot_type,
                            solver_name=experiments[solver_idx][0].solver.name,
                            problem_name=problem_set_name,
                            solve_tol=solve_tol,
                        )
                    )
                elif plot_type == PlotType.QUANTILE_SOLVABILITY:
                    file_list.append(
                        setup_plot(
                            plot_type=plot_type,
                            solver_name=experiments[solver_idx][0].solver.name,
                            problem_name=problem_set_name,
                            beta=beta,
                            solve_tol=solve_tol,
                        )
                    )
                handle = solver_curve.plot()
                if plot_conf_ints or print_max_hw:
                    bs_conf_int_lb_curve, bs_conf_int_ub_curve = bootstrap_procedure(
                        experiments=[experiments[solver_idx]],
                        n_bootstraps=n_bootstraps,
                        conf_level=conf_level,
                        plot_type=plot_type,
                        solve_tol=solve_tol,
                        beta=beta,
                        estimator=solver_curve,
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
                            bs_conf_int_lb_curve, bs_conf_int_ub_curve
                        )
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
                if plot_type == PlotType.CDF_SOLVABILITY:
                    file_list.append(
                        save_plot(
                            solver_name=experiments[solver_idx][0].solver.name,
                            problem_name=problem_set_name,
                            plot_type=plot_type,
                            normalize=True,
                            extra=solve_tol,
                            ext=ext,
                            save_as_pickle=save_as_pickle,
                        )
                    )
                elif plot_type == PlotType.QUANTILE_SOLVABILITY:
                    file_list.append(
                        save_plot(
                            solver_name=experiments[solver_idx][0].solver.name,
                            problem_name=problem_set_name,
                            plot_type=plot_type,
                            normalize=True,
                            extra=[solve_tol, beta],
                            ext=ext,
                            save_as_pickle=save_as_pickle,
                        )
                    )
        if plot_type in [
            PlotType.DIFFERENCE_OF_CDF_SOLVABILITY,
            PlotType.DIFFERENCE_OF_QUANTILE_SOLVABILITY,
        ]:
            if ref_solver is None:
                error_msg = (
                    "Reference solver must be specified for difference profiles."
                )
                raise ValueError(error_msg)
            non_ref_solvers = [
                solver_name for solver_name in solver_names if solver_name != ref_solver
            ]
            ref_solver_idx = solver_names.index(ref_solver)
            for solver_idx in range(n_solvers):
                if solver_idx is not ref_solver_idx:
                    if plot_type == PlotType.DIFFERENCE_OF_CDF_SOLVABILITY:
                        file_list.append(
                            setup_plot(
                                plot_type=plot_type,
                                solver_name=experiments[solver_idx][0].solver.name,
                                problem_name=problem_set_name,
                                solve_tol=solve_tol,
                            )
                        )
                    elif plot_type == PlotType.DIFFERENCE_OF_QUANTILE_SOLVABILITY:
                        file_list.append(
                            setup_plot(
                                plot_type=plot_type,
                                solver_name=experiments[solver_idx][0].solver.name,
                                problem_name=problem_set_name,
                                beta=beta,
                                solve_tol=solve_tol,
                            )
                        )
                    diff_solver_curve = curve_utils.difference_of_curves(
                        solver_curves[solver_idx], solver_curves[ref_solver_idx]
                    )
                    handle = diff_solver_curve.plot()
                    if plot_conf_ints or print_max_hw:
                        bs_conf_int_lb_curve, bs_conf_int_ub_curve = (
                            bootstrap_procedure(
                                experiments=[
                                    experiments[solver_idx],
                                    experiments[ref_solver_idx],
                                ],
                                n_bootstraps=n_bootstraps,
                                conf_level=conf_level,
                                plot_type=plot_type,
                                solve_tol=solve_tol,
                                beta=beta,
                                estimator=diff_solver_curve,
                                normalize=True,
                            )
                        )
                        if plot_conf_ints:
                            if isinstance(
                                bs_conf_int_lb_curve, (int, float)
                            ) or isinstance(bs_conf_int_ub_curve, (int, float)):
                                error_msg = (
                                    "Bootstrap confidence intervals are not available "
                                    "for scalar estimators."
                                )
                                raise ValueError(error_msg)
                            plot_bootstrap_conf_ints(
                                bs_conf_int_lb_curve, bs_conf_int_ub_curve
                            )
                        if print_max_hw:
                            if isinstance(
                                bs_conf_int_lb_curve, (int, float)
                            ) or isinstance(bs_conf_int_ub_curve, (int, float)):
                                error_msg = (
                                    "Max halfwidth is not available for "
                                    "scalar estimators."
                                )
                                raise ValueError(error_msg)
                            report_max_halfwidth(
                                curve_pairs=[
                                    [bs_conf_int_lb_curve, bs_conf_int_ub_curve]
                                ],
                                normalize=True,
                                conf_level=conf_level,
                                difference=True,
                            )
                    if plot_type == PlotType.DIFFERENCE_OF_CDF_SOLVABILITY:
                        file_list.append(
                            save_plot(
                                solver_name=experiments[solver_idx][0].solver.name,
                                problem_name=problem_set_name,
                                plot_type=plot_type,
                                normalize=True,
                                extra=solve_tol,
                                ext=ext,
                                save_as_pickle=save_as_pickle,
                            )
                        )
                    elif plot_type == PlotType.DIFFERENCE_OF_QUANTILE_SOLVABILITY:
                        file_list.append(
                            save_plot(
                                solver_name=experiments[solver_idx][0].solver.name,
                                problem_name=problem_set_name,
                                plot_type=plot_type,
                                normalize=True,
                                extra=[solve_tol, beta],
                                ext=ext,
                                save_as_pickle=save_as_pickle,
                            )
                        )
    return file_list
