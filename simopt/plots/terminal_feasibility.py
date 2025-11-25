"""Terminal feasibility plot."""

from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.bootstrap import compute_bootstrap_conf_int
from simopt.experiment import ProblemSolver
from simopt.plot_type import PlotType

from .utils import save_plot, setup_plot


# TODO: this function is an almost identical copy from the original implementation.
# The code quality is suboptimal and will be refactored together with other plotting
# functions.
def plot_terminal_feasibility(
    experiments: list[list[ProblemSolver]],
    plot_type: Literal["scatter", "violin"] = "scatter",
    score_type: Literal["inf_norm", "norm"] = "inf_norm",
    two_sided: bool = True,
    plot_zero: bool = True,
    plot_optimal: bool = True,
    norm_degree: int = 1,
    all_in_one: bool = True,
    n_bootstraps: int = 100,
    conf_level: float = 0.95,
    plot_conf_ints: bool = True,
    bias_correction: bool = True,
    solver_set_name: str = "SOLVER_SET",
    plot_title: str | None = None,
    legend_loc: str | None = None,
    ext: str = ".png",
    save_as_pickle: bool = False,
) -> list[str]:
    """Plot the feasibility of one solver problem pair. (for now).

    Parameters
    ----------
    experiments : list [list [``experiment_base.ProblemSolver``]]
        Problem-solver pairs used to produce plots.
    plot_type : str, default = 'scatter'
        String indicating which type of plot to produce:
            "scatter" : scatter plot with terminal objective on x-axis and terminal
            feasiblity score on y-axis for all macroreps
            "violin" : violin plot showing density of terminal feasiblity scores for all
            macroreps
    score_type : str, default = "inf_norm"
        "inf_norm" : use infinite norm of lhs of violated constraints to calculate
        feasiblity score
        "norm" : use "norm_degree" to specify degree of norm of lhs of violated
        constriants
    two_sided : bool, default = "True"
        Two-sided or one-sided feasiblity score
    plot_zero: bool, default = True
        Plot a dashed red line a feasiblity score = 0
    plot_optimal : default = True
        Plot a dashed red line at beast feasible objective across all postreplications
    norm_degree : int, default = 1
        if not using inf_norm, specifies degree of norm taken of lhs of violated
        constraints for feasibility score
    all_in_one : bool, default = True
        plot all solvers on same plot
    n_bootstraps : int, default = 100
        number of bootsrap replications for CI construction
    conf_level : float, default = 0.95
        confidence level of created CI
    plot_conf_ints : bool, default = True
        plot CI's for each maroreplication
    bias_correction: bool, default = True
        use bias correction for CI construction
    solver_set_name : str, default = "SOLVER_SET"
        Override solver names in plot, only applies if all_in_one = True
    plot_title : str, optional
        Optional title to override the one that is autmatically generated
    legend_loc : str, default="best"
        specificies location of legend
    ext: str, default = '.png'
         Extension to add to image file path to change file type
    save_as_pickle: bool, default = False
         True if plot should be saved to pickle file, False otherwise.

    Returns:
    -------
    file_list : list [str]
        List compiling path names for plots produced.

    Raises:
    ------
    TypeError
    ValueError

    """
    # define legend location
    if legend_loc is None:
        legend_loc = "best"

    file_list = []
    # Set up plot.
    n_solvers = len(experiments)
    n_problems = len(experiments[0])

    # set up extra file extension
    if score_type == "inf_norm":
        extra = "inf_norm"
    elif score_type == "norm":
        extra = f"{norm_degree}_norm"

    for problem_idx in range(
        n_problems
    ):  # must create new plot for every different problem
        if plot_type == PlotType.FEASIBILITY_SCATTER:
            if all_in_one:  # plot all solvers together
                ref_experiment = experiments[0][problem_idx]
                marker_list = ["o", "v", "s", "*", "P", "X", "D", "V", ">", "<"]
                setup_plot(
                    plot_type=PlotType.FEASIBILITY_SCATTER,
                    solver_name=solver_set_name,
                    problem_name=ref_experiment.problem.name,
                    plot_title=plot_title,
                    normalize=False,
                    feasibility_score_method=score_type,
                    feasibility_norm_degree=norm_degree,
                )
                solver_names = [
                    solver_experiments[0].solver.name
                    for solver_experiments in experiments
                ]
                solver_curve_handles = []
                handle = None
                for solver_idx in range(n_solvers):
                    experiment = experiments[solver_idx][problem_idx]
                    color_str = "C" + str(solver_idx)
                    marker_str = marker_list[
                        solver_idx % len(marker_list)
                    ]  # Cycle through list of marker types.
                    # Compute terminal feasibility scores
                    experiment.feasibility_score_history(
                        score_type, norm_degree, two_sided
                    )  # gives list of feasibility scores for each macrorep
                    term_feas_score = [
                        curve.y_vals[-1] for curve in experiment.feasibility_curves
                    ]
                    # Plot mean of terminal progress.
                    terminals = [
                        curve.y_vals[-1] for curve in experiment.objective_curves
                    ]
                    if plot_conf_ints:
                        bootstrap_rng = MRG32k3a(s_ss_sss_index=[1, 0, 0])
                        all_obj_reps = []
                        all_feas_reps = []
                        for _ in range(n_bootstraps):
                            est_obj, est_feas = (
                                experiment.bootstrap_terminal_objective_and_feasibility(
                                    bootstrap_rng,
                                    score_type,
                                    norm_degree,
                                    two_sided,
                                )
                            )  # estimations for each macrorep in experiment
                            all_obj_reps.append(est_obj)
                            all_feas_reps.append(est_feas)
                        # sort by mrep
                        obj_conf_int_lb = []
                        obj_conf_int_ub = []
                        feas_conf_int_lb = []
                        feas_conf_int_ub = []
                        for mrep in range(experiment.n_macroreps):
                            mrep_est_obj = [rep[mrep] for rep in all_obj_reps]
                            mrep_est_feas = [rep[mrep] for rep in all_feas_reps]
                            lower_obj, upper_obj = compute_bootstrap_conf_int(
                                mrep_est_obj,
                                conf_level=conf_level,
                                bias_correction=bias_correction,
                                overall_estimator=terminals[mrep],
                            )
                            lower_feas, upper_feas = compute_bootstrap_conf_int(
                                mrep_est_feas,
                                conf_level=conf_level,
                                bias_correction=bias_correction,
                                overall_estimator=term_feas_score[mrep],
                            )
                            obj_conf_int_lb.append(lower_obj[0])
                            obj_conf_int_ub.append(upper_obj[0])
                            feas_conf_int_lb.append(lower_feas[0])
                            feas_conf_int_ub.append(upper_feas[0])
                        x_err = [
                            np.abs(np.array(terminals) - np.array(obj_conf_int_lb)),
                            np.abs(np.array(obj_conf_int_ub) - np.array(terminals)),
                        ]
                        y_err = [
                            np.abs(
                                np.array(term_feas_score) - np.array(feas_conf_int_lb)
                            ),
                            np.abs(
                                np.array(feas_conf_int_ub) - np.array(term_feas_score)
                            ),
                        ]
                        handle = plt.errorbar(
                            x=terminals,
                            y=term_feas_score,
                            xerr=x_err,
                            yerr=y_err,
                            color=color_str,
                            marker=marker_str,
                            elinewidth=1,
                            linestyle="none",
                        )
                    else:  # do not plot conf int
                        handle = plt.scatter(
                            x=terminals,
                            y=term_feas_score,
                            color=color_str,
                            marker=marker_str,
                        )
                    solver_curve_handles.append(handle)
                plt.legend(
                    handles=solver_curve_handles,
                    labels=solver_names,
                    loc=legend_loc,
                )
                if plot_zero:
                    plt.axhline(y=0, color="red", linestyle="--", linewidth=0.75)
                if plot_optimal:
                    plt.axvline(
                        x=np.mean(experiment.xstar_postreps),
                        color="red",
                        linestyle="--",
                        linewidth=0.75,
                    )
                file_list.append(
                    save_plot(
                        solver_name=solver_set_name,
                        problem_name=experiment.problem.name,
                        plot_type=PlotType.FEASIBILITY_SCATTER,
                        normalize=False,
                        plot_title=plot_title,
                        extra=extra,
                        ext=ext,
                        save_as_pickle=save_as_pickle,
                    )
                )

            else:
                for solver_idx in range(n_solvers):
                    experiment = experiments[solver_idx][problem_idx]
                    setup_plot(
                        plot_type=PlotType.FEASIBILITY_SCATTER,
                        solver_name=experiment.solver.name,
                        # this part only works for currently using one problem
                        problem_name=experiment.problem.name,
                        normalize=False,
                        feasibility_score_method=score_type,
                        feasibility_norm_degree=norm_degree,
                    )

                    # Compute terminal feasibility scores
                    experiment.feasibility_score_history(
                        score_type, norm_degree, two_sided
                    )  # gives list of feasibility scores for each macrorep
                    term_feas_score = [
                        curve.y_vals[-1] for curve in experiment.feasibility_curves
                    ]
                    # Plot mean and of terminal progress.
                    terminals = [
                        curve.y_vals[-1] for curve in experiment.objective_curves
                    ]
                    if plot_conf_ints:
                        bootstrap_rng = MRG32k3a(s_ss_sss_index=[1, 0, 0])
                        all_obj_reps = []
                        all_feas_reps = []
                        for _ in range(n_bootstraps):
                            est_obj, est_feas = (
                                experiment.bootstrap_terminal_objective_and_feasibility(
                                    bootstrap_rng,
                                    feasibility_score_method=score_type,
                                    feasibility_norm_degree=norm_degree,
                                    feasibility_two_sided=two_sided,
                                )
                            )  # estimations for each macrorep in experiment
                            all_obj_reps.append(est_obj)
                            all_feas_reps.append(est_feas)
                        # sort by mrep
                        obj_conf_int_lb = []
                        obj_conf_int_ub = []
                        feas_conf_int_lb = []
                        feas_conf_int_ub = []
                        for mrep in range(experiment.n_macroreps):
                            mrep_est_obj = [rep[mrep] for rep in all_obj_reps]
                            mrep_est_feas = [rep[mrep] for rep in all_feas_reps]
                            lower_obj, upper_obj = compute_bootstrap_conf_int(
                                mrep_est_obj,
                                conf_level=conf_level,
                                bias_correction=bias_correction,
                                overall_estimator=terminals[mrep],
                            )
                            lower_feas, upper_feas = compute_bootstrap_conf_int(
                                mrep_est_feas,
                                conf_level=conf_level,
                                bias_correction=bias_correction,
                                overall_estimator=term_feas_score[mrep],
                            )
                            obj_conf_int_lb.append(lower_obj[0])
                            obj_conf_int_ub.append(upper_obj[0])
                            feas_conf_int_lb.append(lower_feas[0])
                            feas_conf_int_ub.append(upper_feas[0])
                        x_err = [
                            np.abs(np.array(terminals) - np.array(obj_conf_int_lb)),
                            np.abs(np.array(obj_conf_int_ub) - np.array(terminals)),
                        ]
                        y_err = [
                            np.abs(
                                np.array(term_feas_score) - np.array(feas_conf_int_lb)
                            ),
                            np.abs(
                                np.array(feas_conf_int_ub) - np.array(term_feas_score)
                            ),
                        ]
                        handle = plt.errorbar(
                            x=terminals,
                            y=term_feas_score,
                            xerr=x_err,
                            yerr=y_err,
                            color="C0",
                            marker="o",
                            elinewidth=1,
                            linestyle="none",
                        )
                    else:  # no confidence intervals
                        # edit plot
                        handle = plt.scatter(
                            x=terminals,
                            y=term_feas_score,
                            color="C0",
                            marker="o",
                        )
                    if plot_zero:
                        plt.axhline(y=0, color="red", linestyle="--", linewidth=0.75)
                    if plot_optimal:
                        plt.axvline(
                            x=experiment.fstar,
                            color="red",
                            linestyle="--",
                            linewidth=0.75,
                        )
                    file_list.append(
                        save_plot(
                            solver_name=experiment.solver.name,
                            problem_name=experiment.problem.name,
                            plot_type=PlotType.FEASIBILITY_SCATTER,
                            extra=extra,
                            ext=ext,
                            normalize=False,
                            save_as_pickle=save_as_pickle,
                        )
                    )

        if plot_type == PlotType.FEASIBILITY_VIOLIN:
            if score_type == "inf_norm":
                y_title = "Feasibility Score: Infinite Norm"
            elif score_type == "norm":
                y_title = f"Feasibility Score: {norm_degree} Degree Norm"

            if all_in_one:
                ref_experiment = experiments[0][problem_idx]
                setup_plot(
                    plot_type=PlotType.FEASIBILITY_VIOLIN,
                    solver_name=solver_set_name,
                    problem_name=ref_experiment.problem.name,
                    normalize=False,
                    feasibility_score_method=score_type,
                    feasibility_norm_degree=norm_degree,
                    plot_title=plot_title,
                )

                feas_data = []
                solver_names = []
                for solver_idx in range(n_solvers):
                    experiment = experiments[solver_idx][problem_idx]
                    experiment.feasibility_score_history(
                        score_type, norm_degree, two_sided
                    )  # gives list of feasibility scores for each macrorep
                    term_feas_score = [
                        curve.y_vals[-1] for curve in experiment.feasibility_curves
                    ]
                    feas_data.append(term_feas_score)
                    solver_names.append(experiment.solver.name)

                feas_data_dict = {
                    "Solvers": solver_names,
                    y_title: feas_data,
                }
                feas_data_df = pd.DataFrame(feas_data_dict)
                feas_data_df = feas_data_df.explode(y_title)
                feas_data_df[y_title] = pd.to_numeric(
                    feas_data_df[y_title], errors="coerce"
                )
                sns.violinplot(
                    x="Solvers",
                    y=y_title,
                    data=feas_data_df,
                    inner="stick",
                    density_norm="width",
                    cut=0.1,
                    hue="Solvers",
                )
                if plot_zero:
                    plt.axhline(y=0, color="red", linestyle="--", linewidth=0.75)

                file_list.append(
                    save_plot(
                        solver_name=solver_set_name,
                        problem_name=ref_experiment.problem.name,
                        plot_type=PlotType.FEASIBILITY_VIOLIN,
                        normalize=False,
                        extra=extra,
                        plot_title=plot_title,
                        ext=ext,
                        save_as_pickle=save_as_pickle,
                    )
                )
            else:
                for solver_idx in range(n_solvers):
                    experiment = experiments[solver_idx][problem_idx]
                    setup_plot(
                        plot_type=PlotType.FEASIBILITY_VIOLIN,
                        solver_name=experiment.solver.name,
                        problem_name=experiment.problem.name,
                        normalize=False,
                        feasibility_score_method=score_type,
                        feasibility_norm_degree=norm_degree,
                        plot_title=plot_title,
                    )

                    # Compute terminal feasibility scores
                    experiment.feasibility_score_history(
                        score_type, norm_degree, two_sided
                    )  # gives list of feasibility scores for each macrorep
                    terminal_data = [
                        experiment.feasibility_curves[mrep].y_vals[-1]
                        for mrep in range(experiment.n_macroreps)
                    ]

                    solver_name_rep = [experiment.solver.name for td in terminal_data]
                    terminal_data_dict = {
                        "Solver": solver_name_rep,
                        y_title: terminal_data,
                    }
                    terminal_data_df = pd.DataFrame(terminal_data_dict)
                    sns.violinplot(
                        x="Solver",
                        y=y_title,
                        data=terminal_data_df,
                        density_norm="width",
                        cut=0.1,
                        inner="stick",
                    )
                    if plot_zero:
                        plt.axhline(y=0, color="red", linestyle="--", linewidth=2)
                    file_list.append(
                        save_plot(
                            solver_name=experiment.solver.name,
                            problem_name=experiment.problem.name,
                            plot_type=PlotType.FEASIBILITY_VIOLIN,
                            extra=extra,
                            ext=ext,
                            normalize=False,
                            save_as_pickle=save_as_pickle,
                        )
                    )

    return file_list
