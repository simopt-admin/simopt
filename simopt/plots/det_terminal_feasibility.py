"""Terminal feasibility plot."""

from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np


from simopt.experiment import ProblemSolver
from simopt.plot_type import PlotType

from .utils import save_plot, setup_plot


# TODO: this function is an almost identical copy from the original implementation.
# The code quality is suboptimal and will be refactored together with other plotting
# functions.
def plot_det_terminal_feasibility(
    experiments: list[list[ProblemSolver]],
    plot_type = PlotType.DETERMINISTIC_FEASIBILITY_SCATTER,
    plot_zero: bool = True,
    plot_optimal: bool = False,
    score_type: Literal["value", "norm", "merit"] = "norm",
    floor_feas: bool = False,
    feas_tol: float = 1e-5,
    merit_const: float = 1e8,
    log_scale: bool = True,
    all_in_one: bool = True,
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
    score_type : str, default = "value"
        "value" : return unedited lhs value of constraint (only works for problems with 1 constraint)
        feasiblity score
        "norm" : use return l2 norm of the lhs value of all constraints
        constriants
        "merit": return weighted objective + norm lhs constraints (merit)
    floor_feas: bool, default = False, If true, apply feas_tol
    feas_tol: float, default = 1e-5, if floor_feas = True, lhs values below this point are set to 0 
    merit_const: float, defult = 1e8, value to multiply feas reisdual by for merit plots
    log_scale: bool, default = True, plot y-axis on log scale
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


    for problem_idx in range(
        n_problems
    ):  # must create new plot for every different problem
        if plot_type == PlotType.DETERMINISTIC_FEASIBILITY_SCATTER:
            if all_in_one:  # plot all solvers together
                ref_experiment = experiments[0][problem_idx]
                marker_list = ["o", "v", "s", "*", "P", "X", "D", "V", ">", "<"]
                setup_plot(
                    plot_type=plot_type,
                    solver_name=solver_set_name,
                    problem_name=ref_experiment.problem.name,
                    plot_title=plot_title,
                    normalize=False,
                    feasibility_score_method=score_type
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
                    if score_type == "value":
                        experiment.det_feasibility_history(feas_tol = feas_tol)  # gives list of feasibility scores for each macrorep
                    elif score_type == "merit":
                        experiment.det_feasibility_history(method = "objective", merit_const = merit_const, feas_tol = feas_tol)
                    elif score_type == "norm":
                        experiment.det_feasibility_history(method = "norm", feas_tol_lower = feas_tol, floor_feas = floor_feas)
                    term_feas_score = [
                        curve.y_vals[-1] for curve in experiment.det_feasibility_curves
                    ]
                    # Plot mean of terminal progress.
                    terminals = [
                        curve.y_vals[-1] for curve in experiment.objective_curves
                    ]
                   
                    # CI currenlty not implemented
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
                if log_scale:
                    plt.yscale("log")
                file_list.append(
                    save_plot(
                        solver_name=solver_set_name,
                        problem_name=experiment.problem.name,
                        plot_type=PlotType.DETERMINISTIC_FEASIBILITY_SCATTER,
                        normalize=False,
                        plot_title=plot_title,
                        ext=ext,
                        save_as_pickle=save_as_pickle,
                    )
                )

            else: 
                raise NotImplementedError("Currenlty only plot_all_in_one = True supported.")





    return file_list
