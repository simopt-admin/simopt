"""Feasibility progress plot."""

from typing import Literal

import matplotlib.pyplot as plt

import simopt.curve_utils as curve_utils
from simopt.bootstrap import bootstrap_procedure
from simopt.experiment import ProblemSolver
from simopt.plot_type import PlotType

from .utils import plot_bootstrap_conf_ints, report_max_halfwidth, save_plot, setup_plot


# TODO: this function is an almost identical copy from the original implementation.
# The code quality is suboptimal and will be refactored together with other plotting
# functions.
def plot_feasibility_progress(
    experiments: list[list[ProblemSolver]],
    plot_type: PlotType = PlotType.ALL_FEASIBILITY_PROGRESS,
    score_type: Literal["inf_norm", "norm"] = "inf_norm",
    norm_degree: int = 1,
    two_sided: bool = True,
    plot_zero: bool = True,
    all_in_one: bool = True,
    n_bootstraps: int = 100,
    conf_level: float = 0.95,
    plot_conf_ints: bool = True,
    print_max_hw: bool = True,
    beta: float = 0.5,
    solver_set_name: str = "SOLVER_SET",
    plot_title: str | None = None,
    legend_loc: str | None = None,
    ext: str = ".png",
    save_as_pickle: bool = False,
) -> list[str]:
    """Plot feasibility over solver progress.

    Parameters
    ----------
    experiments : list [list [``experiment_base.ProblemSolver``]]
        Problem-solver pairs used to produce plots.
    plot_type : str, default = 'scatter'
        String indicating which type of plot to produce:
            "all" : show all macroreps
            "mean" : plot mean of all macroreps
            "quantile" : plot quantile of all macroreps
    score_type : str, default = "inf_norm"
        "inf_norm" : use infinite norm of lhs of violated constraints to calculate
        feasiblity score
        "norm" : use "norm_degree" to specify degree of norm of lhs of violated
        constriants
    two_sided : bool, default = "True"
        Two-sided or one-sided feasiblity score. Two-sided only supported for
        plot_type = "all"
    plot_zero: bool, default = True
        Plot a dashed red line a feasiblity score = 0
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
    print_max_hw : bool, default = True
        report max halfwidth for CI's'
    beta : float, default = .5
        quantile to computue must be between 0 and 1
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

    # set up extra for file name
    if score_type == "inf_norm":
        file_name = "inf_norm"
    elif score_type == "norm":
        file_name = f"{norm_degree}_norm"
    extra = (
        [file_name, beta]
        if plot_type == PlotType.QUANTILE_FEASIBILITY_PROGRESS
        else file_name
    )

    # check feasibility score compatibility
    if plot_type != PlotType.ALL_FEASIBILITY_PROGRESS and two_sided:
        raise RuntimeError(
            "Mean and quantile plots not supported for two sided feasibility scores."
        )

    for problem_idx in range(
        n_problems
    ):  # must create new plot for every different problem
        if all_in_one:
            ref_experiment = experiments[0][problem_idx]
            setup_plot(
                plot_type=plot_type,
                solver_name=solver_set_name,
                problem_name=ref_experiment.problem.name,
                budget=ref_experiment.problem.factors["budget"],
                beta=beta,
                plot_title=plot_title,
                feasibility_score_method=score_type,
                feasibility_norm_degree=norm_degree,
                normalize=False,
            )
            solver_curve_handles = []
            curve_pairs = []
            solver_names = []
            for exp_idx in range(n_solvers):
                experiment = experiments[exp_idx][problem_idx]
                experiment.feasibility_score_history(score_type, norm_degree, two_sided)
                color_str = "C" + str(exp_idx)
                estimator = None
                solver_names.append(experiment.solver.name)
                if plot_type == PlotType.ALL_FEASIBILITY_PROGRESS:
                    handle = experiment.feasibility_curves[0].plot(color_str=color_str)
                    for curve in experiment.feasibility_curves[1:]:
                        curve.plot(color_str=color_str)
                elif plot_type == PlotType.MEAN_FEASIBILITY_PROGRESS:
                    # Plot estimated mean progress curve.
                    estimator = curve_utils.mean_of_curves(
                        experiment.feasibility_curves
                    )
                    handle = estimator.plot(color_str=color_str)
                elif (
                    plot_type == PlotType.QUANTILE_FEASIBILITY_PROGRESS
                ):  # Must be quantile.
                    # Plot estimated beta-quantile progress curve.
                    estimator = curve_utils.quantile_of_curves(
                        experiment.feasibility_curves, beta
                    )
                    handle = estimator.plot(color_str=color_str)
                if (
                    plot_conf_ints or print_max_hw
                ) and plot_type != PlotType.ALL_FEASIBILITY_PROGRESS:
                    # Note: "experiments" needs to be a list of list of ProblemSolver
                    # objects.
                    bs_conf_int_lb_curve, bs_conf_int_ub_curve = bootstrap_procedure(
                        experiments=[[experiment]],
                        n_bootstraps=n_bootstraps,
                        conf_level=conf_level,
                        plot_type=plot_type,
                        beta=beta,
                        estimator=estimator,
                        normalize=False,
                        feasibility_score_method=score_type,
                        feasibility_norm_degree=norm_degree,
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
                if plot_zero:
                    plt.axhline(y=0, color="red", linestyle="--", linewidth=0.75)
                solver_curve_handles.append(handle)

            if print_max_hw:
                report_max_halfwidth(
                    curve_pairs=curve_pairs,
                    normalize=False,
                    conf_level=conf_level,
                )
            plt.legend(
                handles=solver_curve_handles,
                labels=solver_names,
                loc=legend_loc,
            )
            file_list.append(
                save_plot(
                    solver_name=solver_set_name,
                    problem_name=ref_experiment.problem.name,
                    plot_type=plot_type,
                    extra=extra,
                    plot_title=plot_title,
                    ext=ext,
                    save_as_pickle=save_as_pickle,
                    normalize=False,
                )
            )

        else:
            for solver_idx in range(n_solvers):
                experiment = experiments[solver_idx][problem_idx]
                setup_plot(
                    plot_type=plot_type,
                    solver_name=experiment.solver.name,
                    problem_name=experiment.problem.name,
                    budget=experiment.problem.factors["budget"],
                    feasibility_score_method=score_type,
                    feasibility_norm_degree=norm_degree,
                    normalize=False,
                    beta=beta,
                )
                # Compute terminal feasibility scores
                experiment.feasibility_score_history(
                    score_type, norm_degree, two_sided
                )  # gives list of feasibility scores for each macrorep

                if plot_type == PlotType.ALL_FEASIBILITY_PROGRESS:
                    for curve in experiment.feasibility_curves:
                        curve.plot()
                elif plot_type == PlotType.MEAN_FEASIBILITY_PROGRESS:
                    estimator = curve_utils.mean_of_curves(
                        experiment.feasibility_curves
                    )
                    estimator.plot()
                elif plot_type == PlotType.QUANTILE_FEASIBILITY_PROGRESS:
                    estimator = curve_utils.quantile_of_curves(
                        experiment.feasibility_curves, beta
                    )
                    estimator.plot()
                if plot_zero:
                    plt.axhline(y=0, color="red", linestyle="--", linewidth=0.75)
                if (
                    plot_conf_ints or print_max_hw
                ) and plot_type != PlotType.ALL_FEASIBILITY_PROGRESS:
                    # Note: "experiments" needs to be a list of list of ProblemSolver
                    # objects.
                    bs_conf_int_lb_curve, bs_conf_int_ub_curve = bootstrap_procedure(
                        experiments=[[experiment]],
                        n_bootstraps=n_bootstraps,
                        conf_level=conf_level,
                        plot_type=plot_type,
                        beta=beta,
                        estimator=estimator,
                        normalize=False,
                        feasibility_score_method=score_type,
                        feasibility_norm_degree=norm_degree,
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
                        )
                    if print_max_hw:
                        curve_pairs.append([bs_conf_int_lb_curve, bs_conf_int_ub_curve])

                file_list.append(
                    save_plot(
                        solver_name=experiment.solver.name,
                        problem_name=experiment.problem.name,
                        plot_type=plot_type,
                        ext=ext,
                        extra=extra,
                        normalize=False,
                        save_as_pickle=save_as_pickle,
                    )
                )

    return file_list
