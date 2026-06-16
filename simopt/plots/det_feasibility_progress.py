"""Feasibility progress plot."""

from typing import Literal

import matplotlib.pyplot as plt

import simopt.curve_utils as curve_utils
from simopt.bootstrap import bootstrap_procedure
from simopt.experiment import ProblemSolver
from simopt.plot_type import PlotType

from .utils import plot_bootstrap_conf_ints, report_max_halfwidth, save_plot, setup_plot


def plot_det_feasibility(
    experiments: list[list[ProblemSolver]],
    plot_type: PlotType = PlotType.DETERMINISTIC_FEASIBILITY_PROGRESS,
    all_in_one: bool = True,
    score_type: Literal["value", "norm", "stationarity"] = "value",
    solver_set_name: str = "SOLVER_SET",
    plot_title: str | None = None,
    legend_loc: str | None = None,
    ext: str = ".png",
    save_as_pickle: bool = False,
) -> list[str]:
   
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
        if all_in_one:
            ref_experiment = experiments[0][problem_idx]
            setup_plot(
                plot_type=plot_type,
                solver_name=solver_set_name,
                problem_name=ref_experiment.problem.name,
                budget=ref_experiment.problem.factors["budget"],
                plot_title=plot_title,
                normalize=False,
            )
            solver_curve_handles = []
            curve_pairs = []
            solver_names = []
            for exp_idx in range(n_solvers):
                experiment = experiments[exp_idx][problem_idx]
                if score_type == "value":
                    experiment.det_feasibility_history()
                    plot_curves = experiment.det_feasibility_curves
                elif score_type == "stationarity":
                    experiment.det_stationarity_history()
                    plot_curves = experiment.stationarity_curves
                color_str = "C" + str(exp_idx)
                estimator = None
                solver_names.append(experiment.solver.name)
                handle = plot_curves[0].plot(color_str=color_str)
                for curve in plot_curves[1:]:
                    curve.plot(color_str=color_str)
                plt.axhline(y=0, color="red", linestyle="--", linewidth=0.75)
                solver_curve_handles.append(handle)

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
                    normalize=False,
                )

                experiment.det_feasibility_history()
                for curve in experiment.det_feasibility_curves:
                    curve.plot()
                
                plt.axhline(y=0, color="red", linestyle="--", linewidth=0.75)
               
                file_list.append(
                    save_plot(
                        solver_name=experiment.solver.name,
                        problem_name=experiment.problem.name,
                        plot_type=plot_type,
                        ext=ext,
                        normalize=False,
                        save_as_pickle=save_as_pickle,
                    )
                )

    return file_list

