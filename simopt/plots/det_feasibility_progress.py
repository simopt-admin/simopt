"""Feasibility progress plot."""

from typing import Literal

import matplotlib.pyplot as plt

import simopt.curve_utils as curve_utils
from simopt.bootstrap import bootstrap_procedure
from simopt.experiment import ProblemSolver
from simopt.plot_type import PlotType
from numpy.linalg import LinAlgError, inv, norm, pinv

from .utils import plot_bootstrap_conf_ints, report_max_halfwidth, save_plot, setup_plot


def plot_det_feasibility(
    experiments: list[list[ProblemSolver]],
    plot_type: PlotType = PlotType.DETERMINISTIC_FEASIBILITY_PROGRESS,
    all_in_one: bool = True,
    score_type: Literal["value", "norm", "merit"] = "norm",
    obj_const: float = 1e8,
    log_scale: bool = True,
    floor_feas: bool = False,
    feas_tol_lower: float = 1e-5,
    feas_tol_upper: float = 1e-2,
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
                feasibility_score_method = score_type,
                normalize=False,
            )
            solver_curve_handles = []
            solver_names = []
            for exp_idx in range(n_solvers):
                experiment = experiments[exp_idx][problem_idx]
            
                if score_type == "value":
                    experiment.det_feasibility_history(
                        method="value",
                    )
            
                elif score_type == "merit":
                    experiment.det_feasibility_history(
                        method="objective",
                        obj_const=obj_const,
                        feas_tol_upper=feas_tol_upper,
                        feas_tol_lower = feas_tol_lower,
                        floor_feas = floor_feas
                    )
            
                elif score_type == "norm":
                    experiment.det_feasibility_history(
                        method="norm",
                        feas_tol_lower=feas_tol_lower,
                        floor_feas = floor_feas
                    )
            
                color_str = "C" + str(exp_idx)
            
                solver_names.append(experiment.solver.name)
            
                handle = experiment.det_feasibility_curves[0].plot(
                    color_str=color_str
                )
            
                for curve in experiment.det_feasibility_curves[1:]:
                    curve.plot(color_str=color_str)
            
                solver_curve_handles.append(handle)
            if log_scale:
                plt.yscale("log")

                #plt.ylim(bottom=0.001)
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
            raise NotImplementedError("Currenlty only plot_all_in_one = True supported.")
    return file_list

