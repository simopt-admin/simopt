"""Terminal progress plot."""

from pathlib import Path

import matplotlib.pyplot as plt

from simopt.experiment import ProblemSolver
from simopt.plot_type import PlotType

from .utils import check_common_problem_and_reference, save_plot, setup_plot


def plot_terminal_progress(
    experiments: list[ProblemSolver],
    plot_type: PlotType = PlotType.VIOLIN,
    normalize: bool = True,
    all_in_one: bool = True,
    plot_title: str | None = None,
    ext: str = ".png",
    save_as_pickle: bool = False,
    solver_set_name: str = "SOLVER_SET",
) -> list[Path]:
    """Plots terminal progress as box or violin plots for solvers on a single problem.

    Args:
        experiments (list[ProblemSolver]): ProblemSolver pairs for different solvers on
            a common problem.
        plot_type (str, optional): Type of plot to generate: "box" or "violin".
            Defaults to "violin".
        normalize (bool, optional): If True, normalize progress curves by optimality
            gaps. Defaults to True.
        all_in_one (bool, optional): If True, plot all curves in one figure.
            Defaults to True.
        plot_title (str, optional): Custom title to override the default. Used only if
            all_in_one is True.
        ext (str, optional): File extension for saved plots (e.g., ".png").
            Defaults to ".png".
        save_as_pickle (bool, optional): If True, save the plot as a pickle file.
            Defaults to False.
        solver_set_name (str, optional): Label for solver group in plot titles.
            Defaults to "SOLVER_SET".

    Returns:
        list[str]: List of file paths for the plots produced.

    Raises:
        ValueError: If an unsupported plot type is specified.
    """
    # Value checking
    if plot_type not in [PlotType.BOX, PlotType.VIOLIN]:
        error_msg = "Plot type must be either 'box' or 'violin'."
        raise ValueError(error_msg)

    # Check if problems are the same with the same x0 and x*.
    check_common_problem_and_reference(experiments)
    file_list = []
    # Set up plot.
    n_experiments = len(experiments)
    if all_in_one:
        ref_experiment = experiments[0]
        setup_plot(
            plot_type=plot_type,
            solver_name=solver_set_name,
            problem_name=ref_experiment.problem.name,
            normalize=normalize,
            budget=ref_experiment.problem.factors["budget"],
            plot_title=plot_title,
        )
        # solver_curve_handles = []
        if normalize:
            terminal_data = [
                [
                    experiment.progress_curves[mrep].y_vals[-1]
                    for mrep in range(experiment.n_macroreps)
                ]
                for experiment in experiments
            ]
        else:
            terminal_data = [
                [
                    experiment.objective_curves[mrep].y_vals[-1]
                    for mrep in range(experiment.n_macroreps)
                ]
                for experiment in experiments
            ]
        if plot_type == PlotType.BOX:
            plt.boxplot(terminal_data)
            plt.xticks(
                range(1, n_experiments + 1),
                labels=[experiment.solver.name for experiment in experiments],
            )
        elif plot_type == PlotType.VIOLIN:
            import seaborn as sns

            # Construct dictionary of lists directly
            terminal_data_dict = {
                "Solvers": [
                    experiments[exp_idx].solver.name
                    for exp_idx in range(n_experiments)
                    for _ in terminal_data[exp_idx]
                ],
                "Terminal": [
                    td
                    for exp_idx in range(n_experiments)
                    for td in terminal_data[exp_idx]
                ],
            }

            sns.violinplot(
                x="Solvers",
                y="Terminal",
                data=terminal_data_dict,
                inner="stick",
                density_norm="width",
                cut=0.1,
                hue="Solvers",
            )

            plt.ylabel("Terminal Progress" if normalize else "Terminal Objective")

        file_list.append(
            save_plot(
                solver_name=solver_set_name,
                problem_name=ref_experiment.problem.name,
                plot_type=plot_type,
                normalize=normalize,
                plot_title=plot_title,
                ext=ext,
                save_as_pickle=save_as_pickle,
            )
        )
    else:  # Plot separately.
        for experiment in experiments:
            setup_plot(
                plot_type=plot_type,
                solver_name=experiment.solver.name,
                problem_name=experiment.problem.name,
                normalize=normalize,
                budget=experiment.problem.factors["budget"],
            )
            if normalize:
                curves = experiment.progress_curves
            else:
                curves = experiment.objective_curves
            terminal_data = [curve.y_vals[-1] for curve in curves]
            if plot_type == PlotType.BOX:
                plt.boxplot(terminal_data)
                plt.xticks([1], labels=[experiment.solver.name])
            if plot_type == PlotType.VIOLIN:
                terminal_data_dict = {
                    "Solver": [experiment.solver.name] * len(terminal_data),
                    "Terminal": terminal_data,
                }
                import seaborn as sns

                sns.violinplot(
                    x=terminal_data_dict["Solver"],
                    y=terminal_data_dict["Terminal"],
                    inner="stick",
                )
            if normalize:
                plt.ylabel("Terminal Progress")
            else:
                plt.ylabel("Terminal Objective")
            file_list.append(
                save_plot(
                    solver_name=experiment.solver.name,
                    problem_name=experiment.problem.name,
                    plot_type=plot_type,
                    normalize=normalize,
                    ext=ext,
                    save_as_pickle=save_as_pickle,
                )
            )
    return file_list
