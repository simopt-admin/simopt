from enum import Enum

from matplotlib import pyplot as plt
from plotting_base import PlottingBase
from simopt.experiment_base import (
    ProblemSolver,
    bootstrap_procedure,
    check_common_problem_and_reference,
    mean_of_curves,
    plot_bootstrap_conf_ints,
    quantile_of_curves,
    report_max_halfwidth,
    save_plot,
    setup_plot,
)


class PlottingTypes(Enum):
    """Enum class for the different types of plots."""

    ALL = "all"
    MEAN = "mean"
    QUANTILE = "quantile"


class ProgressCurve(PlottingBase):
    @property
    def plot_type(self) -> PlottingTypes:
        return self._plot_type

    @plot_type.setter
    def plot_type(self, value: PlottingTypes) -> None:
        if not isinstance(value, PlottingTypes):
            raise ValueError(
                "The plot_type value must be a PlottingTypes object."
            )
        self._plot_type = value

    @property
    def beta(self) -> float:
        return self._beta

    @beta.setter
    def beta(self, value: float) -> None:
        if not isinstance(value, float):
            raise ValueError("The beta value must be a float.")
        if not 0 < value < 1:
            raise ValueError("The beta quantile must be in the range (0, 1).")
        self._beta = value

    @property
    def normalize(self) -> bool:
        return self._normalize

    @normalize.setter
    def normalize(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise ValueError("The normalize value must be a boolean.")
        self._normalize = value

    @property
    def n_bootstraps(self) -> int:
        return self._n_bootstraps

    @n_bootstraps.setter
    def n_bootstraps(self, value: int) -> None:
        if not isinstance(value, int):
            raise ValueError("The n_bootstraps value must be an integer.")
        if value < 1:
            raise ValueError("The n_bootstraps value must be greater than 0.")
        self._n_bootstraps = value

    @property
    def conf_level(self) -> float:
        return self._conf_level

    @conf_level.setter
    def conf_level(self, value: float) -> None:
        if not isinstance(value, float):
            raise ValueError("The conf_level value must be a float.")
        if not 0 < value < 1:
            raise ValueError(
                "The conf_level value must be in the range (0, 1)."
            )
        self._conf_level = value

    @property
    def plot_conf_ints(self) -> bool:
        return self._plot_conf_ints

    @plot_conf_ints.setter
    def plot_conf_ints(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise ValueError("The plot_conf_ints value must be a boolean.")
        self._plot_conf_ints = value

    @property
    def print_max_hw(self) -> bool:
        return self._print_max_hw

    @print_max_hw.setter
    def print_max_hw(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise ValueError("The print_max_hw value must be a boolean.")
        self._print_max_hw = value

    @property
    def legend_loc(self) -> str | None:
        return self._legend_loc

    @legend_loc.setter
    def legend_loc(self, value: str | None) -> None:
        if not isinstance(value, (str, type(None))):
            raise ValueError("The legend_loc value must be a string or None.")
        self._legend_loc = value

    @property
    def solver_set_name(self) -> str:
        return self._solver_set_name

    @solver_set_name.setter
    def solver_set_name(self, value: str) -> None:
        if not isinstance(value, str):
            raise ValueError("The solver_set_name value must be a string.")
        self._solver_set_name = value

    def __init__(
        self,
        experiments: list[ProblemSolver],
        all_in_one: bool,
        title: str,
        plot_type: PlottingTypes,
        beta: float = 0.50,
        normalize: bool = True,
        n_bootstraps: int = 100,
        conf_level: float = 0.95,
        plot_conf_ints: bool = True,
        print_max_hw: bool = True,
        legend_loc: str | None = None,
        solver_set_name: str = "SOLVER_SET",
    ) -> None:
        # Set default values
        if legend_loc is None:
            legend_loc = "best"
        # Check if problems are the same with the same x0 and x*.
        check_common_problem_and_reference(experiments)
        super().__init__(experiments, all_in_one, title)
        self.plot_type = plot_type
        self.beta = beta
        self.normalize = normalize
        self.n_bootstraps = n_bootstraps
        self.conf_level = conf_level
        self.plot_conf_ints = plot_conf_ints
        self.print_max_hw = print_max_hw
        self.legend_loc = legend_loc
        self.solver_set_name = solver_set_name

    def generate_plot(self) -> None:
        if self.all_in_one:
            self.__plot_all_in_one()
        else:
            self.__plot_separately()

    def __plot_all_in_one(self) -> None:
        # Set up plot.
        ref_experiment = self.experiments[0]
        setup_plot(
            plot_type=self.plot_type.value,
            solver_name=self.solver_set_name,
            problem_name=ref_experiment.problem.name,
            normalize=self.normalize,
            budget=ref_experiment.problem.factors["budget"],
            beta=self.beta,
            plot_title=self.title,
        )
        solver_curve_handles = []
        curve_pairs = []
        for exp_idx in range(len(self.experiments)):
            experiment = self.experiments[exp_idx]
            color_str = "C" + str(exp_idx)
            estimator = None
            plot_conf_ints = self.plot_conf_ints or self.print_max_hw
            # Determine which curves to use
            if self.normalize:
                curves = experiment.progress_curves
            else:
                curves = experiment.objective_curves
            # Plot those curves
            if self.plot_type == PlottingTypes.ALL:
                handle = curves[0].plot(color_str=color_str)
                for curve in curves[1:]:
                    curve.plot(color_str=color_str)
                plot_conf_ints = False
            elif self.plot_type == PlottingTypes.MEAN:
                estimator = mean_of_curves(curves)
                handle = estimator.plot(color_str=color_str)
            elif self.plot_type == PlottingTypes.QUANTILE:
                estimator = quantile_of_curves(curves, self.beta)
                handle = estimator.plot(color_str=color_str)
            else:
                raise ValueError("Invalid plot type.")
            solver_curve_handles.append(handle)
            # Plot the confidence intervals and/or report max halfwidth
            if plot_conf_ints:
                # Do some assertions to make type checking happy
                plot_type = self.plot_type.value
                assert plot_type != "all"
                # Note: "experiments" needs to be a list of list of ProblemSolver objects.
                bs_conf_int_lb_curve, bs_conf_int_ub_curve = (
                    bootstrap_procedure(
                        experiments=[[experiment]],
                        n_bootstraps=self.n_bootstraps,
                        conf_level=self.conf_level,
                        plot_type=plot_type,
                        beta=self.beta,
                        estimator=estimator,
                        normalize=self.normalize,
                    )
                )
                if self.plot_conf_ints:
                    if isinstance(
                        bs_conf_int_lb_curve, (int, float)
                    ) or isinstance(bs_conf_int_ub_curve, (int, float)):
                        error_msg = "Bootstrap confidence intervals are not available for scalar estimators."
                        raise ValueError(error_msg)
                    plot_bootstrap_conf_ints(
                        bs_conf_int_lb_curve,
                        bs_conf_int_ub_curve,
                        color_str=color_str,
                    )
                if self.print_max_hw:
                    curve_pairs.append(
                        [bs_conf_int_lb_curve, bs_conf_int_ub_curve]
                    )
        plt.legend(
            handles=solver_curve_handles,
            labels=[experiment.solver.name for experiment in self.experiments],
            loc=self.legend_loc,
        )
        if self.print_max_hw and self.plot_type != "all":
            report_max_halfwidth(
                curve_pairs=curve_pairs,
                normalize=self.normalize,
                conf_level=self.conf_level,
            )
        self._file_list.append(
            save_plot(
                solver_name=self.solver_set_name,
                problem_name=ref_experiment.problem.name,
                plot_type=self.plot_type.value,
                normalize=self.normalize,
                extra=self.beta,
                plot_title=self.title,
                ext="png",
                save_as_pickle=False,
            )
        )

    def __plot_separately(self) -> None:
        for experiment in self.experiments:
            setup_plot(
                plot_type=self.plot_type.value,
                solver_name=experiment.solver.name,
                problem_name=experiment.problem.name,
                normalize=self.normalize,
                budget=experiment.problem.factors["budget"],
                beta=self.beta,
            )
            estimator = None
            if self.plot_type == "all":
                # Plot all estimated progress curves.
                if self.normalize:
                    for curve in experiment.progress_curves:
                        curve.plot()
                else:
                    for curve in experiment.objective_curves:
                        curve.plot()
            elif self.plot_type == "mean":
                # Plot estimated mean progress curve.
                if self.normalize:
                    estimator = mean_of_curves(experiment.progress_curves)
                else:
                    estimator = mean_of_curves(experiment.objective_curves)
                estimator.plot()
            else:  # Must be quantile.
                # Plot estimated beta-quantile progress curve.
                if self.normalize:
                    estimator = quantile_of_curves(
                        experiment.progress_curves, self.beta
                    )
                else:
                    estimator = quantile_of_curves(
                        experiment.objective_curves, self.beta
                    )
                estimator.plot()
            if (
                self.plot_conf_ints or self.print_max_hw
            ) and self.plot_type != "all":
                plot_type = self.plot_type.value
                assert plot_type != "all"
                # Note: "experiments" needs to be a list of list of ProblemSolvers.
                bs_conf_int_lb_curve, bs_conf_int_ub_curve = (
                    bootstrap_procedure(
                        experiments=[[experiment]],
                        n_bootstraps=self.n_bootstraps,
                        conf_level=self.conf_level,
                        plot_type=plot_type,
                        beta=self.beta,
                        estimator=estimator,
                        normalize=self.normalize,
                    )
                )
                if self.plot_conf_ints:
                    if isinstance(
                        bs_conf_int_lb_curve, (int, float)
                    ) or isinstance(bs_conf_int_ub_curve, (int, float)):
                        error_msg = "Bootstrap confidence intervals are not available for scalar estimators."
                        raise ValueError(error_msg)
                    plot_bootstrap_conf_ints(
                        bs_conf_int_lb_curve, bs_conf_int_ub_curve
                    )
                if self.print_max_hw:
                    if isinstance(
                        bs_conf_int_lb_curve, (int, float)
                    ) or isinstance(bs_conf_int_ub_curve, (int, float)):
                        error_msg = "Max halfwidth is not available for scalar estimators."
                        raise ValueError(error_msg)
                    report_max_halfwidth(
                        curve_pairs=[
                            [bs_conf_int_lb_curve, bs_conf_int_ub_curve]
                        ],
                        normalize=self.normalize,
                        conf_level=self.conf_level,
                    )
            self._file_list.append(
                save_plot(
                    solver_name=experiment.solver.name,
                    problem_name=experiment.problem.name,
                    plot_type=self.plot_type.value,
                    normalize=self.normalize,
                    extra=self.beta,
                    ext="png",
                    save_as_pickle=False,
                )
            )
