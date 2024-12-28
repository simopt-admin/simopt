from plotting_base import PlottingBase
from simopt.experiment_base import ProblemSolver


class AreaScatterplot(PlottingBase):
    @property
    def n_bootstraps(self) -> int:
        return self._n_bootstraps

    @n_bootstraps.setter
    def n_bootstraps(self, value: int) -> None:
        if not isinstance(value, int):
            raise TypeError("The n_bootstraps value must be an integer.")
        if value < 1:
            raise ValueError("The n_bootstraps value must be greater than 0.")
        self._n_bootstraps = value

    @property
    def conf_level(self) -> float:
        return self._conf_level

    @conf_level.setter
    def conf_level(self, value: float) -> None:
        if not isinstance(value, float):
            raise TypeError("The conf_level value must be a float.")
        if value <= 0 or value >= 1:
            raise ValueError("The conf_level value must be between 0 and 1.")
        self._conf_level = value

    @property
    def plot_conf_ints(self) -> bool:
        return self._plot_conf_ints

    @plot_conf_ints.setter
    def plot_conf_ints(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("The plot_conf_ints value must be a boolean.")
        self._plot_conf_ints = value

    @property
    def print_max_hw(self) -> bool:
        return self._print_max_hw

    @print_max_hw.setter
    def print_max_hw(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("The print_max_hw value must be a boolean.")
        self._print_max_hw = value

    @property
    def legend_loc(self) -> str:
        return self._legend_loc

    @legend_loc.setter
    def legend_loc(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError("The legend_loc value must be a string.")
        self._legend_loc = value

    @property
    def solver_set_name(self) -> str:
        return self._solver_set_name

    @solver_set_name.setter
    def solver_set_name(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError("The solver_set_name value must be a string.")
        self._solver_set_name = value

    @property
    def problem_set_name(self) -> str:
        return self._problem_set_name

    @problem_set_name.setter
    def problem_set_name(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError("The problem_set_name value must be a string.")
        self._problem_set_name = value

    def __init__(
        self,
        experiments: list[ProblemSolver],
        all_in_one: bool = True,
        title: str | None = None,
        n_bootstraps: int = 100,
        conf_level: float = 0.95,
        plot_conf_ints: bool = True,
        legend_loc: str = "best",
        solver_set_name: str = "SOLVER_SET",
        problem_set_name: str = "PROBLEM_SET",
    ) -> None:
        super().__init__(experiments, all_in_one, title)
        self.n_bootstraps = n_bootstraps
        self.conf_level = conf_level
        self.plot_conf_ints = plot_conf_ints
        self.legend_loc = legend_loc
        self.solver_set_name = solver_set_name
        self.problem_set_name = problem_set_name
