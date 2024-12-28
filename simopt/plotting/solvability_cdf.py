from plotting_base import PlottingBase
from simopt.experiment_base import ProblemSolver

class SolvabilityCDF(PlottingBase):
    @property
    def solve_tol(self) -> float:
        return self._solve_tol
    
    @solve_tol.setter
    def solve_tol(self, value: float) -> None:
        if not isinstance(value, float):
            raise ValueError("The solve_tol value must be a float.")
        self._solve_tol = value

    @property
    def n_bootstraps(self) -> int:
        return self._n_bootstraps
    
    @n_bootstraps.setter
    def n_bootstraps(self, value: int) -> None:
        if not isinstance(value, int):
            raise ValueError("The n_bootstraps value must be an integer.")
        self._n_bootstraps = value

    @property
    def conf_level(self) -> float:
        return self._conf_level

    @conf_level.setter
    def conf_level(self, value: float) -> None:
        if not isinstance(value, float):
            raise ValueError("The conf_level value must be a float.")
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
    def legend_loc(self) -> str:
        return self._legend_loc

    @legend_loc.setter
    def legend_loc(self, value: str) -> None:
        if not isinstance(value, str):
            raise ValueError("The legend_loc value must be a string.")
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
        experiments: list[list[ProblemSolver]],
        all_in_one: bool = True,
        title: str | None = None,
        solve_tol: float = 0.1,
        n_bootstraps: int = 100,
        conf_level: float = 0.95,
        plot_conf_ints: bool = True,
        print_max_hw: bool = True,
        legend_loc: str = "best",
        solver_set_name: str = "SOLVER_SET",
    ) -> None:
        if legend_loc is None:
            legend_loc = "best"
        super().__init__(experiments, all_in_one, title)
        self.solve_tol = solve_tol
        self.n_bootstraps = n_bootstraps
        self.conf_level = conf_level
        self.plot_conf_ints = plot_conf_ints
        self.print_max_hw = print_max_hw
        self.legend_loc = legend_loc
        self.solver_set_name = solver_set_name        

