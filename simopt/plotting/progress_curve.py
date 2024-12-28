from enum import Enum
from plotting_base import PlottingBase
from simopt.experiment_base import ProblemSolver

class PlottingTypes(Enum):
    """Enum class for the different types of plots."""
    all = "all"
    mean = "mean"
    quantile = "quantile"

class ProgressCurve(PlottingBase):
    @property
    def plot_type(self) -> PlottingTypes:
        return self._plot_type
    
    @plot_type.setter
    def plot_type(self, value: PlottingTypes) -> None:
        if not isinstance(value, PlottingTypes):
            raise ValueError("The plot_type value must be a PlottingTypes object.")
        self._plot_type = value

    @property
    def beta(self) -> float:
        return self._beta
    
    @beta.setter
    def beta(self, value: float) -> None:
        if not isinstance(value, float):
            raise ValueError("The beta value must be a float.")
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
        experiments: list[list[ProblemSolver]],
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
        if legend_loc is None:
            legend_loc = "best"
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
        pass
    