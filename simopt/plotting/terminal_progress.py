
from enum import Enum

from simopt.experiment_base import ProblemSolver
from simopt.plotting.plotting_base import PlottingBase


class PlottingType(Enum):
    box = "box"
    violin = "violin"

class TerminalProgress(PlottingBase):
    @property
    def plot_type(self) -> PlottingType:
        return self._plot_type
    
    @plot_type.setter
    def plot_type(self, value: PlottingType) -> None:
        if not isinstance(value, PlottingType):
            raise ValueError("The plot_type value must be a PlottingType object.")
        self._plot_type = value

    @property
    def normalize(self) -> bool:
        return self._normalize
    
    @normalize.setter
    def normalize(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise ValueError("The normalize value must be a boolean.")
        self._normalize = value

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
        title: str | None,
        plot_type: PlottingType,
        normalize: bool,
        solver_set_name: str,
    ) -> None:
        super().__init__(experiments, all_in_one, title)
        self.plot_type = plot_type
        self.normalize = normalize
        self.solver_set_name = solver_set_name