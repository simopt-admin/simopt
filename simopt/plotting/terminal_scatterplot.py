from simopt.experiment_base import ProblemSolver
from simopt.plotting.plotting_base import PlottingBase


class TerminalScatterplot(PlottingBase):
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

    @property
    def problem_set_name(self) -> str:
        return self._problem_set_name

    @problem_set_name.setter
    def problem_set_name(self, value: str) -> None:
        if not isinstance(value, str):
            raise ValueError("The problem_set_name value must be a string.")
        self._problem_set_name = value

    def __init__(
        self,
        experiments: list[list[ProblemSolver]],
        all_in_one: bool = True,
        title: str | None = None,
        legend_loc: str = "best",
        solver_set_name: str = "SOLVER_SET",
        problem_set_name: str = "PROBLEM_SET",
    ) -> None:
        if legend_loc is None:
            legend_loc = "best"
        super().__init__(experiments, all_in_one, title)
        self.legend_loc = legend_loc
        self.solver_set_name = solver_set_name
        self.problem_set_name = problem_set_name
