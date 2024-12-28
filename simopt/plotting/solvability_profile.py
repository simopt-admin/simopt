from enum import Enum
from plotting_base import PlottingBase
from simopt.experiment_base import ProblemSolver

class PlottingTypes(Enum):
    """Enum class for the different types of plots."""
    cdf_solvability = "cdf_solvability"
    quantile_solvability = "quantile_solvability"
    diff_cdf_solvability = "diff_cdf_solvability"
    diff_quantile_solvability = "diff_quantile_solvability"

class SolvabilityProfile(PlottingBase):
    @property
    def plot_type(self) -> PlottingTypes:
        return self._plot_type
    
    @plot_type.setter
    def plot_type(self, value: PlottingTypes) -> None:
        if not isinstance(value, PlottingTypes):
            raise ValueError("The plot_type value must be a PlottingTypes object.")
        self._plot_type = value

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
            raise ValueError("The conf_level value must be in the range (0, 1).")
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
    def beta(self) -> float:
        return self._beta
    
    @beta.setter
    def beta(self, value: float) -> None:
        if not isinstance(value, float):
            raise ValueError("The beta value must be a float.")
        if not 0 < value < 1:
            raise ValueError("The beta value must be in the range (0, 1).")
        self._beta = value

    @property
    def ref_solver(self) -> str | None:
        return self._ref_solver
    
    @ref_solver.setter
    def ref_solver(self, value: str | None) -> None:
        if not isinstance(value, (str, type(None))):
            raise ValueError("The ref_solver value must be a string or None.")
        self._ref_solver = value

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

    @property
    def solve_tol(self) -> float:
        return self._solve_tol
    
    @solve_tol.setter
    def solve_tol(self, value: float) -> None:
        if not isinstance(value, float):
            raise ValueError("The solve_tol value must be a float.")
        if not 0 < value <= 1:
            raise ValueError("The solve_tol value must be in the range (0, 1].")
        self._solve_tol = value

    def __init__(
        self,
        experiments: list[ProblemSolver],
        plot_type: PlottingTypes,
        all_in_one: bool = True,
        title: str | None = None,
        n_bootstraps: int = 100,
        conf_level: float = 0.95,
        plot_conf_ints: bool = True,
        print_max_hw: bool = True,
        legend_loc: str | None = None,
        beta: float = 0.5,
        ref_solver: str | None = None,
        solver_set_name: str = "SOLVER_SET",
        problem_set_name: str = "PROBLEM_SET",
        solve_tol: float = 0.1,
    ) -> None:
        if legend_loc is None:
            legend_loc = "best"
        super().__init__(experiments, all_in_one, title)
        self.plot_type = plot_type
        self.n_bootstraps = n_bootstraps
        self.conf_level = conf_level
        self.plot_conf_ints = plot_conf_ints
        self.print_max_hw = print_max_hw
        self.legend_loc = legend_loc
        self.beta = beta
        self.ref_solver = ref_solver
        self.solver_set_name = solver_set_name
        self.problem_set_name = problem_set_name
        self.solve_tol = solve_tol