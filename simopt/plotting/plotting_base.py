from abc import abstractmethod

from simopt.experiment_base import ProblemSolver


class PlottingBase:
    @property
    def experiments(self) -> list[list[ProblemSolver]]:
        return self._experiments

    @experiments.setter
    def experiments(self, value: list[list[ProblemSolver]]) -> None:
        # Check that the value is a list of lists of ProblemSolver objects
        try:
            assert isinstance(value, list)
            assert all(
                isinstance(problem_solver_list, list)
                for problem_solver_list in value
            )
            assert all(
                isinstance(problem_solver, ProblemSolver)
                for problem_solver_list in value
                for problem_solver in problem_solver_list
            )
        except TypeError as e:
            raise ValueError(
                "The experiments value must be a list of lists of ProblemSolver objects."
            ) from e
        self._experiments = value

    @property
    def all_in_one(self) -> bool:
        return self._all_in_one

    @all_in_one.setter
    def all_in_one(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise ValueError("The all_in_one value must be a boolean.")
        self._all_in_one = value

    @property
    def title(self) -> str | None:
        return self._title

    @title.setter
    def title(self, value: str | None) -> None:
        if not isinstance(value, str):
            raise ValueError("The title value must be a string.")
        self._title = value

    def __init__(
        self,
        experiments: list[list[ProblemSolver]],
        all_in_one: bool,
        title: str | None,
    ) -> None:
        self.experiments = experiments
        self.all_in_one = all_in_one
        self.title = title

    @abstractmethod
    def generate_plot(self) -> None:
        pass

    def save_to_pickle(self, filename: str) -> None:
        pass

    def load_from_pickle(self, filename: str) -> None:
        pass

    def save_to_image(self, filename: str, extension: str) -> None:
        pass
