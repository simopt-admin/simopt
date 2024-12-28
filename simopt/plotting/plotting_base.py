from abc import abstractmethod

from matplotlib.figure import Figure

from simopt.experiment_base import ProblemSolver


class PlottingBase:
    @property
    def plot(self) -> Figure | None:
        return self._plot
    
    @plot.setter
    def plot(self, value: Figure | None) -> None:
        self._plot = value

    @property
    def experiments(self) -> list[ProblemSolver]:
        return self._experiments

    @experiments.setter
    def experiments(self, value: list[ProblemSolver]) -> None:
        try:
            assert isinstance(value, list)
            assert all(
                isinstance(experiment, ProblemSolver) for experiment in value
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
    
    @property
    def file_list(self) -> list[str]:
        return self._file_list

    def __init__(
        self,
        experiments: list[ProblemSolver],
        all_in_one: bool,
        title: str | None,
    ) -> None:
        self.experiments = experiments
        self.all_in_one = all_in_one
        self.title = title
        self._file_list = []
        self._plot = None

    @abstractmethod
    def generate_plot(self) -> None:
        pass

    def save_to_pickle(self, filename: str) -> list[str]:
        return []

    def load_from_pickle(self, filename: str) -> None:
        pass

    def save_to_image(self, filename: str, extension: str) -> None:
        pass
