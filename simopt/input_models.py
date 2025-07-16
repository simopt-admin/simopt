import random
from abc import ABC, abstractmethod


class InputModel(ABC):
    @abstractmethod
    def set_rng(self, rng: random.Random) -> None:
        pass

    @abstractmethod
    def unset_rng(self) -> None:
        pass

    @abstractmethod
    def random(self, *args, **kwargs) -> float:
        pass
