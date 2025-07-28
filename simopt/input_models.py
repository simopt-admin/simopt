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


class Exp(InputModel):
    def set_rng(self, rng: random.Random) -> None:
        self.rng = rng

    def unset_rng(self) -> None:
        self.rng = None

    def random(self, lambda_: float) -> float:
        return self.rng.expovariate(lambda_)
