import bisect
import itertools
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


class Gamma(InputModel):
    def set_rng(self, rng: random.Random) -> None:
        self.rng = rng

    def unset_rng(self) -> None:
        self.rng = None

    def random(self, alpha: float, beta: float) -> float:
        return self.rng.gammavariate(alpha, beta)


class WeightedChoice(InputModel):
    def set_rng(self, rng: random.Random) -> None:
        self.rng = rng

    def unset_rng(self) -> None:
        self.rng = None

    def random(self, population, weights) -> float:
        # Calculate cumulative weights
        cum_weights = list(itertools.accumulate(weights))
        # Generate a value somewhere between 0 and the sum of weights
        x = self.rng.random() * cum_weights[-1]
        # Find the index of the first cumulative weight that is >= x
        # Return the corresponding element from the population
        return population[bisect.bisect(cum_weights, x)]


class Poisson(InputModel):
    def set_rng(self, rng: random.Random) -> None:
        self.rng = rng

    def unset_rng(self) -> None:
        self.rng = None

    def random(self, lam: float) -> int:
        return self.rng.poissonvariate(lam)
