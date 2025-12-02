"""Simple input-model wrappers for common distributions."""

import bisect
import itertools
import math
from abc import abstractmethod
from collections.abc import Sequence
from random import Random
from typing import ParamSpec, Protocol, TypeVar

P = ParamSpec("P")
R = TypeVar("R", covariant=True)


class InputModel(Protocol[P, R]):
    """Abstract base for input models used by simulations."""

    rng: Random | None = None

    def set_rng(self, rng: Random) -> None:
        """Attach a Python RNG to the input model.

        Args:
            rng (random.Random): Random number generator to use for sampling.
        """
        self.rng = rng

    def unset_rng(self) -> None:
        """Detach any RNG currently attached to the input model."""
        self.rng = None

    @abstractmethod
    def random(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Generate a random variate from the input model.

        Returns:
            T: A random variate from the input model.
        """
        pass


class Exp(InputModel):
    """Exponential distribution wrapper."""

    def random(self, lambda_: float) -> float:
        """Sample from Exp(lambda_).

        Args:
            lambda_ (float): Rate parameter (lambda > 0).

        Returns:
            float: An exponential variate.
        """
        assert self.rng is not None
        return self.rng.expovariate(lambda_)


class Gamma(InputModel):
    """Gamma distribution wrapper."""

    def random(self, alpha: float, beta: float) -> float:
        """Sample from Gamma(alpha, beta).

        Args:
            alpha (float): Shape parameter.
            beta (float): Scale parameter.

        Returns:
            float: A gamma variate.
        """
        assert self.rng is not None
        return self.rng.gammavariate(alpha, beta)


class WeightedChoice(InputModel):
    """Discrete weighted choice wrapper."""

    def random(self, population: Sequence[object], weights: Sequence[float]) -> object:
        """Sample an element from ``population`` according to ``weights``.

        Args:
            population (Sequence): Items to choose from.
            weights (Sequence[float]): Nonnegative weights for each item.

        Returns:
            Any: A randomly selected element from ``population``.
        """
        # Calculate cumulative weights
        cum_weights = list(itertools.accumulate(weights))
        # Generate a value somewhere between 0 and the sum of weights
        assert self.rng is not None
        x = self.rng.random() * cum_weights[-1]
        # Find the index of the first cumulative weight that is >= x
        # Return the corresponding element from the population
        return population[bisect.bisect(cum_weights, x)]


class Poisson(InputModel):
    """Poisson distribution wrapper."""

    def _poissonvariate(self, lmbda: float) -> int:
        assert self.rng is not None

        if lmbda >= 35:
            return max(
                math.ceil(lmbda + math.sqrt(lmbda) * self.rng.normalvariate() - 0.5),
                0,
            )
        n = 0
        p = self.rng.random()
        threshold = math.exp(-lmbda)
        while p >= threshold:
            p *= self.rng.random()
            n += 1
        return n

    def random(self, lam: float) -> int:
        """Sample from Poisson(lam).

        Args:
            lam (float): Mean rate parameter (lambda >= 0).

        Returns:
            int: A Poisson variate.
        """
        return self._poissonvariate(lam)


class Beta(InputModel):
    """Beta distribution wrapper."""

    def random(self, alpha: float, beta: float) -> float:
        """Sample from Beta(alpha, beta).

        Args:
            alpha (float): Alpha (>= 0).
            beta (float): Beta (>= 0).

        Returns:
            float: A beta variate in [0, 1].
        """
        assert self.rng is not None
        return self.rng.betavariate(alpha, beta)


class Triangular(InputModel):
    """Triangular distribution wrapper."""

    def random(self, low: float, high: float, mode: float) -> float:
        """Sample from Triangular(low, high, mode).

        Args:
            low (float): Lower bound.
            high (float): Upper bound.
            mode (float): Mode of the distribution.

        Returns:
            float: A triangular variate.
        """
        assert self.rng is not None
        return self.rng.triangular(low, high, mode)


class Uniform(InputModel):
    """Uniform distribution wrapper."""

    def random(self, low: float, high: float) -> float:
        """Sample from Uniform(low, high).

        Args:
            low (float): Lower bound.
            high (float): Upper bound.

        Returns:
            float: A uniform variate in [low, high].
        """
        assert self.rng is not None
        return self.rng.uniform(low, high)


class Normal(InputModel):
    """Normal distribution wrapper."""

    def random(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        """Sample from Normal(mu, sigma).

        Args:
            mu (float): Mean.
            sigma (float): Standard deviation.

        Returns:
            float: A normal variate.
        """
        assert self.rng is not None
        return self.rng.normalvariate(mu, sigma)
