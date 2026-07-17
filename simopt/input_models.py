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
T = TypeVar("T")


class InputModel(Protocol[P, R]):
    """Abstract base for input models used by simulations."""

    @abstractmethod
    def random(self, rng: Random, *args: P.args, **kwargs: P.kwargs) -> R:
        """Generate a random variate from the input model.

        Args:
            rng (random.Random): Random number generator used for sampling.
            *args: Positional distribution parameters.
            **kwargs: Keyword distribution parameters.

        Returns:
            T: A random variate from the input model.
        """
        pass


class Exp(InputModel):
    """Exponential distribution wrapper."""

    def random(self, rng: Random, lambda_: float) -> float:
        """Sample from Exp(lambda_).

        Args:
            rng (random.Random): Random number generator used for sampling.
            lambda_ (float): Rate parameter (lambda > 0).

        Returns:
            float: An exponential variate.
        """
        return rng.expovariate(lambda_)


class Gamma(InputModel):
    """Gamma distribution wrapper."""

    def random(self, rng: Random, alpha: float, beta: float) -> float:
        """Sample from Gamma(alpha, beta).

        Args:
            rng (random.Random): Random number generator used for sampling.
            alpha (float): Shape parameter.
            beta (float): Scale parameter.

        Returns:
            float: A gamma variate.
        """
        return rng.gammavariate(alpha, beta)


class WeightedChoice(InputModel):
    """Discrete weighted choice wrapper."""

    def random(self, rng: Random, population: Sequence[T], weights: Sequence[float]) -> T:
        """Sample an element from ``population`` according to ``weights``.

        Args:
            rng (random.Random): Random number generator used for sampling.
            population (Sequence): Items to choose from.
            weights (Sequence[float]): Nonnegative weights for each item.

        Returns:
            T: A randomly selected element from ``population``.
        """
        # Calculate cumulative weights
        cum_weights = list(itertools.accumulate(weights))
        # Generate a value somewhere between 0 and the sum of weights
        x = rng.random() * cum_weights[-1]
        # Find the index of the first cumulative weight that is >= x
        # Return the corresponding element from the population
        return population[bisect.bisect(cum_weights, x)]


class Poisson(InputModel):
    """Poisson distribution wrapper."""

    def _poissonvariate(self, rng: Random, lmbda: float) -> int:
        if lmbda >= 35:
            return max(
                math.ceil(lmbda + math.sqrt(lmbda) * rng.normalvariate() - 0.5),
                0,
            )
        n = 0
        p = rng.random()
        threshold = math.exp(-lmbda)
        while p >= threshold:
            p *= rng.random()
            n += 1
        return n

    def random(self, rng: Random, lam: float) -> int:
        """Sample from Poisson(lam).

        Args:
            rng (random.Random): Random number generator used for sampling.
            lam (float): Mean rate parameter (lambda >= 0).

        Returns:
            int: A Poisson variate.
        """
        return self._poissonvariate(rng, lam)


class Beta(InputModel):
    """Beta distribution wrapper."""

    def random(self, rng: Random, alpha: float, beta: float) -> float:
        """Sample from Beta(alpha, beta).

        Args:
            rng (random.Random): Random number generator used for sampling.
            alpha (float): Alpha (>= 0).
            beta (float): Beta (>= 0).

        Returns:
            float: A beta variate in [0, 1].
        """
        return rng.betavariate(alpha, beta)


class Triangular(InputModel):
    """Triangular distribution wrapper."""

    def random(self, rng: Random, low: float, high: float, mode: float) -> float:
        """Sample from Triangular(low, high, mode).

        Args:
            rng (random.Random): Random number generator used for sampling.
            low (float): Lower bound.
            high (float): Upper bound.
            mode (float): Mode of the distribution.

        Returns:
            float: A triangular variate.
        """
        return rng.triangular(low, high, mode)


class Uniform(InputModel):
    """Uniform distribution wrapper."""

    def random(self, rng: Random, low: float, high: float) -> float:
        """Sample from Uniform(low, high).

        Args:
            rng (random.Random): Random number generator used for sampling.
            low (float): Lower bound.
            high (float): Upper bound.

        Returns:
            float: A uniform variate in [low, high].
        """
        return rng.uniform(low, high)


class Normal(InputModel):
    """Normal distribution wrapper."""

    def random(self, rng: Random, mu: float = 0.0, sigma: float = 1.0) -> float:
        """Sample from Normal(mu, sigma).

        Args:
            rng (random.Random): Random number generator used for sampling.
            mu (float): Mean.
            sigma (float): Standard deviation.

        Returns:
            float: A normal variate.
        """
        return rng.normalvariate(mu, sigma)
