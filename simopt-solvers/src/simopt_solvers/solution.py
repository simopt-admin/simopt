"""Solutions of simulation-optimization problems and their replication results."""

from collections.abc import Iterable

import numpy as np
import simopt_dsl as dsl
from simopt_dsl.model import ReplicationEvaluation

from mrg32k3a.mrg32k3a import MRG32k3a


def _to_grad_array(gradients: float | Iterable[float]) -> np.ndarray:
    rv = np.array(gradients)
    if rv.ndim == 0:
        rv = rv.reshape((1,))
    rv[np.isnan(rv)] = 0.0
    return rv


class Solution:
    """Base class for solutions in simulation-optimization problems.

    Solutions are represented by a vector of decision variables and a dictionary of
    associated decision factors.
    """

    @property
    def x(self) -> tuple:
        """Vector of decision variables."""
        return self.__x

    @x.setter
    def x(self, value: tuple) -> None:
        self.__x = value
        self.__dim = len(value)

    @property
    def dim(self) -> int:
        """Number of decision variables describing `x`."""
        return self.__dim

    @property
    def objectives_mean(self) -> np.ndarray:
        """Mean of objectives."""
        mean = np.mean(self.objectives[: self.n_reps], axis=0)
        return np.round(mean, 15)

    @property
    def objectives_var(self) -> np.ndarray:
        """Variance of objectives."""
        if self.n_reps == 1:
            return np.zeros(self.objectives.shape[1])
        var = np.var(self.objectives[: self.n_reps], axis=0, ddof=1)
        return np.round(var, 15)

    @property
    def objectives_stderr(self) -> np.ndarray:
        """Standard error of objectives."""
        if self.n_reps == 1:
            return np.zeros(self.objectives.shape[1])
        std = np.std(self.objectives[: self.n_reps], axis=0, ddof=1)
        sqrt = np.sqrt(self.n_reps)
        return np.round(std / sqrt, 15)

    @property
    def objectives_cov(self) -> np.ndarray:
        """Covariance of objectives."""
        if self.n_reps == 1:
            return np.zeros(self.objectives.shape[1])
        cov = np.cov(self.objectives[: self.n_reps], rowvar=False, ddof=1)
        return np.round(cov, 15)

    @property
    def objectives_gradients_mean(self) -> np.ndarray:
        """Mean of gradients of objectives."""
        if self.n_reps == 1:
            return np.zeros(self.objectives.shape[1])
        mean = np.mean(self.objectives_gradients[: self.n_reps], axis=0)
        return np.round(mean, 15)

    @property
    def objectives_gradients_var(self) -> np.ndarray:
        """Variance of gradients of objectives."""
        if self.n_reps == 1:
            return np.zeros(self.objectives.shape[1])
        var = np.var(self.objectives_gradients[: self.n_reps], axis=0, ddof=1)
        return np.round(var, 15)

    @property
    def objectives_gradients_stderr(self) -> np.ndarray:
        """Standard error of gradients of objectives."""
        if self.n_reps == 1:
            return np.zeros(self.objectives.shape[1])
        std = np.std(self.objectives_gradients[: self.n_reps], axis=0, ddof=1)
        sqrt = np.sqrt(self.n_reps)
        return np.round(std / sqrt, 15)

    @property
    def objectives_gradients_cov(self) -> np.ndarray:
        """Covariance of gradients of objectives."""
        if self.n_reps == 1:
            return np.zeros(self.objectives.shape[1])
        return np.array(
            [
                np.cov(
                    self.objectives_gradients[: self.n_reps, obj],
                    rowvar=False,
                    ddof=1,
                )
                for obj in range(self.objectives.shape[1])
            ]
        )

    @property
    def stoch_constraints_mean(self) -> np.ndarray:
        """Mean of stochastic constraints."""
        if self.stoch_constraints is None:
            return np.array([])
        mean = np.mean(self.stoch_constraints[: self.n_reps], axis=0)
        return np.round(mean, 15)

    @property
    def stoch_constraints_var(self) -> np.ndarray:
        """Variance of stochastic constraints."""
        if self.stoch_constraints is None:
            return np.array([])
        var = np.var(self.stoch_constraints[: self.n_reps], axis=0, ddof=1)
        return np.round(var, 15)

    @property
    def stoch_constraints_stderr(self) -> np.ndarray:
        """Standard error of stochastic constraints."""
        if self.stoch_constraints is None:
            return np.array([])
        std = np.std(self.stoch_constraints[: self.n_reps], axis=0, ddof=1)
        sqrt = np.sqrt(self.n_reps)
        return np.round(std / sqrt, 15)

    @property
    def stoch_constraints_cov(self) -> np.ndarray:
        """Covariance of stochastic constraints."""
        if self.stoch_constraints is None:
            return np.array([])
        cov = np.cov(self.stoch_constraints[: self.n_reps], rowvar=False, ddof=1)
        return np.round(cov, 15)

    @property
    def stoch_constraints_gradients(self) -> np.ndarray:
        """Gradients of stochastic constraints."""
        if not self._stoch_constraints_gradients:
            return np.array([])
        return np.array(self._stoch_constraints_gradients)

    @property
    def stoch_constraints_gradients_mean(self) -> np.ndarray:
        """Mean of gradients of stochastic constraints."""
        if not self._stoch_constraints_gradients:
            return np.array([])
        mean = np.mean(self.stoch_constraints_gradients[: self.n_reps], axis=0)
        return np.round(mean, 15)

    # TODO: implement these properties
    # self.stoch_constraints_gradients_mean = np.mean(
    #     self.stoch_constraints_gradients[: self.n_reps], axis=0
    # )
    # self.stoch_constraints_gradients_var = np.var(
    #     self.stoch_constraints_gradients[: self.n_reps], axis=0, ddof=1
    # )
    # self.stoch_constraints_gradients_stderr = np.std(
    #     self.stoch_constraints_gradients[: self.n_reps], axis=0, ddof=1
    # ) / np.sqrt(self.n_reps)
    # self.stoch_constraints_gradients_cov = np.array(
    #     [
    #         np.cov(
    #             self.stoch_constraints_gradients[: self.n_reps, stcon],
    #             rowvar=False,
    #             ddof=1,
    #         )
    #         for stcon in range(len(self.det_stoch_constraints))
    #     ]
    # )

    def __init__(self, x: tuple, rng_list: list[MRG32k3a]) -> None:
        """Initialize a solution object.

        Args:
            x (tuple): Vector of decision variables.
            rng_list (list[MRG32k3a]): List of RNGs used to run simulation replications.
        """
        super().__init__()
        self.x = x
        self.rng_list = rng_list
        # self.n_reps = 0

        self._objectives = []
        self._objectives_gradients = []
        self._stoch_constraints = []
        self._stoch_constraints_gradients = []
        self._objectives_array: np.ndarray | None = None
        self._objectives_gradients_array: np.ndarray | None = None
        self._stoch_constraints_array: np.ndarray | None = None

    @property
    def n_reps(self) -> int:
        """Number of replications."""
        return len(self._objectives)

    @property
    def objectives(self) -> np.ndarray:
        """Objectives."""
        if self._objectives_array is None:
            self._objectives_array = np.array(self._objectives)
        return self._objectives_array

    @property
    def objectives_gradients(self) -> np.ndarray:
        """Objectives gradients."""
        if self._objectives_gradients_array is None:
            self._objectives_gradients_array = np.array(self._objectives_gradients)
        return self._objectives_gradients_array

    @property
    def stoch_constraints(self) -> np.ndarray:
        """Stochastic constraints."""
        if self._stoch_constraints_array is None:
            self._stoch_constraints_array = np.array(self._stoch_constraints)
        return self._stoch_constraints_array

    def add_replicate_result(self, result: ReplicationEvaluation) -> None:
        """Add a replicate result to the solution.

        Args:
            result: The DSL replication evaluation to add.
        """
        stochastic_constraints = result.stochastic_constraints

        self._objectives.append(np.array([result.objective]))
        self._objectives_array = None

        objective_gradient = result.objective_gradient
        objective_gradient = (
            np.zeros(self.dim) if objective_gradient is None else _to_grad_array(objective_gradient)
        )
        self._objectives_gradients.append(np.array([objective_gradient]))
        self._objectives_gradients_array = None

        self._stoch_constraints.append(
            np.array([constraint.value for constraint in stochastic_constraints])
        )
        self._stoch_constraints_array = None

        gradients = []
        for constraint in stochastic_constraints:
            gradient = constraint.gradient
            gradient = np.zeros(self.dim) if gradient is None else _to_grad_array(gradient)
            gradients.append(gradient)
        self._stoch_constraints_gradients.append(np.array(gradients))
        self._stoch_constraints_gradients_array = None

    def simulate(self, model: dsl.Model, num_macroreps: int = 1) -> None:
        """Simulate `m` i.i.d. replications of `model` at this solution.

        Each replication uses the solution's RNGs, which are then advanced to the
        start of the next subsubstream.

        Args:
            model (dsl.Model): Model to replicate.
            num_macroreps (int, optional): Number of macroreplications to simulate at
                `x`. Defaults to 1.
        """
        # NOTE:
        # Gradients of objective function and stochastic constraint LHSs are temporarily
        # commented out. Under development.

        for _ in range(num_macroreps):
            # Generate one replication at the solution.
            result = model.run_replication(self.x, self.rng_list)
            self.add_replicate_result(result)

            # Advance rngs to start of next subsubstream.
            for rng in self.rng_list:
                rng.advance_subsubstream()
