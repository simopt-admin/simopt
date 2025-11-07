"""Base class for simulation models used in simulation optimization problems."""

from abc import ABC, abstractmethod
from typing import ClassVar

from boltons.typeutils import classproperty
from pydantic import BaseModel

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.utils import get_specifications


class Model(ABC):
    """Base class for simulation models used in simulation-optimization problems.

    Each model defines the simulation logic behind a given problem instance.
    """

    class_name_abbr: ClassVar[str]
    """Short name of the model class."""

    class_name: ClassVar[str]
    """Long name of the model class."""

    config_class: ClassVar[type[BaseModel]]
    """Configuration class for the model."""

    n_rngs: ClassVar[int]
    """Number of RNGs used to run a simulation replication."""

    n_responses: ClassVar[int]
    """Number of responses (performance measures)."""

    def __init__(self, fixed_factors: dict | None = None) -> None:
        """Initialize a model object.

        Args:
            fixed_factors (dict | None, optional): Dictionary of user-specified model
                factors.
        """
        # Add all the fixed factors to the model
        fixed_factors = fixed_factors or {}
        self.config = self.config_class(**fixed_factors)
        self._factors = self.config.model_dump(by_alias=True)

    def __eq__(self, other: object) -> bool:
        """Check if two models are equivalent.

        Args:
            other (object): Other object to compare to self.

        Returns:
            bool: True if the two models are equivalent, otherwise False.
        """
        if not isinstance(other, Model):
            return False
        return type(self) is type(other) and self.factors == other.factors

    def __hash__(self) -> int:
        """Return the hash value of the model.

        Returns:
            int: Hash value of the model.
        """
        return hash((self.name, tuple(self.factors.items())))

    @classproperty
    def name(cls) -> str:  # noqa: N805
        """Name of model."""
        return cls.__name__.replace("_", " ")

    @classproperty
    def specifications(cls) -> dict[str, dict]:  # noqa: N805
        """Details of each factor (for GUI, data validation, and defaults)."""
        return get_specifications(cls.config_class)

    @property
    def factors(self) -> dict:
        """Changeable factors of the simulation model."""
        # TODO: this is currently needed because the solver may update the factors
        return self._factors

    def model_created(self) -> None:  # noqa: B027
        """Hook called after the model is constructed.

        Subclasses can override this to use custom input models.
        """
        pass

    @abstractmethod
    def before_replicate(self, rng_list: list[MRG32k3a]) -> None:
        """Prepare the model just before generating a replication.

        Args:
            rng_list (list[MRG32k3a]): RNGs used to drive the simulation.

        Raises:
            NotImplementedError: If the subclass does not implement this hook.
        """
        raise NotImplementedError

    @abstractmethod
    def replicate(self) -> tuple[dict, dict]:
        """Simulate a single replication for the current model factors.

        Returns:
            tuple:
                - dict: Performance measures of interest.
                - dict: Gradient estimates for each response.
        """
        raise NotImplementedError
