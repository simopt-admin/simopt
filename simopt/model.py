"""Base class for simulation models used in simulation optimization problems."""

from abc import ABC, abstractmethod
from typing import ClassVar

from boltons.typeutils import classproperty
from pydantic import BaseModel

from mrg32k3a.mrg32k3a import MRG32k3a


def _get_specifications(config_class: type[BaseModel]) -> dict[str, dict]:
    spec = {}
    for name, field in config_class.model_fields.items():
        spec[name] = {
            "description": field.description,
            "datatype": field.annotation,
            "default": field.default,
        }
    return spec


class Model(ABC):
    """Base class for simulation models used in simulation-optimization problems.

    Each model defines the simulation logic behind a given problem instance.
    """

    config_class: ClassVar[type[BaseModel]]

    @classproperty
    def class_name_abbr(cls) -> str:
        """Short name of the model class."""
        return cls.__name__.capitalize()

    @classproperty
    def class_name(cls) -> str:
        """Long name of the model class."""
        # Insert spaces before capital letters
        import re

        return re.sub(r"(?<!^)(?=[A-Z])", " ", cls.__name__)

    @classproperty
    def name(cls) -> str:
        """Name of model."""
        return cls.__name__.replace("_", " ")

    @classproperty
    @abstractmethod
    def n_rngs(cls) -> int:
        """Number of random-number generators used to run a simulation replication."""
        raise NotImplementedError

    @classproperty
    @abstractmethod
    def n_responses(cls) -> int:
        """Number of responses (performance measures)."""
        raise NotImplementedError

    @classproperty
    def specifications(cls) -> dict[str, dict]:
        """Details of each factor (for GUI, data validation, and defaults)."""
        return _get_specifications(cls.config_class)

    @property
    def factors(self) -> dict:
        """Changeable factors of the simulation model."""
        # TODO: this is currently needed because the solver may update the factors
        return self._factors

    @factors.setter
    def factors(self, value: dict | None) -> None:
        if value is None:
            value = {}
        self.__factors = value

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

        Args:
            rng_list (list[``mrg32k3a.mrg32k3a.MRG32k3a``]): List of random-number
                generators used to generate a random replication.

        Returns:
            tuple:
                - dict: Performance measures of interest.
                - dict: Gradient estimates for each response.
        """
        raise NotImplementedError
