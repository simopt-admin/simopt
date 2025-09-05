"""Enumeration classes for problem types."""

from enum import Enum


class ObjectiveType(Enum):
    """Enum class for objective types."""

    SINGLE = 1
    MULTI = 2

    def symbol(self) -> str:
        """Return the symbol of the objective type."""
        symbol_mapping = {ObjectiveType.SINGLE: "S", ObjectiveType.MULTI: "M"}
        return symbol_mapping.get(self, "?")


class ConstraintType(Enum):
    """Enum class for constraint types."""

    UNCONSTRAINED = 1
    BOX = 2
    DETERMINISTIC = 3
    STOCHASTIC = 4

    def symbol(self) -> str:
        """Return the symbol of the constraint type."""
        symbol_mapping = {
            ConstraintType.UNCONSTRAINED: "U",
            ConstraintType.BOX: "B",
            ConstraintType.DETERMINISTIC: "D",
            ConstraintType.STOCHASTIC: "S",
        }
        return symbol_mapping.get(self, "?")


class VariableType(Enum):
    """Enum class for variable types."""

    DISCRETE = 1
    CONTINUOUS = 2
    MIXED = 3

    def symbol(self) -> str:
        """Return the symbol of the variable type."""
        symbol_mapping = {
            VariableType.DISCRETE: "D",
            VariableType.CONTINUOUS: "C",
            VariableType.MIXED: "M",
        }
        return symbol_mapping.get(self, "?")
