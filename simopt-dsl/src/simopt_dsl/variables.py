"""Scalar and vector decision variables."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import overload

from simopt_dsl.expressions import EvaluationContext, Expression


@dataclass(frozen=True)
class Variable(Expression):
    """A scalar decision variable."""

    name: str
    lb: float
    ub: float
    initial: float
    integer: bool = False

    def evaluate(self, context: EvaluationContext) -> float:
        return context.variables[self]


@dataclass(frozen=True)
class VectorVariable:
    """A one-dimensional decision variable with scalar expression components."""

    name: str
    components: tuple[Variable, ...]

    @property
    def shape(self) -> tuple[int]:
        return (len(self.components),)

    @property
    def lb(self) -> tuple[float, ...]:
        return tuple(component.lb for component in self.components)

    @property
    def ub(self) -> tuple[float, ...]:
        return tuple(component.ub for component in self.components)

    @property
    def initial(self) -> tuple[float, ...]:
        return tuple(component.initial for component in self.components)

    def evaluate(self, context: EvaluationContext) -> tuple[float, ...]:
        return tuple(component.evaluate(context) for component in self.components)

    def __len__(self) -> int:
        return len(self.components)

    def __iter__(self) -> Iterator[Variable]:
        return iter(self.components)

    @overload
    def __getitem__(self, index: int) -> Variable: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[Variable, ...]: ...

    def __getitem__(self, index: int | slice) -> Variable | tuple[Variable, ...]:
        return self.components[index]


DecisionVariable = Variable | VectorVariable


def components(decision: DecisionVariable) -> tuple[Variable, ...]:
    return (decision,) if isinstance(decision, Variable) else decision.components


__all__ = ["Variable", "VectorVariable"]
