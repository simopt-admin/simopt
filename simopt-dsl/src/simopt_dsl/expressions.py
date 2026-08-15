"""Expressions used to declare objectives and constraints."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from simopt_dsl.simulation import Simulation


class Expression:
    """Base class for scalar, replication-level expressions."""

    def evaluate(self, context: EvaluationContext) -> float:
        raise NotImplementedError

    def __add__(self, other: object) -> Expression | AggregateExpression:
        return _binary_expression("+", self, other)

    def __radd__(self, other: object) -> Expression | AggregateExpression:
        return _binary_expression("+", other, self)

    def __sub__(self, other: object) -> Expression | AggregateExpression:
        return _binary_expression("-", self, other)

    def __rsub__(self, other: object) -> Expression | AggregateExpression:
        return _binary_expression("-", other, self)

    def __mul__(self, other: object) -> Expression | AggregateExpression:
        return _binary_expression("*", self, other)

    def __rmul__(self, other: object) -> Expression | AggregateExpression:
        return _binary_expression("*", other, self)

    def __truediv__(self, other: object) -> Expression | AggregateExpression:
        return _binary_expression("/", self, other)

    def __rtruediv__(self, other: object) -> Expression | AggregateExpression:
        return _binary_expression("/", other, self)

    def __neg__(self) -> Expression:
        return BinaryExpression("*", Constant(-1.0), self)

    def __le__(self, other: object) -> Constraint:
        return Constraint(self, "<=", as_expression(other))

    def __ge__(self, other: object) -> Constraint:
        return Constraint(self, ">=", as_expression(other))


@dataclass(frozen=True)
class Constant(Expression):
    value: float

    def evaluate(self, context: EvaluationContext) -> float:
        return self.value


@dataclass(frozen=True)
class Metric(Expression):
    simulation: Simulation
    name: str
    indices: tuple[int, ...] = ()

    def evaluate(self, context: EvaluationContext) -> float:
        value = context.metrics[(self.simulation.name, self.name)]
        for index in self.indices:
            value = value[index]
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            suffix = "".join(f"[{index}]" for index in self.indices)
            raise TypeError(
                f"simulation metric {self.name!r}{suffix} is not scalar; "
                "select a scalar component with metric[index]"
            ) from exc

    def __getitem__(self, index: int) -> Metric:
        if not isinstance(index, int):
            raise TypeError("simulation metric indices must be integers")
        return Metric(self.simulation, self.name, (*self.indices, index))


@dataclass(frozen=True)
class BinaryExpression(Expression):
    operator: str
    left: Expression
    right: Expression

    def evaluate(self, context: EvaluationContext) -> float:
        return apply_binary_operator(
            self.operator,
            self.left.evaluate(context),
            self.right.evaluate(context),
        )


@dataclass(frozen=True)
class Constraint:
    left: Expression
    sense: str
    right: Expression

    def satisfied(self, context: EvaluationContext) -> bool:
        left = self.left.evaluate(context)
        right = self.right.evaluate(context)
        if self.sense == "<=":
            return left <= right
        if self.sense == ">=":
            return left >= right
        raise ValueError(f"unknown constraint sense {self.sense!r}")

    def residual(self) -> Expression:
        """Return an expression whose feasible values are nonpositive."""
        if self.sense == "<=":
            return BinaryExpression("-", self.left, self.right)
        if self.sense == ">=":
            return BinaryExpression("-", self.right, self.left)
        raise ValueError(f"unknown constraint sense {self.sense!r}")


class AggregateExpression:
    """Base class for expressions estimated over simulation replications."""

    def __add__(self, other: object) -> AggregateExpression:
        return BinaryAggregateExpression("+", self, as_aggregate_expression(other))

    def __radd__(self, other: object) -> AggregateExpression:
        return BinaryAggregateExpression("+", as_aggregate_expression(other), self)

    def __sub__(self, other: object) -> AggregateExpression:
        return BinaryAggregateExpression("-", self, as_aggregate_expression(other))

    def __rsub__(self, other: object) -> AggregateExpression:
        return BinaryAggregateExpression("-", as_aggregate_expression(other), self)

    def __mul__(self, other: object) -> AggregateExpression:
        return BinaryAggregateExpression("*", self, as_aggregate_expression(other))

    def __rmul__(self, other: object) -> AggregateExpression:
        return BinaryAggregateExpression("*", as_aggregate_expression(other), self)

    def __truediv__(self, other: object) -> AggregateExpression:
        return BinaryAggregateExpression("/", self, as_aggregate_expression(other))

    def __rtruediv__(self, other: object) -> AggregateExpression:
        return BinaryAggregateExpression("/", as_aggregate_expression(other), self)

    def __neg__(self) -> AggregateExpression:
        return BinaryAggregateExpression("*", Constant(-1.0), self)


@dataclass(frozen=True)
class Mean(AggregateExpression):
    expression: Expression


@dataclass(frozen=True)
class BinaryAggregateExpression(AggregateExpression):
    operator: str
    left: Expression | AggregateExpression
    right: Expression | AggregateExpression


@dataclass
class EvaluationContext:
    variables: dict[Any, float]
    metrics: dict[tuple[str, str], Any]
    metric_derivatives: dict[tuple[str, str, tuple[int, ...], str], float]

    def __init__(self, variables: dict[Any, float]) -> None:
        self.variables = variables
        self.metrics = {}
        self.metric_derivatives = {}


def mean(expression: object) -> AggregateExpression:
    """Return the replication mean of a scalar expression."""
    return Mean(as_expression(expression))


def sum(expressions: Iterable[object]) -> Expression:
    """Return the sum of scalar expressions, or zero when empty."""
    iterator = iter(expressions)
    total = as_expression(next(iterator, 0.0))
    for expression in iterator:
        total = BinaryExpression("+", total, as_expression(expression))
    return total


def as_expression(value: object) -> Expression:
    if isinstance(value, Expression):
        return value
    if isinstance(value, (int, float)):
        return Constant(float(value))
    raise TypeError(f"expected an expression or number, got {type(value).__name__}")


def as_aggregate_expression(value: object) -> Expression | AggregateExpression:
    if isinstance(value, (Expression, AggregateExpression)):
        return value
    if isinstance(value, (int, float)):
        return Constant(float(value))
    raise TypeError(
        f"expected an expression, statistic, or number, got {type(value).__name__}"
    )


def apply_binary_operator(operator: str, left: float, right: float) -> float:
    if operator == "+":
        return left + right
    if operator == "-":
        return left - right
    if operator == "*":
        return left * right
    if operator == "/":
        return left / right
    raise ValueError(f"unknown operator {operator!r}")


def _binary_expression(
    operator: str, left: object, right: object
) -> Expression | AggregateExpression:
    if isinstance(left, AggregateExpression) or isinstance(right, AggregateExpression):
        return BinaryAggregateExpression(
            operator,
            as_aggregate_expression(left),
            as_aggregate_expression(right),
        )
    return BinaryExpression(operator, as_expression(left), as_expression(right))


__all__ = ["mean", "sum"]
