"""Simulation callbacks and result normalization."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np

from simopt_dsl.expressions import EvaluationContext, Metric
from simopt_dsl.variables import DecisionVariable, Variable, components

SimulationResult = tuple[
    Mapping[str, Any],
    Mapping[str, Mapping[str, Any]],
]


@dataclass(frozen=True)
class SimulationEvaluation:
    metrics: dict[str, Any]
    derivatives: dict[tuple[str, tuple[int, ...], str], float]


@dataclass
class Simulation:
    """A simulation callback and its decision-variable bindings."""

    name: str
    run: Callable[..., SimulationResult]
    decisions: dict[str, DecisionVariable]
    n_rngs: int = 1

    def metric(self, name: str) -> Metric:
        """Return a scalar expression referring to a simulation response."""
        if not isinstance(name, str):
            raise TypeError("simulation metric name must be a string")
        if not name:
            raise ValueError("simulation metric name cannot be empty")
        return Metric(self, name)

    def evaluate(
        self, context: EvaluationContext, rngs: Sequence[Any]
    ) -> SimulationEvaluation:
        """Evaluate the callback once and normalize its metrics and derivatives."""
        decisions = {
            name: variable.evaluate(context)
            for name, variable in self.decisions.items()
        }
        result = self.run(decisions, rngs)
        return _coerce_evaluation(result, self.decisions)


def component_items(
    decisions: Mapping[str, DecisionVariable],
) -> tuple[tuple[str, Variable], ...]:
    items: list[tuple[str, Variable]] = []
    seen_names: set[str] = set()
    for decision_name, decision in decisions.items():
        decision_components = components(decision)
        component_names = (
            (decision_name,)
            if isinstance(decision, Variable)
            else tuple(
                f"{decision_name}[{index}]" for index in range(len(decision_components))
            )
        )
        for component_name, component in zip(
            component_names, decision_components, strict=True
        ):
            if component_name in seen_names:
                raise ValueError(
                    f"simulation decision component name {component_name!r} is ambiguous"
                )
            seen_names.add(component_name)
            items.append((component_name, component))
    return tuple(items)


def _coerce_evaluation(
    result: Any, decisions: Mapping[str, DecisionVariable]
) -> SimulationEvaluation:
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError("simulation must return a (responses, gradients) pair")
    raw_metrics, raw_derivatives = result
    if not isinstance(raw_metrics, Mapping):
        raise TypeError("simulation response payload must be a mapping")
    if not isinstance(raw_derivatives, Mapping):
        raise TypeError("simulation gradient payload must be a mapping")

    metrics: dict[str, Any] = {}
    for name, value in raw_metrics.items():
        if not isinstance(name, str):
            raise TypeError("simulation metric names must be strings")
        metrics[name] = value

    return SimulationEvaluation(
        metrics,
        _coerce_derivatives(raw_derivatives, decisions, metrics),
    )


def _coerce_derivatives(
    raw: Any,
    decisions: Mapping[str, DecisionVariable],
    metrics: Mapping[str, Any],
) -> dict[tuple[str, tuple[int, ...], str], float]:
    if not isinstance(raw, Mapping):
        raise TypeError("simulation derivatives must be a mapping")

    derivatives: dict[tuple[str, tuple[int, ...], str], float] = {}
    for metric_name, derivative_values in raw.items():
        if not isinstance(metric_name, str):
            raise TypeError("gradient response names must be strings")
        metric_shape = _metric_shape(metrics, metric_name)
        if not isinstance(derivative_values, Mapping):
            raise TypeError(
                f"gradients for response {metric_name!r} must be a mapping"
            )
        for decision_name, value in derivative_values.items():
            if not isinstance(decision_name, str):
                raise TypeError("gradient decision names must be strings")
            if decision_name in decisions:
                _store_decision_derivatives(
                    derivatives,
                    metric_name,
                    metric_shape,
                    decision_name,
                    decisions[decision_name],
                    value,
                )
    return derivatives


def _store_decision_derivatives(
    derivatives: dict[tuple[str, tuple[int, ...], str], float],
    metric_name: str,
    metric_shape: tuple[int, ...],
    decision_name: str,
    decision: DecisionVariable,
    raw_values: Any,
) -> None:
    items = component_items({decision_name: decision})
    ordered_values = _flatten_values(raw_values)
    indices = _metric_indices(metric_shape)
    expected = len(indices) * len(items)
    if len(ordered_values) != expected:
        raise ValueError(
            f"derivative for decision {decision_name!r} must have "
            f"{expected} component(s), got {len(ordered_values)}"
        )
    offset = 0
    for metric_index in indices:
        for component_name, _ in items:
            derivatives[(metric_name, metric_index, component_name)] = ordered_values[
                offset
            ]
            offset += 1


def _metric_shape(metrics: Mapping[str, Any], metric_name: str) -> tuple[int, ...]:
    if metric_name not in metrics:
        raise ValueError(f"derivative provided for unknown metric {metric_name!r}")
    value = metrics[metric_name]
    if _is_scalar(value):
        return ()
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"simulation metric {metric_name!r} must be scalar or array-like"
        ) from exc
    if array.dtype == object:
        raise TypeError(
            f"simulation metric {metric_name!r} must have a rectangular shape"
        )
    return tuple(int(size) for size in array.shape)


def _metric_indices(shape: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    if not shape:
        return ((),)
    return tuple(
        tuple(int(index) for index in indices) for indices in np.ndindex(shape)
    )


def _is_scalar(value: Any) -> bool:
    if isinstance(value, Real):
        return True
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _flatten_values(raw: Any) -> tuple[float, ...]:
    if _is_scalar(raw):
        return (float(raw),)
    if isinstance(raw, (str, bytes)):
        raise TypeError("derivative values must contain ordered numeric values")
    try:
        values = tuple(raw)
    except TypeError as exc:
        raise TypeError(
            "derivative values must contain ordered numeric values"
        ) from exc
    return tuple(item for value in values for item in _flatten_values(value))


__all__ = ["Simulation"]
