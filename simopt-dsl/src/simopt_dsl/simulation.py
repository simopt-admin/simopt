"""Simulation callbacks and result normalization."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np

from simopt_dsl.expressions import EvaluationContext, Metric
from simopt_dsl.variables import DecisionVariable, Variable, components

SimulationResult = Mapping[str, Any] | tuple[Mapping[str, Any], Mapping[Any, Any]]


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
        result = _invoke(self.run, decisions, rngs)
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


def _invoke(
    callback: Callable[..., SimulationResult],
    decisions: dict[str, float | tuple[float, ...]],
    rngs: Sequence[Any],
) -> SimulationResult:
    candidates = (
        ((), {"decisions": decisions, "rngs": rngs}),
        ((decisions, rngs), {}),
        ((), {"decisions": decisions, "rng": rngs[0]}),
        ((decisions, rngs[0]), {}),
        ((), {**decisions, "rngs": rngs}),
        ((), {**decisions, "rng": rngs[0]}),
        ((), decisions),
    )
    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError):
        return callback(decisions, rngs)

    for args, kwargs in candidates:
        try:
            signature.bind(*args, **kwargs)
        except TypeError:
            continue
        return callback(*args, **kwargs)
    raise TypeError("simulation callback does not accept a supported signature")


_DERIVATIVE_KEYS = ("derivatives", "gradient", "gradients")


def _coerce_evaluation(
    result: Any, decisions: Mapping[str, DecisionVariable]
) -> SimulationEvaluation:
    raw_derivatives: Any = None
    if isinstance(result, tuple) and len(result) == 2:
        raw_metrics, raw_derivatives = result
        if not isinstance(raw_metrics, Mapping):
            raise TypeError("simulation metric payload must be a mapping")
    elif isinstance(result, Mapping):
        raw_metrics = {
            key: value for key, value in result.items() if key not in _DERIVATIVE_KEYS
        }
        raw_derivatives = next(
            (result[key] for key in _DERIVATIVE_KEYS if key in result), None
        )
    else:
        raise TypeError("simulation must return a mapping or (metrics, derivatives)")

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
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise TypeError("simulation derivatives must be a mapping")

    ordered_components = component_items(decisions)
    derivatives: dict[tuple[str, tuple[int, ...], str], float] = {}
    for metric_name, derivative_values in raw.items():
        if isinstance(metric_name, tuple) and len(metric_name) == 2:
            metric, decision_name = metric_name
            if not isinstance(metric, str) or not isinstance(decision_name, str):
                raise TypeError("flat derivative keys must be (metric, decision)")
            if decision_name in decisions:
                _store_decision_derivatives(
                    derivatives,
                    metric,
                    _metric_shape(metrics, metric),
                    decision_name,
                    decisions[decision_name],
                    derivative_values,
                )
            continue

        if not isinstance(metric_name, str):
            raise TypeError("derivative metric names must be strings")
        metric_shape = _metric_shape(metrics, metric_name)
        if isinstance(derivative_values, Mapping):
            derivative_names = tuple(derivative_values)
            if not all(isinstance(name, str) for name in derivative_names):
                raise TypeError("derivative decision names must be strings")
            if any(name in decisions for name in derivative_names):
                for decision_name, value in derivative_values.items():
                    if decision_name in decisions:
                        _store_decision_derivatives(
                            derivatives,
                            metric_name,
                            metric_shape,
                            decision_name,
                            decisions[decision_name],
                            value,
                        )
                continue

        ordered_values = _flatten_values(derivative_values)
        metric_indices = _metric_indices(metric_shape)
        expected = len(metric_indices) * len(ordered_components)
        if len(ordered_values) != expected:
            raise ValueError(
                "ordered derivative values must match the metric shape times "
                "the total number of simulation decision components"
            )
        offset = 0
        for metric_index in metric_indices:
            for component_name, _ in ordered_components:
                derivatives[(metric_name, metric_index, component_name)] = (
                    ordered_values[offset]
                )
                offset += 1
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
    if isinstance(raw, Mapping):
        return tuple(item for value in raw.values() for item in _flatten_values(value))
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
