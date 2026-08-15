"""Simulation-optimization model declarations and replication evaluation."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import isinf
from numbers import Real
from typing import Any, cast

import sympy as sp

from simopt_dsl.expressions import (
    AggregateExpression,
    BinaryAggregateExpression,
    BinaryExpression,
    Constant,
    Constraint,
    EvaluationContext,
    Expression,
    Mean,
    Metric,
    apply_binary_operator,
    mean,
)
from simopt_dsl.simulation import Simulation, SimulationResult, component_items
from simopt_dsl.variables import DecisionVariable, Variable, VectorVariable, components

Number = int | float


@dataclass(frozen=True)
class StochasticConstraintEvaluation:
    value: float
    gradient: tuple[float, ...] | None


@dataclass(frozen=True)
class ReplicationEvaluation:
    objective: float
    objective_gradient: tuple[float, ...] | None
    stochastic_constraints: tuple[StochasticConstraintEvaluation, ...] = ()


class Model:
    """A declarative simulation-optimization model."""

    def __init__(self, name: str = "") -> None:
        if not isinstance(name, str):
            raise TypeError("model name must be a string")
        self.name = name
        self.variables: list[Variable] = []
        self.simulations: list[Simulation] = []
        self.deterministic_constraints: list[tuple[str, Constraint]] = []
        self.stochastic_constraints: list[tuple[str, Constraint]] = []
        self.objective: AggregateExpression | None = None
        self.objective_sense = "minimize"
        self.n_rngs = 1
        self._declared_variables: list[DecisionVariable] = []
        self._evaluation_plan: _EvaluationPlan | None = None

    def __getstate__(self) -> dict[str, Any]:
        """Return serializable state without process-local compiled callables."""
        state = self.__dict__.copy()
        state["_evaluation_plan"] = None
        return state

    def add_continuous_variable(
        self, lb: Number, ub: Number, name: str | None = None, initial: Number | None = None
    ) -> Variable:
        """Add a scalar continuous decision variable."""
        return self._add_scalar_variable(lb, ub, name, initial, integer=False)

    def add_continuous_vector(
        self,
        lb: Number | Iterable[Number],
        ub: Number | Iterable[Number],
        name: str | None = None,
        shape: int | tuple[int] | None = None,
        initial: Number | Iterable[Number] | None = None,
    ) -> VectorVariable:
        """Add a one-dimensional continuous decision variable."""
        return self._add_vector_variable(lb, ub, name, shape, initial, integer=False)

    def add_integer_variable(
        self, lb: Number, ub: Number, name: str | None = None, initial: Number | None = None
    ) -> Variable:
        """Add a scalar integer decision variable."""
        return self._add_scalar_variable(lb, ub, name, initial, integer=True)

    def add_integer_vector(
        self,
        lb: Number | Iterable[Number],
        ub: Number | Iterable[Number],
        name: str | None = None,
        shape: int | tuple[int] | None = None,
        initial: Number | Iterable[Number] | None = None,
    ) -> VectorVariable:
        """Add a one-dimensional integer decision variable."""
        return self._add_vector_variable(lb, ub, name, shape, initial, integer=True)

    def add_linear_constraint(self, constraint: Constraint, name: str = "") -> None:
        """Add a deterministic constraint."""
        self._validate_constraint(constraint, name)
        self.deterministic_constraints.append((name, constraint))
        self._invalidate_evaluation_plan()

    def add_stochastic_constraint(self, constraint: Constraint, name: str = "") -> None:
        """Add a replication-level stochastic constraint."""
        self._validate_constraint(constraint, name)
        self.stochastic_constraints.append((name, constraint))
        self._invalidate_evaluation_plan()

    def add_simulation(
        self,
        name: str | None = None,
        run: Callable[..., SimulationResult] | None = None,
        decisions: Mapping[str, DecisionVariable] | None = None,
        n_rngs: int = 1,
    ) -> Simulation:
        """Add a simulation callback and bind its decision variables."""
        if run is None or not callable(run):
            raise TypeError("simulation run callable is required")
        if decisions is None:
            raise TypeError("simulation decisions are required")
        if not isinstance(n_rngs, int) or isinstance(n_rngs, bool) or n_rngs <= 0:
            raise ValueError("simulation must use at least one RNG")
        simulation_name = self._resolve_name(name, "simulation", self.simulations)

        decision_variables: dict[str, DecisionVariable] = {}
        seen_components: set[int] = set()
        for decision_name, variable in decisions.items():
            if not isinstance(decision_name, str):
                raise TypeError("simulation decision names must be strings")
            if not decision_name:
                raise ValueError("simulation decision names cannot be empty")
            if not isinstance(variable, (Variable, VectorVariable)):
                raise TypeError("simulation decisions must be decision variables")
            if not any(variable is declared for declared in self._declared_variables):
                raise ValueError(f"simulation decision {decision_name!r} is not in this model")
            variable_components = components(variable)
            if any(id(component) in seen_components for component in variable_components):
                raise ValueError("simulation decision variable components must be unique")
            decision_variables[decision_name] = variable
            seen_components.update(id(component) for component in variable_components)

        simulation = Simulation(simulation_name, run, decision_variables, n_rngs)
        self.simulations.append(simulation)
        self.n_rngs = max(self.n_rngs, n_rngs)
        self._invalidate_evaluation_plan()
        return simulation

    def maximize(self, objective: AggregateExpression | Expression) -> None:
        """Set the model's maximization objective."""
        self._set_objective(objective, "maximize")

    def minimize(self, objective: AggregateExpression | Expression) -> None:
        """Set the model's minimization objective."""
        self._set_objective(objective, "minimize")

    def initial_vector(self) -> tuple[float, ...]:
        """Return initial values in flattened solver order."""
        return tuple(variable.initial for variable in self.variables)

    def lower_bounds(self) -> tuple[float, ...]:
        """Return lower bounds in flattened solver order."""
        return tuple(variable.lb for variable in self.variables)

    def upper_bounds(self) -> tuple[float, ...]:
        """Return upper bounds in flattened solver order."""
        return tuple(variable.ub for variable in self.variables)

    def unpack_vector(self, values: Iterable[float]) -> dict[str, float | tuple[float, ...]]:
        """Regroup a flat solver vector by declared scalar and vector variables."""
        flat_values = tuple(float(value) for value in values)
        if len(flat_values) != len(self.variables):
            raise ValueError("solution dimension does not match model variables")

        unpacked: dict[str, float | tuple[float, ...]] = {}
        offset = 0
        for variable in self._declared_variables:
            next_offset = offset + len(components(variable))
            component_values = flat_values[offset:next_offset]
            unpacked[variable.name] = (
                component_values[0] if isinstance(variable, Variable) else component_values
            )
            offset = next_offset
        return unpacked

    def run_replication(
        self, values: Iterable[float], rngs: Sequence[Any]
    ) -> ReplicationEvaluation:
        """Evaluate one replication at a solution using caller-owned RNGs."""
        plan = self._get_evaluation_plan()
        flat_values = tuple(float(value) for value in values)
        if len(flat_values) != len(self.variables):
            raise ValueError("solution dimension does not match model variables")
        for variable, value in zip(self.variables, flat_values, strict=True):
            if not variable.lb <= value <= variable.ub:
                raise ValueError("solution variable value must be within its bounds")
            if variable.integer:
                _validate_integer(value, "integer variable value")
        if len(rngs) < self.n_rngs:
            raise ValueError("not enough RNGs for model simulations")

        context = EvaluationContext(dict(zip(self.variables, flat_values, strict=True)))
        for _, constraint in self.deterministic_constraints:
            if not constraint.satisfied(context):
                raise ValueError("solution violates a deterministic constraint")

        for simulation in self.simulations:
            evaluation = simulation.evaluate(context, rngs[: simulation.n_rngs])
            for metric_name, metric_value in evaluation.metrics.items():
                context.metrics[(simulation.name, metric_name)] = metric_value
            for key, derivative in evaluation.derivatives.items():
                metric_name, metric_indices, decision_name = key
                context.metric_derivatives[
                    (simulation.name, metric_name, metric_indices, decision_name)
                ] = derivative

        objective_value = _evaluate_aggregate(plan.objective, context)
        objective_gradient = plan.estimate_gradient(plan.objective, context)
        stochastic_constraints = tuple(
            StochasticConstraintEvaluation(
                residual.evaluate(context), plan.estimate_gradient(residual, context)
            )
            for residual in plan.stochastic_constraint_residuals
        )
        return ReplicationEvaluation(objective_value, objective_gradient, stochastic_constraints)

    def _add_scalar_variable(
        self, lb: Number, ub: Number, name: str | None, initial: Number | None, *, integer: bool
    ) -> Variable:
        lower = _coerce_number(lb, "variable lower bound")
        upper = _coerce_number(ub, "variable upper bound")
        if integer:
            _validate_integer_bound(lower, "integer variable lower bound")
            _validate_integer_bound(upper, "integer variable upper bound")
        if lower >= upper:
            raise ValueError("variable lower bound must be less than upper bound")
        variable_name = self._resolve_name(name, "variable", self._declared_variables)
        initial_value = lower if initial is None else _coerce_number(initial, "initial value")
        if integer:
            _validate_integer(initial_value, "integer variable initial value")
        _validate_initial(initial_value, lower, upper)
        variable = Variable(variable_name, lower, upper, initial_value, integer)
        self.variables.append(variable)
        self._declared_variables.append(variable)
        self._invalidate_evaluation_plan()
        return variable

    def _add_vector_variable(
        self,
        lb: Number | Iterable[Number],
        ub: Number | Iterable[Number],
        name: str | None,
        shape: int | tuple[int] | None,
        initial: Number | Iterable[Number] | None,
        *,
        integer: bool,
    ) -> VectorVariable:
        if shape is None:
            raise TypeError("vector variable shape is required")
        size = _vector_size(shape)
        lower = _vector_values(lb, size, "lower bound")
        upper = _vector_values(ub, size, "upper bound")
        initial_values = (
            lower if initial is None else _vector_values(initial, size, "initial value")
        )
        if integer:
            for lower_value in lower:
                _validate_integer_bound(lower_value, "integer variable lower bound")
            for upper_value in upper:
                _validate_integer_bound(upper_value, "integer variable upper bound")
            for initial_value in initial_values:
                _validate_integer(initial_value, "integer variable initial value")
        if any(
            lower_value >= upper_value
            for lower_value, upper_value in zip(lower, upper, strict=True)
        ):
            raise ValueError("each vector variable lower bound must be less than its upper bound")
        for initial_value, lower_value, upper_value in zip(
            initial_values, lower, upper, strict=True
        ):
            _validate_initial(initial_value, lower_value, upper_value)

        variable_name = self._resolve_name(name, "variable", self._declared_variables)
        scalar_components = tuple(
            Variable(
                f"{variable_name}[{index}]",
                lower[index],
                upper[index],
                initial_values[index],
                integer,
            )
            for index in range(size)
        )
        variable = VectorVariable(variable_name, scalar_components)
        self.variables.extend(scalar_components)
        self._declared_variables.append(variable)
        self._invalidate_evaluation_plan()
        return variable

    def _set_objective(self, objective: AggregateExpression | Expression, sense: str) -> None:
        if not isinstance(objective, (AggregateExpression, Expression)):
            raise TypeError("objective must be an expression")
        self.objective = (
            objective if isinstance(objective, AggregateExpression) else mean(objective)
        )
        self.objective_sense = sense
        self._invalidate_evaluation_plan()

    def _resolve_name(self, name: str | None, prefix: str, existing: Iterable[Any]) -> str:
        existing_names = {item.name for item in existing}
        if name is None or name == "":
            index = 1
            while f"{prefix}_{index}" in existing_names:
                index += 1
            return f"{prefix}_{index}"
        if not isinstance(name, str):
            raise TypeError(f"{prefix} name must be a string")
        if name in existing_names:
            raise ValueError(f"duplicate {prefix} name {name!r}")
        return name

    @staticmethod
    def _validate_constraint(constraint: Constraint, name: str) -> None:
        if not isinstance(constraint, Constraint):
            raise TypeError("constraint must be created with <= or >=")
        if not isinstance(name, str):
            raise TypeError("constraint name must be a string")

    def _invalidate_evaluation_plan(self) -> None:
        self._evaluation_plan = None

    def _get_evaluation_plan(self) -> _EvaluationPlan:
        if self._evaluation_plan is None:
            self._evaluation_plan = _EvaluationPlan(self)
        return self._evaluation_plan


class _UnsupportedGradient(Exception):
    pass


class _SymbolicGradient:
    def __init__(
        self, variables: Sequence[Variable], expression: Expression | AggregateExpression
    ) -> None:
        self.variables = variables
        self.variable_symbols = {
            variable: sp.Symbol(f"v_{index}", real=True) for index, variable in enumerate(variables)
        }
        self.metric_applications: dict[tuple[str, str, tuple[int, ...]], sp.Expr] = {}
        self.metric_derivative_keys: dict[sp.Expr, tuple[str, str, tuple[int, ...], str]] = {}

        sample_expression = self._expression(expression)
        gradient_expressions = tuple(
            sp.simplify(sp.diff(sample_expression, self.variable_symbols[variable]))
            for variable in variables
        )
        arguments = (
            *self.variable_symbols.values(),
            *self.metric_applications.values(),
            *self.metric_derivative_keys,
        )
        self._compiled_gradient: Callable[..., Any] = sp.lambdify(
            arguments, gradient_expressions, modules="math", dummify=True
        )

    def estimate(self, context: EvaluationContext) -> tuple[float, ...] | None:
        arguments = self._argument_values(context)
        if arguments is None:
            return None
        try:
            return tuple(float(value) for value in self._compiled_gradient(*arguments))
        except (TypeError, ValueError, ZeroDivisionError, OverflowError):
            return None

    def _argument_values(self, context: EvaluationContext) -> tuple[float, ...] | None:
        values = [context.variables[variable] for variable in self.variables]
        for simulation_name, metric_name, metric_indices in self.metric_applications:
            value = context.metrics[(simulation_name, metric_name)]
            for index in metric_indices:
                value = value[index]
            values.append(float(value))
        for derivative_key in self.metric_derivative_keys.values():
            if derivative_key not in context.metric_derivatives:
                return None
            values.append(context.metric_derivatives[derivative_key])
        return tuple(values)

    def _expression(self, expression: Expression | AggregateExpression) -> sp.Expr:
        if isinstance(expression, Constant):
            return sp.Float(expression.value)
        if isinstance(expression, Variable):
            return self.variable_symbols[expression]
        if isinstance(expression, Metric):
            return self._metric_application(expression)
        if isinstance(expression, BinaryExpression):
            return self._binary_expression(expression.operator, expression.left, expression.right)
        if isinstance(expression, Mean):
            return self._expression(expression.expression)
        if isinstance(expression, BinaryAggregateExpression):
            return self._binary_expression(expression.operator, expression.left, expression.right)
        raise _UnsupportedGradient(
            f"unsupported objective expression {type(expression).__name__!r}"
        )

    def _binary_expression(
        self,
        operator: str,
        left: Expression | AggregateExpression,
        right: Expression | AggregateExpression,
    ) -> sp.Expr:
        left_expression = self._expression(left)
        right_expression = self._expression(right)
        if operator == "+":
            return left_expression + right_expression
        if operator == "-":
            return left_expression - right_expression
        if operator == "*":
            return left_expression * right_expression
        if operator == "/":
            return left_expression / right_expression
        raise ValueError(f"unknown operator {operator!r}")

    def _metric_application(self, metric: Metric) -> sp.Expr:
        key = (metric.simulation.name, metric.name, metric.indices)
        if key not in self.metric_applications:
            function: Any = sp.Function(f"metric_{len(self.metric_applications)}")
            items = component_items(metric.simulation.decisions)
            application = function(*(self.variable_symbols[component] for _, component in items))
            self.metric_applications[key] = application
            for component_name, component in items:
                derivative = sp.Derivative(application, self.variable_symbols[component])
                self.metric_derivative_keys[derivative] = (
                    metric.simulation.name,
                    metric.name,
                    metric.indices,
                    component_name,
                )
        return self.metric_applications[key]


class _EvaluationPlan:
    """Compiled, reusable objective and constraint evaluation structure."""

    def __init__(self, model: Model) -> None:
        if model.objective is None:
            raise ValueError("model has no objective")
        self.variables = tuple(model.variables)
        self.objective = model.objective
        self.stochastic_constraint_residuals = tuple(
            constraint.residual() for _, constraint in model.stochastic_constraints
        )
        self._gradients: dict[
            int, tuple[Expression | AggregateExpression, _SymbolicGradient | None]
        ] = {}
        self._gradient(self.objective)
        for residual in self.stochastic_constraint_residuals:
            self._gradient(residual)

    def estimate_gradient(
        self, expression: Expression | AggregateExpression, context: EvaluationContext
    ) -> tuple[float, ...] | None:
        gradient = self._gradient(expression)
        return None if gradient is None else gradient.estimate(context)

    def _gradient(self, expression: Expression | AggregateExpression) -> _SymbolicGradient | None:
        key = id(expression)
        cached = self._gradients.get(key)
        if cached is not None and cached[0] is expression:
            return cached[1]
        try:
            gradient = _SymbolicGradient(self.variables, expression)
        except _UnsupportedGradient:
            gradient = None
        self._gradients[key] = (expression, gradient)
        return gradient


def _evaluate_aggregate(
    expression: Expression | AggregateExpression, context: EvaluationContext
) -> float:
    if isinstance(expression, Mean):
        return expression.expression.evaluate(context)
    if isinstance(expression, BinaryAggregateExpression):
        return apply_binary_operator(
            expression.operator,
            _evaluate_aggregate(expression.left, context),
            _evaluate_aggregate(expression.right, context),
        )
    if isinstance(expression, Expression):
        return expression.evaluate(context)
    raise TypeError(f"unsupported aggregate objective {type(expression).__name__!r}")


def _coerce_number(value: object, parameter_name: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{parameter_name} must be numeric")
    return float(value)


def _validate_initial(initial: float, lower: float, upper: float) -> None:
    if not lower <= initial <= upper:
        raise ValueError("variable initial value must be within its bounds")


def _validate_integer(value: float, parameter_name: str) -> None:
    if value % 1 != 0:
        raise ValueError(f"{parameter_name} must be an integer")


def _validate_integer_bound(value: float, parameter_name: str) -> None:
    if not isinf(value):
        _validate_integer(value, parameter_name)


def _vector_size(shape: int | tuple[int]) -> int:
    if isinstance(shape, bool):
        raise TypeError("vector variable size must be a positive integer")
    if isinstance(shape, int):
        size = shape
    else:
        dimensions = tuple(shape)
        if len(dimensions) != 1:
            raise ValueError("vector variable shape must have exactly one dimension")
        size = dimensions[0]
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ValueError("vector variable size must be a positive integer")
    return size


def _vector_values(
    value: Number | Iterable[Number], size: int, parameter_name: str
) -> tuple[float, ...]:
    if isinstance(value, Real) and not isinstance(value, bool):
        return (float(value),) * size
    if isinstance(value, (str, bytes)):
        raise TypeError(f"vector variable {parameter_name} must be numeric")
    try:
        raw_values = tuple(cast(Iterable[Number], value))
    except TypeError as exc:
        raise TypeError(f"vector variable {parameter_name} must be numeric") from exc
    if len(raw_values) != size:
        raise ValueError(
            f"vector variable {parameter_name} must have {size} values, got {len(raw_values)}"
        )
    return tuple(_coerce_number(item, f"vector variable {parameter_name}") for item in raw_values)


__all__ = ["Model"]
