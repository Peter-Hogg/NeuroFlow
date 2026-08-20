"""Canonical lazy expressions for the supported NumPy-compatible API."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, cast

import dask.array as da
import numpy as np

EXPRESSION_SCHEMA_VERSION = "1"
Casting: TypeAlias = Literal["no", "equiv", "safe", "same_kind", "unsafe"]


@dataclass(frozen=True)
class InputExpr:
    """The single array input bound to a NeuroFlow selection."""

    shape: tuple[int, ...]
    axes: tuple[str, ...]
    dtype: str


@dataclass(frozen=True)
class ScalarExpr:
    """An immutable Python or NumPy scalar operand."""

    value: bool | int | float | complex | np.generic
    scalar_kind: Literal["python", "numpy"]
    dtype: str

    @property
    def shape(self) -> tuple[()]:
        return ()

    @property
    def axes(self) -> tuple[()]:
        return ()


@dataclass(frozen=True)
class UFuncExpr:
    """One supported NumPy ufunc call."""

    operation: str
    operands: tuple[Expression, ...]
    shape: tuple[int, ...]
    axes: tuple[str, ...]
    dtype: str


@dataclass(frozen=True)
class ReductionExpr:
    """One reduction over normalized named axes."""

    operation: str
    operand: Expression
    reduced_axes: tuple[str, ...]
    keepdims: bool
    dtype_argument: str | None
    parameters: tuple[tuple[str, object], ...]
    shape: tuple[int, ...]
    axes: tuple[str, ...]
    dtype: str


@dataclass(frozen=True)
class CastExpr:
    """A dtype conversion."""

    operand: Expression
    dtype: str
    casting: Casting
    shape: tuple[int, ...]
    axes: tuple[str, ...]


Expression: TypeAlias = InputExpr | ScalarExpr | UFuncExpr | ReductionExpr | CastExpr


SUPPORTED_UFUNCS: dict[str, np.ufunc] = {
    name: getattr(np, name)
    for name in (
        "absolute",
        "add",
        "cos",
        "divide",
        "equal",
        "exp",
        "expm1",
        "floor_divide",
        "greater",
        "greater_equal",
        "isfinite",
        "isinf",
        "isnan",
        "less",
        "less_equal",
        "log",
        "log1p",
        "maximum",
        "minimum",
        "multiply",
        "negative",
        "not_equal",
        "positive",
        "power",
        "remainder",
        "sin",
        "sqrt",
        "subtract",
        "tan",
        "true_divide",
    )
}

SUPPORTED_REDUCTIONS = frozenset(
    {"max", "mean", "median", "min", "percentile", "quantile", "sum"}
)

# These operations have associative bounded partials implemented by the staged
# executor. Exact quantiles/medians deliberately remain single-stage because a
# portable bounded combine algorithm is not yet part of NeuroFlow's contract.
SUPPORTED_STAGED_REDUCTIONS = frozenset({"max", "mean", "min", "sum"})


def make_input(
    shape: tuple[int, ...], axes: tuple[str, ...], dtype: str | np.dtype[Any]
) -> InputExpr:
    return InputExpr(shape, axes, np.dtype(dtype).str)


def make_scalar(value: object) -> ScalarExpr:
    if isinstance(value, np.ndarray):
        raise TypeError(
            "NumPy array operands are not supported; use a scalar or another "
            "NeuroArray with the same source selection"
        )
    if isinstance(value, np.generic):
        if value.dtype.kind not in "biufc":
            raise TypeError(f"unsupported NumPy scalar dtype {value.dtype}")
        return ScalarExpr(value, "numpy", value.dtype.str)
    if isinstance(value, bool):
        return ScalarExpr(value, "python", np.dtype(bool).str)
    if isinstance(value, int):
        dtype = np.asarray(value).dtype
        if dtype.kind not in "iu":
            raise OverflowError(
                "Python integer scalar is outside NumPy's fixed-width range"
            )
        return ScalarExpr(value, "python", dtype.str)
    if isinstance(value, float):
        return ScalarExpr(value, "python", np.dtype(float).str)
    if isinstance(value, complex):
        return ScalarExpr(value, "python", np.dtype(complex).str)
    raise TypeError(
        f"unsupported operand type {type(value).__name__}; NeuroFlow currently "
        "supports scalar broadcasting only"
    )


def _sample_value(expression: Expression) -> object:
    if isinstance(expression, ScalarExpr):
        return expression.value
    return np.ones((1,) * len(expression.shape), dtype=np.dtype(expression.dtype))


def make_ufunc(operation: str, operands: tuple[Expression, ...]) -> UFuncExpr:
    ufunc = SUPPORTED_UFUNCS.get(operation)
    if ufunc is None:
        raise TypeError(f"NumPy ufunc {operation!r} is not supported by NeuroFlow")
    if ufunc.nout != 1:
        raise TypeError("multi-output NumPy ufuncs are not supported")
    arrays = [item for item in operands if not isinstance(item, ScalarExpr)]
    if not arrays:
        raise TypeError("a NeuroFlow ufunc requires at least one NeuroArray operand")
    shaped = [item for item in arrays if item.shape]
    first = shaped[0] if shaped else arrays[0]
    if any(
        item.shape and (item.axes != first.axes or item.shape != first.shape)
        for item in arrays
    ):
        raise ValueError(
            "NeuroArray operands must have identical axes and shapes; general "
            "array broadcasting is not supported (global scalar reductions are "
            "the only staged broadcast)"
        )
    with np.errstate(all="ignore"):
        sample = ufunc(*(_sample_value(item) for item in operands))
    dtype = np.asarray(sample).dtype.str
    return UFuncExpr(operation, operands, first.shape, first.axes, dtype)


def _reduction_sample(
    operation: str,
    operand: Expression,
    axis_indices: tuple[int, ...],
    *,
    keepdims: bool,
    dtype_argument: str | None,
    parameters: dict[str, object],
) -> np.ndarray:
    sample = np.ones((1,) * len(operand.shape), dtype=np.dtype(operand.dtype))
    axis: int | tuple[int, ...]
    axis = axis_indices[0] if len(axis_indices) == 1 else axis_indices
    kwargs: dict[str, object] = {"axis": axis, "keepdims": keepdims}
    if dtype_argument is not None:
        kwargs["dtype"] = np.dtype(dtype_argument)
    kwargs.update(parameters)
    function = getattr(np, operation)
    with np.errstate(all="ignore"):
        return np.asarray(function(sample, **kwargs))


def make_reduction(
    operation: str,
    operand: Expression,
    reduced_axes: tuple[str, ...],
    *,
    keepdims: bool = False,
    dtype: str | np.dtype[Any] | None = None,
    parameters: dict[str, object] | None = None,
) -> ReductionExpr:
    if operation not in SUPPORTED_REDUCTIONS:
        raise TypeError(f"NumPy reduction {operation!r} is not supported")
    missing = set(reduced_axes) - set(operand.axes)
    if missing:
        raise ValueError("expression has no axes: " + ", ".join(sorted(missing)))
    axis_indices = tuple(operand.axes.index(axis) for axis in reduced_axes)
    dtype_argument = np.dtype(dtype).str if dtype is not None else None
    values = dict(parameters or {})
    sample = _reduction_sample(
        operation,
        operand,
        axis_indices,
        keepdims=keepdims,
        dtype_argument=dtype_argument,
        parameters=values,
    )
    if keepdims:
        shape = tuple(
            1 if axis in reduced_axes else size
            for axis, size in zip(operand.axes, operand.shape, strict=True)
        )
        axes = operand.axes
    else:
        shape = tuple(
            size
            for axis, size in zip(operand.axes, operand.shape, strict=True)
            if axis not in reduced_axes
        )
        axes = tuple(axis for axis in operand.axes if axis not in reduced_axes)
    return ReductionExpr(
        operation=operation,
        operand=operand,
        reduced_axes=reduced_axes,
        keepdims=keepdims,
        dtype_argument=dtype_argument,
        parameters=tuple(sorted(values.items())),
        shape=shape,
        axes=axes,
        dtype=sample.dtype.str,
    )


def make_cast(
    operand: Expression,
    dtype: str | np.dtype[Any],
    *,
    casting: Casting = "unsafe",
) -> CastExpr:
    target = np.dtype(dtype)
    if casting not in {"no", "equiv", "safe", "same_kind", "unsafe"}:
        raise ValueError(f"invalid casting rule {casting!r}")
    if not np.can_cast(np.dtype(operand.dtype), target, casting=casting):
        raise TypeError(
            f"cannot cast from {np.dtype(operand.dtype)} to {target} "
            f"according to the rule {casting!r}"
        )
    return CastExpr(operand, target.str, casting, operand.shape, operand.axes)


def expression_to_dict(expression: Expression) -> dict[str, object]:
    """Return a canonical JSON-compatible expression representation."""
    common: dict[str, object] = {
        "shape": list(expression.shape),
        "axes": list(expression.axes),
        "dtype": expression.dtype,
    }
    if isinstance(expression, InputExpr):
        return {"kind": "input", **common}
    if isinstance(expression, ScalarExpr):
        return {
            "kind": "scalar",
            "scalar_kind": expression.scalar_kind,
            "dtype": expression.dtype,
            "value": _canonical_scalar(expression.value),
        }
    if isinstance(expression, UFuncExpr):
        return {
            "kind": "ufunc",
            "operation": expression.operation,
            "operands": [expression_to_dict(item) for item in expression.operands],
            **common,
        }
    if isinstance(expression, ReductionExpr):
        return {
            "kind": "reduction",
            "operation": expression.operation,
            "operand": expression_to_dict(expression.operand),
            "reduced_axes": list(expression.reduced_axes),
            "keepdims": expression.keepdims,
            "dtype_argument": expression.dtype_argument,
            "parameters": {
                key: _canonical_value(value) for key, value in expression.parameters
            },
            **common,
        }
    return {
        "kind": "cast",
        "operand": expression_to_dict(expression.operand),
        "casting": expression.casting,
        **common,
    }


def expression_from_dict(value: object) -> Expression:
    """Validate and reconstruct an allowlisted expression representation.

    The parser intentionally rebuilds nodes through the public constructors and
    compares their canonical representation. Workflow files can therefore name
    only NeuroFlow's finite operation registry; they cannot import modules,
    resolve callables, or execute encoded Python.
    """

    node_count = 0

    def parse(raw: object, depth: int = 0) -> Expression:
        nonlocal node_count
        node_count += 1
        if node_count > 1000:
            raise ValueError("expression contains more than 1000 nodes")
        if depth > 64:
            raise ValueError("expression nesting exceeds 64 levels")
        if not isinstance(raw, dict):
            raise ValueError("expression node must be a JSON object")
        kind = raw.get("kind")
        try:
            if kind == "input":
                expression = make_input(
                    _shape(raw.get("shape")),
                    _axes(raw.get("axes")),
                    _dtype(raw.get("dtype")),
                )
            elif kind == "scalar":
                expression = _scalar_from_dict(raw)
            elif kind == "ufunc":
                operation = raw.get("operation")
                operands = raw.get("operands")
                if not isinstance(operation, str) or not isinstance(operands, list):
                    raise ValueError("ufunc requires an operation and operand list")
                if not 1 <= len(operands) <= 8:
                    raise ValueError(
                        "ufunc must contain between one and eight operands"
                    )
                expression = make_ufunc(
                    operation, tuple(parse(item, depth + 1) for item in operands)
                )
            elif kind == "reduction":
                operation = raw.get("operation")
                parameters = raw.get("parameters", {})
                if not isinstance(operation, str) or not isinstance(parameters, dict):
                    raise ValueError("reduction has invalid operation or parameters")
                expression = make_reduction(
                    operation,
                    parse(raw.get("operand"), depth + 1),
                    _axes(raw.get("reduced_axes")),
                    keepdims=_boolean(raw.get("keepdims"), "keepdims"),
                    dtype=(
                        None
                        if raw.get("dtype_argument") is None
                        else _dtype(raw.get("dtype_argument"))
                    ),
                    parameters={
                        key: _parameter_from_canonical(key, item)
                        for key, item in parameters.items()
                        if isinstance(key, str)
                    },
                )
                if len(parameters) != len(dict(expression.parameters)):
                    raise ValueError("reduction parameter names must be strings")
            elif kind == "cast":
                casting = raw.get("casting")
                if casting not in {"no", "equiv", "safe", "same_kind", "unsafe"}:
                    raise ValueError("cast has an invalid casting rule")
                expression = make_cast(
                    parse(raw.get("operand"), depth + 1),
                    _dtype(raw.get("dtype")),
                    casting=cast(Casting, casting),
                )
            else:
                raise ValueError(f"unsupported expression node kind {kind!r}")
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            if isinstance(exc, ValueError) and str(exc).startswith("expression "):
                raise
            raise ValueError(
                f"invalid {kind or 'unknown'} expression node: {exc}"
            ) from exc
        if expression_to_dict(expression) != raw:
            raise ValueError(
                f"{kind} expression metadata is not canonical or does not match "
                "the derived shape/dtype"
            )
        return expression

    return parse(value)


def expression_identity(expression: Expression) -> str:
    """Return a stable identifier for one canonical expression subtree."""
    payload = json.dumps(
        expression_to_dict(expression),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def staged_reductions(expression: Expression) -> tuple[ReductionExpr, ...]:
    """Return supported global scalar reductions in dependency order."""
    found: dict[str, ReductionExpr] = {}

    def visit(item: Expression) -> None:
        if isinstance(item, ReductionExpr):
            visit(item.operand)
            if (
                not item.shape
                and item.operation in SUPPORTED_STAGED_REDUCTIONS
                and set(item.reduced_axes) == set(item.operand.axes)
            ):
                found.setdefault(expression_identity(item), item)
        elif isinstance(item, UFuncExpr):
            for operand in item.operands:
                visit(operand)
        elif isinstance(item, CastExpr):
            visit(item.operand)

    visit(expression)
    return tuple(found.values())


def _shape(value: object) -> tuple[int, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, int) or isinstance(item, bool) or item < 0
        for item in value
    ):
        raise ValueError("shape must be a list of non-negative integers")
    if len(value) > 32:
        raise ValueError("array rank exceeds 32")
    return tuple(value)


def _axes(value: object) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not all(isinstance(item, str) and item for item in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError("axes must be a list of unique non-empty strings")
    return tuple(value)


def _dtype(value: object) -> str:
    if not isinstance(value, str) or len(value) > 128:
        raise ValueError("dtype must be a short string")
    dtype = np.dtype(value)
    if dtype.hasobject:
        raise ValueError("object dtypes are not portable workflow inputs")
    return dtype.str


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _scalar_from_dict(raw: dict[str, object]) -> ScalarExpr:
    kind = raw.get("scalar_kind")
    dtype = np.dtype(_dtype(raw.get("dtype")))
    encoded = raw.get("value")
    if kind == "numpy":
        if not isinstance(encoded, dict) or set(encoded) != {"bytes"}:
            raise ValueError("NumPy scalar requires an encoded byte value")
        byte_value = encoded["bytes"]
        if not isinstance(byte_value, str) or len(byte_value) != dtype.itemsize * 2:
            raise ValueError("NumPy scalar byte value has the wrong size")
        try:
            scalar = np.frombuffer(bytes.fromhex(byte_value), dtype=dtype, count=1)[0]
        except ValueError as exc:
            raise ValueError("NumPy scalar contains invalid hexadecimal bytes") from exc
        return make_scalar(scalar)
    if kind != "python":
        raise ValueError("scalar_kind must be 'python' or 'numpy'")
    if dtype.kind == "b" and isinstance(encoded, bool):
        return make_scalar(encoded)
    if dtype.kind in "iu" and isinstance(encoded, str):
        return make_scalar(int(encoded))
    if dtype.kind == "f" and isinstance(encoded, str):
        return make_scalar(float.fromhex(encoded))
    if dtype.kind == "c" and isinstance(encoded, dict):
        real = encoded.get("real")
        imag = encoded.get("imag")
        if isinstance(real, str) and isinstance(imag, str):
            return make_scalar(complex(float.fromhex(real), float.fromhex(imag)))
    raise ValueError("Python scalar value does not match its declared dtype")


def _parameter_from_canonical(name: str, value: object) -> object:
    if name == "q":
        if not isinstance(value, str):
            raise ValueError("quantile parameter q must be a canonical float")
        return float.fromhex(value)
    if name == "method":
        if not isinstance(value, str):
            raise ValueError("quantile method must be a string")
        return value
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, dict) and "dtype" in value and "value" in value:
        dtype = np.dtype(_dtype(value["dtype"]))
        scalar = _scalar_from_dict(
            {"scalar_kind": "numpy", "dtype": dtype.str, "value": value["value"]}
        )
        return scalar.value
    if isinstance(value, str):
        try:
            return float.fromhex(value)
        except ValueError:
            return value
    # Canonical Python integers are decimal strings and floats are hexadecimal
    # strings. Reduction parameters currently use floats (q) and strings
    # (method), so disambiguate hexadecimal floats by their marker.
    if isinstance(value, dict) and set(value) == {"real", "imag"}:
        real, imag = value.get("real"), value.get("imag")
        if isinstance(real, str) and isinstance(imag, str):
            return complex(float.fromhex(real), float.fromhex(imag))
    return value


def _canonical_scalar(value: bool | int | float | complex | np.generic) -> object:
    if isinstance(value, np.generic):
        return {"bytes": value.tobytes().hex()}
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return value.hex()
    return {"real": value.real.hex(), "imag": value.imag.hex()}


def _canonical_value(value: object) -> object:
    if isinstance(value, np.generic):
        return {
            "dtype": value.dtype.str,
            "value": _canonical_scalar(value),
        }
    if isinstance(value, (bool, int, float, complex)):
        return _canonical_scalar(value)
    if isinstance(value, str) or value is None:
        return value
    raise TypeError(f"cannot serialize expression parameter {type(value).__name__}")


def evaluate_numpy(
    expression: Expression,
    value: np.ndarray,
    *,
    staged_values: dict[str, object] | None = None,
) -> np.ndarray:
    """Evaluate an expression against one bounded in-memory source partition."""

    def evaluate(item: Expression) -> object:
        if isinstance(item, InputExpr):
            return value
        if isinstance(item, ScalarExpr):
            return item.value
        if isinstance(item, UFuncExpr):
            function = SUPPORTED_UFUNCS[item.operation]
            return function(*(evaluate(operand) for operand in item.operands))
        if isinstance(item, CastExpr):
            return np.asarray(evaluate(item.operand)).astype(
                np.dtype(item.dtype), casting=item.casting, copy=False
            )
        stage_id = expression_identity(item)
        if staged_values is not None and stage_id in staged_values:
            return staged_values[stage_id]
        operand = np.asarray(evaluate(item.operand))
        indices = tuple(item.operand.axes.index(axis) for axis in item.reduced_axes)
        axis: int | tuple[int, ...]
        axis = indices[0] if len(indices) == 1 else indices
        kwargs: dict[str, object] = {"axis": axis, "keepdims": item.keepdims}
        if item.dtype_argument is not None:
            kwargs["dtype"] = np.dtype(item.dtype_argument)
        kwargs.update(dict(item.parameters))
        return getattr(np, item.operation)(operand, **kwargs)

    return np.asarray(evaluate(expression))


def evaluate_dask(expression: Expression, value: da.Array) -> da.Array:
    """Compile an expression into the equivalent lazy Dask array graph."""

    def evaluate(item: Expression) -> object:
        if isinstance(item, InputExpr):
            return value
        if isinstance(item, ScalarExpr):
            return item.value
        if isinstance(item, UFuncExpr):
            function = SUPPORTED_UFUNCS[item.operation]
            return function(*(evaluate(operand) for operand in item.operands))
        if isinstance(item, CastExpr):
            operand = cast(da.Array, evaluate(item.operand))
            return operand.astype(
                np.dtype(item.dtype), casting=item.casting, copy=False
            )
        operand = cast(da.Array, evaluate(item.operand))
        indices = tuple(item.operand.axes.index(axis) for axis in item.reduced_axes)
        axis: int | tuple[int, ...]
        axis = indices[0] if len(indices) == 1 else indices
        kwargs: dict[str, object] = {"axis": axis, "keepdims": item.keepdims}
        if item.dtype_argument is not None:
            kwargs["dtype"] = np.dtype(item.dtype_argument)
        kwargs.update(dict(item.parameters))
        if item.operation == "percentile":
            raw_q = kwargs.pop("q")
            if not isinstance(raw_q, (int, float)):
                raise TypeError("percentile q must be numeric")
            q = float(raw_q) / 100.0
            return da.quantile(operand, q, **kwargs)  # pyright: ignore[reportArgumentType]
        return getattr(da, item.operation)(operand, **kwargs)

    return cast(da.Array, evaluate(expression))


def reduced_input_axes(
    expression: Expression, *, exclude_staged_dependencies: bool = False
) -> tuple[str, ...]:
    """Return source axes that every persistence partition must read in full."""
    axes: set[str] = set()

    def visit(item: Expression) -> None:
        if isinstance(item, ReductionExpr):
            if (
                exclude_staged_dependencies
                and item is not expression
                and not item.shape
                and item.operation in SUPPORTED_STAGED_REDUCTIONS
                and set(item.reduced_axes) == set(item.operand.axes)
            ):
                return
            axes.update(item.reduced_axes)
            visit(item.operand)
        elif isinstance(item, UFuncExpr):
            for operand in item.operands:
                visit(operand)
        elif isinstance(item, CastExpr):
            visit(item.operand)

    visit(expression)
    input_axes = input_expression(expression).axes
    return tuple(axis for axis in input_axes if axis in axes)


def input_expression(expression: Expression) -> InputExpr:
    """Return the common input leaf, validating that one exists."""
    if isinstance(expression, InputExpr):
        return expression
    if isinstance(expression, ScalarExpr):
        raise ValueError("a scalar expression has no array input")
    if isinstance(expression, (ReductionExpr, CastExpr)):
        return input_expression(expression.operand)
    for operand in expression.operands:
        if not isinstance(operand, ScalarExpr):
            return input_expression(operand)
    raise ValueError("expression has no array input")


def rebind_input(expression: Expression, replacement: InputExpr) -> Expression:
    """Rebuild an expression after a commuting slice changes its input shape."""
    if isinstance(expression, InputExpr):
        return replacement
    if isinstance(expression, ScalarExpr):
        return expression
    if isinstance(expression, UFuncExpr):
        return make_ufunc(
            expression.operation,
            tuple(rebind_input(item, replacement) for item in expression.operands),
        )
    if isinstance(expression, CastExpr):
        return make_cast(
            rebind_input(expression.operand, replacement),
            expression.dtype,
            casting=expression.casting,
        )
    return make_reduction(
        expression.operation,
        rebind_input(expression.operand, replacement),
        expression.reduced_axes,
        keepdims=expression.keepdims,
        dtype=expression.dtype_argument,
        parameters=dict(expression.parameters),
    )


def estimate_working_memory(
    expression: Expression,
    *,
    input_shape: tuple[int, ...] | None = None,
) -> int:
    """Conservatively estimate live bytes for one expression evaluation."""
    root = input_expression(expression)
    root_shape = input_shape or root.shape
    if len(root_shape) != len(root.axes):
        raise ValueError("input shape rank does not match expression input")
    seen: set[int] = set()

    def visit(item: Expression) -> tuple[tuple[int, ...], tuple[str, ...], int]:
        if isinstance(item, ScalarExpr):
            return (), (), np.dtype(item.dtype).itemsize
        if isinstance(item, InputExpr):
            shape = root_shape
            axes = item.axes
            size = math.prod(shape) * np.dtype(item.dtype).itemsize
        elif isinstance(item, UFuncExpr):
            array_operand = next(
                value for value in item.operands if not isinstance(value, ScalarExpr)
            )
            shape, axes, _ = visit(array_operand)
            for operand in item.operands:
                visit(operand)
            size = math.prod(shape) * np.dtype(item.dtype).itemsize
            if id(item) not in seen:
                seen.add(id(item))
                nonlocal_total[0] += size
            return shape, axes, size
        elif isinstance(item, CastExpr):
            shape, axes, _ = visit(item.operand)
            size = math.prod(shape) * np.dtype(item.dtype).itemsize
            if id(item) not in seen:
                seen.add(id(item))
                nonlocal_total[0] += size
            return shape, axes, size
        else:
            child_shape, child_axes, child_size = visit(item.operand)
            if item.keepdims:
                shape = tuple(
                    1 if axis in item.reduced_axes else extent
                    for axis, extent in zip(child_axes, child_shape, strict=True)
                )
                axes = child_axes
            else:
                shape = tuple(
                    extent
                    for axis, extent in zip(child_axes, child_shape, strict=True)
                    if axis not in item.reduced_axes
                )
                axes = tuple(
                    axis for axis in child_axes if axis not in item.reduced_axes
                )
            size = math.prod(shape) * np.dtype(item.dtype).itemsize
            if item.operation in {"median", "percentile", "quantile"}:
                nonlocal_total[0] += child_size
        if id(item) not in seen:
            seen.add(id(item))
            nonlocal_total[0] += size
        return shape, axes, size

    nonlocal_total = [0]
    visit(expression)
    return nonlocal_total[0]


def output_shape_for_input(
    expression: Expression, input_shape: tuple[int, ...]
) -> tuple[int, ...]:
    """Resolve an expression's output shape for one source partition shape."""
    root = input_expression(expression)
    if len(input_shape) != len(root.axes):
        raise ValueError("input shape rank does not match expression input")

    def visit(item: Expression) -> tuple[tuple[int, ...], tuple[str, ...]]:
        if isinstance(item, ScalarExpr):
            return (), ()
        if isinstance(item, InputExpr):
            return input_shape, item.axes
        if isinstance(item, UFuncExpr):
            operand = next(
                value for value in item.operands if not isinstance(value, ScalarExpr)
            )
            return visit(operand)
        if isinstance(item, CastExpr):
            return visit(item.operand)
        child_shape, child_axes = visit(item.operand)
        if item.keepdims:
            return (
                tuple(
                    1 if axis in item.reduced_axes else extent
                    for axis, extent in zip(child_axes, child_shape, strict=True)
                ),
                child_axes,
            )
        return (
            tuple(
                extent
                for axis, extent in zip(child_axes, child_shape, strict=True)
                if axis not in item.reduced_axes
            ),
            tuple(axis for axis in child_axes if axis not in item.reduced_axes),
        )

    return visit(expression)[0]
