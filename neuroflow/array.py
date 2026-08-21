"""Lazy named arrays with a deliberately bounded NumPy-compatible surface."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin

from neuroflow.adapters import ArrayOutput, ExpressionAdapter
from neuroflow.api import open_array, open_source, run
from neuroflow.execution.resources import resolve_memory_budget
from neuroflow.expression import (
    SUPPORTED_UFUNCS,
    Casting,
    Expression,
    InputExpr,
    estimate_working_memory,
    evaluate_dask,
    make_cast,
    make_input,
    make_reduction,
    make_scalar,
    make_ufunc,
    rebind_input,
    reduced_input_axes,
)
from neuroflow.partition import SpatialTilePlan
from neuroflow.results.workflow import WorkflowResult
from neuroflow.selection import NWBQuery, Selection, absolute_selection_bounds
from neuroflow.source.base import NWBSource
from neuroflow.storage import SegmentationOutput, ZarrOutput

if TYPE_CHECKING:
    from neuroflow.diagnostics.plan import ExecutionPlan
    from neuroflow.traces import TracePlan
    from neuroflow.workflow import WorkflowSpec

DEFAULT_COMPUTE_MEMORY_LIMIT = "1 GiB"
DEFAULT_PERSIST_MEMORY_LIMIT = "2 GiB"
# Segmentation needs its own default because ``memory_limit`` is a *total*
# process target and one loaded ``cpsam`` network measures ~1.9 GiB resident on
# CPU (``benchmarks/results/current-memory-attribution.json``). The 2 GiB
# persist default is therefore consumed entirely by model weights before any
# image data is charged, and the default call would be refused. A single fish
# plane costs only ~175 MiB to segment, so the extra headroom here is model
# residency rather than partition size. Running on CUDA moves ~1.2 GiB into
# VRAM and makes a smaller host target viable.
DEFAULT_SEGMENT_MEMORY_LIMIT = "4 GiB"
_UNSET = object()


def _selection_key(selection: Selection) -> tuple[object, ...]:
    metadata = selection.metadata
    bounds = absolute_selection_bounds(metadata)
    return (
        metadata.source,
        metadata.path,
        bounds,
        metadata.shape,
        metadata.axes,
        np.dtype(metadata.dtype).str,
    )


def _normalize_axes(
    axis: object,
    axes: tuple[str, ...],
) -> tuple[str, ...]:
    if axis is None:
        return axes
    raw = axis if isinstance(axis, tuple) else (axis,)
    normalized: list[str] = []
    for item in raw:
        if isinstance(item, str):
            if item not in axes:
                raise ValueError(f"array has no axis {item!r}")
            name = item
        elif isinstance(item, (int, np.integer)) and not isinstance(
            item, (bool, np.bool_)
        ):
            index = int(item)
            if index < -len(axes) or index >= len(axes):
                raise np.exceptions.AxisError(index, ndim=len(axes))
            name = axes[index % len(axes)]
        else:
            raise TypeError("axis must be None, a name, an integer, or a tuple")
        if name in normalized:
            raise ValueError("duplicate value in 'axis'")
        normalized.append(name)
    selected = set(normalized)
    return tuple(name for name in axes if name in selected)


def _reject_optional_argument(name: str, value: object, default: object) -> None:
    if value is not default:
        raise TypeError(f"NeuroFlow does not support the {name!r} argument yet")


def _validated_quantile(value: object, *, upper: int) -> float:
    if not np.isscalar(value) or isinstance(value, (str, bytes, complex)):
        raise ValueError(f"q must be one finite scalar in [0, {upper}]")
    try:
        numeric = float(value)  # pyright: ignore[reportArgumentType]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"q must be one finite scalar in [0, {upper}]") from exc
    if not math.isfinite(numeric) or not 0 <= numeric <= upper:
        raise ValueError(f"q must be one finite scalar in [0, {upper}]")
    return numeric


def _size_along_axis(array: NeuroArray, axis: object) -> int:
    """Return one named or positional extent with NumPy-compatible validation."""
    if isinstance(axis, str):
        if axis not in array.axes:
            raise ValueError(f"array has no axis {axis!r}")
        axis_name = axis
    elif isinstance(axis, (int, np.integer)) and not isinstance(axis, (bool, np.bool_)):
        axis_name = _normalize_axes(axis, array.axes)[0]
    else:
        raise TypeError("np.size axis must be one name or integer")
    return array.shape[array.axes.index(axis_name)]


@dataclass(eq=False, repr=False)
class NeuroArray(NDArrayOperatorsMixin):
    """A lazy, named-axis array backed by one NWB or NeuroFlow array selection.

    Supported NumPy expressions remain lazy. Numerical reads happen only at an
    explicit :meth:`compute` or :meth:`persist` boundary.
    """

    source: NWBSource
    selection: Selection
    workflow: Any | None = None
    _expression: Expression | None = None

    __array_priority__ = 1000

    def __post_init__(self) -> None:
        if self._expression is None:
            self._expression = make_input(
                self.selection.metadata.shape,
                self.selection.metadata.axes,
                self.selection.metadata.dtype,
            )

    @property
    def expression(self) -> Expression:
        value = self._expression
        if value is None:  # pragma: no cover - guarded by __post_init__
            raise RuntimeError("NeuroArray has no expression")
        return value

    @property
    def axes(self) -> tuple[str, ...]:
        return self.expression.axes

    @property
    def shape(self) -> tuple[int, ...]:
        return self.expression.shape

    @property
    def dtype(self) -> np.dtype[Any]:
        return np.dtype(self.expression.dtype)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return math.prod(self.shape)

    @property
    def nbytes(self) -> int:
        return self.size * self.dtype.itemsize

    def __repr__(self) -> str:
        return (
            f"NeuroArray(shape={self.shape}, axes={self.axes}, "
            f"dtype={self.dtype}, lazy=True)"
        )

    def __array__(
        self, dtype: object = None, copy: object = None
    ) -> np.ndarray:  # pragma: no cover - NumPy raises from this method
        del dtype, copy
        raise TypeError(
            "NeuroArray is lazy; use .compute() for an in-memory NumPy array "
            "or .persist(...) for a durable result"
        )

    def __bool__(self) -> bool:
        raise TypeError(
            "the truth value of a lazy NeuroArray is undefined; use an explicit "
            "reduction followed by .compute()"
        )

    def __iter__(self) -> object:
        raise TypeError(
            "NeuroArray iteration would trigger implicit reads; slice the array "
            "and call .compute() explicitly"
        )

    def __len__(self) -> int:
        if not self.shape:
            raise TypeError("len() of unsized NeuroArray")
        return self.shape[0]

    def _derived(self, expression: Expression) -> NeuroArray:
        return NeuroArray(
            self.source,
            self.selection,
            workflow=None,
            _expression=expression,
        )

    def isel(self, **indexers: slice) -> NeuroArray:
        """Return a lazy, rank-preserving contiguous slice by axis name."""
        unknown = set(indexers) - set(self.axes)
        if unknown:
            raise KeyError("array has no axes: " + ", ".join(sorted(unknown)))
        reduced = set(reduced_input_axes(self.expression))
        blocked = reduced & set(indexers)
        if blocked:
            raise ValueError(
                "cannot slice a reduced keepdims axis; slice before the reduction: "
                + ", ".join(sorted(blocked))
            )
        selection = self.selection.isel(**indexers)
        replacement = make_input(
            selection.metadata.shape,
            selection.metadata.axes,
            selection.metadata.dtype,
        )
        expression = rebind_input(self.expression, replacement)
        return NeuroArray(self.source, selection, _expression=expression)

    def __getitem__(self, key: object) -> NeuroArray:
        """Support NumPy basic slicing without implicit rank changes."""
        raw = key if isinstance(key, tuple) else (key,)
        ellipses = sum(item is Ellipsis for item in raw)
        if ellipses > 1:
            raise IndexError("an index can only have a single ellipsis")
        if any(item is None for item in raw):
            raise IndexError("newaxis is not supported; NeuroFlow preserves axis names")
        provided = len(raw) - ellipses
        if provided > self.ndim:
            raise IndexError("too many indices for NeuroArray")
        expanded: list[object] = []
        for item in raw:
            if item is Ellipsis:
                expanded.extend([slice(None)] * (self.ndim - provided))
            else:
                expanded.append(item)
        expanded.extend([slice(None)] * (self.ndim - len(expanded)))
        if not all(isinstance(item, slice) for item in expanded):
            raise IndexError(
                "NeuroFlow currently supports contiguous slices only; integer, "
                "boolean, and fancy indexing would change the named-axis layout"
            )
        return self.isel(
            **dict(zip(self.axes, expanded, strict=True))  # type: ignore[arg-type]
        )

    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: str,
        *inputs: object,
        **kwargs: object,
    ) -> object:
        if method != "__call__" or ufunc.__name__ not in SUPPORTED_UFUNCS:
            return NotImplemented
        if "out" in kwargs:
            raise TypeError("NeuroArray ufuncs do not support 'out'")
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"unsupported NeuroArray ufunc arguments: {names}")
        expressions: list[Expression] = []
        for value in inputs:
            if isinstance(value, NeuroArray):
                if _selection_key(value.selection) != _selection_key(self.selection):
                    raise ValueError(
                        "NeuroArray operands must use the same source selection; "
                        "persist or align them explicitly first"
                    )
                expressions.append(value.expression)
            else:
                expressions.append(make_scalar(value))
        return self._derived(make_ufunc(ufunc.__name__, tuple(expressions)))

    def __array_function__(
        self,
        function: object,
        types: tuple[type, ...],
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> object:
        if not all(issubclass(item, NeuroArray) for item in types):
            return NotImplemented
        if function in {np.shape, np.ndim}:
            if len(args) != 1 or kwargs:
                function_name = "np.shape" if function is np.shape else "np.ndim"
                raise TypeError(f"{function_name} accepts one NeuroArray")
            array = args[0]
            if not isinstance(array, NeuroArray):
                return NotImplemented
            return array.shape if function is np.shape else array.ndim
        if function is np.size:
            if not 1 <= len(args) <= 2 or set(kwargs) - {"axis"}:
                raise TypeError("np.size accepts one NeuroArray and an optional axis")
            if len(args) == 2 and "axis" in kwargs:
                raise TypeError("np.size received axis more than once")
            array = args[0]
            if not isinstance(array, NeuroArray):
                return NotImplemented
            axis = args[1] if len(args) == 2 else kwargs.get("axis")
            return array.size if axis is None else _size_along_axis(array, axis)
        reductions = {
            np.sum: "sum",
            np.mean: "mean",
            np.min: "min",
            np.amin: "min",
            np.max: "max",
            np.amax: "max",
            np.median: "median",
            np.percentile: "percentile",
            np.quantile: "quantile",
        }
        name = reductions.get(function)
        if name is None:
            return NotImplemented
        if not args or not isinstance(args[0], NeuroArray):
            return NotImplemented
        method = getattr(args[0], name)
        return method(*args[1:], **kwargs)

    def astype(
        self,
        dtype: str | np.dtype[Any],
        order: str = "K",
        casting: Casting = "unsafe",
        subok: bool = True,
        copy: bool = True,
    ) -> NeuroArray:
        """Return a lazy dtype conversion."""
        if order != "K":
            raise TypeError("NeuroFlow astype currently supports order='K' only")
        if not subok:
            raise TypeError("NeuroFlow astype does not support subok=False")
        del copy
        return self._derived(make_cast(self.expression, dtype, casting=casting))

    def _reduce(
        self,
        operation: str,
        axis: object,
        *,
        dtype: str | np.dtype[Any] | None = None,
        keepdims: bool = False,
        parameters: dict[str, object] | None = None,
    ) -> NeuroArray:
        if not isinstance(keepdims, (bool, np.bool_)):
            raise TypeError("keepdims must be a boolean")
        axes = _normalize_axes(axis, self.axes)
        return self._derived(
            make_reduction(
                operation,
                self.expression,
                axes,
                keepdims=bool(keepdims),
                dtype=dtype,
                parameters=parameters,
            )
        )

    def sum(
        self,
        axis: object = None,
        dtype: str | np.dtype[Any] | None = None,
        out: object = None,
        keepdims: bool = False,
        initial: object = _UNSET,
        where: object = _UNSET,
    ) -> NeuroArray:
        if out is not None:
            raise TypeError("NeuroArray reductions do not support 'out'")
        _reject_optional_argument("initial", initial, _UNSET)
        _reject_optional_argument("where", where, _UNSET)
        return self._reduce("sum", axis, dtype=dtype, keepdims=keepdims)

    def mean(
        self,
        axis: object = None,
        dtype: str | np.dtype[Any] | None = None,
        out: object = None,
        keepdims: bool = False,
        where: object = _UNSET,
    ) -> NeuroArray:
        if out is not None:
            raise TypeError("NeuroArray reductions do not support 'out'")
        _reject_optional_argument("where", where, _UNSET)
        return self._reduce("mean", axis, dtype=dtype, keepdims=keepdims)

    def min(
        self,
        axis: object = None,
        out: object = None,
        keepdims: bool = False,
        initial: object = _UNSET,
        where: object = _UNSET,
    ) -> NeuroArray:
        if out is not None:
            raise TypeError("NeuroArray reductions do not support 'out'")
        _reject_optional_argument("initial", initial, _UNSET)
        _reject_optional_argument("where", where, _UNSET)
        return self._reduce("min", axis, keepdims=keepdims)

    def max(
        self,
        axis: object = None,
        out: object = None,
        keepdims: bool = False,
        initial: object = _UNSET,
        where: object = _UNSET,
    ) -> NeuroArray:
        if out is not None:
            raise TypeError("NeuroArray reductions do not support 'out'")
        _reject_optional_argument("initial", initial, _UNSET)
        _reject_optional_argument("where", where, _UNSET)
        return self._reduce("max", axis, keepdims=keepdims)

    def median(
        self,
        axis: object = None,
        out: object = None,
        overwrite_input: bool = False,
        keepdims: bool = False,
        *,
        output: str | Path | None = None,
        chunks: tuple[int, ...] | None = None,
        max_workers: int | None = None,
        memory_limit: int | str | None = DEFAULT_PERSIST_MEMORY_LIMIT,
    ) -> NeuroArray:
        """Build a lazy median, or persist it with the legacy ``output=`` form."""
        if out is not None:
            raise TypeError("NeuroArray reductions do not support 'out'")
        if overwrite_input:
            raise TypeError("NeuroArray reductions do not support overwrite_input")
        result = self._reduce("median", axis, keepdims=keepdims)
        if output is None:
            if chunks is not None or max_workers is not None:
                raise TypeError("chunks and max_workers require output= or .persist()")
            return result
        return result.persist(
            output,
            chunks=chunks,
            max_workers=max_workers,
            memory_limit=memory_limit,
        )

    def percentile(
        self,
        q: float,
        axis: object = None,
        out: object = None,
        overwrite_input: bool = False,
        method: str = "linear",
        keepdims: bool = False,
        *,
        weights: object = None,
    ) -> NeuroArray:
        if out is not None or overwrite_input or weights is not None:
            raise TypeError(
                "NeuroFlow percentile does not support out, overwrite_input, or weights"
            )
        q_value = _validated_quantile(q, upper=100)
        return self._reduce(
            "percentile",
            axis,
            keepdims=keepdims,
            parameters={"q": q_value, "method": method},
        )

    def quantile(
        self,
        q: float,
        axis: object = None,
        out: object = None,
        overwrite_input: bool = False,
        method: str = "linear",
        keepdims: bool = False,
        *,
        weights: object = None,
    ) -> NeuroArray:
        if out is not None or overwrite_input or weights is not None:
            raise TypeError(
                "NeuroFlow quantile does not support out, overwrite_input, or weights"
            )
        q_value = _validated_quantile(q, upper=1)
        return self._reduce(
            "quantile",
            axis,
            keepdims=keepdims,
            parameters={"q": q_value, "method": method},
        )

    def compute(
        self,
        *,
        memory_limit: int | str | None = DEFAULT_COMPUTE_MEMORY_LIMIT,
        max_workers: int | None = None,
    ) -> np.ndarray:
        """Explicitly materialize the expression after a conservative size check."""
        estimate = estimate_working_memory(self.expression)
        if memory_limit is not None:
            # ``memory_limit`` means the same thing here as on ``persist()``: a
            # total process-memory target, not an allowance for array data
            # alone. The materialized result has to share the target with the
            # interpreter, the imported library stack and the read cache, so the
            # expression is checked against ``task_bytes`` rather than the
            # headline total. Comparing against the total would let a 1 GiB
            # request peak near 1.5 GiB of resident set.
            budget = resolve_memory_budget(memory_limit)
            if estimate > budget.task_bytes:
                raise ValueError(
                    f"expression requires an estimated {estimate} bytes, exceeding "
                    f"the {budget.task_bytes} bytes available for array data under "
                    f"a {budget.total_bytes}-byte total process-memory target "
                    f"({budget.reserved_bytes} bytes are committed to process "
                    "overhead); use .persist() for a bounded durable result"
                )
        if max_workers is not None and max_workers < 1:
            raise ValueError("max_workers must be positive")
        lazy = evaluate_dask(self.expression, self.selection.as_dask_array())
        options = {"num_workers": max_workers} if max_workers is not None else {}
        return np.asarray(lazy.compute(scheduler="threads", **options))

    def persist(
        self,
        output: str | Path,
        *,
        chunks: tuple[int, ...] | None = None,
        max_workers: int | None = None,
        memory_limit: int | str | None = DEFAULT_PERSIST_MEMORY_LIMIT,
        scheduler: Literal["threads", "processes", "distributed"] = "threads",
        resume: bool = True,
        mode: Literal["create", "overwrite"] = "create",
    ) -> NeuroArray:
        """Execute bounded tasks and return a lazy handle to durable Zarr output."""
        workflow = self._persist_workflow(
            output,
            chunks=chunks,
            max_workers=max_workers,
            memory_limit=memory_limit,
            scheduler=scheduler,
            resume=resume,
            mode=mode,
        )
        workflow.execute()
        # Execution just finalized every manifest and checksum in this process,
        # so trust those records instead of rereading the complete output here.
        source, selection = open_array(output, verify=False)
        return NeuroArray(source, selection, workflow)

    def to_spec(
        self,
        output: str | Path,
        *,
        chunks: tuple[int, ...] | None = None,
        max_workers: int | None = None,
        memory_limit: int | str | None = DEFAULT_PERSIST_MEMORY_LIMIT,
        scheduler: Literal["threads", "processes", "distributed"] = "threads",
        resume: bool = True,
    ) -> WorkflowSpec:
        """Describe a future persistence run as a safe portable workflow."""
        workflow = self._persist_workflow(
            output,
            chunks=chunks,
            max_workers=max_workers,
            memory_limit=memory_limit,
            scheduler=scheduler,
            resume=resume,
            mode="create",
        )
        return workflow.to_spec()

    def plan(
        self,
        output: str | Path,
        *,
        chunks: tuple[int, ...] | None = None,
        max_workers: int | None = None,
        memory_limit: int | str | None = DEFAULT_PERSIST_MEMORY_LIMIT,
    ) -> ExecutionPlan:
        """Return the validated metadata-only persistence plan."""
        return self._persist_workflow(
            output,
            chunks=chunks,
            max_workers=max_workers,
            memory_limit=memory_limit,
            scheduler="threads",
            resume=True,
            mode="create",
        ).plan

    def _persist_workflow(
        self,
        output: str | Path,
        *,
        chunks: tuple[int, ...] | None,
        max_workers: int | None,
        memory_limit: int | str | None,
        scheduler: Literal["threads", "processes", "distributed"],
        resume: bool,
        mode: Literal["create", "overwrite"],
    ) -> WorkflowResult:
        if chunks is not None:
            if len(chunks) != self.ndim or any(
                not isinstance(size, (int, np.integer))
                or isinstance(size, (bool, np.bool_))
                or size <= 0
                for size in chunks
            ):
                raise ValueError(
                    "chunks must contain one positive integer per output axis"
                )
            chunks = tuple(
                min(extent, int(chunk))
                for extent, chunk in zip(self.shape, chunks, strict=True)
            )
        required_axes = reduced_input_axes(
            self.expression, exclude_staged_dependencies=True
        )
        dropped_axes = tuple(
            axis for axis in self.selection.metadata.axes if axis not in self.axes
        )
        kept_reduced_axes = tuple(axis for axis in required_axes if axis in self.axes)
        native = self.selection.metadata.native_chunks or self.selection.metadata.shape
        output_chunk_by_axis = (
            dict(zip(self.axes, chunks, strict=True)) if chunks is not None else {}
        )
        split_axes: list[str] = []
        tile_shape: list[int] = []
        for axis, extent, native_chunk in zip(
            self.selection.metadata.axes,
            self.selection.metadata.shape,
            native,
            strict=True,
        ):
            if axis in required_axes or axis not in self.axes:
                continue
            tile = min(extent, native_chunk)
            output_chunk = output_chunk_by_axis.get(axis)
            if output_chunk is not None:
                tile = min(extent, math.lcm(tile, output_chunk))
            if tile < extent:
                split_axes.append(axis)
                tile_shape.append(tile)
        adapter = ExpressionAdapter(
            expression=self.expression,
            output=ArrayOutput(
                self.dtype.str,
                name="result",
                reduced_axes=dropped_axes,
                chunks=chunks,
                kept_reduced_axes=kept_reduced_axes,
            ),
            splittable_axes=tuple(
                axis
                for axis in self.selection.metadata.axes
                if axis not in required_axes
            ),
        )
        return run(
            source=self.source,
            selection=self.selection,
            adapter=adapter,
            partition=SpatialTilePlan(
                tuple(tile_shape),
                (0,) * len(tile_shape),
                tuple(split_axes),
            ),
            output=ZarrOutput(str(output), mode=mode),
            scheduler=scheduler,
            resume=resume,
            execute=False,
            max_workers=max_workers,
            memory_limit=memory_limit,
        )

    def segment(
        self,
        adapter: object,
        *,
        output: str | Path,
        tile_shape: tuple[int, ...],
        axes: tuple[str, ...],
        halo: tuple[int, ...] | None = None,
        max_workers: int | None = None,
        allow_unmerged: bool = False,
        memory_limit: int | str | None = None,
    ) -> WorkflowResult:
        """Run a segmentation adapter over bounded named-axis tiles."""
        if not isinstance(self.expression, InputExpr):
            raise ValueError(
                "persist a lazy NumPy expression before passing it to segmentation"
            )
        split_spatial = tuple(
            axis
            for axis, tile in zip(axes, tile_shape, strict=True)
            if axis in {"x", "y"} and tile < self.shape[self.axes.index(axis)]
        )
        if split_spatial and not allow_unmerged:
            raise ValueError(
                "segmentation would split cell-bearing spatial axes "
                f"{split_spatial}; use complete planes, an adapter with internal "
                "tiling, or allow_unmerged=True for explicitly unreconciled labels"
            )
        return run(
            source=self.source,
            selection=self.selection,
            adapter=adapter,  # type: ignore[arg-type]
            partition=SpatialTilePlan(tile_shape, halo or (0,) * len(tile_shape), axes),
            output=SegmentationOutput(str(output)),
            execute=True,
            max_workers=max_workers,
            memory_limit=memory_limit,
        )

    def cellpose(
        self,
        *,
        output: str | Path,
        pretrained_model: str = "cpsam",
        memory_limit: int | str = DEFAULT_SEGMENT_MEMORY_LIMIT,
        max_workers: int = 1,
        **model_settings: object,
    ) -> NeuroArray:
        """Segment a projection with Cellpose using a laptop-safe plane policy.

        Two-dimensional inputs run as one image. A named ``z`` axis runs one
        complete x/y plane per durable partition, avoiding unreconciled spatial
        tiling while retaining resume and integrity semantics.
        """
        from neuroflow_cellpose import CellposeAdapter

        if not isinstance(self.expression, InputExpr):
            raise ValueError("persist the projection before Cellpose segmentation")
        settings = dict(model_settings)
        # The adapter's declared per-task memory is deliberately *not* set from
        # ``memory_limit``. Echoing the user's own number back would make the
        # budget check circular: any limit would appear to be satisfied. The
        # adapter declares its real per-plane cost, and its model residency is
        # reported separately via ``external_memory_reserve_bytes()``.
        if "z" in self.axes:
            z_axis = self.axes.index("z")
            settings.setdefault("squeeze_singleton_axis", z_axis)
            tile_shape = (1,)
            axes = ("z",)
        elif self.ndim == 2:
            tile_shape = self.shape
            axes = self.axes
        else:
            raise ValueError(
                "Cellpose convenience requires a 2-D projection or named z planes"
            )
        adapter_type = cast(Any, CellposeAdapter)
        result = self.segment(
            adapter_type(pretrained_model=pretrained_model, **settings),
            output=output,
            tile_shape=tile_shape,
            axes=axes,
            max_workers=max_workers,
            memory_limit=memory_limit,
        )
        source, selection = open_array(output, component="labels", verify=False)
        return NeuroArray(source, selection, result)

    def extract_traces(
        self,
        labels: NeuroArray,
        *,
        output: str | Path,
        time_chunk: int | None = None,
        memory_limit: int | str = DEFAULT_PERSIST_MEMORY_LIMIT,
    ) -> NeuroArray:
        """Extract mean fluorescence per label with bounded movie reads."""
        if not isinstance(self.expression, InputExpr):
            raise ValueError("persist the movie expression before extracting traces")
        if not isinstance(labels.expression, InputExpr):
            raise ValueError("persist the label expression before extracting traces")
        from neuroflow.traces import extract_traces

        return extract_traces(
            self,
            labels,
            output=output,
            time_chunk=time_chunk,
            memory_limit=memory_limit,
        )

    def plan_traces(
        self,
        labels: NeuroArray,
        *,
        time_chunk: int | None = None,
        memory_limit: int | str = DEFAULT_PERSIST_MEMORY_LIMIT,
    ) -> TracePlan:
        """Plan source-aligned trace extraction without reading movie values."""
        if not isinstance(self.expression, InputExpr):
            raise ValueError("persist the movie expression before extracting traces")
        if not isinstance(labels.expression, InputExpr):
            raise ValueError("persist the label expression before extracting traces")
        from neuroflow.traces import plan_trace_extraction

        return plan_trace_extraction(
            self,
            labels,
            time_chunk=time_chunk,
            memory_limit=memory_limit,
        )

    def close(self) -> None:
        self.source.close()


def load(
    source: str | Path,
    *,
    name: str | None = None,
    asset: str | None = None,
    version: str | None = None,
    storage_options: dict[str, object] | None = None,
) -> NeuroArray:
    """Open one named NWB array as a lazy named-axis ``NeuroArray``."""
    opened = open_source(source, version=version, storage_options=storage_options)
    return NeuroArray(opened, opened.select(NWBQuery(name=name, asset=asset)))
