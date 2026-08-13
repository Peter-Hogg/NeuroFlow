"""Semantic selection value objects."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

import dask.array as da
import numpy as np

from neuroflow.source.base import SourceIdentity


class _SlicedArray:
    """Lazy basic-slice view over an array-like source."""

    def __init__(self, source: Any, slices: tuple[slice, ...]) -> None:
        self.source = source
        self.slices = slices
        self.shape = tuple(item.stop - item.start for item in slices)  # type: ignore[operator]
        self.dtype = source.dtype
        native = getattr(source, "chunks", None)
        self.chunks = (
            tuple(
                min(size, chunk)
                for size, chunk in zip(self.shape, native, strict=True)
            )
            if native
            else None
        )
        self.ndim = len(self.shape)

    def __getitem__(self, key: object) -> object:
        raw_keys = key if isinstance(key, tuple) else (key,)
        if len(raw_keys) != self.ndim or not all(
            isinstance(item, slice) for item in raw_keys
        ):
            raise IndexError("bounded selections support one basic slice per axis")
        keys = cast(tuple[slice, ...], raw_keys)
        mapped: list[slice] = []
        for outer, inner, size in zip(self.slices, keys, self.shape, strict=True):
            start, stop, step = inner.indices(size)
            if step != 1:
                raise IndexError("bounded selections do not support slice steps")
            mapped.append(slice(outer.start + start, outer.start + stop))  # type: ignore[operator]
        return self.source[tuple(mapped)]


@dataclass(frozen=True)
class NWBQuery:
    neurodata_type: str | type | None = None
    name: str | None = None
    path: str | None = None
    asset: str | None = None
    subject: str | None = None
    session_id: str | None = None
    where: Mapping[str, object] | None = None


@dataclass(frozen=True)
class SelectionMetadata:
    source: SourceIdentity
    path: str
    neurodata_type: str
    shape: tuple[int, ...]
    dtype: str
    native_chunks: tuple[int, ...] | None
    axes: tuple[str, ...] = ()
    name: str | None = None
    rate: float | None = None
    starting_time: float | None = None
    timestamps_path: str | None = None
    attributes: Mapping[str, object] | None = None


@dataclass(frozen=True)
class Selection:
    """A semantic NWB selection backed by an unopened numerical Dask graph."""

    metadata: SelectionMetadata
    # Zarr arrays and h5py datasets both provide shape/dtype/chunks and bounded
    # NumPy-style slicing. Keeping the concrete handle here prevents conversion
    # (and therefore prevents accidental full materialization).
    _array: Any
    _timestamps: Any | None = None

    def as_dask_array(
        self,
        *,
        chunks: tuple[int, ...] | Literal["native", "auto"] = "auto",
    ) -> da.Array:
        """Expose numerical data lazily without reading array chunks."""
        native = self._array.chunks
        requested = cast(
            tuple[int, ...] | Literal["auto"],
            (native or "auto") if chunks in ("native", "auto") else chunks,
        )
        return da.from_array(
            self._array,
            chunks=requested,  # pyright: ignore[reportArgumentType] - Dask stub is narrow
            name=False,
            asarray=False,
            fancy=False,
            meta=np.empty((0,) * self._array.ndim, dtype=self._array.dtype),
        )

    def plan(self, partition: object) -> object:
        """Build partitions using a compatible partition plan."""
        build = getattr(partition, "build", None)
        if build is None:
            raise TypeError("partition must implement build(selection)")
        return build(self)

    def as_dask_timestamps(self) -> da.Array | None:
        """Return irregular timestamps lazily, or ``None`` for regular sampling."""
        if self._timestamps is None:
            return None
        chunks = cast(
            tuple[int, ...] | Literal["auto"], self._timestamps.chunks or "auto"
        )
        return da.from_array(
            self._timestamps,
            chunks=chunks,  # pyright: ignore[reportArgumentType] - Dask stub is narrow
            name=False,
            asarray=False,
            fancy=False,
            meta=np.empty((0,), dtype=self._timestamps.dtype),
        )

    def isel(self, **indexers: slice) -> "Selection":
        """Return a lazy, bounded basic-slice selection by named axis."""
        unknown = set(indexers) - set(self.metadata.axes)
        if unknown:
            raise KeyError("selection has no axes: " + ", ".join(sorted(unknown)))
        slices: list[slice] = []
        for axis, size in zip(self.metadata.axes, self.metadata.shape, strict=True):
            requested = indexers.get(axis, slice(0, size))
            start, stop, step = requested.indices(size)
            if step != 1:
                raise ValueError("isel only supports contiguous slices with step 1")
            if stop <= start:
                raise ValueError(f"isel produced an empty {axis!r} axis")
            slices.append(slice(start, stop))
        bounded = tuple(slices)
        shape = tuple(item.stop - item.start for item in bounded)  # type: ignore[operator]
        native = self.metadata.native_chunks
        metadata = SelectionMetadata(
            **{
                **self.metadata.__dict__,
                "shape": shape,
                "native_chunks": (
                    tuple(
                        min(size, chunk)
                        for size, chunk in zip(shape, native, strict=True)
                    )
                    if native
                    else None
                ),
            }
        )
        timestamps = self._timestamps
        if timestamps is not None and "time" in self.metadata.axes:
            time_slice = bounded[self.metadata.axes.index("time")]
            timestamps = _SlicedArray(timestamps, (time_slice,))
        return Selection(metadata, _SlicedArray(self._array, bounded), timestamps)
