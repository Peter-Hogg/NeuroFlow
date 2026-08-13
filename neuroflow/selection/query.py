"""Semantic selection value objects."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

import dask.array as da
import numpy as np

from neuroflow.source.base import SourceIdentity


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
