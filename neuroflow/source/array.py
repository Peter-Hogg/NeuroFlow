"""Generic array source for chaining persisted NeuroFlow results."""

from __future__ import annotations

from pathlib import Path

import fsspec
import zarr

from neuroflow.selection.query import Selection, SelectionMetadata
from neuroflow.source.base import AssetMetadata, SourceIdentity, SourceSummary


class ArraySource:
    """Expose one Zarr array through the standard NeuroFlow source contract."""

    def __init__(
        self,
        uri: str | Path,
        *,
        component: str,
        axes: tuple[str, ...],
    ) -> None:
        self.uri = str(uri)
        self.component = component
        mapper = fsspec.get_mapper(self.uri)
        group = zarr.open_group(mapper, mode="r")
        value = group[component]
        if not isinstance(value, zarr.Array):
            raise TypeError(f"{component!r} is not a Zarr array")
        if len(axes) != value.ndim or len(set(axes)) != len(axes):
            raise ValueError("axes must be unique and match the array rank")
        self._array = value
        self.axes = axes
        self.identity = SourceIdentity(self.uri, None, asset_id=component)

    def assets(self) -> tuple[AssetMetadata, ...]:
        return (AssetMetadata(self.component, self.component, is_zarr=True),)

    def select(self, query: object | None = None) -> Selection:
        metadata = SelectionMetadata(
            source=self.identity,
            path=self.component,
            neurodata_type="Array",
            shape=tuple(self._array.shape),
            dtype=str(self._array.dtype),
            native_chunks=tuple(self._array.chunks),
            axes=self.axes,
            name=self.component,
            attributes={"backend": "zarr-array", "transport": "local-or-fsspec"},
        )
        return Selection(metadata, self._array)

    def inspect(self) -> SourceSummary:
        return SourceSummary(self.identity, assets=self.assets())

    def close(self) -> None:
        return None
