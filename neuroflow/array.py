"""NumPy-like high-level array interface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from neuroflow.adapters import ArrayOutput, FunctionAdapter
from neuroflow.api import open_array, open_source, run
from neuroflow.partition import SpatialTilePlan
from neuroflow.results.workflow import WorkflowResult
from neuroflow.selection import NWBQuery, Selection
from neuroflow.source.base import NWBSource
from neuroflow.storage import SegmentationOutput, ZarrOutput


def _median(value: np.ndarray, *, axis: int) -> np.ndarray:
    return np.asarray(np.median(value, axis=axis), dtype=np.float32)


@dataclass
class NeuroArray:
    """Named-axis lazy array backed by an NWB object or persisted Zarr result."""

    source: NWBSource
    selection: Selection
    workflow: Any | None = None

    @property
    def axes(self) -> tuple[str, ...]:
        return self.selection.metadata.axes

    @property
    def shape(self) -> tuple[int, ...]:
        return self.selection.metadata.shape

    def isel(self, **indexers: slice) -> NeuroArray:
        return NeuroArray(self.source, self.selection.isel(**indexers))

    def median(
        self,
        axis: str,
        *,
        output: str | Path,
        chunks: tuple[int, ...] | None = None,
        max_workers: int | None = None,
        memory_limit: int | str | None = None,
    ) -> NeuroArray:
        """Compute a resumable temporal/spatial median by named axis."""
        if axis not in self.axes:
            raise ValueError(f"array has no axis {axis!r}")
        reduced_index = self.axes.index(axis)
        output_axes = tuple(item for item in self.axes if item != axis)
        output_shape = tuple(
            size for index, size in enumerate(self.shape) if index != reduced_index
        )
        chunks = chunks or output_shape
        native = self.selection.metadata.native_chunks or self.shape
        split_axes = tuple(
            item
            for index, item in enumerate(self.axes)
            if item != axis and native[index] < self.shape[index]
        )
        if not split_axes:
            split_axes = (output_axes[-1],)
        tile_shape = tuple(native[self.axes.index(item)] for item in split_axes)
        adapter = FunctionAdapter(
            function=_median,
            input_kind="array",
            output=ArrayOutput(
                "float32",
                name="median",
                reduced_axes=(axis,),
                chunks=chunks,
            ),
            name="median",
            version="1",
            splittable_axes=split_axes,
            parameters={"axis": reduced_index},
        )
        workflow = run(
            source=self.source,
            selection=self.selection,
            adapter=adapter,
            partition=SpatialTilePlan(
                tile_shape, (0,) * len(tile_shape), split_axes
            ),
            output=ZarrOutput(str(output)),
            execute=True,
            max_workers=max_workers,
            memory_limit=memory_limit,
        )
        source, selection = open_array(output)
        return NeuroArray(source, selection, workflow)

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
            partition=SpatialTilePlan(
                tile_shape, halo or (0,) * len(tile_shape), axes
            ),
            output=SegmentationOutput(str(output)),
            execute=True,
            max_workers=max_workers,
            memory_limit=memory_limit,
        )

    def compute(self) -> np.ndarray:
        return np.asarray(self.selection.as_dask_array().compute())

    def extract_traces(
        self,
        labels: NeuroArray,
        *,
        output: str | Path,
        time_chunk: int = 10,
        memory_limit: int | str | None = None,
    ) -> NeuroArray:
        """Extract mean fluorescence per label with bounded movie reads."""
        from neuroflow.traces import extract_traces

        return extract_traces(
            self,
            labels,
            output=output,
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
    storage_options: dict[str, object] | None = None,
) -> NeuroArray:
    """Open one named NWB array as a lazy named-axis ``NeuroArray``."""
    opened = open_source(source, storage_options=storage_options)
    return NeuroArray(opened, opened.select(NWBQuery(name=name, asset=asset)))
