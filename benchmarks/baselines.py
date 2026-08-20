"""Fair baseline implementations over the same NWB array and selection."""

from __future__ import annotations

import importlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import dask.array as da
import numpy as np
from hdmf_zarr import NWBZarrIO
from pynwb import NWBHDF5IO


@dataclass(frozen=True)
class DirectTracePlan:
    """Manual Dask trace traversal facts for fair workflow comparisons."""

    source_shape: tuple[int, ...]
    source_chunks: tuple[int, ...]
    time_chunk: int
    cell_count: int
    active_spatial_chunks: int
    skipped_spatial_chunks: int
    dask_compute_calls: int
    source_chunks_touched: int
    estimated_uncompressed_bytes_read: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def direct_pynwb_zarr_projection(
    source: Path, *, object_name: str, frames: int
) -> np.ndarray:
    """Materialize the selected local array through PyNWB/HDMF-Zarr."""
    io = cast(Any, NWBZarrIO)(source, mode="r", load_namespaces=True)
    with io:
        nwbfile: Any = io.read()
        dataset = nwbfile.acquisition[object_name].data
        return np.median(np.asarray(dataset[:frames]), axis=0)


def direct_dask_zarr_projection(
    source: Path, *, object_name: str, frames: int
) -> np.ndarray:
    """Run the same median through direct Dask over the HDMF-Zarr array."""
    io = cast(Any, NWBZarrIO)(source, mode="r", load_namespaces=True)
    with io:
        nwbfile: Any = io.read()
        dataset = nwbfile.acquisition[object_name].data
        chunks = getattr(dataset, "chunks", None) or "auto"
        lazy = da.from_array(dataset, chunks=chunks, asarray=False, fancy=False)
        return np.asarray(da.median(lazy[:frames], axis=0).compute())


def lindi_hdf5_projection(
    source: str | Path,
    *,
    object_name: str,
    frames: int,
    z_axis: int | None = None,
) -> np.ndarray:
    """Read a local/remote HDF5 NWB file via LINDI's documented PyNWB bridge.

    When ``z_axis`` is supplied, planes are reduced independently so the
    baseline remains bounded rather than materializing a complete 4-D movie.
    """
    try:
        lindi = importlib.import_module("lindi")
    except ImportError as exc:
        raise RuntimeError(
            "the LINDI baseline requires the 'baselines' extra: "
            "uv sync --extra baselines"
        ) from exc
    lindi_file = lindi.LindiH5pyFile.from_hdf5_file(str(source))
    try:
        with NWBHDF5IO(file=lindi_file, mode="r", load_namespaces=True) as io:
            nwbfile: Any = io.read()
            dataset: Any = nwbfile.acquisition[object_name].data
            if z_axis is None:
                return np.median(np.asarray(dataset[:frames]), axis=0)
            if z_axis != dataset.ndim - 1:
                raise ValueError("the current bounded LINDI baseline expects z last")
            planes = [
                np.median(np.asarray(dataset[:frames, ..., index]), axis=0)
                for index in range(int(dataset.shape[z_axis]))
            ]
            return np.stack(planes, axis=-1)
    finally:
        close = getattr(lindi_file, "close", None)
        if callable(close) and getattr(lindi_file, "_is_open", True):
            close()


def direct_dask_mean_traces(
    dataset: Any,
    labels: np.ndarray,
    *,
    time_chunk: int,
) -> tuple[np.ndarray, np.ndarray, DirectTracePlan]:
    """Manually traverse source chunks with Dask and average each label.

    This intentionally includes the source-index construction, chunk loop, and
    accumulator logic a baseline user must supply. It provides no persistence,
    restart, checksums, memory planning, or provenance.
    """
    shape = tuple(int(value) for value in dataset.shape)
    chunks_value = getattr(dataset, "chunks", None)
    if chunks_value is None:
        raise ValueError("direct Dask trace baseline requires physical chunks")
    chunks = tuple(int(value) for value in chunks_value)
    if len(shape) < 2 or len(chunks) != len(shape):
        raise ValueError("dataset shape/chunks are incompatible")
    if tuple(labels.shape) != shape[1:]:
        raise ValueError("label shape must match dataset spatial shape")
    if labels.dtype.kind not in "ui" or np.any(labels < 0):
        raise ValueError("labels must be non-negative integers")
    if time_chunk < 1 or time_chunk % chunks[0] != 0:
        raise ValueError("time_chunk must be a positive multiple of source time chunks")

    cell_ids, counts = np.unique(labels[labels != 0], return_counts=True)
    if not len(cell_ids):
        raise ValueError("labels contain no cells")
    id_to_column = {int(value): index for index, value in enumerate(cell_ids)}
    spatial_slices = [
        _axis_slices(size, chunk)
        for size, chunk in zip(shape[1:], chunks[1:], strict=True)
    ]
    active: list[tuple[tuple[slice, ...], tuple[int, ...]]] = []
    total_spatial_chunks = math.prod(len(axis) for axis in spatial_slices)
    for index in np.ndindex(*(len(axis) for axis in spatial_slices)):
        key = tuple(spatial_slices[axis][item] for axis, item in enumerate(index))
        local_ids = tuple(int(value) for value in np.unique(labels[key]) if value != 0)
        if local_ids:
            active.append((key, local_ids))

    lazy = cast(Any, da.from_array)(dataset, chunks=chunks, asarray=False, fancy=False)
    traces = np.empty((shape[0], len(cell_ids)), dtype=np.float32)
    compute_calls = 0
    estimated_read = 0
    for start in range(0, shape[0], time_chunk):
        stop = min(shape[0], start + time_chunk)
        sums = np.zeros((stop - start, len(cell_ids)), dtype=np.float64)
        for spatial_key, local_ids in active:
            block = np.asarray(
                lazy[(slice(start, stop), *spatial_key)].compute(
                    scheduler="threads", num_workers=1
                ),
                dtype=np.float32,
            ).reshape(stop - start, -1)
            compute_calls += 1
            estimated_read += int(block.size * np.dtype(dataset.dtype).itemsize)
            flat_labels = labels[spatial_key].reshape(-1)
            for label_id in local_ids:
                sums[:, id_to_column[label_id]] += block[
                    :, flat_labels == label_id
                ].sum(axis=1)
        traces[start:stop] = (sums / counts[None, :]).astype(np.float32)
    source_chunks_touched = len(active) * math.ceil(shape[0] / chunks[0])
    plan = DirectTracePlan(
        source_shape=shape,
        source_chunks=chunks,
        time_chunk=time_chunk,
        cell_count=len(cell_ids),
        active_spatial_chunks=len(active),
        skipped_spatial_chunks=total_spatial_chunks - len(active),
        dask_compute_calls=compute_calls,
        source_chunks_touched=source_chunks_touched,
        estimated_uncompressed_bytes_read=estimated_read,
    )
    return traces, cell_ids.astype(np.uint64), plan


def _axis_slices(size: int, chunk: int) -> tuple[slice, ...]:
    return tuple(
        slice(start, min(size, start + chunk)) for start in range(0, size, chunk)
    )
