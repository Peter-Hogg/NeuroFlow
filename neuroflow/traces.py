"""Durable bounded fluorescence trace extraction from dense labels."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import dask.array as da
import fsspec
import numpy as np
import zarr

from neuroflow.array import NeuroArray
from neuroflow.exceptions import OutputConflictError, ProvenanceMismatchError
from neuroflow.execution.resources import parse_bytes
from neuroflow.partition.base import Partition
from neuroflow.provenance.hashing import stable_hash
from neuroflow.selection.query import absolute_selection_bounds
from neuroflow.storage.base import (
    join_uri,
    read_json,
    validate_output_separation,
    write_json_atomic,
)
from neuroflow.storage.manifest import PartitionManifest


def extract_traces(
    movie: NeuroArray,
    labels: NeuroArray,
    *,
    output: str | Path,
    time_chunk: int = 10,
    memory_limit: int | str | None = None,
) -> NeuroArray:
    """Average movie voxels per nonzero label in resumable time windows."""
    _validate_inputs(movie, labels, time_chunk)
    output_uri = str(output)
    validate_output_separation(
        output_uri,
        {
            "movie": str(
                getattr(movie.source, "uri", movie.selection.metadata.source.uri)
            ),
            "labels": str(
                getattr(labels.source, "uri", labels.selection.metadata.source.uri)
            ),
        },
    )
    label_data = labels.selection.as_dask_array()
    budget = parse_bytes(memory_limit) if memory_limit is not None else None
    ids, counts = _discover_label_counts(label_data, budget=budget)
    if not len(ids):
        raise ValueError("labels contain no cells")
    estimated_memory = _estimated_trace_memory(movie, labels, len(ids), time_chunk)
    if memory_limit is not None and estimated_memory > parse_bytes(memory_limit):
        raise ValueError(
            f"trace window requires an estimated {estimated_memory} bytes, "
            f"exceeding memory_limit={memory_limit!r}"
        )

    workflow_id = stable_hash(
        {
            "operation": "mean-fluorescence-traces",
            "movie": asdict(movie.selection.metadata.source),
            "movie_path": movie.selection.metadata.path,
            "movie_shape": movie.shape,
            "movie_bounds": absolute_selection_bounds(movie.selection.metadata),
            "labels": asdict(labels.selection.metadata.source),
            "labels_path": labels.selection.metadata.path,
            "labels_shape": labels.shape,
            "label_bounds": absolute_selection_bounds(labels.selection.metadata),
            "cell_ids": hashlib.sha256(ids.tobytes()).hexdigest(),
            "time_chunk": time_chunk,
            "schema_version": "1",
        }
    )
    partitions = _time_partitions(movie, len(ids), time_chunk)
    traces = _initialize_trace_output(
        output_uri,
        movie,
        labels,
        ids,
        workflow_id,
        partitions,
        time_chunk,
        estimated_memory,
        memory_limit,
    )
    movie_data = movie.selection.as_dask_array()
    spatial_axes = labels.axes
    z_axis = spatial_axes.index("z") if "z" in spatial_axes else None
    z_values = range(labels.shape[z_axis]) if z_axis is not None else (None,)
    id_to_column = {int(value): index for index, value in enumerate(ids)}

    try:
        for partition in partitions:
            manifest_path = _manifest_uri(output_uri, partition.key)
            existing = read_json(manifest_path)
            if existing is not None and _valid_trace_manifest(
                existing, traces, partition, workflow_id, output_uri
            ):
                continue
            start = partition.output_slices[0].start or 0
            stop = partition.output_slices[0].stop or traces.shape[0]
            sums = np.zeros((stop - start, len(ids)), dtype=np.float64)
            for z_value in z_values:
                label_key = [slice(None)] * len(spatial_axes)
                movie_key = [slice(None)] * len(movie.axes)
                movie_key[movie.axes.index("time")] = slice(start, stop)
                if z_axis is not None:
                    label_key[z_axis] = slice(z_value, z_value + 1)  # type: ignore[operator]
                    movie_key[movie.axes.index("z")] = slice(z_value, z_value + 1)  # type: ignore[operator]
                plane_labels = np.asarray(
                    label_data[tuple(label_key)].compute(
                        scheduler="threads", num_workers=1
                    )
                )
                block = np.asarray(
                    movie_data[tuple(movie_key)].compute(
                        scheduler="threads", num_workers=1
                    ),
                    dtype=np.float32,
                )
                block = np.moveaxis(block, movie.axes.index("time"), 0).reshape(
                    stop - start, -1
                )
                flat_labels = plane_labels.reshape(-1)
                for label_id in np.unique(flat_labels):
                    if label_id == 0:
                        continue
                    column = id_to_column[int(label_id)]
                    sums[:, column] += block[:, flat_labels == label_id].sum(axis=1)
            values = (sums / counts[None, :]).astype(np.float32)
            traces[partition.output_slices] = values
            checksum = hashlib.sha256(values.tobytes(order="C")).hexdigest()
            write_json_atomic(
                manifest_path,
                PartitionManifest(
                    partition.key,
                    workflow_id,
                    "complete",
                    {"traces": output_uri},
                    {"traces": checksum},
                    sizes={"traces": int(values.nbytes)},
                ).to_dict(),
            )
    except Exception as exc:
        _fail_trace_output(output_uri, workflow_id, exc)
        raise
    _finalize_trace_output(output_uri, workflow_id, partitions)
    from neuroflow.api import open_array

    source, selection = open_array(output_uri, verify=False)
    return NeuroArray(source, selection)


def _validate_inputs(movie: NeuroArray, labels: NeuroArray, time_chunk: int) -> None:
    if "time" not in movie.axes:
        raise ValueError("movie requires a time axis")
    spatial_axes = tuple(axis for axis in movie.axes if axis != "time")
    if labels.axes != spatial_axes:
        raise ValueError("label axes must equal the movie axes excluding time")
    expected = tuple(movie.shape[movie.axes.index(axis)] for axis in spatial_axes)
    if labels.shape != expected:
        raise ValueError("label and movie spatial shapes differ")
    if np.dtype(labels.selection.metadata.dtype).kind not in "ui":
        raise TypeError("labels must contain non-negative integers")
    if time_chunk < 1:
        raise ValueError("time_chunk must be positive")


def _discover_label_counts(
    label_data: da.Array,
    *,
    budget: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Count labels one storage chunk at a time with one Dask worker.

    A Python mapping is retained because the number of cells is normally tiny
    relative to the voxel count. Its conservative per-entry budget prevents a
    pathological one-label-per-voxel input from consuming unbounded memory.
    """
    counts_by_id: dict[int, int] = {}
    itemsize = int(label_data.dtype.itemsize)
    entry_bytes = 160
    block_grid = tuple(len(axis_chunks) for axis_chunks in label_data.chunks)
    chunk_slices: list[tuple[slice, ...]] = []
    for axis_chunks in label_data.chunks:
        start = 0
        slices: list[slice] = []
        for chunk_size in axis_chunks:
            stop = start + int(chunk_size)
            slices.append(slice(start, stop))
            start = stop
        chunk_slices.append(tuple(slices))
    for block_index in np.ndindex(*block_grid):
        key = tuple(chunk_slices[axis][index] for axis, index in enumerate(block_index))
        lazy_block = label_data[key]
        block_elements = math.prod(int(size) for size in lazy_block.shape)
        # Input, sort workspace, unique values, int64 counts, and aggregation
        # workspace can coexist during np.unique.
        block_workspace = block_elements * (4 * itemsize + 8)
        if budget is not None and (
            block_workspace + len(counts_by_id) * entry_bytes > budget
        ):
            raise ValueError(
                "label discovery requires an estimated chunk workspace exceeding "
                f"memory_limit={budget} bytes; rechunk labels or raise the limit"
            )
        block = np.asarray(lazy_block.compute(scheduler="threads", num_workers=1))
        if block.dtype.kind == "i" and np.any(block < 0):
            raise ValueError("labels must contain non-negative integers")
        local_ids, local_counts = np.unique(block, return_counts=True)
        foreground = local_ids != 0
        local_ids = local_ids[foreground]
        local_counts = local_counts[foreground]
        new_count = sum(int(value) not in counts_by_id for value in local_ids)
        if budget is not None and (
            block_workspace + (len(counts_by_id) + new_count) * entry_bytes > budget
        ):
            raise ValueError(
                "label discovery found an estimated distinct-label workspace "
                "exceeding "
                f"memory_limit={budget} bytes"
            )
        for label_id, count in zip(local_ids, local_counts, strict=True):
            key = int(label_id)
            counts_by_id[key] = counts_by_id.get(key, 0) + int(count)

    ordered_ids = sorted(counts_by_id)
    ids = np.fromiter(ordered_ids, dtype=np.uint64, count=len(ordered_ids))
    counts = np.fromiter(
        (counts_by_id[label_id] for label_id in ordered_ids),
        dtype=np.int64,
        count=len(ordered_ids),
    )
    return ids, counts


def _time_partitions(
    movie: NeuroArray, cell_count: int, time_chunk: int
) -> tuple[Partition, ...]:
    time_size = movie.shape[movie.axes.index("time")]
    return tuple(
        Partition(
            key=f"trace-{start:08d}",
            read_slices=tuple(
                slice(start, min(time_size, start + time_chunk))
                if axis == "time"
                else slice(0, size)
                for axis, size in zip(movie.axes, movie.shape, strict=True)
            ),
            output_slices=(
                slice(start, min(time_size, start + time_chunk)),
                slice(0, cell_count),
            ),
            trim_slices=(),
            coordinates=(start,),
        )
        for start in range(0, time_size, time_chunk)
    )


def _estimated_trace_memory(
    movie: NeuroArray, labels: NeuroArray, cell_count: int, time_chunk: int
) -> int:
    plane_shape = list(labels.shape)
    if "z" in labels.axes:
        plane_shape[labels.axes.index("z")] = 1
    plane_elements = int(np.prod(plane_shape, dtype=np.int64))
    window_frames = min(time_chunk, movie.shape[movie.axes.index("time")])
    movie_elements = window_frames * plane_elements
    source_movie_bytes = (
        movie_elements * np.dtype(movie.selection.metadata.dtype).itemsize
    )
    float_movie_bytes = movie_elements * np.dtype("float32").itemsize
    label_itemsize = np.dtype(labels.selection.metadata.dtype).itemsize
    label_bytes = plane_elements * label_itemsize
    label_workspace = plane_elements * (2 * label_itemsize + 9)
    accumulator_bytes = time_chunk * cell_count * np.dtype("float64").itemsize
    trace_output_bytes = time_chunk * cell_count * np.dtype("float32").itemsize
    cell_index_bytes = cell_count * 160
    return (
        source_movie_bytes
        + 2 * float_movie_bytes
        + label_bytes
        + label_workspace
        + accumulator_bytes
        + trace_output_bytes
        + cell_index_bytes
    )


def _initialize_trace_output(
    uri: str,
    movie: NeuroArray,
    labels: NeuroArray,
    ids: np.ndarray,
    workflow_id: str,
    partitions: tuple[Partition, ...],
    time_chunk: int,
    estimated_memory: int,
    memory_limit: int | str | None,
) -> zarr.Array:
    existing = read_json(join_uri(uri, ".neuroflow", "provenance.json"))
    filesystem, root_path = fsspec.core.url_to_fs(uri)
    if filesystem.exists(root_path) and existing is None:
        raise OutputConflictError(
            "trace output already exists without matching NeuroFlow provenance"
        )
    if existing is not None and existing.get("workflow_id") != workflow_id:
        raise ProvenanceMismatchError(
            "existing trace output belongs to another workflow"
        )
    shape = (movie.shape[movie.axes.index("time")], len(ids))
    chunks = (min(time_chunk, shape[0]), min(1024, shape[1]))
    timestamp_workspace = shape[0] * np.dtype("float64").itemsize * 2
    coordinate_workspace = int(ids.nbytes) + timestamp_workspace
    if memory_limit is not None and coordinate_workspace > parse_bytes(memory_limit):
        raise ValueError(
            "trace coordinates require an estimated allocation of "
            f"{coordinate_workspace} bytes, exceeding memory_limit={memory_limit!r}"
        )
    mapper = fsspec.get_mapper(uri)
    root = zarr.open_group(mapper, mode="a")
    if "cell_ids" in root:
        cell_ids = root["cell_ids"]
        if (
            not isinstance(cell_ids, zarr.Array)
            or np.dtype(cell_ids.dtype) != np.dtype("uint64")
            or tuple(cell_ids.shape) != ids.shape
            or not np.array_equal(np.asarray(cell_ids), ids)
        ):
            raise ValueError("existing trace output has different cell IDs")
    else:
        root.create_dataset("cell_ids", data=ids)
    if "traces" in root:
        traces = root["traces"]
        if (
            not isinstance(traces, zarr.Array)
            or tuple(traces.shape) != shape
            or np.dtype(traces.dtype) != np.dtype("float32")
            or tuple(traces.chunks) != chunks
        ):
            raise ValueError("existing trace output has an incompatible shape")
    else:
        traces = root.create_dataset(
            "traces", shape=shape, chunks=chunks, dtype="float32", fill_value=np.nan
        )
    timestamps = movie.selection.as_dask_timestamps()
    if timestamps is not None:
        time_values = np.asarray(
            timestamps.compute(scheduler="threads", num_workers=1), dtype=np.float64
        )
    elif movie.selection.metadata.rate is not None:
        time_values = (
            float(movie.selection.metadata.starting_time or 0.0)
            + np.arange(shape[0], dtype=np.float64) / movie.selection.metadata.rate
        )
    else:
        time_values = np.arange(shape[0], dtype=np.float64)
    if "timestamps" in root:
        stored_timestamps = root["timestamps"]
        if (
            not isinstance(stored_timestamps, zarr.Array)
            or np.dtype(stored_timestamps.dtype) != np.dtype("float64")
            or tuple(stored_timestamps.shape) != time_values.shape
            or not np.array_equal(np.asarray(stored_timestamps), time_values)
        ):
            raise ValueError("existing trace output has different timestamps")
    else:
        root.create_dataset("timestamps", data=time_values)
    execution_started = datetime.now(timezone.utc).isoformat()
    current_attempt: dict[str, object] = {
        "execution_started": execution_started,
        "status": "running",
        "memory_limit": memory_limit,
    }
    provenance = {
        "schema_version": "1",
        "workflow_id": workflow_id,
        "source": asdict(movie.selection.metadata.source),
        "nwb_paths": [movie.selection.metadata.path],
        "adapter": {"name": "mean-fluorescence-traces", "version": "1"},
        "parameters": {
            "time_chunk": time_chunk,
            "estimated_memory_per_window": estimated_memory,
            "memory_limit": memory_limit,
        },
        "selection": {
            "shape": movie.shape,
            "axes": movie.axes,
            "dtype": movie.selection.metadata.dtype,
        },
        "labels": {
            "source": asdict(labels.selection.metadata.source),
            "path": labels.selection.metadata.path,
            "shape": labels.shape,
            "axes": labels.axes,
        },
        "partition_plan": {
            "task_count": len(partitions),
            "partition_ids": [item.key for item in partitions],
            "partitions": [
                {"partition_id": item.key, **item.to_dict()} for item in partitions
            ],
        },
        "output": {
            "kind": "array",
            "uri": uri,
            "name": "traces",
            "dtype": "float32",
            "shape": shape,
            "axes": ["time", "cell"],
            "chunks": chunks,
            "coordinates": {"time": "timestamps", "cell": "cell_ids"},
        },
        "status": "running",
        "execution_started": execution_started,
        "execution_attempts": [current_attempt],
    }
    if existing is not None:
        current_attempt["resumed_from_status"] = str(existing.get("status", "unknown"))
        provenance["execution_attempts"] = [
            *_existing_trace_attempts(existing),
            current_attempt,
        ]
        for key in ("execution_started", "execution_finished"):
            if key in existing:
                provenance[key] = existing[key]
    write_json_atomic(join_uri(uri, ".neuroflow", "provenance.json"), provenance)
    return traces


def _manifest_uri(uri: str, partition_id: str) -> str:
    return join_uri(uri, ".neuroflow", "manifests", f"{partition_id}.json")


def _valid_trace_manifest(
    value: dict[str, object],
    traces: zarr.Array,
    partition: Partition,
    workflow_id: str,
    output_uri: str,
) -> bool:
    try:
        manifest = PartitionManifest.from_dict(value)
    except (KeyError, ValueError):
        return False
    if (
        manifest.partition_id != partition.key
        or manifest.workflow_id != workflow_id
        or manifest.status != "complete"
        or dict(manifest.outputs) != {"traces": output_uri}
        or set(manifest.checksums) != {"traces"}
    ):
        return False
    block = np.asarray(traces[partition.output_slices])
    if manifest.sizes and manifest.sizes.get("traces") != int(block.nbytes):
        return False
    actual = hashlib.sha256(block.tobytes(order="C")).hexdigest()
    return manifest.checksums.get("traces") == actual


def _existing_trace_attempts(
    provenance: dict[str, object],
) -> list[dict[str, object]]:
    attempts = provenance.get("execution_attempts")
    if isinstance(attempts, list) and all(isinstance(item, dict) for item in attempts):
        return [dict(item) for item in attempts]
    upgraded: dict[str, object] = {"status": str(provenance.get("status", "unknown"))}
    for key in ("execution_started", "execution_finished", "error"):
        if key in provenance:
            upgraded[key] = provenance[key]
    return [upgraded]


def _finish_trace_attempt(
    provenance: dict[str, object],
    *,
    status: str,
    finished: str,
    error: str | None = None,
) -> str | None:
    attempts = provenance.get("execution_attempts")
    if not isinstance(attempts, list) or not attempts:
        return None
    latest = attempts[-1]
    if not isinstance(latest, dict):
        return None
    latest["status"] = status
    latest["execution_finished"] = finished
    if error is not None:
        latest["error"] = error
    previous = latest.get("resumed_from_status")
    return str(previous) if previous is not None else None


def _finalize_trace_output(
    uri: str, workflow_id: str, partitions: tuple[Partition, ...]
) -> None:
    provenance_uri = join_uri(uri, ".neuroflow", "provenance.json")
    provenance = read_json(provenance_uri)
    if provenance is None or provenance.get("workflow_id") != workflow_id:
        raise RuntimeError("trace provenance disappeared during execution")
    finished = datetime.now(timezone.utc).isoformat()
    previous_status = _finish_trace_attempt(
        provenance, status="complete", finished=finished
    )
    provenance["status"] = "complete"
    provenance["completed_partitions"] = [item.key for item in partitions]
    provenance["failed_partitions"] = []
    if previous_status != "complete":
        provenance["execution_finished"] = finished
    write_json_atomic(provenance_uri, provenance)
    write_json_atomic(
        join_uri(uri, ".neuroflow", "result.json"),
        {
            "schema_version": "1",
            "workflow_id": workflow_id,
            "status": "complete",
            "task_count": len(partitions),
            "output": provenance["output"],
            "provenance": provenance_uri,
        },
    )


def _fail_trace_output(uri: str, workflow_id: str, error: Exception) -> None:
    provenance_uri = join_uri(uri, ".neuroflow", "provenance.json")
    provenance = read_json(provenance_uri)
    if provenance is None or provenance.get("workflow_id") != workflow_id:
        return
    finished = datetime.now(timezone.utc).isoformat()
    error_message = f"{type(error).__name__}: {error}"
    previous_status = _finish_trace_attempt(
        provenance,
        status="failed",
        finished=finished,
        error=error_message,
    )
    provenance["status"] = "failed"
    provenance["error"] = error_message
    if previous_status != "complete":
        provenance["execution_finished"] = finished
    write_json_atomic(provenance_uri, provenance)
