"""Durable bounded fluorescence trace extraction from dense labels."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import dask.array as da
import fsspec
import numpy as np
import zarr

from neuroflow.array import NeuroArray
from neuroflow.execution.resources import parse_bytes
from neuroflow.partition.base import Partition
from neuroflow.provenance.hashing import stable_hash
from neuroflow.storage.base import join_uri, read_json, write_json_atomic
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
    label_data = labels.selection.as_dask_array()
    label_discovery_bytes = int(np.prod(labels.shape)) * np.dtype(
        labels.selection.metadata.dtype
    ).itemsize
    if memory_limit is not None and label_discovery_bytes > parse_bytes(memory_limit):
        raise ValueError(
            "label discovery requires a conservative estimated allocation of "
            f"{label_discovery_bytes} bytes, exceeding memory_limit={memory_limit!r}"
        )
    unique_result = cast(
        tuple[da.Array, da.Array], da.unique(label_data, return_counts=True)
    )
    ids, counts = unique_result
    ids, counts = da.compute(ids, counts)
    foreground = ids != 0
    ids = np.asarray(ids[foreground], dtype=np.uint64)
    counts = np.asarray(counts[foreground], dtype=np.int64)
    order = np.argsort(ids)
    ids, counts = ids[order], counts[order]
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
            "movie_bounds": movie.selection.metadata.selection_bounds,
            "labels": asdict(labels.selection.metadata.source),
            "labels_path": labels.selection.metadata.path,
            "labels_shape": labels.shape,
            "label_bounds": labels.selection.metadata.selection_bounds,
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
                existing, traces, partition, workflow_id
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
                plane_labels = np.asarray(label_data[tuple(label_key)].compute())
                block = np.asarray(
                    movie_data[tuple(movie_key)].compute(), dtype=np.float32
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
                ).to_dict(),
            )
    except Exception as exc:
        _fail_trace_output(output_uri, workflow_id, exc)
        raise
    _finalize_trace_output(output_uri, workflow_id, partitions)
    from neuroflow.source.array import ArraySource

    source = ArraySource(output_uri, component="traces", axes=("time", "cell"))
    return NeuroArray(source, source.select())


def _validate_inputs(movie: NeuroArray, labels: NeuroArray, time_chunk: int) -> None:
    if "time" not in movie.axes:
        raise ValueError("movie requires a time axis")
    spatial_axes = tuple(axis for axis in movie.axes if axis != "time")
    if labels.axes != spatial_axes:
        raise ValueError("label axes must equal the movie axes excluding time")
    expected = tuple(movie.shape[movie.axes.index(axis)] for axis in spatial_axes)
    if labels.shape != expected:
        raise ValueError("label and movie spatial shapes differ")
    if time_chunk < 1:
        raise ValueError("time_chunk must be positive")


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
    movie_bytes = (
        min(time_chunk, movie.shape[movie.axes.index("time")])
        * plane_elements
        * np.dtype(movie.selection.metadata.dtype).itemsize
    )
    label_bytes = plane_elements * np.dtype(labels.selection.metadata.dtype).itemsize
    accumulator_bytes = time_chunk * cell_count * np.dtype("float64").itemsize
    return movie_bytes + label_bytes + accumulator_bytes


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
    if existing is not None and existing.get("workflow_id") != workflow_id:
        raise ValueError("existing trace output belongs to another workflow")
    mapper = fsspec.get_mapper(uri)
    root = zarr.open_group(mapper, mode="a")
    shape = (movie.shape[movie.axes.index("time")], len(ids))
    chunks = (min(time_chunk, shape[0]), min(1024, shape[1]))
    if "cell_ids" in root:
        if not np.array_equal(np.asarray(root["cell_ids"]), ids):
            raise ValueError("existing trace output has different cell IDs")
    else:
        root.create_dataset("cell_ids", data=ids)
    if "traces" in root:
        traces = root["traces"]
        if not isinstance(traces, zarr.Array) or tuple(traces.shape) != shape:
            raise ValueError("existing trace output has an incompatible shape")
    else:
        traces = root.create_dataset(
            "traces", shape=shape, chunks=chunks, dtype="float32", fill_value=np.nan
        )
    timestamps = movie.selection.as_dask_timestamps()
    if timestamps is not None:
        time_values = np.asarray(timestamps.compute(), dtype=np.float64)
    elif movie.selection.metadata.rate is not None:
        time_values = (
            float(movie.selection.metadata.starting_time or 0.0)
            + np.arange(shape[0], dtype=np.float64)
            / movie.selection.metadata.rate
        )
    else:
        time_values = np.arange(shape[0], dtype=np.float64)
    if "timestamps" in root:
        if not np.array_equal(np.asarray(root["timestamps"]), time_values):
            raise ValueError("existing trace output has different timestamps")
    else:
        root.create_dataset("timestamps", data=time_values)
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
        "execution_started": datetime.now(timezone.utc).isoformat(),
    }
    if existing is not None:
        provenance.update(
            {key: existing[key] for key in ("execution_started",) if key in existing}
        )
    write_json_atomic(join_uri(uri, ".neuroflow", "provenance.json"), provenance)
    return traces


def _manifest_uri(uri: str, partition_id: str) -> str:
    return join_uri(uri, ".neuroflow", "manifests", f"{partition_id}.json")


def _valid_trace_manifest(
    value: dict[str, object],
    traces: zarr.Array,
    partition: Partition,
    workflow_id: str,
) -> bool:
    try:
        manifest = PartitionManifest.from_dict(value)
    except (KeyError, ValueError):
        return False
    if manifest.workflow_id != workflow_id or manifest.status != "complete":
        return False
    block = np.asarray(traces[partition.output_slices])
    actual = hashlib.sha256(block.tobytes(order="C")).hexdigest()
    return manifest.checksums.get("traces") == actual


def _finalize_trace_output(
    uri: str, workflow_id: str, partitions: tuple[Partition, ...]
) -> None:
    provenance_uri = join_uri(uri, ".neuroflow", "provenance.json")
    provenance = read_json(provenance_uri)
    if provenance is None or provenance.get("workflow_id") != workflow_id:
        raise RuntimeError("trace provenance disappeared during execution")
    provenance["status"] = "complete"
    provenance["completed_partitions"] = [item.key for item in partitions]
    provenance["failed_partitions"] = []
    provenance["execution_finished"] = datetime.now(timezone.utc).isoformat()
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
    provenance["status"] = "failed"
    provenance["error"] = f"{type(error).__name__}: {error}"
    provenance["execution_finished"] = datetime.now(timezone.utc).isoformat()
    write_json_atomic(provenance_uri, provenance)
