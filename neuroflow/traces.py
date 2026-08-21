"""Durable bounded fluorescence trace extraction from dense labels."""

from __future__ import annotations

import hashlib
import math
import resource
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import dask.array as da
import fsspec
import numpy as np
import zarr

from neuroflow.array import NeuroArray
from neuroflow.exceptions import OutputConflictError, ProvenanceMismatchError
from neuroflow.execution.resources import (
    MemoryBudget,
    parse_bytes,
    resolve_memory_budget,
)
from neuroflow.partition.base import Partition
from neuroflow.provenance.environment import capture_environment
from neuroflow.provenance.hashing import stable_hash
from neuroflow.selection.query import absolute_selection_bounds
from neuroflow.storage.base import (
    join_uri,
    read_json,
    validate_output_separation,
    write_json_atomic,
)
from neuroflow.storage.manifest import PartitionManifest

DEFAULT_TRACE_MEMORY_LIMIT = "2 GiB"


@dataclass(frozen=True)
class ROIChunk:
    """One source-aligned spatial chunk containing at least one label."""

    slices: tuple[slice, ...]
    label_ids: tuple[int, ...]

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple((item.stop or 0) - (item.start or 0) for item in self.slices)

    def to_dict(self) -> dict[str, object]:
        return {
            "slices": [[item.start, item.stop] for item in self.slices],
            "label_ids": list(self.label_ids),
        }


@dataclass(frozen=True)
class TracePlan:
    """Preflight report for bounded source-chunk-oriented trace extraction."""

    source_shape: tuple[int, ...]
    source_dtype: str
    native_chunks: tuple[int, ...] | None
    cell_count: int
    active_spatial_chunks: int
    skipped_empty_spatial_chunks: int
    time_chunk: int
    automatic_time_chunk: bool
    task_count: int
    memory_limit_bytes: int
    memory_budget: MemoryBudget
    estimated_memory_per_task: int
    estimated_source_chunks_touched: int | None
    estimated_total_bytes_read: int | None
    expected_output_shape: tuple[int, int]
    expected_output_bytes: int

    def to_dict(self) -> dict[str, object]:
        def estimate(value: object) -> dict[str, object]:
            return {
                "status": "unknown" if value is None else "estimated",
                "value": value,
            }

        return {
            "schema_version": "1",
            "operation": "mean-fluorescence-traces",
            "source": {
                "shape": list(self.source_shape),
                "dtype": self.source_dtype,
                "physical_chunks": (
                    list(self.native_chunks) if self.native_chunks else None
                ),
                "logical_bytes": estimate(
                    math.prod(self.source_shape) * np.dtype(self.source_dtype).itemsize
                ),
            },
            "roi_index": {
                "cell_count": self.cell_count,
                "active_spatial_chunks": self.active_spatial_chunks,
                "skipped_empty_spatial_chunks": self.skipped_empty_spatial_chunks,
            },
            "partitioning": {
                "time_chunk": self.time_chunk,
                "automatic_time_chunk": self.automatic_time_chunk,
                "task_count": estimate(self.task_count),
                "estimated_source_chunks_touched": estimate(
                    self.estimated_source_chunks_touched
                ),
                "estimated_total_bytes_read": estimate(self.estimated_total_bytes_read),
            },
            "resources": {
                "memory_limit_bytes": self.memory_limit_bytes,
                "memory_budget": self.memory_budget.to_dict(),
                "estimated_memory_per_task": estimate(self.estimated_memory_per_task),
                # The planner controls the task working set; it cannot control
                # allocator retention or third-party residency, so the planned
                # total is an estimate of process peak, not a guarantee.
                "estimated_process_peak_bytes": estimate(
                    self.memory_budget.reserved_bytes
                    + self.estimated_memory_per_task
                ),
                "measured_process_peak_rss_bytes": {
                    "status": "unknown",
                    "value": None,
                    "note": (
                        "populated from execution metrics after a run; planning "
                        "alone cannot measure resident set size"
                    ),
                },
            },
            "output": {
                "axes": ["time", "cell"],
                "shape": list(self.expected_output_shape),
                "estimated_size_bytes": estimate(self.expected_output_bytes),
            },
            "bounded": {
                "status": "estimated",
                "value": self.estimated_memory_per_task
                <= self.memory_budget.task_bytes,
                "reasons": [
                    "movie traversal is aligned to source spatial chunks",
                    "each task covers one finite time window",
                    "empty spatial chunks are omitted after bounded label indexing",
                ],
            },
        }

    def summary(self) -> str:
        read = (
            f"{self.estimated_total_bytes_read} uncompressed bytes"
            if self.estimated_total_bytes_read is not None
            else "unknown"
        )
        return "\n".join(
            (
                f"source: shape={self.source_shape}, dtype={self.source_dtype}",
                f"cells: {self.cell_count}",
                "spatial source chunks: "
                f"{self.active_spatial_chunks} active, "
                f"{self.skipped_empty_spatial_chunks} empty skipped",
                f"tasks: {self.task_count}, time window={self.time_chunk}",
                "memory: total process target "
                f"{self.memory_limit_bytes} bytes = "
                f"{self.memory_budget.reserved_bytes} reserved overhead + "
                f"{self.memory_budget.task_bytes} available per task; "
                f"estimated task working set {self.estimated_memory_per_task} bytes",
                "estimated process peak: "
                f"{self.memory_budget.reserved_bytes + self.estimated_memory_per_task}"
                " bytes (measure the real peak RSS to confirm)",
                f"estimated source read: {read}",
                f"output: shape={self.expected_output_shape}, "
                f"{self.expected_output_bytes} bytes",
                "bounded: yes",
            )
        )


def extract_traces(
    movie: NeuroArray,
    labels: NeuroArray,
    *,
    output: str | Path,
    time_chunk: int | None = None,
    memory_limit: int | str = DEFAULT_TRACE_MEMORY_LIMIT,
) -> NeuroArray:
    """Average movie voxels per label with source-aligned resumable reads."""
    started_clock = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
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
    io_before = _source_bytes_read(movie)
    plan, ids, counts, roi_chunks = _build_trace_plan(
        movie,
        labels,
        time_chunk=time_chunk,
        memory_limit=memory_limit,
    )
    time_chunk = plan.time_chunk

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
            "memory_limit": memory_limit,
            "roi_chunks": [item.to_dict() for item in roi_chunks],
            "schema_version": "2",
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
        plan,
        memory_limit,
    )
    movie_data = movie.selection.as_dask_array()
    label_data = labels.selection.as_dask_array()
    id_to_column = {int(value): index for index, value in enumerate(ids)}
    computed_partitions = 0
    resumed_partitions = 0

    try:
        for partition in partitions:
            manifest_path = _manifest_uri(output_uri, partition.key)
            existing = read_json(manifest_path)
            if existing is not None and _valid_trace_manifest(
                existing, traces, partition, workflow_id, output_uri
            ):
                resumed_partitions += 1
                continue
            start = partition.output_slices[0].start or 0
            stop = partition.output_slices[0].stop or traces.shape[0]
            sums = np.zeros((stop - start, len(ids)), dtype=np.float64)
            for roi_chunk in roi_chunks:
                movie_key = [slice(None)] * len(movie.axes)
                movie_key[movie.axes.index("time")] = slice(start, stop)
                plane_labels = np.asarray(
                    label_data[roi_chunk.slices].compute(
                        scheduler="threads", num_workers=1
                    )
                )
                for axis, spatial_slice in zip(
                    labels.axes, roi_chunk.slices, strict=True
                ):
                    movie_key[movie.axes.index(axis)] = spatial_slice
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
                for label_id in roi_chunk.label_ids:
                    column = id_to_column[label_id]
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
            computed_partitions += 1
    except Exception as exc:
        _fail_trace_output(output_uri, workflow_id, exc)
        raise
    _finalize_trace_output(
        output_uri,
        workflow_id,
        partitions,
        started_at=started_at,
        wall_time_seconds=time.perf_counter() - started_clock,
        computed_partitions=computed_partitions,
        resumed_partitions=resumed_partitions,
        bytes_read=_bytes_delta(io_before, _source_bytes_read(movie)),
        memory_budget=plan.memory_budget,
        estimated_memory_per_task=plan.estimated_memory_per_task,
    )
    from neuroflow.api import open_array

    source, selection = open_array(output_uri, verify=False)
    return NeuroArray(source, selection)


def plan_trace_extraction(
    movie: NeuroArray,
    labels: NeuroArray,
    *,
    time_chunk: int | None = None,
    memory_limit: int | str = DEFAULT_TRACE_MEMORY_LIMIT,
) -> TracePlan:
    """Inspect labels and return a plan without reading movie values."""
    _validate_inputs(movie, labels, time_chunk)
    plan, _, _, _ = _build_trace_plan(
        movie,
        labels,
        time_chunk=time_chunk,
        memory_limit=memory_limit,
    )
    return plan


def _validate_inputs(
    movie: NeuroArray, labels: NeuroArray, time_chunk: int | None
) -> None:
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
    if time_chunk is not None and time_chunk < 1:
        raise ValueError("time_chunk must be positive")


def _build_trace_plan(
    movie: NeuroArray,
    labels: NeuroArray,
    *,
    time_chunk: int | None,
    memory_limit: int | str,
) -> tuple[TracePlan, np.ndarray, np.ndarray, tuple[ROIChunk, ...]]:
    # ``memory_limit`` is a total process-memory target. Only what remains
    # after the unavoidable process overhead may be spent on partition data,
    # so window sizing is driven by ``budget.task_bytes`` rather than by the
    # headline number.
    budget = resolve_memory_budget(memory_limit)
    task_budget = budget.task_bytes
    total_budget = budget.total_bytes
    movie_data = movie.selection.as_dask_array(chunks="native")
    label_data = labels.selection.as_dask_array()
    spatial_chunks = tuple(
        tuple(int(value) for value in movie_data.chunks[movie.axes.index(axis)])
        for axis in labels.axes
    )
    # Label indexing allocates real per-task working memory (one label block
    # plus the distinct-label mapping), so it is bounded by what remains after
    # process overhead, not by the headline target.
    ids, counts, roi_chunks, total_spatial_chunks = _discover_label_counts(
        label_data,
        spatial_chunks=spatial_chunks,
        budget=task_budget,
    )
    if not len(ids):
        raise ValueError("labels contain no cells")
    maximum_spatial_shape = tuple(
        max(item.shape[axis] for item in roi_chunks) for axis in range(len(labels.axes))
    )
    time_size = movie.shape[movie.axes.index("time")]
    native = movie.selection.metadata.native_chunks
    native_time = native[movie.axes.index("time")] if native is not None else 1
    automatic_time_chunk = time_chunk is None
    if time_chunk is None:
        time_chunk = _automatic_time_chunk(
            movie,
            labels,
            len(ids),
            maximum_spatial_shape,
            task_budget,
            native_time,
        )
    estimated_memory = _estimated_trace_memory(
        movie,
        labels,
        len(ids),
        time_chunk,
        maximum_spatial_shape,
        native_time,
    )
    if estimated_memory > task_budget:
        raise ValueError(
            f"trace window requires an estimated {estimated_memory} bytes of "
            f"task working memory, exceeding the {task_budget} bytes that "
            f"remain of memory_limit={memory_limit!r} after an estimated "
            f"{budget.reserved_bytes} bytes of process overhead"
        )
    task_count = math.ceil(time_size / time_chunk)
    if native is None:
        chunks_touched = None
        total_read = None
    else:
        time_chunk_touches = sum(
            math.ceil(min(time_size, start + time_chunk) / native_time)
            - start // native_time
            for start in range(0, time_size, time_chunk)
        )
        chunks_touched = len(roi_chunks) * time_chunk_touches
        total_read = (
            chunks_touched
            * math.prod(native)
            * np.dtype(movie.selection.metadata.dtype).itemsize
        )
    output_shape = (time_size, len(ids))
    output_bytes = (
        math.prod(output_shape) * np.dtype("float32").itemsize
        + len(ids) * np.dtype("uint64").itemsize
        + time_size * np.dtype("float64").itemsize
    )
    plan = TracePlan(
        source_shape=movie.shape,
        source_dtype=movie.selection.metadata.dtype,
        native_chunks=native,
        cell_count=len(ids),
        active_spatial_chunks=len(roi_chunks),
        skipped_empty_spatial_chunks=total_spatial_chunks - len(roi_chunks),
        time_chunk=time_chunk,
        automatic_time_chunk=automatic_time_chunk,
        task_count=task_count,
        memory_limit_bytes=total_budget,
        memory_budget=budget,
        estimated_memory_per_task=estimated_memory,
        estimated_source_chunks_touched=chunks_touched,
        estimated_total_bytes_read=total_read,
        expected_output_shape=output_shape,
        expected_output_bytes=output_bytes,
    )
    return plan, ids, counts, roi_chunks


def _discover_label_counts(
    label_data: da.Array,
    *,
    spatial_chunks: tuple[tuple[int, ...], ...],
    budget: int,
) -> tuple[np.ndarray, np.ndarray, tuple[ROIChunk, ...], int]:
    """Index labels one source-aligned spatial chunk at a time.

    A Python mapping is retained because the number of cells is normally tiny
    relative to the voxel count. Its conservative per-entry budget prevents a
    pathological one-label-per-voxel input from consuming unbounded memory.
    """
    counts_by_id: dict[int, int] = {}
    roi_chunks: list[ROIChunk] = []
    membership_count = 0
    itemsize = int(label_data.dtype.itemsize)
    entry_bytes = 160
    membership_bytes = 24
    block_grid = tuple(len(axis_chunks) for axis_chunks in spatial_chunks)
    chunk_slices: list[tuple[slice, ...]] = []
    for axis_chunks in spatial_chunks:
        start = 0
        slices: list[slice] = []
        for chunk_size in axis_chunks:
            stop = start + int(chunk_size)
            slices.append(slice(start, stop))
            start = stop
        chunk_slices.append(tuple(slices))
    for block_index in np.ndindex(*block_grid):
        block_key = tuple(
            chunk_slices[axis][index] for axis, index in enumerate(block_index)
        )
        lazy_block = label_data[block_key]
        block_elements = math.prod(int(size) for size in lazy_block.shape)
        # Input, sort workspace, unique values, int64 counts, and aggregation
        # workspace can coexist during np.unique.
        block_workspace = block_elements * (4 * itemsize + 8)
        if block_workspace + len(counts_by_id) * entry_bytes > budget:
            raise ValueError(
                "label discovery requires an estimated chunk workspace exceeding "
                f"the {budget} bytes of task memory available under the current "
                "memory_limit; rechunk labels or raise the limit"
            )
        block = np.asarray(lazy_block.compute(scheduler="threads", num_workers=1))
        if block.dtype.kind == "i" and np.any(block < 0):
            raise ValueError("labels must contain non-negative integers")
        local_ids, local_counts = np.unique(block, return_counts=True)
        foreground = local_ids != 0
        local_ids = local_ids[foreground]
        local_counts = local_counts[foreground]
        new_count = sum(int(value) not in counts_by_id for value in local_ids)
        local_label_ids = tuple(int(value) for value in local_ids)
        new_memberships = len(local_label_ids)
        if (
            block_workspace
            + (len(counts_by_id) + new_count) * entry_bytes
            + (membership_count + new_memberships) * membership_bytes
            > budget
        ):
            raise ValueError(
                "label discovery found an estimated distinct-label workspace "
                f"exceeding the {budget} bytes of task memory available under "
                "the current memory_limit"
            )
        for label_id, count in zip(local_ids, local_counts, strict=True):
            label_key = int(label_id)
            counts_by_id[label_key] = counts_by_id.get(label_key, 0) + int(count)
        if local_label_ids:
            roi_chunks.append(ROIChunk(block_key, local_label_ids))
            membership_count += new_memberships

    ordered_ids = sorted(counts_by_id)
    ids = np.fromiter(ordered_ids, dtype=np.uint64, count=len(ordered_ids))
    counts = np.fromiter(
        (counts_by_id[label_id] for label_id in ordered_ids),
        dtype=np.int64,
        count=len(ordered_ids),
    )
    return ids, counts, tuple(roi_chunks), math.prod(block_grid)


def _automatic_time_chunk(
    movie: NeuroArray,
    labels: NeuroArray,
    cell_count: int,
    spatial_shape: tuple[int, ...],
    budget: int,
    native_time: int,
) -> int:
    time_size = movie.shape[movie.axes.index("time")]
    one = _estimated_trace_memory(
        movie,
        labels,
        cell_count,
        1,
        spatial_shape,
        native_time,
    )
    if one > budget:
        raise ValueError(
            "one source-aligned trace chunk requires an estimated "
            f"{one} bytes of task working memory, exceeding the {budget} bytes "
            "left for tasks after process overhead; raise memory_limit"
        )
    two = _estimated_trace_memory(
        movie,
        labels,
        cell_count,
        2,
        spatial_shape,
        native_time,
    )
    incremental = max(1, two - one)
    maximum = min(time_size, 1 + (budget - one) // incremental)
    if native_time > 1 and maximum >= native_time:
        maximum = max(native_time, (maximum // native_time) * native_time)
    return max(1, maximum)


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
    movie: NeuroArray,
    labels: NeuroArray,
    cell_count: int,
    time_chunk: int,
    spatial_shape: tuple[int, ...],
    native_time: int,
) -> int:
    spatial_elements = math.prod(spatial_shape)
    requested_frames = min(time_chunk, movie.shape[movie.axes.index("time")])
    time_size = movie.shape[movie.axes.index("time")]
    loaded_frames = max(requested_frames, min(native_time, time_size))
    movie_elements = loaded_frames * spatial_elements
    source_movie_bytes = (
        movie_elements * np.dtype(movie.selection.metadata.dtype).itemsize
    )
    float_movie_bytes = movie_elements * np.dtype("float32").itemsize
    label_itemsize = np.dtype(labels.selection.metadata.dtype).itemsize
    label_bytes = spatial_elements * label_itemsize
    label_workspace = spatial_elements * (2 * label_itemsize + 9)
    accumulator_bytes = requested_frames * cell_count * np.dtype("float64").itemsize
    trace_output_bytes = requested_frames * cell_count * np.dtype("float32").itemsize
    cell_index_bytes = cell_count * 160
    # No scheduler or cache reserve is added here. Those costs are process
    # overhead, not per-task working set, and are accounted once in
    # ``MemoryBudget.process_overhead_bytes``; including them again would
    # double-count them and shrink the window for no reason.
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
    plan: TracePlan,
    memory_limit: int | str,
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
    upstream_provenance = (
        labels.workflow.provenance if labels.workflow is not None else None
    )
    segmentation_identity = None
    if isinstance(upstream_provenance, dict):
        segmentation_identity = {
            "workflow_id": upstream_provenance.get("workflow_id"),
            "adapter": upstream_provenance.get("adapter"),
            "parameters": upstream_provenance.get("parameters"),
            "external_libraries": upstream_provenance.get("external_libraries"),
            "result_checksum": upstream_provenance.get("result_checksum"),
        }
    current_attempt: dict[str, object] = {
        "execution_started": execution_started,
        "status": "running",
        "memory_limit": memory_limit,
    }
    provenance = {
        "schema_version": "1",
        "workflow_id": workflow_id,
        "source": asdict(movie.selection.metadata.source),
        "source_backend": (movie.selection.metadata.attributes or {}).get("transport"),
        "nwb_paths": [movie.selection.metadata.path],
        "adapter": {"name": "mean-fluorescence-traces", "version": "1"},
        "parameters": {
            "time_chunk": time_chunk,
            "automatic_time_chunk": plan.automatic_time_chunk,
            "estimated_memory_per_window": plan.estimated_memory_per_task,
            "memory_limit": memory_limit,
            "memory_budget": plan.memory_budget.to_dict(),
        },
        "preflight_plan": plan.to_dict(),
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
            "segmentation_workflow": segmentation_identity,
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
        "environment": capture_environment(),
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
    uri: str,
    workflow_id: str,
    partitions: tuple[Partition, ...],
    *,
    started_at: str,
    wall_time_seconds: float,
    computed_partitions: int,
    resumed_partitions: int,
    bytes_read: int | None,
    memory_budget: MemoryBudget,
    estimated_memory_per_task: int,
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
    manifest_checksums: list[dict[str, object]] = []
    for partition in partitions:
        manifest = read_json(_manifest_uri(uri, partition.key))
        if manifest is None:
            raise RuntimeError(f"trace manifest disappeared: {partition.key}")
        manifest_checksums.append(
            {
                "partition_id": partition.key,
                "checksums": manifest.get("checksums"),
            }
        )
    provenance["result_checksum"] = stable_hash(manifest_checksums)
    provenance["integrity_verified"] = True
    execution_metrics = {
        "started_at": started_at,
        "finished_at": finished,
        "wall_time_seconds": wall_time_seconds,
        "completed_task_count": len(partitions),
        "computed_task_count": computed_partitions,
        "resumed_task_count": resumed_partitions,
        "partitions_completed": len(partitions),
        "bytes_read": bytes_read,
        "output_bytes": provenance["preflight_plan"]["output"][  # type: ignore[index]
            "estimated_size_bytes"
        ]["value"],  # type: ignore[index]
        "peak_rss_bytes": _peak_rss_bytes(),
        # Planned versus measured, side by side, so the gap is always visible
        # rather than something a reader has to reconstruct.
        "memory": {
            "budget": memory_budget.to_dict(),
            "planned_task_working_bytes": estimated_memory_per_task,
            "planned_process_peak_bytes": (
                memory_budget.reserved_bytes + estimated_memory_per_task
            ),
            "measured_process_peak_rss_bytes": _peak_rss_bytes(),
            "measurement_scope": (
                "whole process high-water mark for this interpreter, including "
                "any work done before or after trace extraction"
            ),
        },
    }
    provenance["execution_metrics"] = execution_metrics
    attempts = provenance.get("execution_attempts")
    if isinstance(attempts, list) and attempts and isinstance(attempts[-1], dict):
        attempts[-1]["execution_metrics"] = execution_metrics
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


def _source_bytes_read(movie: NeuroArray) -> int | None:
    stats = getattr(movie.source, "io_stats", None)
    if not callable(stats):
        return None
    value = stats()
    if not isinstance(value, dict):
        return None
    byte_count = value.get("response_content_bytes")
    return byte_count if isinstance(byte_count, int) else None


def _bytes_delta(before: int | None, after: int | None) -> int | None:
    if before is None or after is None:
        return None
    return max(0, after - before)


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024
