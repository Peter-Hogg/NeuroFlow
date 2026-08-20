"""Explicit execution and durable result initialization."""

from __future__ import annotations

import importlib.metadata
import resource
import sys
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import fsspec
import numpy as np
import zarr
from dask.base import annotate, compute
from dask.delayed import Delayed, delayed

from neuroflow import __version__
from neuroflow.adapters.base import AnalysisAdapter, LoadedPartition, TaskContext
from neuroflow.adapters.numpy import ArrayOutput, TableOutput
from neuroflow.adapters.segmentation import SegmentationOutputSchema
from neuroflow.diagnostics.estimates import slice_shape
from neuroflow.diagnostics.plan import ExecutionPlan
from neuroflow.exceptions import (
    IncompletePartitionError,
    OutputConflictError,
    ProvenanceMismatchError,
)
from neuroflow.partition.base import Partition
from neuroflow.provenance.environment import capture_environment
from neuroflow.provenance.hashing import stable_hash
from neuroflow.selection.query import Selection, absolute_selection_bounds
from neuroflow.source.base import NWBSource
from neuroflow.storage.base import (
    join_uri,
    read_json,
    validate_component_name,
    validate_output_separation,
    write_json_atomic,
)
from neuroflow.storage.manifest import PartitionManifest
from neuroflow.storage.parquet import ParquetOutput
from neuroflow.storage.segmentation import SegmentationOutput
from neuroflow.storage.validation import (
    OutputStorageKind,
    output_component_kinds,
    validate_partition_manifest,
)
from neuroflow.storage.writer import (
    ArrayPartitionWriter,
    SegmentationPartitionWriter,
    TablePartitionWriter,
)
from neuroflow.storage.zarr import ZarrOutput

ExecutionOutput = ZarrOutput | ParquetOutput | SegmentationOutput


def _schema_output_kinds(schema: object) -> dict[str, OutputStorageKind]:
    if isinstance(schema, ArrayOutput):
        return {schema.name: "array"}
    if isinstance(schema, TableOutput):
        return {schema.name: "table"}
    if isinstance(schema, SegmentationOutputSchema):
        return {schema.labels_name: "array", schema.objects_name: "table"}
    raise TypeError("adapter has no supported output schema")


def _validate_recursive_delete_target(
    fs: object, path: str, target_uri: str | None = None
) -> None:
    """Reject recursive deletion of roots and dangerously broad local paths."""
    stripped = path.rstrip("/")
    if not stripped or stripped == "/":
        raise OutputConflictError("refusing to recursively overwrite a storage root")
    if target_uri is not None:
        parsed = urlsplit(target_uri)
        if parsed.scheme and not parsed.path.rstrip("/"):
            raise OutputConflictError(
                "refusing to recursively overwrite an object-store root"
            )
    protocol = getattr(fs, "protocol", "file")
    protocols = (protocol,) if isinstance(protocol, str) else tuple(protocol)
    if "file" in protocols or "local" in protocols:
        resolved = Path(stripped).expanduser().resolve()
        protected = {Path("/").resolve(), Path.home().resolve(), Path.cwd().resolve()}
        if resolved in protected:
            raise OutputConflictError(
                f"refusing to recursively overwrite protected path {resolved}"
            )


def partition_identity(workflow_id: str, partition: Partition) -> str:
    return stable_hash(
        {
            "workflow_id": workflow_id,
            "partition": partition.to_dict(),
        }
    )


def manifest_uri(uri: str, partition_id: str) -> str:
    validate_component_name(partition_id)
    return join_uri(uri, ".neuroflow", "manifests", f"{partition_id}.json")


def _execute_partition(
    source_array: object,
    source_timestamps: object | None,
    sampling_rate: float | None,
    starting_time: float | None,
    adapter: AnalysisAdapter,
    output: ExecutionOutput,
    workflow_id: str,
    partition: Partition,
    selection_axes: tuple[str, ...],
) -> dict[str, object]:
    partition_id = partition_identity(workflow_id, partition)
    schema = getattr(adapter, "output", None)
    output_kinds = _schema_output_kinds(schema)
    existing = read_json(manifest_uri(output.uri, partition_id))
    if existing is not None:
        manifest = PartitionManifest.from_dict(existing)
        if manifest.workflow_id != workflow_id or manifest.partition_id != partition_id:
            raise ProvenanceMismatchError(
                "partition manifest belongs to another workflow"
            )
        if manifest.status == "complete" and not validate_partition_manifest(
            manifest,
            partition,
            output_root=output.uri,
            output_kinds=output_kinds,
            checksums=True,
        ):
            resumed = manifest.to_dict()
            resumed["_execution"] = "skipped"
            return resumed
    seed = getattr(adapter, "random_seed", None)
    context = TaskContext(
        partition_id=partition_id,
        parameters=dict(getattr(adapter, "parameters", None) or {}),
        random_seed=seed,
    )
    if isinstance(schema, ArrayOutput):
        reduced_axis_indices = tuple(
            selection_axes.index(axis) for axis in schema.reduced_axes
        )
        singleton_axis_indices = tuple(
            selection_axes.index(axis) for axis in schema.kept_reduced_axes
        )
        writer: object = ArrayPartitionWriter(
            uri=output.uri,
            array_name=schema.name,
            partition=partition,
            workflow_id=workflow_id,
            partition_id=partition_id,
            reduced_axis_indices=reduced_axis_indices,
            singleton_axis_indices=singleton_axis_indices,
        )
    elif isinstance(schema, TableOutput):
        if not isinstance(output, ParquetOutput):
            raise TypeError("table adapters require ParquetOutput storage")
        writer = TablePartitionWriter(
            uri=output.uri,
            table_name=getattr(schema, "name", "result"),
            workflow_id=workflow_id,
            partition_id=partition_id,
            partition_on=output.partition_on,
        )
    elif isinstance(schema, SegmentationOutputSchema):
        writer = SegmentationPartitionWriter(
            uri=output.uri,
            labels_name=schema.labels_name,
            objects_name=schema.objects_name,
            partition=partition,
            workflow_id=workflow_id,
            partition_id=partition_id,
        )
    else:
        raise TypeError("adapter has no supported output schema")
    try:
        source_free = bool(getattr(adapter, "source_free_after_stages", False))
        data = (
            np.empty((0,) * len(selection_axes), dtype=getattr(source_array, "dtype"))
            if source_free
            else np.asarray(source_array[partition.read_slices])  # type: ignore[index]
        )
        time_axis = selection_axes.index("time") if "time" in selection_axes else None
        time_slice = partition.read_slices[time_axis] if time_axis is not None else None
        if source_free:
            timestamps = None
        elif source_timestamps is not None and time_slice is not None:
            timestamps = np.asarray(source_timestamps[time_slice])  # type: ignore[index]
        elif sampling_rate is not None and time_slice is not None:
            start = time_slice.start or 0
            stop = time_slice.stop or data.shape[time_axis]  # type: ignore[index]
            timestamps = (starting_time or 0.0) + np.arange(start, stop) / sampling_rate
        else:
            timestamps = None
        loaded = LoadedPartition(
            data=data,
            read_slices=partition.read_slices,
            output_slices=partition.output_slices,
            trim_slices=partition.trim_slices,
            timestamps=timestamps,
        )
        prepared = adapter.prepare(loaded, context)
        task_output = adapter.run(prepared, context)
        manifest = adapter.persist(task_output, writer, context)
        if not isinstance(manifest, PartitionManifest):
            raise TypeError("adapter.persist() must return PartitionManifest")
        if (
            manifest.partition_id != partition_id
            or manifest.workflow_id != workflow_id
            or manifest.status != "complete"
        ):
            raise ValueError(
                "adapter.persist() returned a manifest with invalid identity or status"
            )
        committed_value = read_json(manifest_uri(output.uri, partition_id))
        if committed_value is None:
            raise ValueError("adapter.persist() did not commit its returned manifest")
        try:
            committed_manifest = PartitionManifest.from_dict(committed_value)
        except (KeyError, ValueError) as exc:
            raise ValueError(
                f"adapter.persist() committed an invalid manifest: {exc}"
            ) from exc
        if committed_manifest != manifest:
            raise ValueError(
                "adapter.persist() returned a manifest that differs from its commit"
            )
        manifest_errors = validate_partition_manifest(
            committed_manifest,
            partition,
            output_root=output.uri,
            output_kinds=output_kinds,
            checksums=False,
        )
        if manifest_errors:
            raise ValueError(
                "adapter.persist() returned an invalid manifest: "
                + "; ".join(manifest_errors)
            )
        completed = manifest.to_dict()
        completed["_execution"] = "computed"
        return completed
    except Exception as exc:
        failure = PartitionManifest(
            partition_id=partition_id,
            workflow_id=workflow_id,
            status="failed",
            outputs={},
            checksums={},
        )
        failure_value = failure.to_dict()
        failure_value["error"] = f"{type(exc).__name__}: {exc}"
        write_json_atomic(manifest_uri(output.uri, partition_id), failure_value)
        raise


def build_tasks(
    *,
    selection: Selection,
    adapter: AnalysisAdapter,
    output: ExecutionOutput,
    execution_plan: ExecutionPlan,
) -> tuple[Delayed, ...]:
    tasks: list[Delayed] = []
    requirements = adapter.requirements()
    resources: dict[str, float] = {}
    if requirements.resources.gpu:
        resources["GPU"] = float(requirements.resources.gpu)
    for partition in execution_plan.partitions:
        annotation = annotate(resources=resources) if resources else nullcontext()
        with annotation:
            tasks.append(
                delayed(_execute_partition, pure=False)(
                    selection._array,
                    selection._timestamps,
                    selection.metadata.rate,
                    selection.metadata.starting_time,
                    adapter,
                    output,
                    execution_plan.workflow_id,
                    partition,
                    selection.metadata.axes,
                )
            )
    return tuple(tasks)


def initialize_output(
    *,
    source: NWBSource,
    selection: Selection,
    adapter: AnalysisAdapter,
    output: ExecutionOutput,
    execution_plan: ExecutionPlan,
    scheduler: str,
    resume: bool,
    max_workers: int | None,
    memory_limit: int | str | None,
    stages: tuple[dict[str, object], ...] = (),
) -> None:
    physical_source_uri = str(getattr(source, "uri", source.identity.uri))
    validate_output_separation(output.uri, {"source": physical_source_uri})
    provenance_uri = join_uri(output.uri, ".neuroflow", "provenance.json")
    existing = read_json(provenance_uri)
    fs, root_path = fsspec.core.url_to_fs(output.uri)
    root_exists = fs.exists(root_path)
    mode = output.mode
    if root_exists and mode == "overwrite" and not resume:
        _validate_recursive_delete_target(fs, root_path, output.uri)
        fs.rm(root_path, recursive=True)
        existing = None
        root_exists = False
    if existing is not None:
        if existing.get("workflow_id") != execution_plan.workflow_id:
            if mode == "overwrite":
                _validate_recursive_delete_target(fs, root_path, output.uri)
                fs.rm(root_path, recursive=True)
                existing = None
                root_exists = False
            else:
                raise ProvenanceMismatchError(
                    "existing output provenance does not match this workflow"
                )
        if existing is not None and not resume:
            raise OutputConflictError("output exists and resume=False")
    elif root_exists:
        if mode == "overwrite":
            _validate_recursive_delete_target(fs, root_path, output.uri)
            fs.rm(root_path, recursive=True)
        else:
            raise OutputConflictError(
                "output path exists without matching NeuroFlow provenance"
            )
    schema = getattr(adapter, "output", None)
    name = getattr(schema, "name", "result")
    output_metadata: dict[str, object]
    if isinstance(output, ZarrOutput) and isinstance(schema, ArrayOutput):
        mapper = fsspec.get_mapper(output.uri)
        group = zarr.open_group(mapper, mode="a")
        dtype = np.dtype(schema.dtype)
        requested_chunks = schema.chunks or slice_shape(
            execution_plan.partitions[0].output_slices
        )
        planned_chunks = tuple(
            min(full, chunk)
            for full, chunk in zip(
                execution_plan.output_shape, requested_chunks, strict=True
            )
        )
        if name in group:
            array = group[name]
            if not isinstance(array, zarr.Array):
                raise OutputConflictError("result component exists but is not an array")
            if (
                tuple(array.shape) != execution_plan.output_shape
                or np.dtype(array.dtype) != dtype
            ):
                raise OutputConflictError(
                    "existing result array has an incompatible schema"
                )
            if tuple(array.chunks) != planned_chunks:
                raise OutputConflictError(
                    "existing result array chunks do not match the planned chunks"
                )
        else:
            create_options: dict[str, object] = {}
            if output.compressor == "none":
                create_options["compressor"] = None
            group.create_dataset(
                name,
                shape=execution_plan.output_shape,
                chunks=planned_chunks,
                dtype=dtype,
                overwrite=False,
                **create_options,
            )
        output_metadata = {
            "kind": "array",
            "uri": output.uri,
            "name": name,
            "dtype": str(dtype),
            "shape": execution_plan.output_shape,
            "axes": execution_plan.output_axes,
            "chunks": planned_chunks,
        }
    elif isinstance(output, ParquetOutput) and isinstance(schema, TableOutput):
        output_metadata = {"kind": "table", "uri": output.uri, "name": name}
    elif isinstance(output, SegmentationOutput) and isinstance(
        schema, SegmentationOutputSchema
    ):
        mapper = fsspec.get_mapper(output.uri)
        group = zarr.open_group(mapper, mode="a")
        dtype = np.dtype(schema.label_dtype)
        planned_chunks = tuple(
            min(full, chunk)
            for full, chunk in zip(
                selection.metadata.shape,
                execution_plan.processing_partition_shape,
                strict=True,
            )
        )
        if schema.labels_name in group:
            label_array = group[schema.labels_name]
            if not isinstance(label_array, zarr.Array):
                raise OutputConflictError(
                    "segmentation label component exists but is not an array"
                )
            if (
                tuple(label_array.shape) != selection.metadata.shape
                or np.dtype(label_array.dtype) != dtype
            ):
                raise OutputConflictError(
                    "existing segmentation labels have an incompatible schema"
                )
            if tuple(label_array.chunks) != planned_chunks:
                raise OutputConflictError(
                    "existing segmentation label chunks do not match the planned chunks"
                )
        else:
            create_options = {}
            if output.compressor == "none":
                create_options["compressor"] = None
            group.create_dataset(
                schema.labels_name,
                shape=selection.metadata.shape,
                chunks=planned_chunks,
                dtype=dtype,
                fill_value=0,
                overwrite=False,
                **create_options,
            )
        output_metadata = {
            "kind": "segmentation",
            "uri": output.uri,
            "shape": selection.metadata.shape,
            "axes": selection.metadata.axes,
            "bounds": absolute_selection_bounds(selection.metadata),
            "arrays": {
                schema.labels_name: {
                    "name": schema.labels_name,
                    "dtype": str(dtype),
                    "chunks": planned_chunks,
                }
            },
            "tables": {schema.objects_name: {"name": schema.objects_name}},
            "merge_status": "unmerged",
        }
    else:
        raise TypeError("adapter schema and output storage are incompatible")
    identity_parameters = getattr(adapter, "identity_parameters", None)
    adapter_parameters = (
        identity_parameters()
        if callable(identity_parameters)
        else (getattr(adapter, "parameters", None) or {})
    )
    if not isinstance(adapter_parameters, Mapping):
        raise TypeError("adapter identity parameters must be a mapping")
    execution_started = datetime.now(timezone.utc).isoformat()
    execution_policy = {
        "max_workers": max_workers,
        "memory_limit": memory_limit,
    }
    environment = capture_environment()
    current_attempt: dict[str, object] = {
        "execution_started": execution_started,
        "scheduler": scheduler,
        "execution_policy": execution_policy,
        "environment": environment,
        "status": "running",
    }
    provenance = {
        "schema_version": "1",
        "workflow_id": execution_plan.workflow_id,
        "neuroflow_version": __version__,
        "source": asdict(selection.metadata.source),
        "nwb_paths": [selection.metadata.path],
        "adapter": {"name": adapter.name, "version": adapter.version},
        "parameters": dict(adapter_parameters),
        "random_seeds": (
            [getattr(adapter, "random_seed")]
            if getattr(adapter, "random_seed", None) is not None
            else []
        ),
        "external_libraries": _external_library_versions(adapter),
        "selection": {
            "neurodata_type": selection.metadata.neurodata_type,
            "shape": selection.metadata.shape,
            "dtype": selection.metadata.dtype,
            "native_chunks": selection.metadata.native_chunks,
            "axes": selection.metadata.axes,
            "bounds": absolute_selection_bounds(selection.metadata),
        },
        "partition_plan": {
            "processing_shape": execution_plan.processing_partition_shape,
            "overlap": execution_plan.overlap,
            "task_count": execution_plan.task_count,
            "partition_ids": [
                partition_identity(execution_plan.workflow_id, partition)
                for partition in execution_plan.partitions
            ],
            "partitions": [
                {
                    "partition_id": partition_identity(
                        execution_plan.workflow_id, partition
                    ),
                    **partition.to_dict(),
                }
                for partition in execution_plan.partitions
            ],
        },
        "stages": list(stages),
        "scheduler": scheduler,
        "execution_policy": execution_policy,
        "environment": environment,
        "output": output_metadata,
        "execution_started": execution_started,
        "execution_attempts": [current_attempt],
        "completed_partitions": [],
        "failed_partitions": [],
        "status": "running",
    }
    if existing is not None:
        previous_status = str(existing.get("status", "unknown"))
        current_attempt["resumed_from_status"] = previous_status
        provenance["execution_attempts"] = [
            *_existing_execution_attempts(existing),
            current_attempt,
        ]
        for key in (
            "execution_started",
            "execution_finished",
            "scheduler",
            "execution_policy",
            "environment",
        ):
            if key in existing:
                provenance[key] = existing[key]
    write_json_atomic(provenance_uri, provenance)


def _existing_execution_attempts(
    provenance: Mapping[str, object],
) -> list[dict[str, object]]:
    """Return attempt history, upgrading provenance written before it existed."""
    existing = provenance.get("execution_attempts")
    if isinstance(existing, list) and all(
        isinstance(item, Mapping) for item in existing
    ):
        return [dict(item) for item in existing]
    attempt: dict[str, object] = {
        "status": str(provenance.get("status", "unknown")),
    }
    for key in (
        "execution_started",
        "execution_finished",
        "scheduler",
        "execution_policy",
        "environment",
        "error",
    ):
        if key in provenance:
            attempt[key] = provenance[key]
    return [attempt]


def _finish_current_attempt(
    provenance: dict[str, object],
    *,
    status: Literal["complete", "failed"],
    finished: str,
    error: str | None = None,
) -> str | None:
    """Finish the latest attempt and return the status it resumed from."""
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


def _external_library_versions(adapter: AnalysisAdapter) -> dict[str, str]:
    function = getattr(adapter, "function", None)
    module_name = getattr(function, "__module__", "")
    package = str(module_name).split(".", 1)[0]
    packages: set[str] = set(getattr(adapter, "external_packages", ()))
    if package and package not in {"__main__", "builtins"}:
        packages.add(package)
    versions: dict[str, str] = {}
    for item in packages:
        try:
            versions[item] = importlib.metadata.version(item)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def execute_tasks(
    tasks: tuple[Delayed, ...],
    scheduler: Literal["threads", "processes", "distributed"],
    *,
    max_workers: int | None = None,
) -> tuple[dict[str, object], ...]:
    if max_workers is not None and max_workers < 1:
        raise ValueError("max_workers must be positive")
    options = {"num_workers": max_workers} if max_workers is not None else {}
    values = compute(*tasks, scheduler=scheduler, **options)  # pyright: ignore[reportArgumentType]
    return tuple(values)


def finalize_output(
    output: ExecutionOutput,
    workflow_id: str,
    partitions: tuple[Partition, ...],
    *,
    task_results: tuple[dict[str, object], ...] = (),
    source_metrics: Mapping[str, object] | None = None,
) -> None:
    provenance_uri = join_uri(output.uri, ".neuroflow", "provenance.json")
    provenance = read_json(provenance_uri)
    if provenance is None or provenance.get("workflow_id") != workflow_id:
        raise ProvenanceMismatchError("cannot finalize output without valid provenance")
    partition_plan = provenance.get("partition_plan")
    if not isinstance(partition_plan, Mapping):
        raise IncompletePartitionError(
            "cannot finalize output without a partition plan"
        )
    expected_ids = [partition_identity(workflow_id, item) for item in partitions]
    raw_ids = partition_plan.get("partition_ids")
    raw_partitions = partition_plan.get("partitions")
    if (
        not partitions
        or partition_plan.get("task_count") != len(partitions)
        or raw_ids != expected_ids
        or not isinstance(raw_partitions, list)
        or len(raw_partitions) != len(partitions)
    ):
        raise IncompletePartitionError(
            "cannot finalize output with a mismatched partition plan"
        )
    for raw_partition, partition_id, partition in zip(
        raw_partitions,
        expected_ids,
        partitions,
        strict=True,
    ):
        try:
            if not isinstance(raw_partition, Mapping):
                raise ValueError("partition descriptor is not a mapping")
            declared_partition_id = raw_partition.get("partition_id")
            declared_partition = Partition.from_dict(raw_partition)
        except (KeyError, ValueError) as exc:
            raise IncompletePartitionError(
                f"cannot finalize output with an invalid partition plan: {exc}"
            ) from exc
        if declared_partition_id != partition_id or declared_partition != partition:
            raise IncompletePartitionError(
                "cannot finalize output with a mismatched partition plan"
            )
    raw_output = provenance.get("output")
    if not isinstance(raw_output, Mapping):
        raise IncompletePartitionError(
            "cannot finalize output without an output schema"
        )
    try:
        output_kinds = output_component_kinds(raw_output)
    except ValueError as exc:
        raise IncompletePartitionError(f"invalid output schema: {exc}") from exc
    errors: list[str] = []
    output_bytes = 0
    for partition in partitions:
        partition_id = partition_identity(workflow_id, partition)
        raw_manifest = read_json(manifest_uri(output.uri, partition_id))
        if raw_manifest is None:
            errors.append(f"missing manifest for {partition_id}")
            continue
        try:
            manifest = PartitionManifest.from_dict(raw_manifest)
        except (KeyError, ValueError) as exc:
            errors.append(f"invalid manifest for {partition_id}: {exc}")
            continue
        if (
            manifest.partition_id != partition_id
            or manifest.workflow_id != workflow_id
            or manifest.status != "complete"
        ):
            errors.append(f"manifest identity or status mismatch for {partition_id}")
            continue
        partition_errors = validate_partition_manifest(
            manifest,
            partition,
            output_root=output.uri,
            output_kinds=output_kinds,
            checksums=False,
        )
        output_bytes += sum(manifest.sizes.values())
        errors.extend(f"{partition_id}: {error}" for error in partition_errors)
    if errors:
        raise IncompletePartitionError(
            "cannot finalize incomplete or invalid partitions: " + "; ".join(errors[:3])
        )
    finished = datetime.now(timezone.utc).isoformat()
    attempts = provenance.get("execution_attempts")
    attempt_started: str | None = None
    if isinstance(attempts, list) and attempts and isinstance(attempts[-1], dict):
        raw_started = attempts[-1].get("execution_started")
        attempt_started = raw_started if isinstance(raw_started, str) else None
    previous_status = _finish_current_attempt(
        provenance, status="complete", finished=finished
    )
    if previous_status != "complete":
        provenance["execution_finished"] = finished
    complete, failed = _partition_statuses(output, provenance)
    provenance["completed_partitions"] = complete
    provenance["failed_partitions"] = failed
    provenance["status"] = "complete"
    if task_results:
        computed_count = sum(
            item.get("_execution") == "computed" for item in task_results
        )
        resumed_count = sum(
            item.get("_execution") == "skipped" for item in task_results
        )
        wall_time = None
        if attempt_started is not None:
            try:
                wall_time = (
                    datetime.fromisoformat(finished)
                    - datetime.fromisoformat(attempt_started)
                ).total_seconds()
            except ValueError:
                pass
        response_bytes = (
            source_metrics.get("response_content_bytes")
            if source_metrics is not None
            else None
        )
        stage_execution = provenance.get("stage_execution", [])
        provenance["execution_metrics"] = {
            "schema_version": "1",
            "execution_started": attempt_started,
            "execution_finished": finished,
            "wall_time_seconds": wall_time,
            "completed_task_count": len(task_results),
            "computed_task_count": computed_count,
            "resumed_task_count": resumed_count,
            "partitions_completed": computed_count,
            "bytes_read": response_bytes,
            "bytes_read_status": (
                "measured" if isinstance(response_bytes, int) else "unknown"
            ),
            "output_bytes": output_bytes,
            "peak_rss_bytes": _peak_rss_bytes(),
            "stages": stage_execution if isinstance(stage_execution, list) else [],
            "integrity_verification": {
                "status": "measured",
                "manifests": "valid",
                "checksums": "not-run-during-finalization",
            },
        }
    write_json_atomic(provenance_uri, provenance)
    write_json_atomic(
        join_uri(output.uri, ".neuroflow", "result.json"),
        {
            "schema_version": "1",
            "workflow_id": workflow_id,
            "status": "complete",
            "task_count": len(partitions),
            "output": provenance["output"],
            "provenance": provenance_uri,
        },
    )


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def fail_output(output: ExecutionOutput, workflow_id: str, error: Exception) -> None:
    """Record a terminal workflow failure without masking the original error."""
    provenance_uri = join_uri(output.uri, ".neuroflow", "provenance.json")
    provenance = read_json(provenance_uri)
    if provenance is None or provenance.get("workflow_id") != workflow_id:
        return
    finished = datetime.now(timezone.utc).isoformat()
    error_message = f"{type(error).__name__}: {error}"
    previous_status = _finish_current_attempt(
        provenance,
        status="failed",
        finished=finished,
        error=error_message,
    )
    if previous_status != "complete":
        provenance["execution_finished"] = finished
    complete, failed = _partition_statuses(output, provenance)
    provenance["completed_partitions"] = complete
    provenance["failed_partitions"] = failed
    provenance["status"] = "failed"
    provenance["error"] = error_message
    write_json_atomic(provenance_uri, provenance)


def _partition_statuses(
    output: ExecutionOutput, provenance: Mapping[str, object]
) -> tuple[list[str], list[str]]:
    plan = provenance.get("partition_plan")
    if not isinstance(plan, Mapping):
        return [], []
    identities = plan.get("partition_ids", ())
    if not isinstance(identities, list):
        return [], []
    complete: list[str] = []
    failed: list[str] = []
    for value in identities:
        partition_id = str(value)
        manifest = read_json(manifest_uri(output.uri, partition_id))
        status = manifest.get("status") if manifest else None
        if status == "complete":
            complete.append(partition_id)
        elif status == "failed":
            failed.append(partition_id)
    return complete, failed
