"""Explicit execution and durable result initialization."""

from __future__ import annotations

import importlib.metadata
import os
import platform
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Literal

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
from neuroflow.exceptions import OutputConflictError, ProvenanceMismatchError
from neuroflow.partition.base import Partition
from neuroflow.provenance.hashing import stable_hash
from neuroflow.selection.query import Selection
from neuroflow.source.base import NWBSource
from neuroflow.storage.base import join_uri, read_json, write_json_atomic
from neuroflow.storage.manifest import PartitionManifest
from neuroflow.storage.parquet import ParquetOutput
from neuroflow.storage.segmentation import SegmentationOutput
from neuroflow.storage.validation import validate_partition_manifest
from neuroflow.storage.writer import (
    ArrayPartitionWriter,
    SegmentationPartitionWriter,
    TablePartitionWriter,
)
from neuroflow.storage.zarr import ZarrOutput

ExecutionOutput = ZarrOutput | ParquetOutput | SegmentationOutput


def partition_identity(workflow_id: str, partition: Partition) -> str:
    return stable_hash(
        {
            "workflow_id": workflow_id,
            "key": partition.key,
            "coordinates": partition.coordinates,
        }
    )


def manifest_uri(uri: str, partition_id: str) -> str:
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
    existing = read_json(manifest_uri(output.uri, partition_id))
    if existing is not None:
        manifest = PartitionManifest.from_dict(existing)
        if manifest.workflow_id != workflow_id:
            raise ProvenanceMismatchError(
                "partition manifest belongs to another workflow"
            )
        if manifest.status == "complete" and not validate_partition_manifest(
            manifest, partition, checksums=True
        ):
            return manifest.to_dict()
    seed = getattr(adapter, "random_seed", None)
    context = TaskContext(
        partition_id=partition_id,
        parameters=dict(getattr(adapter, "parameters", None) or {}),
        random_seed=seed,
    )
    schema = getattr(adapter, "output", None)
    if isinstance(schema, ArrayOutput):
        reduced_axis_indices = tuple(
            selection_axes.index(axis) for axis in schema.reduced_axes
        )
        writer: object = ArrayPartitionWriter(
            uri=output.uri,
            array_name=schema.name,
            partition=partition,
            workflow_id=workflow_id,
            partition_id=partition_id,
            reduced_axis_indices=reduced_axis_indices,
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
        data = np.asarray(source_array[partition.read_slices])  # type: ignore[index]
        time_slice = partition.read_slices[0]
        if source_timestamps is not None:
            timestamps = np.asarray(source_timestamps[time_slice])  # type: ignore[index]
        elif sampling_rate is not None:
            start = time_slice.start or 0
            stop = time_slice.stop or data.shape[0]
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
        return manifest.to_dict()
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
) -> None:
    provenance_uri = join_uri(output.uri, ".neuroflow", "provenance.json")
    existing = read_json(provenance_uri)
    fs, root_path = fsspec.core.url_to_fs(output.uri)
    root_exists = fs.exists(root_path)
    mode = output.mode
    if root_exists and mode == "overwrite" and not resume:
        fs.rm(root_path, recursive=True)
        existing = None
        root_exists = False
    if existing is not None:
        if existing.get("workflow_id") != execution_plan.workflow_id:
            if mode == "overwrite":
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
            chunks = tuple(array.chunks)
        else:
            requested_chunks = schema.chunks or slice_shape(
                execution_plan.partitions[0].output_slices
            )
            chunks = tuple(
                min(full, chunk)
                for full, chunk in zip(
                    execution_plan.output_shape, requested_chunks, strict=True
                )
            )
            create_options: dict[str, object] = {}
            if output.compressor == "none":
                create_options["compressor"] = None
            group.create_dataset(
                name,
                shape=execution_plan.output_shape,
                chunks=chunks,
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
            "chunks": chunks,
        }
    elif isinstance(output, ParquetOutput) and isinstance(schema, TableOutput):
        output_metadata = {"kind": "table", "uri": output.uri, "name": name}
    elif isinstance(output, SegmentationOutput) and isinstance(
        schema, SegmentationOutputSchema
    ):
        mapper = fsspec.get_mapper(output.uri)
        group = zarr.open_group(mapper, mode="a")
        dtype = np.dtype(schema.label_dtype)
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
        else:
            chunks = tuple(
                min(full, chunk)
                for full, chunk in zip(
                    selection.metadata.shape,
                    execution_plan.processing_partition_shape,
                    strict=True,
                )
            )
            create_options = {}
            if output.compressor == "none":
                create_options["compressor"] = None
            group.create_dataset(
                schema.labels_name,
                shape=selection.metadata.shape,
                chunks=chunks,
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
            "arrays": {
                schema.labels_name: {
                    "name": schema.labels_name,
                    "dtype": str(dtype),
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
    provenance = {
        "schema_version": "1",
        "workflow_id": execution_plan.workflow_id,
        "neuroflow_version": __version__,
        "source": asdict(source.identity),
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
        "scheduler": scheduler,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pid": str(os.getpid()),
        },
        "output": output_metadata,
        "execution_started": datetime.now(timezone.utc).isoformat(),
        "completed_partitions": [],
        "failed_partitions": [],
        "status": "running",
    }
    if existing is not None:
        provenance["execution_started"] = existing.get(
            "execution_started", provenance["execution_started"]
        )
    write_json_atomic(provenance_uri, provenance)


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


def finalize_output(output: ExecutionOutput, workflow_id: str, task_count: int) -> None:
    provenance_uri = join_uri(output.uri, ".neuroflow", "provenance.json")
    provenance = read_json(provenance_uri)
    if provenance is None or provenance.get("workflow_id") != workflow_id:
        raise ProvenanceMismatchError("cannot finalize output without valid provenance")
    provenance["execution_finished"] = datetime.now(timezone.utc).isoformat()
    complete, failed = _partition_statuses(output, provenance)
    provenance["completed_partitions"] = complete
    provenance["failed_partitions"] = failed
    provenance["status"] = "complete"
    write_json_atomic(provenance_uri, provenance)
    write_json_atomic(
        join_uri(output.uri, ".neuroflow", "result.json"),
        {
            "schema_version": "1",
            "workflow_id": workflow_id,
            "status": "complete",
            "task_count": task_count,
            "output": provenance["output"],
            "provenance": provenance_uri,
        },
    )


def fail_output(output: ExecutionOutput, workflow_id: str, error: Exception) -> None:
    """Record a terminal workflow failure without masking the original error."""
    provenance_uri = join_uri(output.uri, ".neuroflow", "provenance.json")
    provenance = read_json(provenance_uri)
    if provenance is None or provenance.get("workflow_id") != workflow_id:
        return
    provenance["execution_finished"] = datetime.now(timezone.utc).isoformat()
    complete, failed = _partition_statuses(output, provenance)
    provenance["completed_partitions"] = complete
    provenance["failed_partitions"] = failed
    provenance["status"] = "failed"
    provenance["error"] = f"{type(error).__name__}: {error}"
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
