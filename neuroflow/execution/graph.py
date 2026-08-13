"""Metadata-only workflow validation and planning."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np

from neuroflow.adapters.base import AnalysisAdapter
from neuroflow.adapters.numpy import ArrayOutput, TableOutput
from neuroflow.adapters.segmentation import SegmentationOutputSchema
from neuroflow.diagnostics.estimates import element_count, slice_shape
from neuroflow.diagnostics.plan import ExecutionPlan
from neuroflow.exceptions import AdapterCompatibilityError, PartitionValidationError
from neuroflow.partition.base import Partition, PartitionPlan
from neuroflow.provenance.hashing import stable_hash
from neuroflow.selection.query import Selection
from neuroflow.source.base import NWBSource
from neuroflow.storage.base import OutputSpec
from neuroflow.storage.parquet import ParquetOutput
from neuroflow.storage.segmentation import SegmentationOutput
from neuroflow.storage.zarr import ZarrOutput


def _value_spec(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return repr(value)


def build_plan(
    *,
    source: NWBSource,
    selection: Selection,
    adapter: AnalysisAdapter,
    partition: PartitionPlan,
    output: OutputSpec,
) -> ExecutionPlan:
    """Validate a workflow and estimate it using metadata only."""
    validation = partition.validate(selection)
    if not validation.valid:
        raise PartitionValidationError("; ".join(validation.errors))
    partitions = tuple(partition.build(selection))
    if not partitions:
        raise PartitionValidationError("partition plan produced no work")
    requirements = adapter.requirements()
    schema = getattr(adapter, "output", None)
    if isinstance(schema, ArrayOutput) and not isinstance(output, ZarrOutput):
        raise AdapterCompatibilityError("array outputs require ZarrOutput")
    if isinstance(schema, TableOutput) and not isinstance(output, ParquetOutput):
        raise AdapterCompatibilityError("table outputs require ParquetOutput")
    if isinstance(schema, SegmentationOutputSchema) and not isinstance(
        output, SegmentationOutput
    ):
        raise AdapterCompatibilityError(
            "segmentation outputs require SegmentationOutput"
        )
    reduced_indices: tuple[int, ...] = ()
    output_axes = selection.metadata.axes
    output_shape = selection.metadata.shape
    if isinstance(schema, ArrayOutput) and schema.reduced_axes:
        reduced_indices = _validate_reduced_axes(
            schema.reduced_axes, selection, partitions
        )
        output_axes = tuple(
            axis
            for index, axis in enumerate(selection.metadata.axes)
            if index not in reduced_indices
        )
        output_shape = tuple(
            size
            for index, size in enumerate(selection.metadata.shape)
            if index not in reduced_indices
        )
        partitions = tuple(
            Partition(
                key=item.key,
                read_slices=item.read_slices,
                output_slices=tuple(
                    value
                    for index, value in enumerate(item.output_slices)
                    if index not in reduced_indices
                ),
                trim_slices=item.trim_slices,
                coordinates=item.coordinates,
            )
            for item in partitions
        )
    if isinstance(schema, ArrayOutput) and schema.chunks is not None:
        if len(schema.chunks) != len(output_shape) or any(
            size <= 0 for size in schema.chunks
        ):
            raise AdapterCompatibilityError(
                "array output chunks must contain one positive size per output axis"
            )
    _validate_required_overlap(partition, requirements.requires_overlap)
    processing_axes = _processing_axes(partitions, selection.metadata.axes)
    unsupported = sorted(set(processing_axes) - set(requirements.splittable_axes))
    if unsupported:
        raise AdapterCompatibilityError(
            f"adapter {adapter.name!r} does not declare splittable axes: "
            f"{', '.join(unsupported)}"
        )
    identity_parameters = getattr(adapter, "identity_parameters", None)
    adapter_parameters = (
        identity_parameters()
        if callable(identity_parameters)
        else (getattr(adapter, "parameters", None) or {})
    )
    identity_payload: dict[str, Any] = {
        "source": asdict(source.identity),
        "selection": {
            "path": selection.metadata.path,
            "type": selection.metadata.neurodata_type,
            "shape": selection.metadata.shape,
            "dtype": selection.metadata.dtype,
            "bounds": selection.metadata.selection_bounds,
        },
        "adapter": {
            "name": adapter.name,
            "version": adapter.version,
            "parameters": adapter_parameters,
            "random_seed": getattr(adapter, "random_seed", None),
        },
        "partition": _value_spec(partition),
        "output_schema": _value_spec(getattr(adapter, "output", None)),
        "schema_version": "1",
    }
    workflow_id = stable_hash(identity_payload)
    itemsize = np.dtype(selection.metadata.dtype).itemsize
    read_elements = [
        element_count(slice_shape(item.read_slices)) for item in partitions
    ]
    output_elements = element_count(output_shape)
    memory_per_task = max(read_elements) * itemsize
    first_shape = tuple(
        (item.stop or 0) - (item.start or 0)
        for item in partitions[0].trim_slices
    )
    source_size = next((item.size for item in source.assets() if item.size), None)
    warnings = list(validation.warnings)
    if sum(read_elements) / output_elements > 1.5:
        warnings.append("overlap causes more than 1.5x source read amplification")
    if selection.metadata.native_chunks:
        if any(
            processing < native
            for processing, native in zip(
                first_shape, selection.metadata.native_chunks, strict=True
            )
        ):
            warnings.append(
                "processing partitions are smaller than native chunks on at least "
                "one axis"
            )
    return ExecutionPlan(
        workflow_id=workflow_id,
        source_size=source_size,
        selected_shape=selection.metadata.shape,
        output_shape=output_shape,
        output_axes=output_axes,
        dtype=selection.metadata.dtype,
        native_chunks=selection.metadata.native_chunks,
        processing_partition_shape=first_shape,
        overlap=tuple(
            read - output
            for read, output in zip(
                slice_shape(partitions[0].read_slices), first_shape, strict=True
            )
        ),
        task_count=len(partitions),
        memory_per_task=memory_per_task,
        read_amplification=sum(read_elements) / output_elements,
        expected_output_size=(
            output_elements * np.dtype(schema.dtype).itemsize
            if isinstance(schema, ArrayOutput)
            else (
                output_elements * np.dtype(schema.label_dtype).itemsize
                if isinstance(schema, SegmentationOutputSchema)
                else None
            )
        ),
        warnings=tuple(warnings),
        partitions=partitions,
        resources=requirements.resources,
    )


def _processing_axes(
    partitions: tuple[object, ...], axes: tuple[str, ...]
) -> tuple[str, ...]:
    if len(partitions) <= 1:
        return ()
    return tuple(
        axis
        for index, axis in enumerate(axes)
        if len(
            {(getattr(item, "read_slices")[index].start or 0) for item in partitions}
        )
        > 1
    )


def _validate_reduced_axes(
    reduced_axes: tuple[str, ...],
    selection: Selection,
    partitions: tuple[Partition, ...],
) -> tuple[int, ...]:
    if len(set(reduced_axes)) != len(reduced_axes):
        raise AdapterCompatibilityError("reduced axes must be unique")
    missing = set(reduced_axes) - set(selection.metadata.axes)
    if missing:
        raise AdapterCompatibilityError(
            "selection has no reduced axes: " + ", ".join(sorted(missing))
        )
    indices = tuple(selection.metadata.axes.index(axis) for axis in reduced_axes)
    for axis, index in zip(reduced_axes, indices, strict=True):
        full = slice(0, selection.metadata.shape[index])
        if any(item.read_slices[index] != full for item in partitions):
            raise AdapterCompatibilityError(
                f"reduced axis {axis!r} must not be split by the partition plan"
            )
    return indices


def _validate_required_overlap(
    partition: PartitionPlan, required: Mapping[str, int | str]
) -> None:
    axes = getattr(partition, "axes", ())
    halo = getattr(partition, "halo", ())
    actual = dict(zip(axes, halo, strict=True)) if axes and halo else {}
    if hasattr(partition, "overlap"):
        actual["time"] = getattr(partition, "overlap")
    missing: list[str] = []
    for axis, needed in required.items():
        available = actual.get(axis)
        if isinstance(needed, int) and isinstance(available, int):
            valid = available >= needed
        else:
            valid = available == needed
        if not valid:
            missing.append(f"{axis}={needed!r} (plan provides {available!r})")
    if missing:
        raise AdapterCompatibilityError(
            "partition overlap does not satisfy adapter requirements: "
            + ", ".join(missing)
        )
