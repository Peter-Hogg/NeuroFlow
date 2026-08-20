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
from neuroflow.selection.query import Selection, absolute_selection_bounds
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
    _validate_partition_descriptors(partitions, selection.metadata.shape)
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
    if isinstance(schema, ArrayOutput) and (
        schema.reduced_axes or schema.kept_reduced_axes
    ):
        reduced_indices, kept_reduced_indices = _validate_reduced_axes(
            schema.reduced_axes,
            schema.kept_reduced_axes,
            selection,
            partitions,
        )
        output_axes = tuple(
            axis
            for index, axis in enumerate(selection.metadata.axes)
            if index not in reduced_indices
        )
        output_shape = tuple(
            1 if index in kept_reduced_indices else size
            for index, size in enumerate(selection.metadata.shape)
            if index not in reduced_indices
        )
        partitions = tuple(
            Partition(
                key=item.key,
                read_slices=item.read_slices,
                output_slices=tuple(
                    (slice(0, 1) if index in kept_reduced_indices else value)
                    for index, value in enumerate(item.output_slices)
                    if index not in reduced_indices
                ),
                trim_slices=item.trim_slices,
                coordinates=item.coordinates,
            )
            for item in partitions
        )
    _validate_partition_coverage(partitions, output_shape)
    if isinstance(schema, ArrayOutput) and schema.chunks is not None:
        if len(schema.chunks) != len(output_shape) or any(
            size <= 0 for size in schema.chunks
        ):
            raise AdapterCompatibilityError(
                "array output chunks must contain one positive size per output axis"
            )
    declared_output_axes = getattr(adapter, "output_axes", None)
    declared_output_shape = getattr(adapter, "output_shape", None)
    if declared_output_axes is not None and tuple(declared_output_axes) != output_axes:
        raise AdapterCompatibilityError(
            "adapter expression axes do not match its declared reduction schema"
        )
    if (
        declared_output_shape is not None
        and tuple(declared_output_shape) != output_shape
    ):
        raise AdapterCompatibilityError(
            "adapter expression shape does not match its declared reduction schema"
        )
    _validate_required_overlap(partition, requirements.requires_overlap)
    processing_axes = _processing_axes(partitions, selection.metadata.axes)
    unsupported = sorted(set(processing_axes) - set(requirements.splittable_axes))
    if unsupported:
        raise AdapterCompatibilityError(
            f"adapter {adapter.name!r} does not declare splittable axes: "
            f"{', '.join(unsupported)}"
        )
    if isinstance(schema, ArrayOutput):
        effective_chunks = schema.chunks or slice_shape(partitions[0].output_slices)
        _validate_chunk_isolation(partitions, effective_chunks, output_axes)
    identity_parameters = getattr(adapter, "identity_parameters", None)
    adapter_parameters = (
        identity_parameters()
        if callable(identity_parameters)
        else (getattr(adapter, "parameters", None) or {})
    )
    identity_payload: dict[str, Any] = {
        "source": asdict(selection.metadata.source),
        "selection": {
            "path": selection.metadata.path,
            "type": selection.metadata.neurodata_type,
            "shape": selection.metadata.shape,
            "dtype": selection.metadata.dtype,
            "bounds": absolute_selection_bounds(selection.metadata),
        },
        "adapter": {
            "name": adapter.name,
            "version": adapter.version,
            "parameters": adapter_parameters,
            "random_seed": getattr(adapter, "random_seed", None),
        },
        "partition": _value_spec(partition),
        "partitions": [item.to_dict() for item in partitions],
        "output_schema": _value_spec(getattr(adapter, "output", None)),
        "schema_version": "2",
    }
    workflow_id = stable_hash(identity_payload)
    itemsize = np.dtype(selection.metadata.dtype).itemsize
    read_elements = [
        element_count(slice_shape(item.read_slices)) for item in partitions
    ]
    selected_elements = element_count(selection.metadata.shape)
    output_elements = element_count(output_shape)
    memory_estimator = getattr(adapter, "estimate_task_memory", None)
    if callable(memory_estimator):
        estimates = [
            memory_estimator(slice_shape(item.read_slices)) for item in partitions
        ]
        if not all(isinstance(value, int) for value in estimates):
            raise TypeError("adapter memory estimates must be integers")
        memory_per_task = max(value for value in estimates if isinstance(value, int))
    else:
        memory_per_task = max(read_elements) * itemsize
    first_shape = tuple(
        (item.stop or 0) - (item.start or 0) for item in partitions[0].trim_slices
    )
    selected_asset = selection.metadata.source.asset_id
    source_size = next(
        (
            item.size
            for item in source.assets()
            if item.size and (selected_asset is None or item.asset_id == selected_asset)
        ),
        None,
    )
    warnings = list(validation.warnings)
    read_amplification = sum(read_elements) / selected_elements
    maximum_partition_shape = tuple(
        max(slice_shape(item.read_slices)[index] for item in partitions)
        for index in range(len(selection.metadata.shape))
    )
    logical_bytes_read = sum(read_elements) * itemsize
    if selection.metadata.native_chunks is None:
        source_chunks_touched = None
        estimated_total_bytes_read = None
    else:
        source_chunks_touched = sum(
            _chunks_touched(item.read_slices, selection.metadata.native_chunks)
            for item in partitions
        )
        estimated_total_bytes_read = (
            source_chunks_touched
            * element_count(selection.metadata.native_chunks)
            * itemsize
        )
    if read_amplification > 1.5:
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
        read_amplification=read_amplification,
        maximum_logical_partition_shape=maximum_partition_shape,
        estimated_logical_bytes_read=logical_bytes_read,
        estimated_source_chunks_touched=source_chunks_touched,
        estimated_total_bytes_read=estimated_total_bytes_read,
        bounded=True,
        bounded_reasons=(
            "every task has finite validated source slices",
            "memory_per_task is derived before numerical I/O",
        ),
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


def _chunks_touched(slices: tuple[slice, ...], chunks: tuple[int, ...]) -> int:
    touched = 1
    for item, chunk in zip(slices, chunks, strict=True):
        start = item.start or 0
        stop = item.stop or 0
        touched *= max(0, (stop - 1) // chunk - start // chunk + 1)
    return touched


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


def _validate_partition_descriptors(
    partitions: tuple[object, ...], source_shape: tuple[int, ...]
) -> None:
    """Validate source-space partition metadata before using it for planning."""
    rank = len(source_shape)
    if any(
        not isinstance(size, int) or isinstance(size, bool) or size <= 0
        for size in source_shape
    ):
        raise PartitionValidationError(
            "selection shape must contain positive integer dimensions"
        )

    seen_keys: set[str] = set()
    for number, item in enumerate(partitions):
        if not isinstance(item, Partition):
            raise PartitionValidationError(
                f"partition {number} is not a Partition descriptor"
            )
        if not isinstance(item.key, str) or not item.key:
            raise PartitionValidationError(
                f"partition {number} key must be a non-empty string"
            )
        if item.key in seen_keys:
            raise PartitionValidationError(f"duplicate partition key {item.key!r}")
        seen_keys.add(item.key)
        if not isinstance(item.coordinates, tuple) or any(
            isinstance(value, bool) or not isinstance(value, (int, str))
            for value in item.coordinates
        ):
            raise PartitionValidationError(
                f"partition {item.key!r} coordinates must be a tuple of integers "
                "or strings"
            )

        read_bounds = _validate_slices(
            item.read_slices,
            source_shape,
            partition_key=item.key,
            field_name="read_slices",
        )
        output_bounds = _validate_slices(
            item.output_slices,
            source_shape,
            partition_key=item.key,
            field_name="output_slices",
        )
        read_shape = tuple(stop - start for start, stop in read_bounds)
        trim_bounds = _validate_slices(
            item.trim_slices,
            read_shape,
            partition_key=item.key,
            field_name="trim_slices",
        )
        if not (len(read_bounds) == len(output_bounds) == len(trim_bounds) == rank):
            # Each call above reports rank errors; this is only a type-narrowing
            # guard for malformed runtime objects.
            raise PartitionValidationError(
                f"partition {item.key!r} descriptors must have rank {rank}"
            )
        for axis, ((read_start, _), (output_start, output_stop), trim) in enumerate(
            zip(read_bounds, output_bounds, trim_bounds, strict=True)
        ):
            trim_start, trim_stop = trim
            if (
                read_start + trim_start != output_start
                or read_start + trim_stop != output_stop
            ):
                raise PartitionValidationError(
                    f"partition {item.key!r} trim_slices do not map read_slices "
                    f"onto output_slices on axis {axis}"
                )


def _validate_slices(
    values: object,
    shape: tuple[int, ...],
    *,
    partition_key: str,
    field_name: str,
) -> tuple[tuple[int, int], ...]:
    if not isinstance(values, tuple) or len(values) != len(shape):
        raise PartitionValidationError(
            f"partition {partition_key!r} {field_name} must be a tuple with "
            f"rank {len(shape)}"
        )
    bounds: list[tuple[int, int]] = []
    for axis, (value, size) in enumerate(zip(values, shape, strict=True)):
        if not isinstance(value, slice):
            raise PartitionValidationError(
                f"partition {partition_key!r} {field_name}[{axis}] must be a slice"
            )
        step = value.step
        if step is not None and (
            not isinstance(step, int) or isinstance(step, bool) or step != 1
        ):
            raise PartitionValidationError(
                f"partition {partition_key!r} {field_name}[{axis}] must use a "
                "unit slice"
            )
        start = value.start
        stop = value.stop
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(stop, int)
            or isinstance(stop, bool)
        ):
            raise PartitionValidationError(
                f"partition {partition_key!r} {field_name}[{axis}] must have "
                "explicit integer bounds"
            )
        if start < 0 or stop <= start or stop > size:
            raise PartitionValidationError(
                f"partition {partition_key!r} {field_name}[{axis}] bounds "
                f"[{start}, {stop}) are outside [0, {size}) or empty"
            )
        bounds.append((start, stop))
    return tuple(bounds)


def _validate_partition_coverage(
    partitions: tuple[Partition, ...], output_shape: tuple[int, ...]
) -> None:
    """Prove exact, disjoint output ownership without an output-sized bitmap."""
    rectangles = tuple(
        _validate_slices(
            item.output_slices,
            output_shape,
            partition_key=item.key,
            field_name="lowered output_slices",
        )
        for item in partitions
    )
    expected_volume = element_count(output_shape)
    actual_volume = sum(
        element_count(tuple(stop - start for start, stop in rectangle))
        for rectangle in rectangles
    )
    if actual_volume != expected_volume:
        raise PartitionValidationError(
            "partition output_slices do not exactly cover the declared output: "
            f"owned volume {actual_volume}, expected {expected_volume}"
        )

    if not output_shape:
        if len(rectangles) != 1:
            raise PartitionValidationError(
                "scalar output must be owned by exactly one partition"
            )
        return

    # Sweep on the most subdivided axis. Only rectangles whose sweep intervals
    # overlap remain active; checking their other axes proves disjointness while
    # memory remains proportional to the number of partitions.
    sweep_axis = max(
        range(len(output_shape)),
        key=lambda axis: len({rectangle[axis] for rectangle in rectangles}),
    )
    ordered = sorted(
        zip(partitions, rectangles, strict=True),
        key=lambda value: (
            value[1][sweep_axis][0],
            value[1][sweep_axis][1],
            value[0].key,
        ),
    )
    active: list[tuple[Partition, tuple[tuple[int, int], ...]]] = []
    for partition, rectangle in ordered:
        sweep_start = rectangle[sweep_axis][0]
        active = [
            candidate
            for candidate in active
            if candidate[1][sweep_axis][1] > sweep_start
        ]
        for other_partition, other_rectangle in active:
            if all(
                left_start < right_stop and right_start < left_stop
                for axis, (
                    (left_start, left_stop),
                    (right_start, right_stop),
                ) in enumerate(zip(rectangle, other_rectangle, strict=True))
                if axis != sweep_axis
            ):
                raise PartitionValidationError(
                    "partition output_slices overlap between "
                    f"{other_partition.key!r} and {partition.key!r}"
                )
        active.append((partition, rectangle))


def _validate_reduced_axes(
    reduced_axes: tuple[str, ...],
    kept_reduced_axes: tuple[str, ...],
    selection: Selection,
    partitions: tuple[Partition, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    full_axes = (*reduced_axes, *kept_reduced_axes)
    if len(set(full_axes)) != len(full_axes):
        raise AdapterCompatibilityError("reduced axes must be unique")
    missing = set(full_axes) - set(selection.metadata.axes)
    if missing:
        raise AdapterCompatibilityError(
            "selection has no reduced axes: " + ", ".join(sorted(missing))
        )
    indices = tuple(selection.metadata.axes.index(axis) for axis in full_axes)
    for axis, index in zip(full_axes, indices, strict=True):
        if any(
            item.read_slices[index].start != 0
            or item.read_slices[index].stop != selection.metadata.shape[index]
            for item in partitions
        ):
            raise AdapterCompatibilityError(
                f"reduced axis {axis!r} must not be split by the partition plan"
            )
    dropped_count = len(reduced_axes)
    return indices[:dropped_count], indices[dropped_count:]


def _validate_chunk_isolation(
    partitions: tuple[Partition, ...],
    chunks: tuple[int, ...],
    output_axes: tuple[str, ...],
) -> None:
    """Require every parallel partition boundary to align to a Zarr chunk."""
    if len(chunks) != len(output_axes):
        return
    for index, (axis, chunk) in enumerate(zip(output_axes, chunks, strict=True)):
        starts = {item.output_slices[index].start or 0 for item in partitions}
        unaligned = sorted(start for start in starts if start and start % chunk)
        if unaligned:
            raise AdapterCompatibilityError(
                f"output chunk {chunk} on axis {axis!r} crosses a processing "
                f"partition boundary at {unaligned[0]}; choose aligned chunks"
            )


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
