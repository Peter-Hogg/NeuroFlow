"""Top-level orchestration entry points.

Selection and planning are metadata-only. Work begins only through ``execute()``
or an explicit ``execute=True`` call.
"""

from __future__ import annotations

import os
import re
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from neuroflow.adapters.base import AnalysisAdapter
    from neuroflow.diagnostics.plan import ExecutionPlan
    from neuroflow.partition.base import PartitionPlan
    from neuroflow.selection.query import Selection
    from neuroflow.source.base import NWBSource, SourceSpec
    from neuroflow.storage.base import OutputSpec

from neuroflow.adapters.numpy import ExpressionAdapter
from neuroflow.exceptions import (
    IncompletePartitionError,
    OutputConflictError,
    UnsupportedBackendError,
)
from neuroflow.execution.graph import build_plan
from neuroflow.execution.resources import parse_bytes, resolve_memory_budget
from neuroflow.execution.stages import build_reduction_stage_plans
from neuroflow.results.workflow import PersistedResult, WorkflowResult
from neuroflow.source.array import ArraySource
from neuroflow.source.base import SourceSpec
from neuroflow.source.dandi import DandiNWBSource
from neuroflow.source.hdf5 import NWBHDF5Source
from neuroflow.source.local import LocalNWBZarrSource
from neuroflow.storage.base import join_uri, read_json
from neuroflow.storage.parquet import ParquetOutput
from neuroflow.storage.segmentation import SegmentationOutput
from neuroflow.storage.zarr import ZarrOutput


def open_source(
    source: str | Path | SourceSpec,
    *,
    version: str | None = None,
    storage_options: dict[str, object] | None = None,
) -> NWBSource:
    """Resolve source metadata without reading numerical datasets."""
    if isinstance(source, SourceSpec):
        if (
            version is not None
            and source.version is not None
            and version != source.version
        ):
            raise ValueError("version conflicts with SourceSpec.version")
        version = version or source.version
        options = dict(source.storage_options)
        options.update(storage_options or {})
        storage_options = options
        source = source.uri
    value = str(source)
    match = re.fullmatch(r"DANDI:(\d{6})(?:@([^/]+))?", value, re.IGNORECASE)
    if match:
        embedded_version = match.group(2)
        if version and embedded_version and version != embedded_version:
            raise ValueError("version conflicts with the DANDI identifier")
        return DandiNWBSource(
            match.group(1),
            version=version or embedded_version,
            storage_options=storage_options,
        )
    source_class = (
        NWBHDF5Source
        if value.lower().split("?", 1)[0].endswith(".nwb")
        else LocalNWBZarrSource
    )
    return source_class(
        value,
        version=version,
        storage_options=storage_options,
    )


def open_dandi(
    dandiset: str,
    *,
    version: str | None = None,
    backend: Literal["auto", "lindi", "remfile"] = "auto",
    storage_options: dict[str, object] | None = None,
) -> DandiNWBSource:
    """Open a Dandiset with replaceable HDF5 transport semantics.

    ``backend='auto'`` retains the conservative remfile default for HDF5 blob
    assets. LINDI is opt-in because backend availability and archive-side LINDI
    artifacts can change independently of NeuroFlow.
    """
    match = re.fullmatch(
        r"(?:DANDI:)?(\d{1,6})(?:@([^/]+))?", dandiset, re.IGNORECASE
    )
    if match is None:
        raise ValueError("dandiset must be a numeric ID or DANDI:<ID>[@<version>]")
    embedded_version = match.group(2)
    if version and embedded_version and version != embedded_version:
        raise ValueError("version conflicts with the DANDI identifier")
    return DandiNWBSource(
        match.group(1),
        version=version or embedded_version,
        backend=backend,
        storage_options=storage_options,
    )


def plan(
    *,
    source: NWBSource,
    selection: Selection,
    adapter: AnalysisAdapter,
    partition: PartitionPlan,
    output: OutputSpec,
) -> ExecutionPlan:
    """Validate and describe a workflow without executing it."""
    return build_plan(
        source=source,
        selection=selection,
        adapter=adapter,
        partition=partition,
        output=output,
    )


def _adapter_external_reserve_bytes(adapter: AnalysisAdapter) -> int:
    """Bytes an adapter needs resident but does not allocate per partition.

    Adapters that load a large third-party model may implement
    ``external_memory_reserve_bytes()``. The value is subtracted from the total
    process-memory target so that a budget which cannot physically hold the
    model is refused with an explanation instead of appearing to succeed.
    """
    hook = getattr(adapter, "external_memory_reserve_bytes", None)
    if not callable(hook):
        return 0
    value = hook()
    if not isinstance(value, int) or value < 0:
        raise TypeError(
            "external_memory_reserve_bytes() must return a non-negative integer"
        )
    return value


def run(
    *,
    source: NWBSource,
    selection: Selection,
    adapter: AnalysisAdapter,
    partition: PartitionPlan,
    output: OutputSpec,
    scheduler: Literal["threads", "processes", "distributed"] = "threads",
    resume: bool = True,
    execute: bool = False,
    max_workers: int | None = None,
    memory_limit: int | str | None = None,
) -> WorkflowResult:
    """Construct a lazy workflow; execution is always explicitly requested."""
    attributes = selection.metadata.attributes or {}
    if attributes.get("backend") == "nwb-hdf5" and scheduler != "threads":
        raise UnsupportedBackendError(
            "NWB-HDF5 selections currently require scheduler='threads'; open "
            "h5py and remote file handles are not safely serializable"
        )
    if not isinstance(output, (ZarrOutput, ParquetOutput, SegmentationOutput)):
        raise TypeError(
            "output must be ZarrOutput, ParquetOutput, or SegmentationOutput"
        )
    if str(output.uri).rstrip("/") == source.identity.uri.rstrip("/"):
        raise OutputConflictError("output cannot overwrite its source")
    execution_plan = plan(
        source=source,
        selection=selection,
        adapter=adapter,
        partition=partition,
        output=output,
    )
    if max_workers is not None and max_workers < 1:
        raise ValueError("max_workers must be positive")
    if memory_limit is not None:
        # ``memory_limit`` is a total process-memory target, so the bytes
        # available for concurrent task working sets are what remains after the
        # process floor.
        budget = resolve_memory_budget(memory_limit)
        # Any residency the adapter declares but does not allocate per
        # partition -- a loaded Cellpose network, for instance -- is charged
        # *per worker*, not once. Adapters cache such models in thread-local
        # state, so N concurrent workers hold N copies; subtracting a single
        # copy from the target would understate concurrent runs by a factor of
        # N and let a large worker count silently exceed the budget.
        reserve = _adapter_external_reserve_bytes(adapter)
        declared = execution_plan.resources.memory
        per_worker = (
            max(
                execution_plan.memory_per_task,
                parse_bytes(declared) if declared is not None else 0,
            )
            + reserve
        )
        if per_worker > budget.task_bytes:
            raise ValueError(
                f"one task requires an estimated {per_worker} bytes"
                + (
                    f" (including {reserve} bytes of declared per-worker model "
                    "residency)"
                    if reserve
                    else ""
                )
                + f", exceeding the {budget.task_bytes} bytes available for "
                f"tasks under a {budget.total_bytes}-byte total process-memory "
                f"target ({budget.process_overhead_bytes} bytes of process "
                "overhead)"
            )
        safe_workers = max(1, budget.task_bytes // max(per_worker, 1))
        # ``max_workers`` is an availability ceiling, not a demand. The contract
        # is that a user states the resources they have and the planner picks
        # partitioning and concurrency to fit; refusing the call would push them
        # back to hand-tuning a low-level knob to rediscover a number the
        # planner already computed. Concurrency is therefore clamped down to
        # what the memory target affords rather than rejected. The clamp is not
        # silent: the granted count is recorded as ``execution.max_workers`` in
        # provenance, so a reduced run stays auditable after the fact.
        available_workers = max(1, os.cpu_count() or 1)
        max_workers = int(
            min(max_workers or available_workers, safe_workers, available_workers)
        )
    if isinstance(adapter, ExpressionAdapter):
        stage_plans = build_reduction_stage_plans(
            selection,
            adapter.expression,
            memory_limit=memory_limit,
        )
        execution_plan = replace(
            execution_plan,
            stages=tuple(item.to_dict() for item in stage_plans),
        )
    result = WorkflowResult(
        source=source,
        selection=selection,
        adapter=adapter,
        output=output,
        plan=execution_plan,
        partition=partition,
        scheduler=scheduler,
        resume_enabled=resume,
        max_workers=max_workers,
        memory_limit=memory_limit,
    )
    return result.execute() if execute else result


def open_result(uri: str | Path) -> PersistedResult:
    """Open a persisted result lazily."""
    value = str(uri)
    metadata = read_json(join_uri(value, ".neuroflow", "result.json"))
    provenance = read_json(join_uri(value, ".neuroflow", "provenance.json"))
    if provenance is None:
        raise IncompletePartitionError(f"{value} does not contain NeuroFlow provenance")
    if metadata is not None and metadata.get("workflow_id") != provenance.get(
        "workflow_id"
    ):
        raise IncompletePartitionError("result and provenance identities do not match")
    if metadata is None:
        metadata = {
            "workflow_id": provenance.get("workflow_id"),
            "status": provenance.get("status", "partial"),
        }
    return PersistedResult(value, metadata, provenance)


def open_array(
    uri: str | Path,
    *,
    component: str | None = None,
    axes: tuple[str, ...] | None = None,
    verify: bool = True,
) -> tuple[ArraySource, Selection]:
    """Open a complete persisted array as a composable workflow input.

    Partition checksums are verified by default.  ``verify=False`` trusts the
    recorded checksums and is intended only for an output that successfully
    finished in the current process, such as the internal return path from
    :meth:`NeuroArray.persist`.
    """
    if not isinstance(verify, bool):
        raise TypeError("verify must be a boolean")
    result = open_result(uri)
    output = result.provenance.get("output", {})
    if not isinstance(output, dict) or output.get("kind") not in {
        "array",
        "segmentation",
    }:
        raise TypeError("result does not contain a composable array")
    content_identity = result.array_source_identity(verify_checksums=verify)
    kind = output["kind"]
    raw_shape = output.get("shape")
    raw_axes = output.get("axes")
    if (
        not isinstance(raw_shape, list)
        or any(
            not isinstance(size, int) or isinstance(size, bool) or size < 0
            for size in raw_shape
        )
        or not isinstance(raw_axes, list)
        or not all(isinstance(axis, str) for axis in raw_axes)
        or len(raw_axes) != len(raw_shape)
        or len(set(raw_axes)) != len(raw_axes)
    ):
        raise IncompletePartitionError(
            "persisted array has invalid shape or axis metadata"
        )
    declared_shape = tuple(raw_shape)
    declared_axes = tuple(raw_axes)
    if component is None:
        if kind == "array":
            name = output.get("name")
            if not isinstance(name, str) or not name:
                raise IncompletePartitionError(
                    "persisted array has no declared component"
                )
            component = name
        else:
            arrays = output.get("arrays", {})
            if not isinstance(arrays, dict) or len(arrays) != 1:
                raise ValueError("component is required for this result")
            component = str(next(iter(arrays)))
    if kind == "array":
        if component != output.get("name"):
            raise ValueError(f"result has no declared array component {component!r}")
        declared_dtype = output.get("dtype")
        declared_chunks = output.get("chunks")
    else:
        arrays = output.get("arrays")
        if not isinstance(arrays, dict) or component not in arrays:
            raise ValueError(f"result has no declared array component {component!r}")
        component_metadata = arrays[component]
        if not isinstance(component_metadata, dict):
            raise IncompletePartitionError(
                f"persisted component {component!r} has invalid metadata"
            )
        declared_dtype = component_metadata.get("dtype")
        declared_chunks = component_metadata.get("chunks")
    if not isinstance(declared_dtype, str):
        raise IncompletePartitionError(
            f"persisted component {component!r} has invalid dtype metadata"
        )
    try:
        expected_dtype = np.dtype(declared_dtype)
    except TypeError as exc:
        raise IncompletePartitionError(
            f"persisted component {component!r} has invalid dtype metadata"
        ) from exc
    expected_chunks: tuple[int, ...] | None = None
    if declared_chunks is not None:
        if (
            not isinstance(declared_chunks, list)
            or len(declared_chunks) != len(declared_shape)
            or any(
                not isinstance(size, int) or isinstance(size, bool) or size < 1
                for size in declared_chunks
            )
        ):
            raise IncompletePartitionError(
                f"persisted component {component!r} has invalid chunk metadata"
            )
        expected_chunks = tuple(declared_chunks)
    if axes is None:
        axes = declared_axes
    elif tuple(axes) != declared_axes:
        raise ValueError("axes do not match the persisted result metadata")
    try:
        source = ArraySource(
            uri,
            component=component,
            axes=axes,
            content_identity=content_identity,
        )
        selection = source.select()
    except (KeyError, TypeError, ValueError) as exc:
        raise IncompletePartitionError(
            f"could not open persisted component {component!r}: {exc}"
        ) from exc
    if (
        selection.metadata.shape != declared_shape
        or np.dtype(selection.metadata.dtype) != expected_dtype
        or (
            expected_chunks is not None
            and selection.metadata.native_chunks != expected_chunks
        )
    ):
        source.close()
        raise IncompletePartitionError(
            f"persisted component {component!r} does not match its declared schema"
        )
    return source, selection
