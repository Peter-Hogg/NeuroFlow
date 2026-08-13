"""Top-level orchestration entry points.

Selection and planning are metadata-only. Work begins only through ``execute()``
or an explicit ``execute=True`` call.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from neuroflow.adapters.base import AnalysisAdapter
    from neuroflow.diagnostics.plan import ExecutionPlan
    from neuroflow.partition.base import PartitionPlan
    from neuroflow.selection.query import Selection
    from neuroflow.source.base import NWBSource, SourceSpec
    from neuroflow.storage.base import OutputSpec

from neuroflow.exceptions import (
    IncompletePartitionError,
    OutputConflictError,
    UnsupportedBackendError,
)
from neuroflow.execution.graph import build_plan
from neuroflow.execution.resources import parse_bytes
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
        budget = parse_bytes(memory_limit)
        declared = execution_plan.resources.memory
        per_worker = max(
            execution_plan.memory_per_task,
            parse_bytes(declared) if declared is not None else 0,
        )
        if per_worker > budget:
            raise ValueError(
                f"one task requires an estimated {per_worker} bytes, exceeding "
                f"the {budget}-byte workflow memory limit"
            )
        safe_workers = max(1, budget // max(per_worker, 1))
        if max_workers is not None and max_workers > safe_workers:
            raise ValueError(
                f"max_workers={max_workers} exceeds the memory-safe limit "
                f"of {safe_workers}"
            )
        available_workers = max(1, os.cpu_count() or 1)
        max_workers = int(
            min(max_workers or available_workers, safe_workers, available_workers)
        )
    result = WorkflowResult(
        source=source,
        selection=selection,
        adapter=adapter,
        output=output,
        plan=execution_plan,
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
) -> tuple[ArraySource, Selection]:
    """Open a persisted NeuroFlow array as a composable workflow input."""
    result = open_result(uri)
    output = result.provenance.get("output", {})
    if not isinstance(output, dict) or output.get("kind") not in {
        "array",
        "segmentation",
    }:
        raise TypeError("result does not contain a composable array")
    if component is None:
        if output.get("kind") == "array":
            component = str(output["name"])
        else:
            arrays = output.get("arrays", {})
            if not isinstance(arrays, dict) or len(arrays) != 1:
                raise ValueError("component is required for this result")
            component = str(next(iter(arrays)))
    if axes is None:
        raw_axes = output.get("axes")
        if not isinstance(raw_axes, list):
            raise ValueError("result has no axis metadata; pass axes explicitly")
        axes = tuple(str(axis) for axis in raw_axes)
    source = ArraySource(uri, component=component, axes=axes)
    return source, source.select()
