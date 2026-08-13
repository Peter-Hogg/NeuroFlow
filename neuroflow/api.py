"""Top-level orchestration entry points.

Selection and planning are metadata-only. Work begins only through ``execute()``
or an explicit ``execute=True`` call.
"""

from __future__ import annotations

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

from neuroflow.exceptions import IncompletePartitionError, OutputConflictError
from neuroflow.execution.graph import build_plan
from neuroflow.results.workflow import PersistedResult, WorkflowResult
from neuroflow.source.base import SourceSpec
from neuroflow.source.dandi import DandiNWBSource
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
    return LocalNWBZarrSource(
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
) -> WorkflowResult:
    """Construct a lazy workflow; execution is always explicitly requested."""
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
    result = WorkflowResult(
        source=source,
        selection=selection,
        adapter=adapter,
        output=output,
        plan=execution_plan,
        scheduler=scheduler,
        resume_enabled=resume,
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
