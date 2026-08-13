"""Executable and reopenable workflow result handles."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from dask.delayed import Delayed

from neuroflow.adapters.base import AnalysisAdapter
from neuroflow.adapters.numpy import ArrayOutput, TableOutput
from neuroflow.adapters.segmentation import SegmentationOutputSchema
from neuroflow.diagnostics.plan import ExecutionPlan
from neuroflow.execution.runner import (
    build_tasks,
    execute_tasks,
    fail_output,
    finalize_output,
    initialize_output,
    manifest_uri,
    partition_identity,
)
from neuroflow.partition.base import Partition
from neuroflow.results.array import ArrayResult
from neuroflow.results.base import ResultStatus
from neuroflow.results.table import TableResult
from neuroflow.results.verification import VerificationReport
from neuroflow.selection.query import Selection
from neuroflow.source.base import NWBSource
from neuroflow.storage.base import join_uri, read_json
from neuroflow.storage.manifest import PartitionManifest
from neuroflow.storage.parquet import ParquetOutput
from neuroflow.storage.segmentation import SegmentationOutput
from neuroflow.storage.validation import validate_partition_manifest
from neuroflow.storage.zarr import ZarrOutput


@dataclass
class WorkflowResult:
    source: NWBSource
    selection: Selection
    adapter: AnalysisAdapter
    output: ZarrOutput | ParquetOutput | SegmentationOutput
    plan: ExecutionPlan
    scheduler: Literal["threads", "processes", "distributed"] = "threads"
    resume_enabled: bool = True
    max_workers: int | None = None
    memory_limit: int | str | None = None
    _tasks: tuple[Delayed, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self._tasks:
            self._tasks = build_tasks(
                selection=self.selection,
                adapter=self.adapter,
                output=self.output,
                execution_plan=self.plan,
            )

    @property
    def arrays(self) -> dict[str, ArrayResult]:
        schema = getattr(self.adapter, "output", None)
        if isinstance(schema, ArrayOutput):
            return {schema.name: ArrayResult(self.output.uri, schema.name)}
        if isinstance(schema, SegmentationOutputSchema):
            return {
                schema.labels_name: ArrayResult(self.output.uri, schema.labels_name)
            }
        return {}

    @property
    def tables(self) -> dict[str, TableResult]:
        schema = getattr(self.adapter, "output", None)
        if isinstance(schema, TableOutput):
            return {schema.name: TableResult(self.output.uri, schema.name)}
        if isinstance(schema, SegmentationOutputSchema):
            return {
                schema.objects_name: TableResult(self.output.uri, schema.objects_name)
            }
        return {}

    @property
    def provenance(self) -> dict[str, object] | None:
        return read_json(join_uri(self.output.uri, ".neuroflow", "provenance.json"))

    @property
    def status(self) -> ResultStatus:
        complete: list[str] = []
        failed: list[str] = []
        for partition in self.plan.partitions:
            partition_id = partition_identity(self.plan.workflow_id, partition)
            value = read_json(manifest_uri(self.output.uri, partition_id))
            if value is None:
                continue
            if value.get("status") == "complete":
                complete.append(partition_id)
            elif value.get("status") == "failed":
                failed.append(partition_id)
        if len(complete) == self.plan.task_count:
            state = "complete"
        elif failed and not complete:
            state = "failed"
        elif complete or failed:
            state = "partial"
        else:
            state = "planned"
        return ResultStatus(state, tuple(complete), tuple(failed))

    @property
    def failed_partitions(self) -> tuple[str, ...]:
        return self.status.failed_partitions

    def execute(self) -> WorkflowResult:
        initialize_output(
            source=self.source,
            selection=self.selection,
            adapter=self.adapter,
            output=self.output,
            execution_plan=self.plan,
            scheduler=self.scheduler,
            resume=self.resume_enabled,
            max_workers=self.max_workers,
            memory_limit=self.memory_limit,
        )
        try:
            execute_tasks(self._tasks, self.scheduler, max_workers=self.max_workers)
        except Exception as exc:
            fail_output(self.output, self.plan.workflow_id, exc)
            raise
        finalize_output(self.output, self.plan.workflow_id, self.plan.task_count)
        return self

    def resume(self) -> WorkflowResult:
        self.resume_enabled = True
        return self.execute()

    def verify(self, *, checksums: bool = True) -> VerificationReport:
        entries = tuple(
            (
                partition_identity(self.plan.workflow_id, partition),
                partition,
            )
            for partition in self.plan.partitions
        )
        return _verify_partitions(
            self.output.uri,
            self.plan.workflow_id,
            entries,
            checksums=checksums,
        )


@dataclass(frozen=True)
class PersistedResult:
    uri: str
    metadata: dict[str, object]
    provenance_record: dict[str, object]

    @property
    def arrays(self) -> dict[str, ArrayResult]:
        output = self.provenance_record.get("output", {})
        if not isinstance(output, dict):
            return {}
        name = output.get("name")
        if output.get("kind") == "array" and name:
            return {str(name): ArrayResult(self.uri, str(name))}
        arrays = output.get("arrays")
        if output.get("kind") == "segmentation" and isinstance(arrays, dict):
            return {str(key): ArrayResult(self.uri, str(key)) for key in arrays}
        return {}

    @property
    def tables(self) -> dict[str, TableResult]:
        output = self.provenance_record.get("output", {})
        if not isinstance(output, dict):
            return {}
        name = output.get("name")
        if output.get("kind") == "table" and name:
            return {str(name): TableResult(self.uri, str(name))}
        tables = output.get("tables")
        if output.get("kind") == "segmentation" and isinstance(tables, dict):
            return {str(key): TableResult(self.uri, str(key)) for key in tables}
        return {}

    @property
    def provenance(self) -> dict[str, object]:
        return self.provenance_record

    @property
    def status(self) -> ResultStatus:
        partition_plan = self.provenance_record.get("partition_plan", {})
        if not isinstance(partition_plan, dict):
            return ResultStatus("partial")
        raw_ids = partition_plan.get("partition_ids", [])
        if not isinstance(raw_ids, list):
            return ResultStatus("partial")
        expected = tuple(str(item) for item in raw_ids)
        complete: list[str] = []
        failed: list[str] = []
        for partition_id in expected:
            value = read_json(manifest_uri(self.uri, partition_id))
            if value is None:
                continue
            if value.get("status") == "complete":
                complete.append(partition_id)
            elif value.get("status") == "failed":
                failed.append(partition_id)
        if expected and len(complete) == len(expected):
            state = "complete"
        elif failed and not complete:
            state = "failed"
        elif complete or failed:
            state = "partial"
        else:
            state = (
                "running"
                if self.provenance_record.get("status") == "running"
                else "partial"
            )
        return ResultStatus(state, tuple(complete), tuple(failed))

    @property
    def failed_partitions(self) -> tuple[str, ...]:
        return self.status.failed_partitions

    def execute(self) -> PersistedResult:
        return self

    def resume(self) -> PersistedResult:
        raise RuntimeError(
            "a reopened result lacks the in-memory adapter function; rerun the "
            "original "
            "analysis definition with resume=True"
        )

    def verify(self, *, checksums: bool = True) -> VerificationReport:
        workflow_id = str(self.provenance_record.get("workflow_id", ""))
        partition_plan = self.provenance_record.get("partition_plan", {})
        if not isinstance(partition_plan, dict):
            return VerificationReport(False, (), ("missing partition plan",))
        raw = partition_plan.get("partitions", [])
        if not isinstance(raw, list):
            return VerificationReport(False, (), ("invalid partition descriptors",))
        selection_record = self.provenance_record.get("selection", {})
        output_record = self.provenance_record.get("output", {})
        selection_shape = (
            tuple(selection_record.get("shape", ()))
            if isinstance(selection_record, dict)
            else ()
        )
        output_shape = (
            tuple(output_record.get("shape", ()))
            if isinstance(output_record, dict)
            else ()
        )
        entries: list[tuple[str, Partition]] = []
        errors: list[str] = []
        for value in raw:
            if not isinstance(value, dict):
                errors.append("invalid partition descriptor")
                continue
            try:
                partition = Partition.from_dict(value)
                if not _partition_within_shape(
                    partition.read_slices, selection_shape
                ) or (
                    output_shape
                    and not _partition_within_shape(
                        partition.output_slices, output_shape
                    )
                ):
                    raise ValueError("partition slices exceed declared result bounds")
                entries.append((str(value["partition_id"]), partition))
            except (KeyError, ValueError) as exc:
                errors.append(f"invalid partition descriptor: {exc}")
        report = _verify_partitions(
            self.uri,
            workflow_id,
            tuple(entries),
            checksums=checksums,
        )
        return VerificationReport(
            report.valid and not errors,
            report.checked_partitions,
            (*errors, *report.errors),
        )


def _partition_within_shape(
    slices: tuple[slice, ...], shape: tuple[object, ...]
) -> bool:
    if len(slices) != len(shape) or not all(isinstance(size, int) for size in shape):
        return False
    return all(
        (item.start or 0) >= 0
        and item.stop is not None
        and item.stop <= size
        and item.stop >= (item.start or 0)
        for item, size in zip(slices, shape, strict=True)
    )
def _verify_partitions(
    uri: str,
    workflow_id: str,
    entries: tuple[tuple[str, Partition], ...],
    *,
    checksums: bool,
) -> VerificationReport:
    checked: list[str] = []
    errors: list[str] = []
    if not entries:
        return VerificationReport(False, (), ("result has no partition descriptors",))
    for partition_id, partition in entries:
        value = read_json(manifest_uri(uri, partition_id))
        if value is None:
            errors.append(f"missing manifest for {partition_id}")
            continue
        try:
            manifest = PartitionManifest.from_dict(value)
        except (KeyError, ValueError) as exc:
            errors.append(f"invalid manifest for {partition_id}: {exc}")
            continue
        if manifest.workflow_id != workflow_id:
            errors.append(f"workflow mismatch for {partition_id}")
            continue
        if manifest.partition_id != partition_id:
            errors.append(f"partition identity mismatch for {partition_id}")
            continue
        partition_errors = validate_partition_manifest(
            manifest, partition, output_root=uri, checksums=checksums
        )
        if partition_errors:
            errors.extend(f"{partition_id}: {error}" for error in partition_errors)
        else:
            checked.append(partition_id)
    return VerificationReport(not errors, tuple(checked), tuple(errors))
