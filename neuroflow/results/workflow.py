"""Executable and reopenable workflow result handles."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, cast

from dask.delayed import Delayed

from neuroflow.adapters.base import AnalysisAdapter
from neuroflow.adapters.numpy import ArrayOutput, ExpressionAdapter, TableOutput
from neuroflow.adapters.segmentation import SegmentationOutputSchema
from neuroflow.diagnostics.plan import ExecutionPlan
from neuroflow.exceptions import IncompletePartitionError
from neuroflow.execution.runner import (
    _schema_output_kinds,
    build_tasks,
    execute_tasks,
    fail_output,
    finalize_output,
    initialize_output,
    manifest_uri,
    partition_identity,
)
from neuroflow.execution.stages import (
    build_reduction_stage_plans,
    execute_reduction_stages,
    verify_reduction_stages,
)
from neuroflow.partition.base import Partition
from neuroflow.provenance.hashing import stable_hash
from neuroflow.results.array import ArrayResult
from neuroflow.results.base import ResultStatus
from neuroflow.results.table import TableResult
from neuroflow.results.verification import VerificationReport
from neuroflow.selection.query import Selection
from neuroflow.source.base import NWBSource
from neuroflow.storage.base import join_uri, read_json, write_json_atomic
from neuroflow.storage.manifest import PartitionManifest
from neuroflow.storage.parquet import ParquetOutput
from neuroflow.storage.segmentation import SegmentationOutput
from neuroflow.storage.validation import (
    OutputStorageKind,
    output_component_kinds,
    validate_partition_manifest,
)
from neuroflow.storage.zarr import ZarrOutput

if TYPE_CHECKING:
    from neuroflow.workflow import WorkflowSpec


@dataclass
class WorkflowResult:
    source: NWBSource
    selection: Selection
    adapter: AnalysisAdapter
    output: ZarrOutput | ParquetOutput | SegmentationOutput
    plan: ExecutionPlan
    partition: object | None = None
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
        stage_plans = (
            build_reduction_stage_plans(
                self.selection,
                self.adapter.expression,
                memory_limit=self.memory_limit,
            )
            if isinstance(self.adapter, ExpressionAdapter)
            else ()
        )
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
            stages=tuple(item.to_dict() for item in stage_plans),
        )
        try:
            if isinstance(self.adapter, ExpressionAdapter):
                stage_execution = execute_reduction_stages(
                    selection=self.selection,
                    staged_values=self.adapter.staged_values,
                    output_uri=self.output.uri,
                    workflow_id=self.plan.workflow_id,
                    plans=stage_plans,
                )
                provenance_uri = join_uri(
                    self.output.uri, ".neuroflow", "provenance.json"
                )
                provenance = read_json(provenance_uri)
                if provenance is not None:
                    provenance["stage_execution"] = list(stage_execution)
                    write_json_atomic(provenance_uri, provenance)
            task_results = execute_tasks(
                self._tasks, self.scheduler, max_workers=self.max_workers
            )
            io_stats = getattr(self.source, "io_stats", None)
            raw_source_metrics = io_stats() if callable(io_stats) else {}
            source_metrics = (
                cast(Mapping[str, object], raw_source_metrics)
                if isinstance(raw_source_metrics, Mapping)
                else {}
            )
            finalize_output(
                self.output,
                self.plan.workflow_id,
                self.plan.partitions,
                task_results=task_results,
                source_metrics=source_metrics,
            )
        except Exception as exc:
            fail_output(self.output, self.plan.workflow_id, exc)
            raise
        return self

    def resume(self) -> WorkflowResult:
        self.resume_enabled = True
        return self.execute()

    def to_spec(self) -> WorkflowSpec:
        """Return a validated portable specification for a safe workflow."""
        from neuroflow.workflow import WorkflowSpec

        return WorkflowSpec.from_workflow(self)

    def verify(self, *, checksums: bool = True) -> VerificationReport:
        entries = tuple(
            (
                partition_identity(self.plan.workflow_id, partition),
                partition,
            )
            for partition in self.plan.partitions
        )
        report = _verify_partitions(
            self.output.uri,
            self.plan.workflow_id,
            entries,
            output_kinds=_schema_output_kinds(getattr(self.adapter, "output", None)),
            checksums=checksums,
        )
        provenance = self.provenance or {}
        stage_errors = verify_reduction_stages(self.output.uri, provenance)
        return VerificationReport(
            report.valid and not stage_errors,
            report.checked_partitions,
            (*stage_errors, *report.errors),
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
        try:
            if not isinstance(output_record, dict):
                raise ValueError("output schema is not a mapping")
            output_kinds = output_component_kinds(output_record)
        except ValueError as exc:
            output_kinds = {}
            errors.append(f"invalid output schema: {exc}")
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
            output_kinds=output_kinds,
            checksums=checksums,
        )
        stage_errors = verify_reduction_stages(self.uri, self.provenance_record)
        return VerificationReport(
            report.valid and not errors and not stage_errors,
            report.checked_partitions,
            (*errors, *stage_errors, *report.errors),
        )

    def array_source_identity(self, *, verify_checksums: bool = True) -> str:
        """Return a canonical identity for a complete, composable result.

        The identity binds downstream workflows to the verified partition
        descriptors and manifest checksums, rather than only to the upstream
        workflow declaration.  ``verify_checksums=False`` is a trusted fast
        path for a result that has just completed in this process; structural
        and manifest checks still run, but output bytes are not reread.
        """
        workflow_id = self.provenance_record.get("workflow_id")
        result_workflow_id = self.metadata.get("workflow_id")
        plan = self.provenance_record.get("partition_plan")
        output = self.provenance_record.get("output")
        result_output = self.metadata.get("output")
        errors: list[str] = []
        if not isinstance(workflow_id, str) or not workflow_id:
            errors.append("missing workflow identity")
        if result_workflow_id != workflow_id:
            errors.append("result and provenance identities do not match")
        if self.metadata.get("status") != "complete":
            errors.append("result metadata is not complete")
        if self.provenance_record.get("status") != "complete":
            errors.append("result provenance is not complete")
        if not isinstance(output, dict) or result_output != output:
            errors.append("result and provenance output schemas do not match")
        if not isinstance(plan, dict):
            errors.append("missing partition plan")
            plan = {}

        raw_ids = plan.get("partition_ids")
        raw_partitions = plan.get("partitions")
        task_count = plan.get("task_count")
        result_task_count = self.metadata.get("task_count")
        if not isinstance(raw_ids, list) or not raw_ids:
            errors.append("partition plan has no partition identities")
            raw_ids = []
        if not isinstance(raw_partitions, list) or not raw_partitions:
            errors.append("partition plan has no partition descriptors")
            raw_partitions = []
        if (
            not isinstance(task_count, int)
            or isinstance(task_count, bool)
            or task_count < 1
        ):
            errors.append("partition plan has an invalid task count")
        elif (
            result_task_count != task_count
            or task_count != len(raw_ids)
            or task_count != len(raw_partitions)
        ):
            errors.append("partition counts do not match")

        entries: list[tuple[str, Partition]] = []
        descriptor_ids: list[str] = []
        for value in raw_partitions:
            if not isinstance(value, dict):
                errors.append("invalid partition descriptor")
                continue
            try:
                partition_id = str(value["partition_id"])
                partition = Partition.from_dict(value)
            except (KeyError, ValueError) as exc:
                errors.append(f"invalid partition descriptor: {exc}")
                continue
            descriptor_ids.append(partition_id)
            entries.append((partition_id, partition))
        expected_ids = [str(value) for value in raw_ids]
        if expected_ids != descriptor_ids or len(set(expected_ids)) != len(
            expected_ids
        ):
            errors.append("partition identities and descriptors do not match")

        if errors:
            raise IncompletePartitionError("; ".join(errors))

        report = self.verify(checksums=verify_checksums)
        if not report.valid:
            detail = "; ".join(report.errors[:3])
            raise IncompletePartitionError(
                "persisted array verification failed"
                + (f": {detail}" if detail else "")
            )

        canonical_manifests: list[dict[str, object]] = []
        for partition_id, partition in entries:
            value = read_json(manifest_uri(self.uri, partition_id))
            if value is None:
                raise IncompletePartitionError(f"missing manifest for {partition_id}")
            try:
                manifest = PartitionManifest.from_dict(value)
            except (KeyError, ValueError) as exc:
                raise IncompletePartitionError(
                    f"invalid manifest for {partition_id}: {exc}"
                ) from exc
            if (
                manifest.status != "complete"
                or manifest.workflow_id != workflow_id
                or manifest.partition_id != partition_id
            ):
                raise IncompletePartitionError(
                    f"manifest identity or status mismatch for {partition_id}"
                )
            if (
                not manifest.outputs
                or set(manifest.outputs) != set(manifest.checksums)
                or not set(manifest.sizes).issubset(manifest.outputs)
            ):
                raise IncompletePartitionError(
                    f"manifest outputs and checksums do not match for {partition_id}"
                )
            if any(
                len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                for digest in manifest.checksums.values()
            ):
                raise IncompletePartitionError(
                    f"manifest has an invalid checksum for {partition_id}"
                )
            canonical_manifests.append(
                {
                    "partition_id": partition_id,
                    "partition": partition.to_dict(),
                    "schema_version": manifest.schema_version,
                    "checksums": dict(sorted(manifest.checksums.items())),
                    "sizes": dict(sorted(manifest.sizes.items())),
                }
            )
        canonical_manifests.sort(key=lambda item: str(item["partition_id"]))
        return stable_hash(
            {
                "schema_version": "1",
                "workflow_id": workflow_id,
                "output": output,
                "manifests": canonical_manifests,
            }
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
    output_kinds: Mapping[str, OutputStorageKind],
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
            manifest,
            partition,
            output_root=uri,
            output_kinds=output_kinds,
            checksums=checksums,
        )
        if partition_errors:
            errors.extend(f"{partition_id}: {error}" for error in partition_errors)
        else:
            checked.append(partition_id)
    return VerificationReport(not errors, tuple(checked), tuple(errors))
