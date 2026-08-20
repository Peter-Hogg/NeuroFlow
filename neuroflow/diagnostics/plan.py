"""Immutable description of a lazy workflow plan."""

from dataclasses import asdict, dataclass

from neuroflow.execution.resources import ResourceSpec
from neuroflow.partition.base import Partition


@dataclass(frozen=True)
class ExecutionPlan:
    workflow_id: str
    source_size: int | None
    selected_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    output_axes: tuple[str, ...]
    dtype: str
    native_chunks: tuple[int, ...] | None
    processing_partition_shape: tuple[int, ...]
    overlap: tuple[int, ...]
    task_count: int
    memory_per_task: int
    read_amplification: float
    maximum_logical_partition_shape: tuple[int, ...]
    estimated_logical_bytes_read: int
    estimated_source_chunks_touched: int | None
    estimated_total_bytes_read: int | None
    bounded: bool
    bounded_reasons: tuple[str, ...] = ()
    stages: tuple[dict[str, object], ...] = ()
    expected_output_size: int | None = None
    warnings: tuple[str, ...] = ()
    partitions: tuple[Partition, ...] = ()
    resources: ResourceSpec = ResourceSpec()

    def to_dict(self) -> dict[str, object]:
        """Return a machine-readable dry-run report with estimate status."""

        def measurement(value: object, *, note: str | None = None) -> dict[str, object]:
            result: dict[str, object] = {
                "status": "unknown" if value is None else "estimated",
                "value": value,
            }
            if note is not None:
                result["note"] = note
            return result

        return {
            "schema_version": "1",
            "workflow_id": self.workflow_id,
            "source": {
                "size_bytes": measurement(
                    self.source_size,
                    note="archive object size when metadata provides it",
                )
            },
            "selection": {
                "shape": list(self.selected_shape),
                "dtype": self.dtype,
                "physical_chunks": (
                    list(self.native_chunks) if self.native_chunks is not None else None
                ),
            },
            "partitioning": {
                "estimated_partitions": measurement(self.task_count),
                "maximum_logical_partition_shape": list(
                    self.maximum_logical_partition_shape
                ),
                "estimated_bytes_per_partition": measurement(self.memory_per_task),
                "estimated_source_chunks_touched": measurement(
                    self.estimated_source_chunks_touched,
                    note="counts repeated touches across logical partitions",
                ),
                "estimated_logical_bytes_read": measurement(
                    self.estimated_logical_bytes_read
                ),
                "estimated_total_bytes_read": measurement(
                    self.estimated_total_bytes_read,
                    note=(
                        "uncompressed physical-chunk bytes; compression, caches, and "
                        "HTTP transport can change actual transfer"
                    ),
                ),
                "read_amplification": measurement(self.read_amplification),
            },
            "output": {
                "shape": list(self.output_shape),
                "axes": list(self.output_axes),
                "estimated_size_bytes": measurement(self.expected_output_size),
            },
            "resources": asdict(self.resources),
            "bounded": {
                "status": "estimated",
                "value": self.bounded,
                "reasons": list(self.bounded_reasons),
            },
            "stages": list(self.stages),
            "warnings": list(self.warnings),
        }

    def summary(self) -> str:
        output = (
            f"{self.expected_output_size} bytes"
            if self.expected_output_size is not None
            else "unknown"
        )
        resource = (
            f"cpu={self.resources.cpu}, gpu={self.resources.gpu}, "
            f"memory={self.resources.memory or 'unspecified'}"
        )
        lines = [
            f"workflow: {self.workflow_id}",
            f"source size: {self.source_size or 'unknown'} bytes",
            f"selection: shape={self.selected_shape}, dtype={self.dtype}",
            f"output: shape={self.output_shape}, axes={self.output_axes}",
            "chunks: "
            f"native={self.native_chunks}, "
            f"processing={self.processing_partition_shape}",
            f"tasks: {self.task_count}, memory/task={self.memory_per_task} bytes",
            f"bounded: {self.bounded}",
            f"read amplification: {self.read_amplification:.2f}x",
            f"expected output: {output}",
            f"resources: {resource}",
        ]
        lines.extend(f"warning: {warning}" for warning in self.warnings)
        lines.extend(
            f"stage: {stage.get('operation', 'unknown')} "
            f"({stage.get('task_count', 'unknown')} bounded partials)"
            for stage in self.stages
        )
        return "\n".join(lines)
