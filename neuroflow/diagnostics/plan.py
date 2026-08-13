"""Immutable description of a lazy workflow plan."""

from dataclasses import dataclass

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
    expected_output_size: int | None = None
    warnings: tuple[str, ...] = ()
    partitions: tuple[Partition, ...] = ()
    resources: ResourceSpec = ResourceSpec()

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
            f"read amplification: {self.read_amplification:.2f}x",
            f"expected output: {output}",
            f"resources: {resource}",
        ]
        lines.extend(f"warning: {warning}" for warning in self.warnings)
        return "\n".join(lines)
