"""Adapter contracts and task context types."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from neuroflow.execution.resources import ResourceSpec


@dataclass(frozen=True)
class LoadedPartition:
    data: object
    read_slices: tuple[slice, ...]
    output_slices: tuple[slice, ...]
    trim_slices: tuple[slice, ...]
    timestamps: object | None = None


@dataclass(frozen=True)
class AdapterRequirements:
    input_kinds: tuple[str, ...]
    splittable_axes: tuple[str, ...]
    requires_overlap: Mapping[str, int | str]
    output_kinds: tuple[str, ...]
    resources: ResourceSpec = field(default_factory=ResourceSpec)
    deterministic: bool = True
    requires_local_path: bool = False


@dataclass(frozen=True)
class TaskContext:
    partition_id: str
    parameters: Mapping[str, object] = field(default_factory=dict)
    random_seed: int | None = None


@dataclass(frozen=True)
class BoundarySummary:
    partition_id: str
    payload: Mapping[str, object]


@dataclass(frozen=True)
class MergeContext:
    workflow_id: str
    parameters: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class MergeManifest:
    workflow_id: str
    merged_partitions: tuple[str, ...]
    outputs: Mapping[str, str]


class AnalysisAdapter(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    def requirements(self) -> AdapterRequirements: ...

    def prepare(self, partition: LoadedPartition, context: TaskContext) -> object: ...

    def run(self, prepared: object, context: TaskContext) -> object: ...

    def persist(
        self, output: object, writer: object, context: TaskContext
    ) -> object: ...


class MergeableAdapter(AnalysisAdapter, Protocol):
    def boundary_summary(self, manifest: object) -> BoundarySummary: ...

    def merge(
        self,
        neighbors: Sequence[BoundarySummary],
        writer: object,
        context: MergeContext,
    ) -> MergeManifest: ...
