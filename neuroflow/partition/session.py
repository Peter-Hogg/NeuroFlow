"""Asset- and session-level processing partitions."""

from collections.abc import Mapping
from dataclasses import dataclass, field

from neuroflow.partition.base import Partition, ValidationReport
from neuroflow.selection.query import Selection


def _whole(selection: Selection, key: str) -> tuple[Partition, ...]:
    slices = tuple(slice(0, size) for size in selection.metadata.shape)
    return (Partition(key, slices, slices, slices, (0,)),)


@dataclass(frozen=True)
class AssetPlan:
    filter: Mapping[str, object] = field(default_factory=dict)

    def validate(self, selection: Selection) -> ValidationReport:
        return ValidationReport(True)

    def build(self, selection: Selection) -> tuple[Partition, ...]:
        return _whole(selection, "asset-00000000")


@dataclass(frozen=True)
class SessionPlan:
    group_by: tuple[str, ...] = ("subject_id", "session_id")

    def validate(self, selection: Selection) -> ValidationReport:
        return ValidationReport(True)

    def build(self, selection: Selection) -> tuple[Partition, ...]:
        return _whole(selection, "session-00000000")
