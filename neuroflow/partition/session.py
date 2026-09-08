"""Asset- and session-level processing partitions.

Both plans process the whole selection as a single partition. Asset filtering
and session grouping are not implemented; rather than accept parameters the
planner would silently ignore, the plans take none.
"""

from dataclasses import dataclass

from neuroflow.partition.base import Partition, ValidationReport
from neuroflow.selection.query import Selection


def _whole(selection: Selection, key: str) -> tuple[Partition, ...]:
    slices = tuple(slice(0, size) for size in selection.metadata.shape)
    return (Partition(key, slices, slices, slices, (0,)),)


@dataclass(frozen=True)
class AssetPlan:
    def validate(self, selection: Selection) -> ValidationReport:
        return ValidationReport(True)

    def build(self, selection: Selection) -> tuple[Partition, ...]:
        return _whole(selection, "asset-00000000")


@dataclass(frozen=True)
class SessionPlan:
    def validate(self, selection: Selection) -> ValidationReport:
        return ValidationReport(True)

    def build(self, selection: Selection) -> tuple[Partition, ...]:
        return _whole(selection, "session-00000000")
