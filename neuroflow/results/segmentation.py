"""Composite lazy segmentation results for optional integrations."""

from collections.abc import Mapping
from dataclasses import dataclass

from neuroflow.results.array import ArrayResult
from neuroflow.results.table import TableResult


@dataclass(frozen=True)
class SegmentationResult:
    labels: ArrayResult
    objects: TableResult
    provenance: Mapping[str, object]
    masks: object | None = None
