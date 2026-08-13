"""Composite lazy trace results for optional integrations."""

from collections.abc import Mapping
from dataclasses import dataclass

from neuroflow.results.array import ArrayResult
from neuroflow.results.table import TableResult


@dataclass(frozen=True)
class TraceResult:
    traces: ArrayResult
    cells: TableResult
    provenance: Mapping[str, object]
    timestamps: ArrayResult | object | None = None
