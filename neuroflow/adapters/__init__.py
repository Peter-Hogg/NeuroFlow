"""Contracts for external scientific libraries."""

from neuroflow.adapters.base import (
    AdapterRequirements,
    AnalysisAdapter,
    BoundarySummary,
    LoadedPartition,
    MergeableAdapter,
    MergeContext,
    MergeManifest,
    TaskContext,
)
from neuroflow.adapters.numpy import (
    ArrayOutput,
    ExpressionAdapter,
    FunctionAdapter,
    TableOutput,
)
from neuroflow.adapters.segmentation import (
    SegmentationFunctionAdapter,
    SegmentationOutputSchema,
    SegmentationTaskOutput,
)

__all__ = [
    "AdapterRequirements",
    "AnalysisAdapter",
    "BoundarySummary",
    "ArrayOutput",
    "ExpressionAdapter",
    "FunctionAdapter",
    "LoadedPartition",
    "MergeableAdapter",
    "MergeContext",
    "MergeManifest",
    "TableOutput",
    "TaskContext",
    "SegmentationFunctionAdapter",
    "SegmentationOutputSchema",
    "SegmentationTaskOutput",
]
