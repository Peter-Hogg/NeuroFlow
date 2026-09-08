"""Contracts for external scientific libraries."""

from neuroflow.adapters.base import (
    AdapterRequirements,
    AnalysisAdapter,
    LoadedPartition,
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
    "ArrayOutput",
    "ExpressionAdapter",
    "FunctionAdapter",
    "LoadedPartition",
    "TableOutput",
    "TaskContext",
    "SegmentationFunctionAdapter",
    "SegmentationOutputSchema",
    "SegmentationTaskOutput",
]
