"""Public API for NeuroFlow."""

from neuroflow._version import __version__
from neuroflow.adapters import FunctionAdapter
from neuroflow.api import open_array, open_dandi, open_result, open_source, plan, run
from neuroflow.array import NeuroArray, load
from neuroflow.exceptions import (
    AdapterCompatibilityError,
    AmbiguousSelectionError,
    IncompletePartitionError,
    NeuroFlowError,
    ObjectNotFoundError,
    OutputConflictError,
    PartitionValidationError,
    ProvenanceMismatchError,
    SourceResolutionError,
    UnsupportedBackendError,
    WorkflowSpecError,
)
from neuroflow.partition import SpatialTilePlan, TimeWindowPlan
from neuroflow.selection import NWBQuery
from neuroflow.storage import ParquetOutput, SegmentationOutput, ZarrOutput
from neuroflow.validation import SegmentationMetrics, compare_segmentations
from neuroflow.workflow import WorkflowSpec, reproduce

__all__ = [
    "AdapterCompatibilityError",
    "AmbiguousSelectionError",
    "FunctionAdapter",
    "IncompletePartitionError",
    "NeuroFlowError",
    "NeuroArray",
    "NWBQuery",
    "ObjectNotFoundError",
    "OutputConflictError",
    "ParquetOutput",
    "PartitionValidationError",
    "ProvenanceMismatchError",
    "SegmentationMetrics",
    "SegmentationOutput",
    "SourceResolutionError",
    "SpatialTilePlan",
    "TimeWindowPlan",
    "UnsupportedBackendError",
    "WorkflowSpec",
    "WorkflowSpecError",
    "ZarrOutput",
    "__version__",
    "compare_segmentations",
    "load",
    "open_array",
    "open_dandi",
    "open_result",
    "open_source",
    "plan",
    "run",
    "reproduce",
]
