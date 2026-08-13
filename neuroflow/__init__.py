"""Public API for NeuroFlow."""

from neuroflow._version import __version__
from neuroflow.api import open_array, open_result, open_source, plan, run
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
)

__all__ = [
    "AdapterCompatibilityError",
    "AmbiguousSelectionError",
    "IncompletePartitionError",
    "NeuroFlowError",
    "NeuroArray",
    "ObjectNotFoundError",
    "OutputConflictError",
    "PartitionValidationError",
    "ProvenanceMismatchError",
    "SourceResolutionError",
    "UnsupportedBackendError",
    "__version__",
    "open_result",
    "open_array",
    "open_source",
    "plan",
    "run",
    "load",
]
