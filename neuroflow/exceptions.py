"""Domain exceptions exposed by NeuroFlow."""


class NeuroFlowError(Exception):
    """Base class for framework errors."""


class SourceResolutionError(NeuroFlowError):
    """A source could not be resolved unambiguously."""


class UnsupportedBackendError(NeuroFlowError):
    """The requested storage backend is unsupported."""


class AmbiguousSelectionError(NeuroFlowError):
    """A semantic query matched more than one NWB object."""


class ObjectNotFoundError(NeuroFlowError):
    """A semantic query matched no NWB object."""


class PartitionValidationError(NeuroFlowError):
    """A partition plan is invalid for the selected data."""


class AdapterCompatibilityError(NeuroFlowError):
    """An adapter cannot consume the requested partitions."""


class OutputConflictError(NeuroFlowError):
    """An output already exists and cannot safely be reused."""


class IncompletePartitionError(NeuroFlowError):
    """A partition lacks a valid atomic completion manifest."""


class ProvenanceMismatchError(NeuroFlowError):
    """Stored provenance does not match the requested workflow."""


class WorkflowSpecError(NeuroFlowError, ValueError):
    """A portable workflow file is invalid or unsupported."""
