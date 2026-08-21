"""Optional Cellpose integration for NeuroFlow."""

from neuroflow_cellpose.adapter import (
    MEASURED_MODEL_HOST_RESERVE_BYTES,
    CellposeAdapter,
)
from neuroflow_cellpose.device import (
    DEVICE_CHOICES,
    CellposeDevice,
    cellpose_version,
    resolve_cellpose_device,
)

__all__ = [
    "DEVICE_CHOICES",
    "MEASURED_MODEL_HOST_RESERVE_BYTES",
    "CellposeAdapter",
    "CellposeDevice",
    "cellpose_version",
    "resolve_cellpose_device",
]
