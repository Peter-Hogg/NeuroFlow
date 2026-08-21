"""Device selection and reporting for Cellpose execution.

Kept separate from the adapter so that a benchmark, a direct-Cellpose reference
run and the NeuroFlow-mediated path all resolve the device through one function.
Equivalence claims are only meaningful when both sides ran on the same device,
and duplicating the selection logic is the easiest way to break that.
"""

from __future__ import annotations

import importlib
import importlib.metadata
from dataclasses import dataclass
from typing import Any, Literal

DeviceChoice = Literal["auto", "cpu", "cuda"]
DEVICE_CHOICES: tuple[DeviceChoice, ...] = ("auto", "cpu", "cuda")


@dataclass(frozen=True)
class CellposeDevice:
    """A resolved Cellpose execution device and the evidence behind it."""

    requested: str
    selected: Literal["cpu", "cuda"]
    cuda_available: bool
    torch_version: str | None
    cuda_version: str | None
    gpu_name: str | None
    gpu_total_memory_bytes: int | None

    @property
    def gpu(self) -> bool:
        """The ``gpu=`` flag Cellpose expects for this device."""
        return self.selected == "cuda"

    def to_dict(self) -> dict[str, object]:
        return {
            "requested_device": self.requested,
            "selected_device": self.selected,
            "cuda_available": self.cuda_available,
            "torch_version": self.torch_version,
            "cuda_version": self.cuda_version,
            "gpu_name": self.gpu_name,
            "gpu_total_memory_bytes": self.gpu_total_memory_bytes,
            "note": (
                "GPU memory is a separate resource from the NeuroFlow host "
                "memory budget and is never counted against memory_limit"
            ),
        }


def _torch() -> Any | None:
    try:
        return importlib.import_module("torch")
    except ImportError:
        return None


def resolve_cellpose_device(requested: DeviceChoice | str = "auto") -> CellposeDevice:
    """Resolve ``auto``/``cpu``/``cuda`` into a concrete device.

    ``auto`` selects CUDA when PyTorch reports a usable CUDA device and CPU
    otherwise, so the same command works on a workstation and on a laptop.
    An explicit ``cuda`` request fails loudly rather than silently degrading:
    a benchmark that quietly ran on CPU would misattribute its timings.
    """
    if requested not in DEVICE_CHOICES:
        raise ValueError(
            f"cellpose device must be one of {', '.join(DEVICE_CHOICES)}; "
            f"got {requested!r}"
        )
    torch = _torch()
    torch_version = str(getattr(torch, "__version__", None)) if torch else None
    cuda_available = bool(torch.cuda.is_available()) if torch is not None else False
    cuda_version: str | None = None
    gpu_name: str | None = None
    gpu_total_memory_bytes: int | None = None
    if torch is not None and cuda_available:
        cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
        cuda_version = str(cuda_version) if cuda_version else None
        try:
            gpu_name = str(torch.cuda.get_device_name(0))
            gpu_total_memory_bytes = int(
                torch.cuda.get_device_properties(0).total_memory
            )
        except (RuntimeError, AssertionError, IndexError):
            # A CUDA build with no reachable device: treat it as unavailable
            # rather than reporting a device that cannot be used.
            cuda_available = False
            gpu_name = None
            gpu_total_memory_bytes = None
    if requested == "cuda" and not cuda_available:
        raise RuntimeError(
            "cellpose device 'cuda' was requested but PyTorch reports no usable "
            f"CUDA device (torch={torch_version or 'not installed'}). Pass "
            "'cpu' or 'auto' to run on CPU."
        )
    selected: Literal["cpu", "cuda"] = (
        "cuda" if (requested == "cuda" or (requested == "auto" and cuda_available))
        else "cpu"
    )
    return CellposeDevice(
        requested=str(requested),
        selected=selected,
        cuda_available=cuda_available,
        torch_version=torch_version,
        cuda_version=cuda_version if selected == "cuda" else cuda_version,
        gpu_name=gpu_name,
        gpu_total_memory_bytes=gpu_total_memory_bytes,
    )


def cellpose_version() -> str:
    return importlib.metadata.version("cellpose")
