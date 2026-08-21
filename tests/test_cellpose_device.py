"""Device selection for Cellpose execution.

Both the NeuroFlow-mediated run and the direct-Cellpose reference must resolve
their device through one function, and the choice must be recorded, because a
software-equivalence claim is only meaningful for a stated execution path.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from neuroflow_cellpose import CellposeAdapter, resolve_cellpose_device
from neuroflow_cellpose import device as device_module


def _fake_torch(
    *,
    available: bool,
    name: str = "Fake GPU",
    total_memory: int = 8 * 1024**3,
    cuda_version: str = "13.0",
    version: str = "9.9.9+cu130",
) -> SimpleNamespace:
    return SimpleNamespace(
        __version__=version,
        version=SimpleNamespace(cuda=cuda_version),
        cuda=SimpleNamespace(
            is_available=lambda: available,
            get_device_name=lambda index: name,
            get_device_properties=lambda index: SimpleNamespace(
                total_memory=total_memory
            ),
        ),
    )


def test_auto_selects_cuda_when_a_device_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        device_module, "_torch", lambda: _fake_torch(available=True)
    )

    device = resolve_cellpose_device("auto")

    assert device.selected == "cuda"
    assert device.gpu is True
    assert device.gpu_name == "Fake GPU"
    assert device.cuda_version == "13.0"
    assert device.torch_version == "9.9.9+cu130"
    assert device.gpu_total_memory_bytes == 8 * 1024**3


def test_auto_falls_back_to_cpu_without_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        device_module, "_torch", lambda: _fake_torch(available=False)
    )

    device = resolve_cellpose_device("auto")

    assert device.selected == "cpu"
    assert device.gpu is False
    assert device.cuda_available is False
    assert device.gpu_name is None


def test_auto_falls_back_to_cpu_when_torch_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(device_module, "_torch", lambda: None)

    device = resolve_cellpose_device("auto")

    assert device.selected == "cpu"
    assert device.torch_version is None


def test_explicit_cpu_is_honoured_even_with_a_gpu_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CPU must stay fully supported and must not be silently upgraded."""
    monkeypatch.setattr(
        device_module, "_torch", lambda: _fake_torch(available=True)
    )

    device = resolve_cellpose_device("cpu")

    assert device.selected == "cpu"
    assert device.gpu is False
    # Availability is still reported, so the record shows a GPU was declined.
    assert device.cuda_available is True


def test_explicit_cuda_fails_loudly_instead_of_degrading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A silent CPU fallback would misattribute benchmark timings."""
    monkeypatch.setattr(
        device_module, "_torch", lambda: _fake_torch(available=False)
    )

    with pytest.raises(RuntimeError, match="no usable"):
        resolve_cellpose_device("cuda")


def test_unreachable_cuda_device_is_treated_as_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A CUDA-enabled build with no working device must not be reported usable."""

    def broken() -> SimpleNamespace:
        torch = _fake_torch(available=True)

        def raise_error(index: int) -> str:
            raise RuntimeError("no CUDA-capable device is detected")

        torch.cuda.get_device_name = raise_error
        return torch

    monkeypatch.setattr(device_module, "_torch", broken)

    device = resolve_cellpose_device("auto")

    assert device.selected == "cpu"
    assert device.cuda_available is False


def test_unknown_device_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="cellpose device must be one of"):
        resolve_cellpose_device("mps")


def test_report_separates_gpu_memory_from_the_host_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        device_module, "_torch", lambda: _fake_torch(available=True)
    )

    report = resolve_cellpose_device("auto").to_dict()

    assert report["gpu_total_memory_bytes"] == 8 * 1024**3
    # VRAM must never be presented as host memory available to memory_limit.
    assert "never counted against memory_limit" in str(report["note"])


def test_report_records_requested_and_selected_device_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        device_module, "_torch", lambda: _fake_torch(available=False)
    )

    report = resolve_cellpose_device("auto").to_dict()

    # 'auto' resolving to CPU must be distinguishable from an explicit CPU run.
    assert report["requested_device"] == "auto"
    assert report["selected_device"] == "cpu"


def test_gpu_execution_lowers_the_declared_host_reserve() -> None:
    """Moving weights to VRAM is what makes a 2 GiB host target feasible."""
    cpu = CellposeAdapter(pretrained_model="cpsam", gpu=False)
    gpu = CellposeAdapter(pretrained_model="cpsam", gpu=True)

    assert gpu.external_memory_reserve_bytes() < cpu.external_memory_reserve_bytes()
    assert gpu.requirements().resources.gpu == 1
    assert cpu.requirements().resources.gpu == 0
