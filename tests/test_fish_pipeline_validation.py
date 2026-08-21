from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

import neuroflow
from benchmarks import benchmark_fish_pipeline
from neuroflow.source.array import ArraySource
from neuroflow_cellpose import resolve_cellpose_device


def _array(
    path: Path,
    component: str,
    values: np.ndarray,
    axes: tuple[str, ...],
) -> neuroflow.NeuroArray:
    group = zarr.open_group(str(path), mode="w")
    group.create_dataset(component, data=values, chunks=values.shape)
    source = ArraySource(path, component=component, axes=axes)
    return neuroflow.NeuroArray(source, source.select())


def test_direct_numpy_fish_subset_validation_uses_global_plane_ids(
    tmp_path: Path,
) -> None:
    movie_values = np.arange(2 * 2 * 2 * 2, dtype=np.float32).reshape(2, 2, 2, 2)
    labels_values = np.zeros((2, 2, 2), dtype=np.uint64)
    labels_values[:1, :, 0] = (np.uint64(1) << np.uint64(32)) + np.uint64(1)
    labels_values[1:, :, 1] = (np.uint64(2) << np.uint64(32)) + np.uint64(1)
    movie = _array(
        tmp_path / "movie.zarr",
        "movie",
        movie_values,
        ("time", "y", "x", "z"),
    )
    labels = _array(
        tmp_path / "labels.zarr",
        "labels",
        labels_values,
        ("y", "x", "z"),
    )
    cell_ids = np.array([2**32 + 1, 2 * 2**32 + 1], dtype=np.uint64)
    expected = np.column_stack(
        [
            movie_values[:, :, :, 0][:, labels_values[:, :, 0] != 0].mean(axis=1),
            movie_values[:, :, :, 1][:, labels_values[:, :, 1] != 0].mean(axis=1),
        ]
    ).astype(np.float32)
    trace_path = tmp_path / "traces.zarr"
    trace_group = zarr.open_group(str(trace_path), mode="w")
    trace_group.create_dataset("traces", data=expected, chunks=expected.shape)
    trace_group.create_dataset("cell_ids", data=cell_ids)
    trace_source = ArraySource(
        trace_path,
        component="traces",
        axes=("time", "cell"),
    )
    traces = neuroflow.NeuroArray(trace_source, trace_source.select())

    result = benchmark_fish_pipeline._direct_numpy_trace_validation(
        movie,
        labels,
        traces,
        trace_output=trace_path,
        frames=2,
    )

    assert result["valid"] is True
    assert result["maximum_absolute_error"] == 0.0
    traces.close()
    labels.close()
    movie.close()


def test_direct_cellpose_comparison_removes_partition_namespace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projection_values = np.array([[[0, 1], [1, 0]], [[0, 0], [0, 1]]], dtype=np.float32)
    local = (projection_values > 0).astype(np.int32)
    global_labels = np.zeros_like(local, dtype=np.uint64)
    for plane in range(2):
        mask = local[:, :, plane] != 0
        namespace = np.uint64(plane + 1) << np.uint64(32)
        global_labels[:, :, plane][mask] = namespace + np.uint64(1)
    projection = _array(
        tmp_path / "projection.zarr",
        "projection",
        projection_values,
        ("y", "x", "z"),
    )
    labels = _array(
        tmp_path / "segmentation.zarr",
        "labels",
        global_labels,
        ("y", "x", "z"),
    )

    class FakeModel:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def eval(self, value: np.ndarray, **kwargs: object) -> tuple[np.ndarray]:
            del kwargs
            return ((value > 0).astype(np.int32),)

    # Resolve the device before faking importlib: the fake intercepts every
    # import_module call, including the one that probes for torch.
    device = resolve_cellpose_device("cpu")
    monkeypatch.setattr(
        benchmark_fish_pipeline.importlib,
        "import_module",
        lambda name: SimpleNamespace(CellposeModel=FakeModel),
    )
    monkeypatch.setattr(
        benchmark_fish_pipeline.importlib.metadata,
        "version",
        lambda name: "test-version",
    )

    result = benchmark_fish_pipeline._direct_cellpose_equivalence(
        projection,
        labels,
        model_name="test-model",
        device=device,
    )

    assert result["valid"] is True
    assert result["mismatched_voxels"] == 0
    assert result["direct_object_count"] == 2
    # The device the comparison ran on must be recorded, otherwise an
    # equivalence claim cannot be tied to a specific execution path.
    assert result["device"] == device.to_dict()
    labels.close()
    projection.close()
