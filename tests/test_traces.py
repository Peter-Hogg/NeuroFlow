from pathlib import Path

import numpy as np
import pytest
import zarr

import neuroflow
from neuroflow.source.array import ArraySource


def _array(
    path: Path, name: str, data: np.ndarray, axes: tuple[str, ...]
) -> neuroflow.NeuroArray:
    group = zarr.open_group(str(path), mode="w")
    group.create_dataset(name, data=data)
    source = ArraySource(path, component=name, axes=axes)
    return neuroflow.NeuroArray(source, source.select())


def test_trace_extraction_combines_one_label_across_z_planes(tmp_path: Path) -> None:
    movie_values = np.array(
        [
            [[[2, 4]], [[10, 20]]],
            [[[6, 8]], [[30, 40]]],
        ],
        dtype=np.float32,
    )
    label_values = np.array([[[1, 1]], [[0, 2]]], dtype=np.uint64)
    movie = _array(
        tmp_path / "movie.zarr", "movie", movie_values, ("time", "y", "x", "z")
    )
    labels = _array(
        tmp_path / "labels.zarr", "labels", label_values, ("y", "x", "z")
    )

    traces = movie.extract_traces(labels, output=tmp_path / "traces.zarr", time_chunk=1)

    np.testing.assert_array_equal(
        traces.compute(), np.array([[3, 20], [7, 40]], dtype=np.float32)
    )
    assert neuroflow.open_result(tmp_path / "traces.zarr").verify().valid
    traces.close()
    labels.close()
    movie.close()


def test_trace_memory_limit_is_enforced_before_output(tmp_path: Path) -> None:
    movie = _array(
        tmp_path / "movie.zarr",
        "movie",
        np.ones((2, 4, 4), dtype=np.float32),
        ("time", "y", "x"),
    )
    labels = _array(
        tmp_path / "labels.zarr",
        "labels",
        np.ones((4, 4), dtype=np.uint64),
        ("y", "x"),
    )
    with pytest.raises(ValueError, match="estimated"):
        movie.extract_traces(
            labels,
            output=tmp_path / "traces.zarr",
            time_chunk=2,
            memory_limit=1,
        )
    assert not (tmp_path / "traces.zarr").exists()
    labels.close()
    movie.close()


def test_trace_memory_limit_rejects_label_discovery_before_compute(
    tmp_path: Path,
) -> None:
    movie = _array(
        tmp_path / "movie.zarr",
        "movie",
        np.ones((1, 16, 16), dtype=np.float32),
        ("time", "y", "x"),
    )
    labels = _array(
        tmp_path / "labels.zarr",
        "labels",
        np.ones((16, 16), dtype=np.uint64),
        ("y", "x"),
    )
    with pytest.raises(ValueError, match="label discovery"):
        movie.extract_traces(labels, output=tmp_path / "traces.zarr", memory_limit=1)
    assert not (tmp_path / "traces.zarr").exists()
    labels.close()
    movie.close()
