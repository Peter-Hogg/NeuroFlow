from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

from benchmarks.baselines import (
    direct_dask_zarr_projection,
    direct_pynwb_zarr_projection,
    lindi_hdf5_projection,
)


def test_direct_baselines_use_the_same_operation(
    nwb_zarr: tuple[Path, np.ndarray],
) -> None:
    source, values = nwb_zarr
    expected = np.median(values[:5], axis=0)

    pynwb_result = direct_pynwb_zarr_projection(
        source, object_name="movie", frames=5
    )
    dask_result = direct_dask_zarr_projection(
        source, object_name="movie", frames=5
    )

    np.testing.assert_allclose(pynwb_result, expected)
    np.testing.assert_allclose(dask_result, expected)


def test_lindi_baseline_explains_its_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = importlib.import_module

    def reject_lindi(name: str):  # type: ignore[no-untyped-def]
        if name == "lindi":
            raise ImportError(name)
        return original_import(name)

    monkeypatch.setattr(importlib, "import_module", reject_lindi)

    with pytest.raises(RuntimeError, match="uv sync --extra baselines"):
        lindi_hdf5_projection("unused.nwb", object_name="movie", frames=1)


def test_dask_trace_baseline_agrees_with_numpy_and_neuroflow(
    tmp_path: Path,
) -> None:
    """The fair-baseline trace loop must itself be numerically correct.

    ``direct_dask_mean_traces`` is what the publication baseline record is
    built on; if it drifted, the NeuroFlow-versus-baseline comparison would be
    measuring a bug instead of a design difference. Three independent
    computations of the same means must agree exactly: plain NumPy, the manual
    Dask chunk loop, and NeuroFlow's planned extraction over identical labels.
    """
    import zarr

    import neuroflow
    from benchmarks.baselines import direct_dask_mean_traces
    from neuroflow.source.array import ArraySource

    rng = np.random.default_rng(7)
    frames, height, width, planes = 12, 8, 10, 4
    movie_values = rng.integers(
        0, 4096, size=(frames, height, width, planes), dtype=np.int16
    )
    # One (y, x) plane per source chunk, like the fish asset, so the baseline's
    # chunk traversal and NeuroFlow's chunk-aware planner both have real spatial
    # chunk boundaries to respect.
    labels = np.zeros((height, width, planes), dtype=np.uint64)
    labels[0:3, 0:4, 0] = 1
    labels[5:8, 6:10, 0] = 2
    labels[2:5, 3:7, 1] = 3  # a different plane, away from chunk edges
    labels[0:2, 0:2, 2] = 2  # one cell spanning two z planes / source chunks
    # Plane 3 carries no labels at all, so both sides must account for one
    # skipped source chunk rather than silently reading it.

    group = zarr.open_group(str(tmp_path / "movie.zarr"), mode="w")
    dataset = group.create_dataset(
        "movie", data=movie_values, chunks=(1, height, width, 1)
    )

    cell_ids_expected = np.array([1, 2, 3], dtype=np.uint64)
    expected = np.empty((frames, len(cell_ids_expected)), dtype=np.float32)
    for column, cell in enumerate(cell_ids_expected):
        voxels = movie_values[:, labels == cell].astype(np.float64)
        expected[:, column] = (voxels.sum(axis=1) / voxels.shape[1]).astype(
            np.float32
        )

    baseline_traces, baseline_ids, plan = direct_dask_mean_traces(
        dataset, labels, time_chunk=4
    )
    np.testing.assert_array_equal(baseline_ids, cell_ids_expected)
    np.testing.assert_array_equal(baseline_traces, expected)
    # The plan must reflect the geometry it claims to have traversed: three
    # labelled planes read, the empty fourth plane skipped.
    assert plan.cell_count == 3
    assert plan.active_spatial_chunks == 3
    assert plan.skipped_spatial_chunks == 1

    label_group = zarr.open_group(str(tmp_path / "labels.zarr"), mode="w")
    label_group.create_dataset("labels", data=labels)
    movie_source = ArraySource(
        tmp_path / "movie.zarr", component="movie", axes=("time", "y", "x", "z")
    )
    label_source = ArraySource(
        tmp_path / "labels.zarr", component="labels", axes=("y", "x", "z")
    )
    movie = neuroflow.NeuroArray(movie_source, movie_source.select())
    label_array = neuroflow.NeuroArray(label_source, label_source.select())

    neuroflow_traces = movie.extract_traces(
        label_array, output=tmp_path / "traces.zarr", time_chunk=4
    )
    result = zarr.open_group(str(tmp_path / "traces.zarr"), mode="r")
    np.testing.assert_array_equal(
        np.asarray(result["cell_ids"]), cell_ids_expected
    )
    np.testing.assert_array_equal(
        np.asarray(result["traces"], dtype=np.float32), expected
    )
    neuroflow_traces.close()
    label_array.close()
    movie.close()
