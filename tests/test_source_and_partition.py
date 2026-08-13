from pathlib import Path

import numpy as np
import pytest
import zarr

import neuroflow
from neuroflow.exceptions import AmbiguousSelectionError, ObjectNotFoundError
from neuroflow.partition import SpatialTilePlan, TimeWindowPlan
from neuroflow.selection import NWBQuery


def test_open_and_plan_do_not_read_selected_numerical_data(
    nwb_zarr: tuple[Path, np.ndarray], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _ = nwb_zarr
    reads: list[str] = []
    original = zarr.Array.__getitem__

    def recorded(array: zarr.Array, key: object) -> object:
        if array.path == "acquisition/movie/data":
            reads.append(repr(key))
        return original(array, key)

    monkeypatch.setattr(zarr.Array, "__getitem__", recorded)
    source = neuroflow.open_source(path)
    movie = source.select(NWBQuery(name="movie", neurodata_type="TimeSeries"))
    plan = TimeWindowPlan(size=4, overlap=1)
    assert plan.validate(movie).valid
    assert len(plan.build(movie)) == 3
    objects = source.inspect().objects
    assert {item.name for item in objects} == {"movie", "other", "irregular"}
    assert next(item for item in objects if item.name == "movie").native_chunks == (
        2,
        3,
        4,
    )
    assert reads == []
    source.close()


def test_semantic_selection_reports_missing_and_ambiguous(
    nwb_zarr: tuple[Path, np.ndarray],
) -> None:
    source = neuroflow.open_source(nwb_zarr[0])
    with pytest.raises(AmbiguousSelectionError):
        source.select(NWBQuery(neurodata_type="TimeSeries"))
    with pytest.raises(ObjectNotFoundError):
        source.select(NWBQuery(name="missing"))
    source.close()


def test_lazy_array_reads_only_when_computed(
    nwb_zarr: tuple[Path, np.ndarray],
) -> None:
    source = neuroflow.open_source(nwb_zarr[0])
    movie = source.select(NWBQuery(name="movie"))
    lazy = movie.as_dask_array(chunks="native")
    assert lazy.shape == nwb_zarr[1].shape
    np.testing.assert_array_equal(lazy[:2].compute(), nwb_zarr[1][:2])
    source.close()


def test_irregular_timestamps_are_exposed_lazily(
    nwb_zarr: tuple[Path, np.ndarray], monkeypatch: pytest.MonkeyPatch
) -> None:
    source = neuroflow.open_source(nwb_zarr[0])
    irregular = source.select(NWBQuery(name="irregular"))
    reads = 0
    original = zarr.Array.__getitem__

    def recorded(array: zarr.Array, key: object) -> object:
        nonlocal reads
        if array.path == "acquisition/irregular/timestamps":
            reads += 1
        return original(array, key)

    monkeypatch.setattr(zarr.Array, "__getitem__", recorded)
    timestamps = irregular.as_dask_timestamps()
    assert timestamps is not None
    assert reads == 0
    np.testing.assert_array_equal(timestamps.compute(), [0.0, 0.4, 1.1, 2.0])
    assert reads > 0
    assert (
        not TimeWindowPlan(size="1 s", align_to="timestamps").validate(irregular).valid
    )
    source.close()


def test_time_and_spatial_partitions_separate_read_and_output_slices(
    nwb_zarr: tuple[Path, np.ndarray],
) -> None:
    source = neuroflow.open_source(nwb_zarr[0])
    movie = source.select(NWBQuery(name="movie"))
    temporal = TimeWindowPlan(size="2 s", overlap="0.5 s").build(movie)
    assert temporal[1].read_slices[0] == slice(3, 9)
    assert temporal[1].output_slices[0] == slice(4, 8)
    spatial = SpatialTilePlan((2, 2), (1, 1), ("y", "x")).build(movie)
    assert len(spatial) == 4
    assert spatial[0].output_slices[1:] == (slice(0, 2), slice(0, 2))
    source.close()
