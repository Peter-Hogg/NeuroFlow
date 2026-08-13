from pathlib import Path

import h5py
import numpy as np
import pytest

import neuroflow
from neuroflow.adapters import ArrayOutput, FunctionAdapter
from neuroflow.exceptions import UnsupportedBackendError
from neuroflow.partition import TimeWindowPlan
from neuroflow.selection import NWBQuery
from neuroflow.source.hdf5 import NWBHDF5Source, _open_remote_file
from neuroflow.storage import ZarrOutput


def test_local_hdf5_selection_stays_dataset_backed(
    nwb_hdf5: tuple[Path, np.ndarray],
) -> None:
    path, expected = nwb_hdf5
    with NWBHDF5Source(path) as source:
        movie = source.select(NWBQuery(name="movie"))

        assert isinstance(movie._array, h5py.Dataset)
        assert movie.metadata.shape == expected.shape
        attributes = movie.metadata.attributes
        assert attributes is not None
        assert attributes["backend"] == "nwb-hdf5"
        assert "bounded-array" in source.inspect().capabilities

        lazy = movie.as_dask_array()
        assert lazy.shape == expected.shape
        np.testing.assert_array_equal(lazy[:2].compute(), expected[:2])


def test_open_source_routes_nwb_files_to_hdf5(
    nwb_hdf5: tuple[Path, np.ndarray],
) -> None:
    path, _ = nwb_hdf5
    source = neuroflow.open_source(path)
    try:
        assert isinstance(source, NWBHDF5Source)
    finally:
        source.close()


def test_hdf5_rejects_process_scheduler(
    nwb_hdf5: tuple[Path, np.ndarray], tmp_path: Path
) -> None:
    path, _ = nwb_hdf5
    source = neuroflow.open_source(path)
    try:
        movie = source.select(NWBQuery(name="movie"))
        adapter = FunctionAdapter(
            function=np.asarray,
            input_kind="array",
            output=ArrayOutput("float32"),
            splittable_axes=("time",),
        )
        with pytest.raises(UnsupportedBackendError, match="scheduler='threads'"):
            neuroflow.run(
                source=source,
                selection=movie,
                adapter=adapter,
                partition=TimeWindowPlan(size=2),
                output=ZarrOutput(str(tmp_path / "result.zarr")),
                scheduler="processes",
            )
    finally:
        source.close()


def test_remote_hdf5_prefers_bounded_remfile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    sentinel = object()

    def fake_file(uri: str, **kwargs: object) -> object:
        calls.append((uri, kwargs))
        return sentinel

    monkeypatch.setattr("neuroflow.source.hdf5.remfile.File", fake_file)
    remote, transport = _open_remote_file(
        "https://example.test/session.nwb",
        {"block_size": 262_144, "cache_size": 8_388_608},
    )

    assert remote is sentinel
    assert transport == "remfile"
    assert calls == [
        (
            "https://example.test/session.nwb",
            {"_min_chunk_size": 262_144, "_max_cache_size": 8_388_608},
        )
    ]
