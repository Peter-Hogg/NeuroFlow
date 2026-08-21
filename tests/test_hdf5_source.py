import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import h5py
import numpy as np
import pytest

import neuroflow
from neuroflow.adapters import ArrayOutput, FunctionAdapter
from neuroflow.exceptions import UnsupportedBackendError
from neuroflow.partition import TimeWindowPlan
from neuroflow.selection import NWBQuery
from neuroflow.source.hdf5 import NWBHDF5Source, _array_metadata, _open_remote_file
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


def test_lindi_transport_uses_optional_documented_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    sentinel = object()
    module = ModuleType("lindi")

    class FakeLindiH5pyFile:
        @staticmethod
        def from_hdf5_file(uri: str) -> object:
            calls.append(uri)
            return sentinel

    module.LindiH5pyFile = FakeLindiH5pyFile  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "lindi", module)

    remote, transport = _open_remote_file(
        "https://example.test/session.nwb", {"transport": "lindi"}
    )

    assert remote is sentinel
    assert transport == "lindi"
    assert calls == ["https://example.test/session.nwb"]


def test_array_capability_check_accepts_non_h5py_dataset() -> None:
    value = SimpleNamespace(
        shape=(10, 3, 4),
        dtype=np.dtype("float32"),
        ndim=3,
        chunks=(2, 3, 4),
        __getitem__=lambda key: key,
    )

    assert _array_metadata(value) == (
        (10, 3, 4),
        np.dtype("float32"),
        (2, 3, 4),
    )


def test_array_capability_check_skips_partial_array_lookalikes() -> None:
    """Container datasets missing array attributes are skipped, not fatal.

    Real NWB files hold hdmf HDF5 object-reference wrappers that expose
    ``shape`` and ``dtype`` but no ``ndim``. Discovery over DANDI:000223
    crashed on one; such objects are not selectable science arrays and must
    make ``_array_metadata`` answer ``None``.
    """
    value = SimpleNamespace(
        shape=(5,),
        dtype=np.dtype("object"),
        # no ndim attribute, like ContainerH5ReferenceDataset
        __getitem__=lambda key: key,
    )

    assert _array_metadata(value) is None


def test_real_lindi_local_bridge_preserves_lazy_array_semantics(
    nwb_hdf5: tuple[Path, np.ndarray],
) -> None:
    pytest.importorskip("lindi")
    path, expected = nwb_hdf5

    with NWBHDF5Source(
        path, storage_options={"transport": "lindi"}
    ) as source:
        movie = source.select(NWBQuery(name="movie"))
        assert type(movie._array).__name__ == "LindiH5pyDataset"
        assert movie.metadata.shape == expected.shape
        assert movie.metadata.attributes is not None
        assert movie.metadata.attributes["transport"] == "lindi"
        np.testing.assert_array_equal(movie.as_dask_array()[:2].compute(), expected[:2])


def test_remote_response_metrics_count_headers_without_reading_bodies() -> None:
    source = NWBHDF5Source.__new__(NWBHDF5Source)
    source.transport = "remfile"
    source._metrics_lock = threading.Lock()
    source._http_responses = 0
    source._response_content_bytes = 0

    source._record_response(SimpleNamespace(headers={"Content-Length": "4096"}))

    assert source.io_stats() == {
        "transport": "remfile",
        "http_responses": 1,
        "response_content_bytes": 4096,
    }
