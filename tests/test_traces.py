from pathlib import Path

import numpy as np
import pytest
import zarr

import neuroflow
from neuroflow.exceptions import OutputConflictError, ProvenanceMismatchError
from neuroflow.source.array import ArraySource
from neuroflow.storage.base import read_json, write_json_atomic


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
    labels = _array(tmp_path / "labels.zarr", "labels", label_values, ("y", "x", "z"))

    traces = movie.extract_traces(labels, output=tmp_path / "traces.zarr", time_chunk=1)

    np.testing.assert_array_equal(
        traces.compute(), np.array([[3, 20], [7, 40]], dtype=np.float32)
    )
    assert traces.source.identity.checksum
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


def test_trace_label_discovery_caps_pathological_distinct_ids(
    tmp_path: Path,
) -> None:
    movie = _array(
        tmp_path / "movie.zarr",
        "movie",
        np.ones((1, 4, 4), dtype=np.float32),
        ("time", "y", "x"),
    )
    labels = _array(
        tmp_path / "labels.zarr",
        "labels",
        np.arange(1, 17, dtype=np.uint64).reshape(4, 4),
        ("y", "x"),
    )

    with pytest.raises(ValueError, match="distinct-label workspace"):
        movie.extract_traces(
            labels,
            output=tmp_path / "traces.zarr",
            memory_limit=1000,
        )

    assert not (tmp_path / "traces.zarr").exists()
    labels.close()
    movie.close()


def test_completed_trace_resume_preserves_original_execution_record(
    tmp_path: Path,
) -> None:
    movie = _array(
        tmp_path / "movie.zarr",
        "movie",
        np.arange(8, dtype=np.float32).reshape(2, 2, 2),
        ("time", "y", "x"),
    )
    labels = _array(
        tmp_path / "labels.zarr",
        "labels",
        np.array([[1, 1], [0, 2]], dtype=np.uint64),
        ("y", "x"),
    )
    output = tmp_path / "traces.zarr"
    first = movie.extract_traces(labels, output=output, time_chunk=1)
    original = read_json(str(output / ".neuroflow" / "provenance.json"))
    assert original is not None

    resumed = movie.extract_traces(labels, output=output, time_chunk=1)
    provenance = read_json(str(output / ".neuroflow" / "provenance.json"))

    assert provenance is not None
    assert provenance["execution_started"] == original["execution_started"]
    assert provenance["execution_finished"] == original["execution_finished"]
    attempts = provenance["execution_attempts"]
    assert isinstance(attempts, list)
    assert [attempt["status"] for attempt in attempts] == ["complete", "complete"]
    first.close()
    resumed.close()
    labels.close()
    movie.close()


def test_trace_output_cannot_overlap_an_input(tmp_path: Path) -> None:
    movie_path = tmp_path / "movie.zarr"
    movie = _array(
        movie_path,
        "movie",
        np.ones((2, 2, 2), dtype=np.float32),
        ("time", "y", "x"),
    )
    labels = _array(
        tmp_path / "labels.zarr",
        "labels",
        np.ones((2, 2), dtype=np.uint64),
        ("y", "x"),
    )

    with pytest.raises(OutputConflictError, match="movie input"):
        movie.extract_traces(labels, output=movie_path, time_chunk=1)

    assert sorted(zarr.open_group(str(movie_path), mode="r").array_keys()) == ["movie"]
    labels.close()
    movie.close()


def test_trace_output_rejects_unmanaged_and_mismatched_existing_roots(
    tmp_path: Path,
) -> None:
    movie = _array(
        tmp_path / "movie.zarr",
        "movie",
        np.ones((2, 2, 2), dtype=np.float32),
        ("time", "y", "x"),
    )
    labels = _array(
        tmp_path / "labels.zarr",
        "labels",
        np.ones((2, 2), dtype=np.uint64),
        ("y", "x"),
    )
    unmanaged = tmp_path / "unmanaged.zarr"
    unmanaged.mkdir()
    sentinel = unmanaged / "keep.txt"
    sentinel.write_text("do not modify")

    with pytest.raises(OutputConflictError, match="without matching"):
        movie.extract_traces(labels, output=unmanaged, time_chunk=1)
    assert sentinel.read_text() == "do not modify"
    assert not (unmanaged / "traces").exists()

    mismatched = tmp_path / "mismatched.zarr"
    mismatched.mkdir()
    write_json_atomic(
        str(mismatched / ".neuroflow" / "provenance.json"),
        {"workflow_id": "another-workflow"},
    )
    with pytest.raises(ProvenanceMismatchError, match="another workflow"):
        movie.extract_traces(labels, output=mismatched, time_chunk=1)
    assert not (mismatched / "traces").exists()

    labels.close()
    movie.close()
