from pathlib import Path

import numpy as np
import pytest
import zarr

import neuroflow
from neuroflow.exceptions import OutputConflictError, ProvenanceMismatchError
from neuroflow.selection import Selection
from neuroflow.source.array import ArraySource
from neuroflow.storage.base import read_json, write_json_atomic


def _array(
    path: Path,
    name: str,
    data: np.ndarray,
    axes: tuple[str, ...],
    *,
    chunks: tuple[int, ...] | None = None,
) -> neuroflow.NeuroArray:
    group = zarr.open_group(str(path), mode="w")
    group.create_dataset(name, data=data, chunks=chunks)
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


def test_trace_plan_skips_empty_source_chunks_and_does_not_read_movie(
    tmp_path: Path,
) -> None:
    movie_values = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)
    movie = _array(
        tmp_path / "movie-chunked.zarr",
        "movie",
        movie_values,
        ("time", "y", "x"),
        chunks=(1, 2, 2),
    )
    labels = _array(
        tmp_path / "labels-chunked.zarr",
        "labels",
        np.array(
            [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            dtype=np.uint64,
        ),
        ("y", "x"),
        chunks=(2, 2),
    )

    class CountingArray:
        def __init__(self, value: object) -> None:
            self.value = value
            self.shape = value.shape  # type: ignore[attr-defined]
            self.dtype = value.dtype  # type: ignore[attr-defined]
            self.ndim = value.ndim  # type: ignore[attr-defined]
            self.chunks = value.chunks  # type: ignore[attr-defined]
            self.reads: list[tuple[slice, ...]] = []

        def __getitem__(self, key: tuple[slice, ...]) -> object:
            self.reads.append(key)
            return self.value[key]  # type: ignore[index]

    counting = CountingArray(movie.selection._array)
    movie.selection = Selection(movie.selection.metadata, counting)

    plan = movie.plan_traces(labels, time_chunk=1, memory_limit="512 MiB")

    assert counting.reads == []
    assert plan.active_spatial_chunks == 1
    assert plan.skipped_empty_spatial_chunks == 3
    assert plan.estimated_source_chunks_touched == 2
    traces = movie.extract_traces(
        labels,
        output=tmp_path / "sparse-traces.zarr",
        time_chunk=1,
        memory_limit="512 MiB",
    )
    expected = movie_values[:, :2, :2].mean(axis=(1, 2), keepdims=False)[:, None]
    np.testing.assert_array_equal(traces.compute(), expected)
    assert len(counting.reads) == 2
    assert all(key[1:] == (slice(0, 2), slice(0, 2)) for key in counting.reads)
    provenance = neuroflow.open_result(
        tmp_path / "sparse-traces.zarr"
    ).provenance
    assert provenance["preflight_plan"]["roi_index"][  # type: ignore[index]
        "skipped_empty_spatial_chunks"
    ] == 3
    assert provenance["execution_metrics"]["computed_task_count"] == 2  # type: ignore[index]
    traces.close()
    labels.close()
    movie.close()


def test_trace_plan_counts_repeated_native_chunk_touches(
    tmp_path: Path,
) -> None:
    movie = _array(
        tmp_path / "movie-time-chunked.zarr",
        "movie",
        np.ones((2, 2, 2), dtype=np.float32),
        ("time", "y", "x"),
        chunks=(2, 2, 2),
    )
    labels = _array(
        tmp_path / "labels-time-chunked.zarr",
        "labels",
        np.ones((2, 2), dtype=np.uint64),
        ("y", "x"),
        chunks=(2, 2),
    )

    plan = movie.plan_traces(labels, time_chunk=1, memory_limit="512 MiB")

    # Two one-frame compute calls each request the same two-frame source chunk.
    assert plan.estimated_source_chunks_touched == 2
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
    assert attempts[0]["execution_metrics"]["computed_task_count"] == 2
    assert attempts[1]["execution_metrics"]["resumed_task_count"] == 2
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
