import os
from dataclasses import replace
from pathlib import Path
from typing import Literal
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
import zarr
from distributed import Client, LocalCluster

import neuroflow
import neuroflow.results.workflow as workflow_results
from neuroflow.adapters import ArrayOutput, FunctionAdapter, TableOutput
from neuroflow.exceptions import (
    IncompletePartitionError,
    OutputConflictError,
    ProvenanceMismatchError,
)
from neuroflow.execution.runner import partition_identity
from neuroflow.partition import SpatialTilePlan, TimeWindowPlan
from neuroflow.selection import NWBQuery
from neuroflow.storage import ParquetOutput, ZarrOutput
from neuroflow.storage.manifest import PartitionManifest


def _adapter(function: object, **parameters: object) -> FunctionAdapter:
    return FunctionAdapter(
        function=function,  # type: ignore[arg-type]
        input_kind="array",
        output=ArrayOutput("float32"),
        name="test-transform",
        version="1",
        splittable_axes=("time",),
        parameters=parameters,
    )


class _AdversarialManifestAdapter:
    """Test adapter that violates one manifest commit contract at a time."""

    name = "adversarial-manifest"
    version = "1"
    deterministic = True
    random_seed = None

    def __init__(
        self,
        behavior: Literal["failed-return", "wrong-identity", "mismatched-return"],
    ) -> None:
        self.behavior = behavior
        self._delegate = _adapter(lambda value: value)
        self.output = self._delegate.output

    def requirements(self):  # type: ignore[no-untyped-def]
        return self._delegate.requirements()

    def prepare(self, partition, context):  # type: ignore[no-untyped-def]
        return self._delegate.prepare(partition, context)

    def run(self, prepared, context):  # type: ignore[no-untyped-def]
        return self._delegate.run(prepared, context)

    def persist(self, output, writer, context):  # type: ignore[no-untyped-def]
        manifest = self._delegate.persist(output, writer, context)
        if not isinstance(manifest, PartitionManifest):
            raise TypeError("delegate did not return a partition manifest")
        if self.behavior == "failed-return":
            return replace(manifest, status="failed")
        if self.behavior == "wrong-identity":
            return replace(manifest, partition_id="wrong-partition")
        return replace(manifest, checksums={"result": "0" * 64})


def test_end_to_end_execution_resume_and_lazy_reopen(
    nwb_zarr: tuple[Path, np.ndarray], tmp_path: Path
) -> None:
    source = neuroflow.open_source(nwb_zarr[0])
    movie = source.select(NWBQuery(name="movie"))
    calls: list[int] = []

    def scale(value: np.ndarray, factor: float) -> np.ndarray:
        calls.append(int(value[0, 0, 0]))
        return value * factor

    output = ZarrOutput(str(tmp_path / "result.zarr"))
    result = neuroflow.run(
        source=source,
        selection=movie,
        adapter=_adapter(scale, factor=2.0),
        partition=TimeWindowPlan(size=4, overlap=1),
        output=output,
    )
    assert not (tmp_path / "result.zarr").exists()
    assert result.status.state == "planned"
    result.execute()
    assert result.status.state == "complete"
    assert len(calls) == 3
    np.testing.assert_array_equal(
        result.arrays["result"].as_dask_array().compute(), nwb_zarr[1] * 2
    )
    provenance = result.provenance
    assert provenance is not None
    assert provenance["source"]["uri"] == str(nwb_zarr[0].resolve())  # type: ignore[index]
    calls.clear()
    result.resume()
    assert calls == []
    reopened = neuroflow.open_result(output.uri)
    assert reopened.status.state == "complete"
    np.testing.assert_array_equal(
        reopened.arrays["result"][:2].compute(), nwb_zarr[1][:2] * 2
    )
    source.close()


def test_completed_resume_preserves_original_execution_metadata(
    nwb_zarr: tuple[Path, np.ndarray], tmp_path: Path
) -> None:
    source = neuroflow.open_source(nwb_zarr[0])
    movie = source.select(NWBQuery(name="movie"))
    result = neuroflow.run(
        source=source,
        selection=movie,
        adapter=_adapter(lambda value: value),
        partition=TimeWindowPlan(size=5),
        output=ZarrOutput(str(tmp_path / "attempt-history.zarr")),
        execute=True,
        max_workers=1,
        memory_limit="1 MiB",
    )
    original = result.provenance
    assert original is not None

    result.max_workers = 2
    result.memory_limit = "2 MiB"
    result.resume()
    resumed = result.provenance
    assert resumed is not None

    for key in (
        "execution_started",
        "execution_finished",
        "scheduler",
        "execution_policy",
        "environment",
    ):
        assert resumed[key] == original[key]
    attempts = resumed["execution_attempts"]
    assert isinstance(attempts, list)
    assert len(attempts) == 2
    assert [attempt["status"] for attempt in attempts] == ["complete", "complete"]
    assert attempts[0]["execution_policy"] == {
        "max_workers": 1,
        "memory_limit": "1 MiB",
    }
    assert attempts[1]["execution_policy"] == {
        "max_workers": 2,
        "memory_limit": "2 MiB",
    }
    assert attempts[1]["resumed_from_status"] == "complete"
    assert "execution_finished" in attempts[1]
    source.close()


def test_bounded_selection_and_tiled_axis_reduction(
    nwb_zarr: tuple[Path, np.ndarray], tmp_path: Path
) -> None:
    source = neuroflow.open_source(nwb_zarr[0])
    movie = source.select(NWBQuery(name="movie")).isel(time=slice(0, 5))

    def temporal_median(tile: np.ndarray) -> np.ndarray:
        return np.median(tile, axis=0)

    result = neuroflow.run(
        source=source,
        selection=movie,
        adapter=FunctionAdapter(
            function=temporal_median,
            input_kind="array",
            output=ArrayOutput(
                "float32",
                name="median_projection",
                reduced_axes=("time",),
                chunks=(2, 2),
            ),
            name="temporal-median",
            version="1",
            splittable_axes=("y", "x"),
        ),
        partition=SpatialTilePlan((2, 2), (0, 0), ("y", "x")),
        output=ZarrOutput(str(tmp_path / "projection.zarr")),
        execute=True,
    )

    assert movie.metadata.shape == (5, 3, 4)
    assert result.plan.output_shape == (3, 4)
    assert result.plan.output_axes == ("y", "x")
    assert result.plan.task_count == 4
    assert result.arrays["median_projection"].as_dask_array().chunks == (
        (2, 1),
        (2, 2),
    )
    np.testing.assert_array_equal(
        result.arrays["median_projection"].as_dask_array().compute(),
        np.median(nwb_zarr[1][:5], axis=0),
    )
    assert result.verify().valid
    assert neuroflow.open_result(result.output.uri).verify().valid
    chained_source, chained = neuroflow.open_array(result.output.uri)
    assert chained.metadata.axes == ("y", "x")
    np.testing.assert_array_equal(
        chained.as_dask_array().compute(), np.median(nwb_zarr[1][:5], axis=0)
    )
    chained_source.close()
    source.close()


def test_equal_shaped_slices_have_distinct_workflow_identities(
    nwb_zarr: tuple[Path, np.ndarray], tmp_path: Path
) -> None:
    source = neuroflow.open_source(nwb_zarr[0])
    movie = source.select(NWBQuery(name="movie"))
    first = neuroflow.run(
        source=source,
        selection=movie.isel(time=slice(0, 5)),
        adapter=_adapter(lambda value: value),
        partition=TimeWindowPlan(size=5),
        output=ZarrOutput(str(tmp_path / "first.zarr")),
    )
    second = neuroflow.run(
        source=source,
        selection=movie.isel(time=slice(5, 10)),
        adapter=_adapter(lambda value: value),
        partition=TimeWindowPlan(size=5),
        output=ZarrOutput(str(tmp_path / "second.zarr")),
    )
    assert first.plan.workflow_id != second.plan.workflow_id
    assert first.selection.metadata.selection_bounds == ((0, 5), (0, 3), (0, 4))
    assert second.selection.metadata.selection_bounds == ((5, 10), (0, 3), (0, 4))
    assert first.selection.metadata.starting_time == 0
    assert second.selection.metadata.starting_time == 2.5
    source.close()


def test_numpy_like_neuroarray_median(
    nwb_zarr: tuple[Path, np.ndarray], tmp_path: Path
) -> None:
    movie = neuroflow.load(nwb_zarr[0], name="movie").isel(time=slice(0, 5))
    projection = movie.median(
        "time", output=tmp_path / "friendly.zarr", chunks=(2, 2), max_workers=1
    )

    assert projection.axes == ("y", "x")
    assert projection.shape == (3, 4)
    np.testing.assert_array_equal(
        projection.compute(), np.median(nwb_zarr[1][:5], axis=0)
    )
    projection.close()
    movie.close()


def test_workflow_memory_limit_clamps_unsafe_concurrency(
    nwb_zarr: tuple[Path, np.ndarray], tmp_path: Path
) -> None:
    """Stated worker availability is reduced to fit, not rejected.

    ``max_workers`` describes the resources the caller has, so a request the
    memory target cannot afford is clamped and the granted count recorded.
    Refusing instead would make the caller hand-tune the very number the
    planner just derived.
    """
    source = neuroflow.open_source(nwb_zarr[0])
    movie = source.select(NWBQuery(name="movie"))
    result = neuroflow.run(
        source=source,
        selection=movie,
        adapter=_adapter(lambda value: value),
        partition=TimeWindowPlan(size=5),
        output=ZarrOutput(str(tmp_path / "memory.zarr")),
        max_workers=2,
        # A 600-byte total target tapers to 300 bytes of task memory, which
        # fits one 240-byte task but not two concurrent ones.
        memory_limit=600,
    )
    assert result.max_workers == 1
    source.close()


def test_workflow_memory_limit_derives_a_bounded_worker_count(
    nwb_zarr: tuple[Path, np.ndarray], tmp_path: Path
) -> None:
    source = neuroflow.open_source(nwb_zarr[0])
    movie = source.select(NWBQuery(name="movie"))
    result = neuroflow.run(
        source=source,
        selection=movie,
        adapter=_adapter(lambda value: value),
        partition=TimeWindowPlan(size=5),
        output=ZarrOutput(str(tmp_path / "bounded-memory.zarr")),
        memory_limit="1 GiB",
    )
    assert result.max_workers is not None
    assert 1 <= result.max_workers <= (os.cpu_count() or 1)
    source.close()


def test_neuroarray_extracts_traces_in_bounded_time_windows(
    nwb_zarr: tuple[Path, np.ndarray], tmp_path: Path
) -> None:
    movie = neuroflow.load(nwb_zarr[0], name="movie")
    labels_path = tmp_path / "labels.zarr"
    group = zarr.open_group(str(labels_path), mode="w")
    group.create_dataset(
        "labels",
        data=np.array([[1, 1, 0, 0], [1, 1, 2, 2], [0, 0, 2, 2]], dtype="uint64"),
    )
    from neuroflow.source.array import ArraySource

    label_source = ArraySource(labels_path, component="labels", axes=("y", "x"))
    labels = neuroflow.NeuroArray(label_source, label_source.select())
    traces = movie.extract_traces(labels, output=tmp_path / "traces.zarr", time_chunk=3)

    expected = np.column_stack(
        [
            nwb_zarr[1][:, :2, :2].mean(axis=(1, 2)),
            nwb_zarr[1][:, 1:, 2:].mean(axis=(1, 2)),
        ]
    )
    np.testing.assert_array_equal(traces.compute(), expected)
    persisted = neuroflow.open_result(tmp_path / "traces.zarr")
    assert persisted.verify().valid
    stored = zarr.open_group(str(tmp_path / "traces.zarr"), mode="r")
    np.testing.assert_array_equal(stored["cell_ids"][:], [1, 2])
    np.testing.assert_array_equal(stored["timestamps"][:], np.arange(10) / 2.0)
    writable = zarr.open_group(str(tmp_path / "traces.zarr"), mode="a")
    writable["traces"][0, 0] = np.float32(999)
    assert not neuroflow.open_result(tmp_path / "traces.zarr").verify().valid
    repaired = movie.extract_traces(
        labels, output=tmp_path / "traces.zarr", time_chunk=3
    )
    np.testing.assert_array_equal(repaired.compute(), expected)
    assert neuroflow.open_result(tmp_path / "traces.zarr").verify().valid
    repaired.close()
    traces.close()
    labels.close()
    movie.close()


def test_resume_rejects_changed_parameters(
    nwb_zarr: tuple[Path, np.ndarray], tmp_path: Path
) -> None:
    source = neuroflow.open_source(nwb_zarr[0])
    movie = source.select(NWBQuery(name="movie"))
    output = ZarrOutput(str(tmp_path / "conflict.zarr"))

    def scale(value: np.ndarray, factor: float) -> np.ndarray:
        return value * factor

    neuroflow.run(
        source=source,
        selection=movie,
        adapter=_adapter(scale, factor=2.0),
        partition=TimeWindowPlan(size=5),
        output=output,
        execute=True,
    )
    changed = neuroflow.run(
        source=source,
        selection=movie,
        adapter=_adapter(scale, factor=3.0),
        partition=TimeWindowPlan(size=5),
        output=output,
    )
    with pytest.raises(ProvenanceMismatchError):
        changed.execute()
    source.close()


def test_partitioned_parquet_output_reopens_lazily(
    nwb_zarr: tuple[Path, np.ndarray], tmp_path: Path
) -> None:
    source = neuroflow.open_source(nwb_zarr[0])
    movie = source.select(NWBQuery(name="movie"))

    def summarize(value: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(
            {"minimum": [float(value.min())], "maximum": [float(value.max())]}
        )

    adapter = FunctionAdapter(
        function=summarize,
        input_kind="array",
        output=TableOutput("summaries"),
        name="summary-table",
        version="1",
        splittable_axes=("time",),
    )
    output = ParquetOutput(str(tmp_path / "tables"))
    result = neuroflow.run(
        source=source,
        selection=movie,
        adapter=adapter,
        partition=TimeWindowPlan(size=5),
        output=output,
        execute=True,
    )
    table = result.tables["summaries"].as_dask_dataframe()
    assert table.npartitions == 2
    assert len(table.compute()) == 2
    assert result.verify().valid
    reopened = neuroflow.open_result(output.uri)
    assert len(reopened.tables["summaries"].as_dask_dataframe().compute()) == 2
    assert reopened.verify().valid
    source.close()


def test_failed_workflow_resumes_without_repeating_completed_partitions(
    nwb_zarr: tuple[Path, np.ndarray], tmp_path: Path
) -> None:
    source = neuroflow.open_source(nwb_zarr[0])
    movie = source.select(NWBQuery(name="movie"))
    should_fail = True
    calls: list[int] = []

    def sometimes_fails(value: np.ndarray) -> np.ndarray:
        nonlocal should_fail
        marker = int(value[0, 0, 0])
        calls.append(marker)
        if marker == 36 and should_fail:
            raise RuntimeError("injected partition failure")
        return value

    result = neuroflow.run(
        source=source,
        selection=movie,
        adapter=_adapter(sometimes_fails),
        partition=TimeWindowPlan(size=4, overlap=1),
        output=ZarrOutput(str(tmp_path / "resumable.zarr")),
    )
    with pytest.raises(RuntimeError, match="injected partition failure"):
        result.execute()
    completed_before = set(result.status.completed_partitions)
    reopened_failure = neuroflow.open_result(result.output.uri)
    assert reopened_failure.status.state in {"failed", "partial"}
    assert reopened_failure.failed_partitions
    assert reopened_failure.provenance["status"] == "failed"
    assert "execution_finished" in reopened_failure.provenance
    assert "injected partition failure" in str(reopened_failure.provenance["error"])
    assert not reopened_failure.verify().valid
    should_fail = False
    calls.clear()
    result.resume()
    markers = {
        partition_identity(result.plan.workflow_id, partition): (
            partition.read_slices[0].start or 0
        )
        * 12
        for partition in result.plan.partitions
    }
    assert all(markers[item] not in calls for item in completed_before)
    assert result.status.state == "complete"
    source.close()


@pytest.mark.parametrize("behavior", ["failed-return", "wrong-identity"])
def test_adapter_returned_manifest_must_have_expected_identity_and_status(
    nwb_zarr: tuple[Path, np.ndarray],
    tmp_path: Path,
    behavior: Literal["failed-return", "wrong-identity"],
) -> None:
    source = neuroflow.open_source(nwb_zarr[0])
    movie = source.select(NWBQuery(name="movie"))
    output = tmp_path / "invalid-returned-manifest.zarr"
    result = neuroflow.run(
        source=source,
        selection=movie,
        adapter=_AdversarialManifestAdapter(behavior),
        partition=TimeWindowPlan(size=5),
        output=ZarrOutput(str(output)),
    )

    with pytest.raises(ValueError, match="invalid identity or status"):
        result.execute()

    provenance = result.provenance
    assert provenance is not None
    assert provenance["status"] == "failed"
    assert not (output / ".neuroflow" / "result.json").exists()
    source.close()


def test_adapter_returned_manifest_must_match_its_commit(
    nwb_zarr: tuple[Path, np.ndarray], tmp_path: Path
) -> None:
    source = neuroflow.open_source(nwb_zarr[0])
    movie = source.select(NWBQuery(name="movie")).isel(time=slice(0, 5))
    output = tmp_path / "missing-manifest.zarr"
    result = neuroflow.run(
        source=source,
        selection=movie,
        adapter=_AdversarialManifestAdapter("mismatched-return"),
        partition=TimeWindowPlan(size=5),
        output=ZarrOutput(str(output)),
    )

    with pytest.raises(ValueError, match="differs from its commit"):
        result.execute()

    provenance = result.provenance
    assert provenance is not None
    assert provenance["status"] == "failed"
    assert not (output / ".neuroflow" / "result.json").exists()
    source.close()


def test_finalization_requires_every_expected_manifest_commit(
    nwb_zarr: tuple[Path, np.ndarray],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = neuroflow.open_source(nwb_zarr[0])
    movie = source.select(NWBQuery(name="movie")).isel(time=slice(0, 5))
    output = tmp_path / "missing-final-manifest.zarr"
    result = neuroflow.run(
        source=source,
        selection=movie,
        adapter=_adapter(lambda value: value),
        partition=TimeWindowPlan(size=5),
        output=ZarrOutput(str(output)),
    )
    execute_tasks = workflow_results.execute_tasks

    def execute_then_remove_manifest(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        values = execute_tasks(*args, **kwargs)  # type: ignore[arg-type]
        manifests = list((output / ".neuroflow" / "manifests").glob("*.json"))
        assert len(manifests) == 1
        manifests[0].unlink()
        return values

    monkeypatch.setattr(workflow_results, "execute_tasks", execute_then_remove_manifest)

    with pytest.raises(IncompletePartitionError, match="missing manifest"):
        result.execute()

    provenance = result.provenance
    assert provenance is not None
    assert provenance["status"] == "failed"
    assert not (output / ".neuroflow" / "result.json").exists()
    source.close()


def test_partitioned_table_output_and_uncompressed_array(
    nwb_zarr: tuple[Path, np.ndarray], tmp_path: Path
) -> None:
    source = neuroflow.open_source(nwb_zarr[0])
    movie = source.select(NWBQuery(name="movie"))

    def classify(value: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(
            {"kind": ["low", "high"], "value": [value.min(), value.max()]}
        )

    table_output = ParquetOutput(
        str(tmp_path / "partitioned-tables"), partition_on=("kind",)
    )
    table_result = neuroflow.run(
        source=source,
        selection=movie,
        adapter=FunctionAdapter(
            function=classify,
            input_kind="array",
            output=TableOutput("classified"),
            name="classifier",
            version="1",
            splittable_axes=("time",),
        ),
        partition=TimeWindowPlan(size=5),
        output=table_output,
        execute=True,
    )
    frame = table_result.tables["classified"].as_dask_dataframe().compute()
    assert sorted(frame["kind"].tolist()) == ["high", "high", "low", "low"]
    assert len(list((tmp_path / "partitioned-tables").glob("**/partitions/*"))) == 2
    assert table_result.verify().valid

    array_output = ZarrOutput(str(tmp_path / "uncompressed.zarr"), compressor="none")
    neuroflow.run(
        source=source,
        selection=movie,
        adapter=_adapter(lambda value: value),
        partition=TimeWindowPlan(size=5),
        output=array_output,
        execute=True,
    )
    stored = zarr.open_group(str(tmp_path / "uncompressed.zarr"), mode="r")["result"]
    assert stored.compressor is None
    source.close()


def test_execution_reads_bounded_source_slices(
    nwb_zarr: tuple[Path, np.ndarray],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = neuroflow.open_source(nwb_zarr[0])
    movie = source.select(NWBQuery(name="movie"))
    reads: list[tuple[slice, ...]] = []
    original = zarr.Array.__getitem__

    def recorded(array: zarr.Array, key: object) -> object:
        if array.path == "acquisition/movie/data" and isinstance(key, tuple):
            reads.append(key)  # type: ignore[arg-type]
        return original(array, key)

    monkeypatch.setattr(zarr.Array, "__getitem__", recorded)
    neuroflow.run(
        source=source,
        selection=movie,
        adapter=_adapter(lambda value: value),
        partition=TimeWindowPlan(size=4, overlap=1),
        output=ZarrOutput(str(tmp_path / "bounded.zarr")),
        execute=True,
    )
    assert len(reads) == 3
    assert all((item[0].stop or 0) - (item[0].start or 0) <= 6 for item in reads)
    source.close()


def test_overwrite_mode_replaces_incompatible_workflow(
    nwb_zarr: tuple[Path, np.ndarray], tmp_path: Path
) -> None:
    source = neuroflow.open_source(nwb_zarr[0])
    movie = source.select(NWBQuery(name="movie"))
    uri = str(tmp_path / "overwrite.zarr")

    def scale(value: np.ndarray, factor: float) -> np.ndarray:
        return value * factor

    neuroflow.run(
        source=source,
        selection=movie,
        adapter=_adapter(scale, factor=2.0),
        partition=TimeWindowPlan(size=5),
        output=ZarrOutput(uri),
        execute=True,
    )
    replacement = neuroflow.run(
        source=source,
        selection=movie,
        adapter=_adapter(scale, factor=3.0),
        partition=TimeWindowPlan(size=5),
        output=ZarrOutput(uri, mode="overwrite"),
        execute=True,
    )
    np.testing.assert_array_equal(
        replacement.arrays["result"].as_dask_array().compute(), nwb_zarr[1] * 3
    )
    source.close()


def test_create_mode_rejects_unmanaged_existing_output(
    nwb_zarr: tuple[Path, np.ndarray], tmp_path: Path
) -> None:
    source = neuroflow.open_source(nwb_zarr[0])
    movie = source.select(NWBQuery(name="movie"))
    output_path = tmp_path / "occupied.zarr"
    output_path.mkdir()
    (output_path / "unmanaged").write_text("data")
    result = neuroflow.run(
        source=source,
        selection=movie,
        adapter=_adapter(lambda value: value),
        partition=TimeWindowPlan(size=5),
        output=ZarrOutput(str(output_path)),
    )
    with pytest.raises(OutputConflictError):
        result.execute()
    source.close()


def test_distributed_scheduler_and_memory_object_store(
    nwb_zarr: tuple[Path, np.ndarray],
) -> None:
    source = neuroflow.open_source(nwb_zarr[0])
    movie = source.select(NWBQuery(name="movie"))
    uri = f"memory://neuroflow-{uuid4().hex}/result.zarr"
    with (
        LocalCluster(n_workers=2, threads_per_worker=1, processes=False) as cluster,
        Client(cluster),
    ):
        result = neuroflow.run(
            source=source,
            selection=movie,
            adapter=_adapter(lambda value: value + 1),
            partition=TimeWindowPlan(size=5),
            output=ZarrOutput(uri),
            scheduler="distributed",
            execute=True,
        )
        np.testing.assert_array_equal(
            result.arrays["result"].as_dask_array().compute(), nwb_zarr[1] + 1
        )
    source.close()


def test_checksum_verification_detects_and_repairs_corrupt_partition(
    nwb_zarr: tuple[Path, np.ndarray], tmp_path: Path
) -> None:
    source = neuroflow.open_source(nwb_zarr[0])
    movie = source.select(NWBQuery(name="movie"))
    calls: list[int] = []

    def identity(value: np.ndarray) -> np.ndarray:
        calls.append(int(value[0, 0, 0]))
        return value

    output = ZarrOutput(str(tmp_path / "verified.zarr"))
    result = neuroflow.run(
        source=source,
        selection=movie,
        adapter=_adapter(identity),
        partition=TimeWindowPlan(size=5),
        output=output,
        execute=True,
    )
    assert result.verify().valid
    reads = 0
    original = zarr.Array.__getitem__

    def recorded(array: zarr.Array, key: object) -> object:
        nonlocal reads
        reads += 1
        return original(array, key)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(zarr.Array, "__getitem__", recorded)
    assert result.verify(checksums=False).valid
    assert reads == 0
    monkeypatch.undo()
    group = zarr.open_group(output.uri, mode="a")
    group["result"][0, 0, 0] = np.float32(999)
    corrupt = neuroflow.open_result(output.uri)
    report = corrupt.verify()
    assert not report.valid
    assert any("checksum mismatch" in error for error in report.errors)

    calls.clear()
    result.resume()
    assert len(calls) == 1
    assert result.verify().valid
    np.testing.assert_array_equal(
        result.arrays["result"].as_dask_array().compute(), nwb_zarr[1]
    )
    source.close()


def test_transport_bytes_are_separated_from_source_chunk_bytes() -> None:
    """The plan must not present chunk-level bytes as expected transfer.

    On DANDI:000223 the chunk-level read estimate was 192 MiB while the
    measured HTTP transfer was 3.26 GiB: a block transport moves whole blocks
    per chunk touch, and 32 KiB chunks fetched through 1 MiB blocks cost the
    block. The two quantities are reported separately, with the transport
    figure a no-reuse model that the measured transfer stays below.
    """
    from neuroflow.execution.graph import estimate_transport_bytes

    # The DANDI:000223 smoke geometry: 6,144 touches of 32 KiB chunks through
    # the 1 MiB default block. The model gives the no-reuse figure and the
    # retained measurement (3,422,379,957 bytes) sits below it, while the
    # chunk-level figure (201,326,592 bytes) sits far below the measurement.
    modelled = estimate_transport_bytes(6144, 32 * 1024 * 2, 1_048_576)
    assert modelled == 6144 * 1_048_576
    assert 201_326_592 < 3_422_379_957 < modelled

    # Chunks larger than a block round up to whole blocks (fish geometry:
    # 3.47 MiB chunks through 256 KiB blocks cost 14 blocks per touch).
    fish_chunk = 888 * 2048 * 2
    assert estimate_transport_bytes(1, fish_chunk, 262_144) == 14 * 262_144

    # No block model means no number: transports that manage their own remote
    # access (LINDI) and local files must stay unknown rather than echoing the
    # chunk-level figure as transfer.
    assert estimate_transport_bytes(6144, 32 * 1024 * 2, None) is None
    assert estimate_transport_bytes(None, 32 * 1024 * 2, 1_048_576) is None


def test_persisted_results_are_identical_across_worker_counts(
    tmp_path: Path,
) -> None:
    """Concurrency must not change the numbers.

    Staged reductions combine bounded partials, and partial-combination order
    is exactly where parallel execution could diverge from serial execution.
    Every equivalence claim in the publication evidence silently assumes this
    invariance, so it is pinned here: the same expression persisted at one and
    at several workers must produce bitwise-identical, checksum-equal output.
    """
    rng = np.random.default_rng(11)
    values = rng.integers(0, 4096, size=(24, 64, 64, 2), dtype=np.int16)
    group = zarr.open_group(str(tmp_path / "movie.zarr"), mode="w")
    group.create_dataset("movie", data=values, chunks=(1, 64, 64, 1))
    from neuroflow.source.array import ArraySource

    outputs: dict[int, np.ndarray] = {}
    for workers in (1, 4):
        source = ArraySource(
            tmp_path / "movie.zarr", component="movie", axes=("time", "y", "x", "z")
        )
        movie = neuroflow.NeuroArray(source, source.select())
        expression = np.mean(movie, axis="time").astype(  # type: ignore[call-overload]
            np.float32
        )
        output = tmp_path / f"mean-w{workers}.zarr"
        result = expression.persist(
            output, max_workers=workers, memory_limit="1 GiB"
        )
        reopened_source, reopened = neuroflow.open_array(output)
        outputs[workers] = np.asarray(reopened.as_dask_array().compute())
        reopened_source.close()
        result.close()
        source.close()

    np.testing.assert_array_equal(outputs[1], outputs[4])
    assert outputs[1].dtype == outputs[4].dtype
