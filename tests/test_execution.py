from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
import zarr
from distributed import Client, LocalCluster

import neuroflow
from neuroflow.adapters import ArrayOutput, FunctionAdapter, TableOutput
from neuroflow.exceptions import OutputConflictError, ProvenanceMismatchError
from neuroflow.execution.runner import partition_identity
from neuroflow.partition import TimeWindowPlan
from neuroflow.selection import NWBQuery
from neuroflow.storage import ParquetOutput, ZarrOutput


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
