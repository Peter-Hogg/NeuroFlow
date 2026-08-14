import hashlib
from pathlib import Path

import numpy as np
import pytest
import zarr

import neuroflow
from neuroflow.adapters import ArrayOutput, FunctionAdapter
from neuroflow.exceptions import IncompletePartitionError, OutputConflictError
from neuroflow.partition import TimeWindowPlan
from neuroflow.partition.base import Partition
from neuroflow.results.workflow import WorkflowResult
from neuroflow.selection import NWBQuery
from neuroflow.source.base import NWBSource
from neuroflow.storage import ZarrOutput
from neuroflow.storage.base import join_uri, read_json, write_json_atomic


def _identity_adapter(*, chunks: tuple[int, ...]) -> FunctionAdapter:
    return FunctionAdapter(
        function=lambda value: value,
        input_kind="array",
        output=ArrayOutput("float32", chunks=chunks),
        name="persisted-array-safety",
        version="1",
        splittable_axes=("time",),
    )


def _persist_array(
    nwb_zarr: tuple[Path, np.ndarray], tmp_path: Path
) -> tuple[NWBSource, WorkflowResult]:
    source = neuroflow.open_source(nwb_zarr[0])
    selection = source.select(NWBQuery(name="movie"))
    result = neuroflow.run(
        source=source,
        selection=selection,
        adapter=_identity_adapter(chunks=(5, 3, 4)),
        partition=TimeWindowPlan(5),
        output=ZarrOutput(str(tmp_path / "persisted.zarr")),
        execute=True,
    )
    return source, result


def test_open_array_verifies_checksums_by_default_and_has_explicit_trusted_path(
    nwb_zarr: tuple[Path, np.ndarray],
    tmp_path: Path,
) -> None:
    source, result = _persist_array(nwb_zarr, tmp_path)
    group = zarr.open_group(result.output.uri, mode="a")
    group["result"][0, 0, 0] = np.float32(999)

    with pytest.raises(IncompletePartitionError, match="checksum mismatch"):
        neuroflow.open_array(result.output.uri)

    trusted_source, trusted = neuroflow.open_array(result.output.uri, verify=False)
    assert trusted.metadata.shape == nwb_zarr[1].shape
    assert trusted_source.identity.checksum
    trusted_source.close()
    source.close()


def test_open_array_rejects_table_decoy_for_declared_array_component(
    nwb_zarr: tuple[Path, np.ndarray],
    tmp_path: Path,
) -> None:
    source, result = _persist_array(nwb_zarr, tmp_path)
    provenance = neuroflow.open_result(result.output.uri).provenance
    partition_plan = provenance["partition_plan"]
    assert isinstance(partition_plan, dict)
    partition_ids = partition_plan["partition_ids"]
    assert isinstance(partition_ids, list)
    partition_id = str(partition_ids[0])

    decoy = (
        Path(result.output.uri) / "tables" / "result" / f"part-{partition_id}.parquet"
    )
    decoy.parent.mkdir(parents=True)
    payload = b"decoy table bytes"
    decoy.write_bytes(payload)
    manifest_uri = join_uri(
        result.output.uri,
        ".neuroflow",
        "manifests",
        f"{partition_id}.json",
    )
    manifest = read_json(manifest_uri)
    assert manifest is not None
    manifest["outputs"] = {"result": str(decoy)}
    manifest["checksums"] = {"result": hashlib.sha256(payload).hexdigest()}
    manifest["sizes"] = {"result": len(payload)}
    write_json_atomic(manifest_uri, manifest)

    group = zarr.open_group(result.output.uri, mode="a")
    group["result"][:] = np.float32(999)

    with pytest.raises(IncompletePartitionError, match="declared array location"):
        neuroflow.open_array(result.output.uri)
    source.close()


def test_trusted_open_still_requires_a_structurally_complete_result(
    nwb_zarr: tuple[Path, np.ndarray],
    tmp_path: Path,
) -> None:
    source, result = _persist_array(nwb_zarr, tmp_path)
    result_uri = join_uri(result.output.uri, ".neuroflow", "result.json")
    metadata = read_json(result_uri)
    assert metadata is not None
    metadata["status"] = "partial"
    write_json_atomic(result_uri, metadata)

    with pytest.raises(IncompletePartitionError, match="metadata is not complete"):
        neuroflow.open_array(result.output.uri, verify=False)
    source.close()


def test_fresh_persist_uses_manifest_trust_without_rereading_output(
    nwb_zarr: tuple[Path, np.ndarray],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_reads: list[object] = []
    original = zarr.Array.__getitem__

    def recorded(array: zarr.Array, key: object) -> object:
        if array.path == "result":
            output_reads.append(key)
        return original(array, key)

    monkeypatch.setattr(zarr.Array, "__getitem__", recorded)
    movie = neuroflow.load(nwb_zarr[0], name="movie")
    persisted = (movie + 1).persist(
        tmp_path / "fresh.zarr", memory_limit="64 MiB", max_workers=1
    )
    assert output_reads == []

    reopened_source, _ = neuroflow.open_array(tmp_path / "fresh.zarr")
    assert output_reads
    reopened_source.close()
    persisted.close()
    movie.close()


def test_downstream_identity_includes_verified_partition_checksums(
    nwb_zarr: tuple[Path, np.ndarray],
    tmp_path: Path,
) -> None:
    source, result = _persist_array(nwb_zarr, tmp_path)
    first_source, first_selection = neuroflow.open_array(result.output.uri)
    adapter = _identity_adapter(chunks=(5, 3, 4))
    first_plan = neuroflow.plan(
        source=first_source,
        selection=first_selection,
        adapter=adapter,
        partition=TimeWindowPlan(5),
        output=ZarrOutput(str(tmp_path / "downstream.zarr")),
    )
    first_content_identity = first_source.identity.checksum
    first_source.close()

    provenance = neuroflow.open_result(result.output.uri).provenance
    plan = provenance["partition_plan"]
    assert isinstance(plan, dict)
    raw_partitions = plan["partitions"]
    assert isinstance(raw_partitions, list)
    raw_partition = raw_partitions[0]
    assert isinstance(raw_partition, dict)
    partition = Partition.from_dict(raw_partition)
    partition_id = str(raw_partition["partition_id"])
    group = zarr.open_group(result.output.uri, mode="a")
    array = group["result"]
    assert isinstance(array, zarr.Array)
    first_index = tuple(item.start or 0 for item in partition.output_slices)
    current_value = np.asarray(array[first_index], dtype=np.float32).item()
    array[first_index] = np.float32(current_value + 10)
    partition_value = np.asarray(array[partition.output_slices])
    digest = hashlib.sha256(partition_value.tobytes(order="C")).hexdigest()
    manifest_uri = join_uri(
        result.output.uri,
        ".neuroflow",
        "manifests",
        f"{partition_id}.json",
    )
    manifest = read_json(manifest_uri)
    assert manifest is not None
    checksums = manifest["checksums"]
    assert isinstance(checksums, dict)
    checksums["result"] = digest
    write_json_atomic(manifest_uri, manifest)

    second_source, second_selection = neuroflow.open_array(result.output.uri)
    second_plan = neuroflow.plan(
        source=second_source,
        selection=second_selection,
        adapter=adapter,
        partition=TimeWindowPlan(5),
        output=ZarrOutput(str(tmp_path / "downstream.zarr")),
    )
    assert second_source.identity.checksum != first_content_identity
    assert second_plan.workflow_id != first_plan.workflow_id
    second_source.close()
    source.close()


def test_resume_rejects_an_existing_array_with_changed_zarr_chunks(
    nwb_zarr: tuple[Path, np.ndarray],
    tmp_path: Path,
) -> None:
    source, result = _persist_array(nwb_zarr, tmp_path)
    group = zarr.open_group(result.output.uri, mode="a")
    values = np.asarray(group["result"])
    del group["result"]
    group.create_dataset(
        "result",
        data=values,
        chunks=(2, 3, 4),
        dtype="float32",
    )

    with pytest.raises(OutputConflictError, match="chunks do not match"):
        result.resume()
    source.close()
