import hashlib
from pathlib import Path

import fsspec
import numpy as np
import pytest
import zarr

import neuroflow
from neuroflow.exceptions import OutputConflictError, SourceResolutionError
from neuroflow.execution.runner import _validate_recursive_delete_target
from neuroflow.partition.base import Partition
from neuroflow.results.workflow import _partition_within_shape
from neuroflow.source.array import ArraySource
from neuroflow.source.dandi import _validate_dandi_url
from neuroflow.source.hdf5 import _redacted_uri
from neuroflow.storage.base import validate_output_separation
from neuroflow.storage.manifest import PartitionManifest
from neuroflow.storage.validation import validate_partition_manifest


def test_recursive_delete_rejects_broad_local_targets(tmp_path: Path) -> None:
    filesystem = fsspec.filesystem("file")
    with pytest.raises(OutputConflictError, match="storage root"):
        _validate_recursive_delete_target(filesystem, "/")
    with pytest.raises(OutputConflictError, match="protected path"):
        _validate_recursive_delete_target(filesystem, str(Path.cwd()))
    with pytest.raises(OutputConflictError, match="object-store root"):
        _validate_recursive_delete_target(
            type("ObjectStore", (), {"protocol": "s3"})(),
            "bucket",
            "s3://bucket",
        )
    _validate_recursive_delete_target(filesystem, str(tmp_path / "specific-result"))


def test_output_separation_rejects_local_and_remote_containment(
    tmp_path: Path,
) -> None:
    source = tmp_path / "inputs" / "movie.zarr"
    with pytest.raises(OutputConflictError, match="overlaps"):
        validate_output_separation(str(tmp_path / "inputs"), {"movie": str(source)})
    with pytest.raises(OutputConflictError, match="overlaps"):
        validate_output_separation(str(source / "derived"), {"movie": str(source)})
    with pytest.raises(OutputConflictError, match="overlaps"):
        validate_output_separation(
            "s3://archive/results/source.zarr/derived",
            {"movie": "s3://archive/results/source.zarr"},
        )

    validate_output_separation(str(tmp_path / "outputs"), {"movie": str(source)})
    validate_output_separation(
        "s3://other-bucket/results", {"movie": "s3://archive/results"}
    )


def test_workflow_overwrite_cannot_remove_a_directory_containing_its_source(
    tmp_path: Path,
) -> None:
    container = tmp_path / "container"
    source_path = container / "movie.zarr"
    group = zarr.open_group(str(source_path), mode="w")
    group.create_dataset("movie", data=np.ones((2, 2), dtype=np.float32), chunks=(1, 2))
    source = ArraySource(source_path, component="movie", axes=("time", "x"))
    movie = neuroflow.NeuroArray(source, source.select())

    with pytest.raises(OutputConflictError, match="overlaps"):
        movie.persist(
            container,
            mode="overwrite",
            memory_limit="16 MiB",
            max_workers=1,
        )

    assert source_path.exists()
    np.testing.assert_array_equal(movie.compute(), np.ones((2, 2), dtype=np.float32))
    movie.close()


def test_manifest_validation_does_not_follow_external_output(tmp_path: Path) -> None:
    partition = Partition(
        "part-0",
        (slice(0, 1),),
        (slice(0, 1),),
        (slice(0, 1),),
        (0,),
    )
    manifest = PartitionManifest(
        "partition-id",
        "workflow-id",
        "complete",
        {"result": "https://127.0.0.1/private.zarr"},
        {"result": "0" * 64},
        sizes={"result": 1},
    )
    errors = validate_partition_manifest(
        manifest,
        partition,
        output_root=str(tmp_path / "result.zarr"),
        output_kinds={"result": "array"},
    )
    assert errors == ("output 'result' does not match its declared array location",)


def test_table_verification_refuses_oversized_partition_before_read(
    tmp_path: Path,
) -> None:
    root = tmp_path / "result"
    table = root / "tables" / "cells" / "part-partition-id.parquet"
    table.parent.mkdir(parents=True)
    table.write_bytes(b"not-parquet-but-deliberately-too-large")
    partition = Partition(
        "part-0",
        (slice(0, 1),),
        (slice(0, 1),),
        (slice(0, 1),),
        (0,),
    )
    manifest = PartitionManifest(
        "partition-id",
        "workflow-id",
        "complete",
        {"cells": str(table)},
        {"cells": "0" * 64},
        sizes={"cells": table.stat().st_size},
    )

    errors = validate_partition_manifest(
        manifest,
        partition,
        output_root=str(root),
        output_kinds={"cells": "table"},
        max_output_bytes=8,
    )

    assert errors == (
        "output 'cells' is 38 bytes, exceeding the 8-byte verification limit",
    )


def test_legacy_v1_manifest_without_sizes_remains_verifiable(
    tmp_path: Path,
) -> None:
    root = tmp_path / "legacy.zarr"
    values = np.array([1.0], dtype=np.float32)
    group = zarr.open_group(str(root), mode="w")
    group.create_dataset("result", data=values, chunks=(1,))
    partition = Partition(
        "part-0",
        (slice(0, 1),),
        (slice(0, 1),),
        (slice(0, 1),),
        (0,),
    )
    manifest = PartitionManifest(
        "partition-id",
        "workflow-id",
        "complete",
        {"result": str(root)},
        {"result": hashlib.sha256(values.tobytes()).hexdigest()},
        schema_version="1",
    )

    assert not validate_partition_manifest(
        manifest,
        partition,
        output_root=str(root),
        output_kinds={"result": "array"},
    )
    current_manifest = PartitionManifest(
        "partition-id",
        "workflow-id",
        "complete",
        {"result": str(root)},
        {"result": hashlib.sha256(values.tobytes()).hexdigest()},
    )
    assert validate_partition_manifest(
        current_manifest,
        partition,
        output_root=str(root),
        output_kinds={"result": "array"},
        checksums=False,
    ) == ("outputs have no sizes: result",)


def test_persisted_partition_slices_must_fit_declared_shape() -> None:
    assert _partition_within_shape((slice(0, 5), slice(2, 4)), (5, 4))
    assert not _partition_within_shape((slice(0, 6), slice(2, 4)), (5, 4))
    assert not _partition_within_shape((slice(0, 5),), (5, 4))


def test_remote_url_redaction_removes_credentials() -> None:
    assert (
        _redacted_uri("https://person:secret@example.org/data.nwb?token=secret#part")
        == "https://example.org/data.nwb"
    )


def test_dandi_url_policy_rejects_non_dandi_hosts() -> None:
    assert _validate_dandi_url("https://api.dandiarchive.org/api/assets/")
    assert _validate_dandi_url("https://dandiarchive.s3.amazonaws.com/blob")
    with pytest.raises(SourceResolutionError, match="approved hosts"):
        _validate_dandi_url("https://127.0.0.1/internal")
    with pytest.raises(SourceResolutionError, match="unsupported URL"):
        _validate_dandi_url("http://api.dandiarchive.org/api/assets/")
