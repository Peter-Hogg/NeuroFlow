from pathlib import Path

import fsspec
import pytest

from neuroflow.exceptions import OutputConflictError, SourceResolutionError
from neuroflow.execution.runner import _validate_recursive_delete_target
from neuroflow.partition.base import Partition
from neuroflow.results.workflow import _partition_within_shape
from neuroflow.source.dandi import _validate_dandi_url
from neuroflow.source.hdf5 import _redacted_uri
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
        {"result": "unused"},
    )
    errors = validate_partition_manifest(
        manifest, partition, output_root=str(tmp_path / "result.zarr")
    )
    assert errors == ("output 'result' escapes the declared result root",)


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
