"""Bounded validation for durable partition outputs."""

import hashlib

import fsspec
import numpy as np
import zarr

from neuroflow.partition.base import Partition
from neuroflow.storage.manifest import PartitionManifest


def validate_partition_manifest(
    manifest: PartitionManifest,
    partition: Partition,
    *,
    checksums: bool = True,
) -> tuple[str, ...]:
    """Validate only the outputs owned by one processing partition."""
    errors: list[str] = []
    if manifest.status != "complete":
        return (f"partition {manifest.partition_id} is not complete",)
    for name, uri in manifest.outputs.items():
        try:
            if uri.endswith(".parquet"):
                fs, path = fsspec.core.url_to_fs(uri)
                if not fs.exists(path):
                    errors.append(f"missing table output: {uri}")
                    continue
                if checksums:
                    with fs.open(path, "rb") as stream:
                        digest = hashlib.sha256()
                        while chunk := stream.read(1024 * 1024):
                            digest.update(chunk)
                    actual = digest.hexdigest()
                else:
                    continue
            else:
                mapper = fsspec.get_mapper(uri)
                group = zarr.open_group(mapper, mode="r")
                component = group[name] if name in group else None
                if not isinstance(component, zarr.Array):
                    errors.append(f"missing array component {name!r} at {uri}")
                    continue
                if checksums:
                    value = np.asarray(component[partition.output_slices])
                    actual = hashlib.sha256(value.tobytes(order="C")).hexdigest()
                else:
                    continue
            expected = manifest.checksums.get(name)
            if expected is None:
                errors.append(f"manifest has no checksum for {name!r}")
            elif actual != expected:
                errors.append(f"checksum mismatch for {name!r}")
        except Exception as exc:
            errors.append(f"could not validate {name!r}: {type(exc).__name__}: {exc}")
    missing = set(manifest.checksums) - set(manifest.outputs)
    if missing:
        errors.append("checksums have no output entries: " + ", ".join(sorted(missing)))
    return tuple(errors)
