"""Bounded validation for durable partition outputs."""

import hashlib
import posixpath
import re
from collections.abc import Mapping
from typing import Literal

import fsspec
import numpy as np
import zarr

from neuroflow.partition.base import Partition
from neuroflow.storage.base import validate_component_name
from neuroflow.storage.manifest import PartitionManifest

DEFAULT_MAX_VERIFY_BYTES = 2 * 1024**3
OutputStorageKind = Literal["array", "table"]
_TABLE_BUCKET = re.compile(r"^[0-9a-f]{16}$")


def output_component_kinds(
    output: Mapping[str, object],
) -> dict[str, OutputStorageKind]:
    """Return the declared storage kind for each provenance component."""
    kind = output.get("kind")
    if kind == "array":
        name = output.get("name")
        if not isinstance(name, str):
            raise ValueError("output schema has no component name")
        validate_component_name(name)
        return {name: "array"}
    if kind == "table":
        name = output.get("name")
        if not isinstance(name, str):
            raise ValueError("output schema has no component name")
        validate_component_name(name)
        return {name: "table"}
    if kind != "segmentation":
        raise ValueError("output schema has an unsupported kind")
    arrays = output.get("arrays")
    tables = output.get("tables")
    if not isinstance(arrays, Mapping) or not isinstance(tables, Mapping):
        raise ValueError("segmentation output schema has invalid components")
    declared: dict[str, OutputStorageKind] = {}
    for name, metadata in arrays.items():
        if not isinstance(name, str):
            raise ValueError("output component names must be strings")
        component = name
        validate_component_name(component)
        if not isinstance(metadata, Mapping) or metadata.get("name") != component:
            raise ValueError(f"array component {component!r} has mismatched metadata")
        declared[component] = "array"
    for name, metadata in tables.items():
        if not isinstance(name, str):
            raise ValueError("output component names must be strings")
        component = name
        validate_component_name(component)
        if not isinstance(metadata, Mapping) or metadata.get("name") != component:
            raise ValueError(f"table component {component!r} has mismatched metadata")
        if component in declared:
            raise ValueError("output component is declared as both array and table")
        declared[component] = "table"
    if not declared:
        raise ValueError("output schema has no components")
    return declared


def _is_expected_output(
    root_uri: str,
    name: str,
    output_uri: str,
    storage_kind: OutputStorageKind,
    partition_id: str,
) -> bool:
    root_fs, root_path = fsspec.core.url_to_fs(root_uri)
    output_fs, output_path = fsspec.core.url_to_fs(output_uri)
    if type(root_fs) is not type(output_fs):
        return False
    normalized_root = posixpath.normpath(root_path)
    normalized_output = posixpath.normpath(output_path)
    if storage_kind == "array":
        return ":" not in name and normalized_output == normalized_root

    table_name, separator, suffix = name.partition(":")
    if separator and not suffix.isdigit():
        return False
    table_root = posixpath.join(normalized_root, "tables", table_name)
    try:
        relative = posixpath.relpath(normalized_output, table_root)
    except ValueError:
        return False
    parts = relative.split("/")
    filename = f"part-{partition_id}.parquet"
    return parts == [filename] or (
        len(parts) == 3
        and parts[0] == "partitions"
        and _TABLE_BUCKET.fullmatch(parts[1]) is not None
        and parts[2] == filename
    )


def validate_partition_manifest(
    manifest: PartitionManifest,
    partition: Partition,
    *,
    output_root: str,
    output_kinds: Mapping[str, OutputStorageKind],
    checksums: bool = True,
    max_output_bytes: int = DEFAULT_MAX_VERIFY_BYTES,
) -> tuple[str, ...]:
    """Validate only the outputs owned by one processing partition."""
    errors: list[str] = []
    if manifest.status != "complete":
        return (f"partition {manifest.partition_id} is not complete",)
    if manifest.schema_version not in {"1", "2"}:
        errors.append(
            f"partition manifest has unsupported schema version "
            f"{manifest.schema_version!r}"
        )
    output_names = set(manifest.outputs)
    checksum_names = set(manifest.checksums)
    size_names = set(manifest.sizes)
    missing_checksums = output_names - checksum_names
    if missing_checksums:
        errors.append(
            "outputs have no checksums: " + ", ".join(sorted(missing_checksums))
        )
    extra_checksums = checksum_names - output_names
    if extra_checksums:
        errors.append(
            "checksums have no output entries: " + ", ".join(sorted(extra_checksums))
        )
    missing_sizes = output_names - size_names
    if manifest.schema_version == "2" and missing_sizes:
        errors.append("outputs have no sizes: " + ", ".join(sorted(missing_sizes)))
    extra_sizes = size_names - output_names
    if extra_sizes:
        errors.append("sizes have no output entries: " + ", ".join(sorted(extra_sizes)))
    for name, digest in manifest.checksums.items():
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            errors.append(f"manifest has an invalid checksum for {name!r}")
    for name, size in manifest.sizes.items():
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            errors.append(f"manifest has an invalid size for {name!r}")

    seen_components: set[str] = set()
    component_outputs: dict[str, list[str]] = {}
    for name, uri in manifest.outputs.items():
        if not isinstance(name, str):
            errors.append(f"manifest contains non-string component name {name!r}")
            continue
        component_name, _, _ = name.partition(":")
        try:
            validate_component_name(component_name)
        except ValueError:
            errors.append(f"manifest contains unsafe component name {name!r}")
            continue
        storage_kind = output_kinds.get(component_name)
        if storage_kind is None:
            errors.append(f"manifest contains undeclared component {name!r}")
            continue
        seen_components.add(component_name)
        component_outputs.setdefault(component_name, []).append(name)
        if not _is_expected_output(
            output_root,
            name,
            uri,
            storage_kind,
            manifest.partition_id,
        ):
            errors.append(
                f"output {name!r} does not match its declared {storage_kind} location"
            )
            continue
        try:
            if storage_kind == "table":
                fs, path = fsspec.core.url_to_fs(uri)
                if not fs.exists(path):
                    errors.append(f"missing table output: {uri}")
                    continue
                actual_size = int(fs.size(path))
                expected_size = manifest.sizes.get(name)
                if expected_size is not None and actual_size != expected_size:
                    errors.append(f"size mismatch for {name!r}")
                    continue
                if actual_size > max_output_bytes:
                    errors.append(
                        f"output {name!r} is {actual_size} bytes, exceeding the "
                        f"{max_output_bytes}-byte verification limit"
                    )
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
                output_shape = tuple(
                    (item.stop or 0) - (item.start or 0)
                    for item in partition.output_slices
                )
                actual_size = int(np.prod(output_shape, dtype=np.int64)) * int(
                    component.dtype.itemsize
                )
                expected_size = manifest.sizes.get(name)
                if expected_size is not None and actual_size != expected_size:
                    errors.append(f"size mismatch for {name!r}")
                    continue
                if actual_size > max_output_bytes:
                    errors.append(
                        f"output {name!r} is {actual_size} bytes, exceeding the "
                        f"{max_output_bytes}-byte verification limit"
                    )
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
    for component_name, names in component_outputs.items():
        if output_kinds[component_name] != "table":
            continue
        if component_name in names:
            if len(names) != 1:
                errors.append(
                    f"table component {component_name!r} mixes base and split outputs"
                )
            continue
        raw_suffixes = [name.partition(":")[2] for name in names]
        if not all(suffix.isdigit() for suffix in raw_suffixes):
            continue
        suffixes = sorted(int(suffix) for suffix in raw_suffixes)
        if suffixes != list(range(len(names))):
            errors.append(
                f"table component {component_name!r} has non-canonical output names"
            )
    output_uris = list(manifest.outputs.values())
    if len(output_uris) != len(set(output_uris)):
        errors.append("manifest maps multiple components to the same output location")
    missing_components = set(output_kinds) - seen_components
    if missing_components:
        errors.append(
            "manifest has no outputs for declared components: "
            + ", ".join(sorted(missing_components))
        )
    return tuple(errors)
