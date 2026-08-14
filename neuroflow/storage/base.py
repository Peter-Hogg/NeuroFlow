"""Backend-neutral durable output contracts and metadata I/O."""

import json
import posixpath
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import unquote, urlsplit

import fsspec

from neuroflow.exceptions import OutputConflictError

_COMPONENT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class _StorageLocation:
    namespace: tuple[str, ...]
    parts: tuple[str, ...]


class OutputSpec(Protocol):
    @property
    def uri(self) -> str: ...


def join_uri(root: str, *parts: str) -> str:
    return "/".join((root.rstrip("/"), *(part.strip("/") for part in parts)))


def validate_component_name(name: str) -> str:
    """Require a storage component name to be one non-traversing path segment."""
    if not _COMPONENT_NAME.fullmatch(name) or name in {".", ".."}:
        raise ValueError(
            "component names must use only letters, digits, '.', '_', and '-' "
            "and may not contain path traversal"
        )
    return name


def _storage_location(uri: str) -> _StorageLocation:
    """Normalize a local path or hierarchical URI for containment checks."""
    parsed = urlsplit(uri)
    if parsed.scheme.lower() in {"", "file"}:
        raw_path = unquote(parsed.path) if parsed.scheme else uri
        resolved = Path(raw_path).expanduser().resolve()
        return _StorageLocation(("file",), tuple(str(part) for part in resolved.parts))

    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").lower()
    port = str(parsed.port) if parsed.port is not None else ""
    normalized_path = posixpath.normpath("/" + unquote(parsed.path)).lstrip("/")
    parts = () if normalized_path in {"", "."} else tuple(normalized_path.split("/"))
    return _StorageLocation((scheme, host, port), parts)


def _contains_location(parent: _StorageLocation, child: _StorageLocation) -> bool:
    return parent.namespace == child.namespace and child.parts[: len(parent.parts)] == (
        parent.parts
    )


def validate_output_separation(
    output_uri: str,
    input_uris: Mapping[str, str],
) -> None:
    """Reject an output equal to, inside, or containing any active input.

    The comparison is lexical for hierarchical remote URIs and uses resolved
    paths locally. It performs no network or numerical reads.
    """
    output = _storage_location(output_uri)
    for label, input_uri in input_uris.items():
        source = _storage_location(input_uri)
        if _contains_location(output, source) or _contains_location(source, output):
            raise OutputConflictError(
                f"output overlaps the active {label} input; choose a separate path"
            )


def read_json(uri: str) -> dict[str, object] | None:
    fs, path = fsspec.core.url_to_fs(uri)
    if not fs.exists(path):
        return None
    with fs.open(path, "rb") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object at {uri}")
    return value


def write_json_atomic(uri: str, value: Mapping[str, object]) -> None:
    """Commit JSON last, using rename locally and copy-on-object-stores."""
    fs, path = fsspec.core.url_to_fs(uri)
    parent = path.rsplit("/", 1)[0] if "/" in path else ""
    if parent:
        fs.makedirs(parent, exist_ok=True)
    temporary = f"{path}.tmp-{uuid.uuid4().hex}"
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    try:
        with fs.open(temporary, "wb") as stream:
            stream.write(payload)
        if fs.exists(path):
            fs.rm(path)
        fs.mv(temporary, path)
    finally:
        if fs.exists(temporary):
            fs.rm(temporary)
