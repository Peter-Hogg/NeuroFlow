"""Backend-neutral durable output contracts and metadata I/O."""

import json
import re
import uuid
from collections.abc import Mapping
from typing import Protocol

import fsspec

_COMPONENT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


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
