"""Deterministic identity helpers for workflows and partitions."""

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def _json_default(value: Any) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(f"cannot create a stable hash for {type(value).__name__}")


def stable_hash(value: object) -> str:
    """Return a stable SHA-256 digest for JSON-compatible metadata."""
    payload = json.dumps(
        value,
        default=_json_default,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
