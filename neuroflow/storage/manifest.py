"""Atomic partition completion metadata."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Literal


@dataclass(frozen=True)
class PartitionManifest:
    partition_id: str
    workflow_id: str
    status: Literal["complete", "failed"]
    outputs: Mapping[str, str]
    checksums: Mapping[str, str]
    schema_version: str = "2"
    sizes: Mapping[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "PartitionManifest":
        outputs = value.get("outputs", {})
        checksums = value.get("checksums", {})
        sizes = value.get("sizes", {})
        if not isinstance(outputs, Mapping):
            raise ValueError("manifest outputs, checksums, and sizes must be mappings")
        if not isinstance(checksums, Mapping):
            raise ValueError("manifest outputs, checksums, and sizes must be mappings")
        if not isinstance(sizes, Mapping):
            raise ValueError("manifest outputs, checksums, and sizes must be mappings")
        normalized_sizes: dict[str, int] = {}
        for key, item in sizes.items():
            if not isinstance(item, int) or isinstance(item, bool) or item < 0:
                raise ValueError("manifest sizes must be non-negative integers")
            normalized_sizes[str(key)] = item
        status = value.get("status")
        if status not in ("complete", "failed"):
            raise ValueError("invalid partition status")
        return cls(
            partition_id=str(value["partition_id"]),
            workflow_id=str(value["workflow_id"]),
            status=status,
            outputs={str(key): str(item) for key, item in outputs.items()},
            checksums={str(key): str(item) for key, item in checksums.items()},
            schema_version=str(value.get("schema_version", "1")),
            sizes=normalized_sizes,
        )
