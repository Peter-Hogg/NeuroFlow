"""Atomic partition completion metadata."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Literal


@dataclass(frozen=True)
class PartitionManifest:
    partition_id: str
    workflow_id: str
    status: Literal["complete", "failed"]
    outputs: Mapping[str, str]
    checksums: Mapping[str, str]
    schema_version: str = "1"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "PartitionManifest":
        outputs = value.get("outputs", {})
        checksums = value.get("checksums", {})
        if not isinstance(outputs, Mapping) or not isinstance(checksums, Mapping):
            raise ValueError("manifest outputs and checksums must be mappings")
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
        )
