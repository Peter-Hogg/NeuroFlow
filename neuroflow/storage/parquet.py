"""Partitioned table output configuration."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ParquetOutput:
    uri: str
    partition_on: tuple[str, ...] = ()
    mode: Literal["create", "overwrite"] = "create"

    def __post_init__(self) -> None:
        if self.mode not in {"create", "overwrite"}:
            raise ValueError("mode must be 'create' or 'overwrite'")
