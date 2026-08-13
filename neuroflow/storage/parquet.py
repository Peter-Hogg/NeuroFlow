"""Partitioned table output configuration."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ParquetOutput:
    uri: str
    partition_on: tuple[str, ...] = ()
    mode: Literal["create", "overwrite", "append"] = "create"
