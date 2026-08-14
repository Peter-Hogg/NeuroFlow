"""Dense chunked-array output configuration."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ZarrOutput:
    uri: str
    mode: Literal["create", "overwrite"] = "create"
    compressor: str = "default"

    def __post_init__(self) -> None:
        if self.mode not in {"create", "overwrite"}:
            raise ValueError("mode must be 'create' or 'overwrite'")
        if self.compressor not in {"default", "none"}:
            raise ValueError("compressor must be 'default' or 'none'")
