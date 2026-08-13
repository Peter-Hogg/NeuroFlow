"""Dense chunked-array output configuration."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ZarrOutput:
    uri: str
    mode: Literal["create", "overwrite", "append"] = "create"
    compressor: str = "default"

    def __post_init__(self) -> None:
        if self.compressor not in {"default", "none"}:
            raise ValueError("compressor must be 'default' or 'none'")
