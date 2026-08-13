"""Composite durable storage for tiled segmentation results."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SegmentationOutput:
    uri: str
    mode: Literal["create", "overwrite", "append"] = "create"
    compressor: Literal["default", "none"] = "default"
