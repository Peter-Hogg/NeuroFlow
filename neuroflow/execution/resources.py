"""Worker resource declarations."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ResourceSpec:
    cpu: int = 1
    memory: str | None = None
    gpu: int = 0
    local_scratch: str | None = None

    def __post_init__(self) -> None:
        if self.cpu < 1 or self.gpu < 0:
            raise ValueError("cpu must be positive and gpu cannot be negative")
