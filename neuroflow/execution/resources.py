"""Worker resource declarations."""

import re
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


def parse_bytes(value: int | str) -> int:
    """Parse a positive byte count such as ``512 MiB`` or ``2 GB``."""
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("memory limit must be positive")
        return value
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([KMGT]?i?B)\s*", value, re.I)
    if match is None:
        raise ValueError(f"invalid memory size: {value!r}")
    amount = float(match.group(1))
    unit = match.group(2).upper()
    powers = {
        "B": 0,
        "KB": 1,
        "KIB": 1,
        "MB": 2,
        "MIB": 2,
        "GB": 3,
        "GIB": 3,
        "TB": 4,
        "TIB": 4,
    }
    base = 1024 if "I" in unit else 1000
    return int(amount * base ** powers[unit])
