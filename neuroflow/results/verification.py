"""Result integrity report."""

from dataclasses import dataclass


@dataclass(frozen=True)
class VerificationReport:
    valid: bool
    checked_partitions: tuple[str, ...]
    errors: tuple[str, ...] = ()
