"""Worker resource declarations and total-process memory budgeting."""

from __future__ import annotations

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


# Resident-set floor that exists before any partition is loaded. Every figure
# is a rounded-up envelope over the measurements in
# ``benchmarks/memory_attribution.py`` on the fish geometry (see
# ``PUBLICATION_READINESS.md``); the measured value is quoted beside each one so
# the envelope can be re-derived rather than taken on trust.
PROCESS_OVERHEAD_COMPONENTS: dict[str, int] = {
    # CPython + numpy baseline (30 MB) plus the neuroflow import chain,
    # which pulls in dask, zarr, h5py, fsspec and pynwb (167 MB).
    "interpreter_and_libraries": 224 * 1024 * 1024,
    # Lazy dask graphs and scheduler state for a multi-chunk selection (113 MB).
    "dask_runtime": 128 * 1024 * 1024,
    # The bounded remote read cache. This is the configured remfile ceiling
    # rather than the 36 MB observed on one asset, because the cache is allowed
    # to fill.
    "source_read_cache": 64 * 1024 * 1024,
    # Zarr store objects plus one compressed output window (12 MB).
    "output_write_buffers": 32 * 1024 * 1024,
}

DEFAULT_PROCESS_OVERHEAD_BYTES = sum(PROCESS_OVERHEAD_COMPONENTS.values())


@dataclass(frozen=True)
class MemoryBudget:
    """Split a total process-memory target into overhead and task working set.

    ``memory_limit`` is a *total process* target: the number a laptop user
    means when they say "stay under 2 GiB". A planner cannot spend all of it on
    partition data, because an already-running Python process holds an
    interpreter, imported libraries, lazy dask graphs, a bounded remote read
    cache and output buffers before the first byte of science data arrives.
    ``total_bytes`` therefore decomposes as::

        total_bytes = process_overhead_bytes + task_bytes

    and only ``task_bytes`` is available for partition working sets.

    This is a target rather than an enforced ceiling. NeuroFlow deliberately
    does not install an OS-level memory cap: exceeding the target should be
    visible as a reported number, not as a killed process. Third-party
    residency that the planner does not allocate -- most importantly a loaded
    Cellpose/PyTorch network -- is declared through ``external_reserve_bytes``
    so that it is subtracted honestly instead of silently ignored.
    """

    total_bytes: int
    process_overhead_bytes: int
    external_reserve_bytes: int = 0
    measured_process_floor_bytes: int = DEFAULT_PROCESS_OVERHEAD_BYTES

    def __post_init__(self) -> None:
        if self.total_bytes <= 0:
            raise ValueError("memory limit must be positive")
        if self.process_overhead_bytes < 0 or self.external_reserve_bytes < 0:
            raise ValueError("memory reserves cannot be negative")

    @property
    def reserved_bytes(self) -> int:
        return self.process_overhead_bytes + self.external_reserve_bytes

    @property
    def task_bytes(self) -> int:
        """Bytes a single task may use for partition data, never below zero."""
        return max(0, self.total_bytes - self.reserved_bytes)

    @property
    def attainable(self) -> bool:
        """Whether a real process could plausibly meet this total target.

        A target below the measured process floor cannot be met by any CPython
        process that has imported this library, no matter how small the
        partitions are. Such a target still usefully bounds the task working
        set, so it is accepted, but it is reported as unattainable rather than
        being silently presented as a satisfied total.
        """
        return self.total_bytes >= self.measured_process_floor_bytes + (
            self.external_reserve_bytes
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "interpretation": "approximate-total-process-target",
            "enforcement": "planning-only; no OS-level memory cap is installed",
            "total_bytes": self.total_bytes,
            "process_overhead_bytes": self.process_overhead_bytes,
            "process_overhead_components": dict(PROCESS_OVERHEAD_COMPONENTS),
            "measured_process_floor_bytes": self.measured_process_floor_bytes,
            "external_reserve_bytes": self.external_reserve_bytes,
            "task_bytes": self.task_bytes,
            "total_target_attainable": self.attainable,
            "note": (
                "task_bytes bounds partition working sets; the total target is "
                "attainable"
                if self.attainable
                else (
                    "this target is below the measured process floor, so the "
                    "total cannot be met; only the task working set is bounded"
                )
            ),
        }


def resolve_memory_budget(
    memory_limit: int | str,
    *,
    external_reserve_bytes: int = 0,
    process_overhead_bytes: int | None = None,
) -> MemoryBudget:
    """Interpret a user ``memory_limit`` as a total process-memory target."""
    total = parse_bytes(memory_limit)
    floor = (
        DEFAULT_PROCESS_OVERHEAD_BYTES
        if process_overhead_bytes is None
        else process_overhead_bytes
    )
    # Charge the full measured floor once the target is large enough to absorb
    # it, and taper to half the target below that. Tapering keeps small targets
    # usable on small data -- a four-element test array should not be told it
    # needs 448 MiB -- while keeping ``task_bytes`` monotonically increasing in
    # ``total``, so raising a limit never reduces the window it permits. The
    # ``attainable`` flag records when the taper is in effect.
    overhead = min(floor, total // 2)
    budget = MemoryBudget(
        total_bytes=total,
        process_overhead_bytes=overhead,
        external_reserve_bytes=external_reserve_bytes,
        measured_process_floor_bytes=floor,
    )
    if budget.task_bytes == 0:
        raise ValueError(
            "memory_limit is a total process-memory target and this one leaves "
            f"nothing for partition data: an estimated {budget.reserved_bytes} "
            "bytes are already committed to process overhead (interpreter and "
            "libraries, dask runtime, source read cache, output buffers"
            + (
                f", plus {budget.external_reserve_bytes} bytes of declared "
                "external model residency"
                if budget.external_reserve_bytes
                else ""
            )
            + f") under memory_limit={memory_limit!r}. "
            f"Raise memory_limit above {budget.reserved_bytes} bytes."
        )
    return budget
