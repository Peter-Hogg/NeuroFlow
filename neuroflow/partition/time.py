"""Temporal processing partitions."""

import re
from dataclasses import dataclass
from typing import Literal

from neuroflow.exceptions import PartitionValidationError
from neuroflow.partition.base import Partition, ValidationReport
from neuroflow.selection.query import Selection


@dataclass(frozen=True)
class TimeWindowPlan:
    size: int | str
    overlap: int | str = 0
    units: Literal["samples"] | None = None
    align_to: Literal["timestamps"] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.size, int) and self.size <= 0:
            raise ValueError("size must be positive")
        if isinstance(self.overlap, int) and self.overlap < 0:
            raise ValueError("overlap cannot be negative")
        if isinstance(self.size, int) and isinstance(self.overlap, int):
            if self.overlap >= self.size:
                raise ValueError("overlap must be smaller than size")

    def _samples(self, value: int | str, selection: Selection) -> int:
        if isinstance(value, int):
            return value
        match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(ms|s|min)\s*", value)
        if match is None:
            raise ValueError(f"invalid duration: {value!r}")
        if selection.metadata.rate is None:
            raise ValueError("duration partitions require a regular NWB sampling rate")
        scale = {"ms": 0.001, "s": 1.0, "min": 60.0}[match.group(2)]
        return max(1, round(float(match.group(1)) * scale * selection.metadata.rate))

    def validate(self, selection: Selection) -> ValidationReport:
        errors: list[str] = []
        warnings: list[str] = []
        if not selection.metadata.shape:
            errors.append("the selection has no time axis")
        if self.align_to == "timestamps" and selection.metadata.timestamps_path:
            errors.append(
                "irregular timestamp alignment is not supported without reading "
                "coordinates"
            )
        try:
            size = self._samples(self.size, selection)
            overlap = self._samples(self.overlap, selection)
            if overlap >= size:
                errors.append("overlap must be smaller than size")
            if selection.metadata.shape and size > selection.metadata.shape[0]:
                warnings.append("window size exceeds the selected time dimension")
        except ValueError as exc:
            errors.append(str(exc))
        return ValidationReport(not errors, tuple(errors), tuple(warnings))

    def build(self, selection: Selection) -> tuple[Partition, ...]:
        report = self.validate(selection)
        if not report.valid:
            raise PartitionValidationError("; ".join(report.errors))
        size = self._samples(self.size, selection)
        overlap = self._samples(self.overlap, selection)
        length = selection.metadata.shape[0]
        tail = tuple(slice(0, extent) for extent in selection.metadata.shape[1:])
        partitions: list[Partition] = []
        for index, start in enumerate(range(0, length, size)):
            stop = min(length, start + size)
            read_start = max(0, start - overlap)
            read_stop = min(length, stop + overlap)
            partitions.append(
                Partition(
                    key=f"time-{index:08d}",
                    read_slices=(slice(read_start, read_stop), *tail),
                    output_slices=(slice(start, stop), *tail),
                    trim_slices=(
                        slice(start - read_start, start - read_start + stop - start),
                        *tail,
                    ),
                    coordinates=(index, start, stop),
                )
            )
        return tuple(partitions)
