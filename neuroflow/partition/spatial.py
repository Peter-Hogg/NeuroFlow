"""Spatial processing partitions."""

from dataclasses import dataclass
from itertools import product

from neuroflow.exceptions import PartitionValidationError
from neuroflow.partition.base import Partition, ValidationReport
from neuroflow.selection.query import Selection


@dataclass(frozen=True)
class SpatialTilePlan:
    tile_shape: tuple[int, ...]
    halo: tuple[int, ...]
    axes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not (len(self.tile_shape) == len(self.halo) == len(self.axes)):
            raise ValueError("tile_shape, halo, and axes must have equal length")
        if any(size <= 0 for size in self.tile_shape):
            raise ValueError("tile dimensions must be positive")
        if any(width < 0 for width in self.halo):
            raise ValueError("halo dimensions cannot be negative")

    def validate(self, selection: Selection) -> ValidationReport:
        errors: list[str] = []
        if len(set(self.axes)) != len(self.axes):
            errors.append("spatial axes must be unique")
        missing = [axis for axis in self.axes if axis not in selection.metadata.axes]
        if missing:
            errors.append(f"selection has no axes: {', '.join(missing)}")
        return ValidationReport(not errors, tuple(errors))

    def build(self, selection: Selection) -> tuple[Partition, ...]:
        report = self.validate(selection)
        if not report.valid:
            raise PartitionValidationError("; ".join(report.errors))
        axis_indices = tuple(selection.metadata.axes.index(axis) for axis in self.axes)
        starts = [
            range(0, selection.metadata.shape[index], tile)
            for index, tile in zip(axis_indices, self.tile_shape, strict=True)
        ]
        partitions: list[Partition] = []
        for number, coordinates in enumerate(product(*starts)):
            read = [slice(0, extent) for extent in selection.metadata.shape]
            output = list(read)
            trim = list(read)
            for dim, start, tile, halo in zip(
                axis_indices,
                coordinates,
                self.tile_shape,
                self.halo,
                strict=True,
            ):
                stop = min(selection.metadata.shape[dim], start + tile)
                read_start = max(0, start - halo)
                read_stop = min(selection.metadata.shape[dim], stop + halo)
                read[dim] = slice(read_start, read_stop)
                output[dim] = slice(start, stop)
                trim[dim] = slice(start - read_start, start - read_start + stop - start)
            partitions.append(
                Partition(
                    key=f"spatial-{number:08d}",
                    read_slices=tuple(read),
                    output_slices=tuple(output),
                    trim_slices=tuple(trim),
                    coordinates=tuple(coordinates),
                )
            )
        return tuple(partitions)
