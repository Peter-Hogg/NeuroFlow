from pathlib import Path

import numpy as np
import pytest

import neuroflow
from neuroflow.adapters.numpy import ArrayOutput, FunctionAdapter
from neuroflow.exceptions import PartitionValidationError
from neuroflow.partition import Partition, ValidationReport
from neuroflow.selection import NWBQuery
from neuroflow.storage.zarr import ZarrOutput


class _StaticPartitionPlan:
    def __init__(self, partitions: tuple[object, ...]) -> None:
        self.partitions = partitions

    def validate(self, selection: object) -> ValidationReport:
        return ValidationReport(True)

    def build(self, selection: object) -> tuple[object, ...]:
        return self.partitions


def _region(
    key: str, output_slices: tuple[slice, ...], *, coordinates: tuple[object, ...] = ()
) -> Partition:
    trim_slices = tuple(
        slice(0, value.stop - value.start)  # type: ignore[operator]
        for value in output_slices
    )
    return Partition(
        key=key,
        read_slices=output_slices,
        output_slices=output_slices,
        trim_slices=trim_slices,
        coordinates=coordinates,  # type: ignore[arg-type]
    )


def _plan(
    source_path: Path,
    output_path: Path,
    partitions: tuple[object, ...],
    *,
    adapter: FunctionAdapter | None = None,
) -> object:
    source = neuroflow.open_source(source_path)
    try:
        movie = source.select(NWBQuery(name="movie"))
        return neuroflow.plan(
            source=source,
            selection=movie,
            adapter=adapter
            or FunctionAdapter(
                function=lambda values: values,
                input_kind="array",
                output=ArrayOutput(dtype="float32"),
                splittable_axes=("time", "y", "x"),
            ),
            partition=_StaticPartitionPlan(partitions),  # type: ignore[arg-type]
            output=ZarrOutput(str(output_path)),
        )
    finally:
        source.close()


def test_custom_partition_plan_must_cover_output_exactly(
    nwb_zarr: tuple[Path, np.ndarray], tmp_path: Path
) -> None:
    gap = (
        _region("left", (slice(0, 4), slice(0, 3), slice(0, 4))),
        _region("right", (slice(5, 10), slice(0, 3), slice(0, 4))),
    )
    with pytest.raises(PartitionValidationError, match="exactly cover"):
        _plan(nwb_zarr[0], tmp_path / "gap.zarr", gap)


def test_equal_volume_gap_and_overlap_is_rejected(
    nwb_zarr: tuple[Path, np.ndarray], tmp_path: Path
) -> None:
    # The overlap at time 5 exactly balances the missing time 9, so a volume
    # equality check alone cannot detect this malformed partition plan.
    gap_and_overlap = (
        _region("left", (slice(0, 6), slice(0, 3), slice(0, 4))),
        _region("right", (slice(5, 9), slice(0, 3), slice(0, 4))),
    )
    with pytest.raises(PartitionValidationError, match="overlap"):
        _plan(nwb_zarr[0], tmp_path / "overlap.zarr", gap_and_overlap)


def test_partition_keys_must_be_nonempty_and_unique(
    nwb_zarr: tuple[Path, np.ndarray], tmp_path: Path
) -> None:
    empty_key = (_region("", (slice(0, 10), slice(0, 3), slice(0, 4))),)
    with pytest.raises(PartitionValidationError, match="non-empty string"):
        _plan(nwb_zarr[0], tmp_path / "empty-key.zarr", empty_key)

    duplicate_keys = (
        _region("same", (slice(0, 5), slice(0, 3), slice(0, 4))),
        _region("same", (slice(5, 10), slice(0, 3), slice(0, 4))),
    )
    with pytest.raises(PartitionValidationError, match="duplicate partition key"):
        _plan(nwb_zarr[0], tmp_path / "duplicate-key.zarr", duplicate_keys)


@pytest.mark.parametrize(
    ("partition", "message"),
    [
        (
            _region(
                "coordinates",
                (slice(0, 10), slice(0, 3), slice(0, 4)),
                coordinates=(True,),
            ),
            "coordinates",
        ),
        (
            Partition(
                "rank",
                (slice(0, 10), slice(0, 3)),
                (slice(0, 10), slice(0, 3), slice(0, 4)),
                (slice(0, 10), slice(0, 3)),
                (),
            ),
            "rank 3",
        ),
        (
            Partition(
                "step",
                (slice(0, 10, 2), slice(0, 3), slice(0, 4)),
                (slice(0, 10), slice(0, 3), slice(0, 4)),
                (slice(0, 10), slice(0, 3), slice(0, 4)),
                (),
            ),
            "unit slice",
        ),
        (
            _region("bounds", (slice(0, 11), slice(0, 3), slice(0, 4))),
            "outside",
        ),
        (
            Partition(
                "trim",
                (slice(0, 10), slice(0, 3), slice(0, 4)),
                (slice(0, 10), slice(0, 3), slice(0, 4)),
                (slice(0, 10), slice(0, 2), slice(0, 4)),
                (),
            ),
            "do not map",
        ),
    ],
)
def test_partition_descriptors_are_validated_before_planning(
    nwb_zarr: tuple[Path, np.ndarray],
    tmp_path: Path,
    partition: Partition,
    message: str,
) -> None:
    with pytest.raises(PartitionValidationError, match=message):
        _plan(nwb_zarr[0], tmp_path / "invalid.zarr", (partition,))


def test_partition_plan_must_return_partition_descriptors(
    nwb_zarr: tuple[Path, np.ndarray], tmp_path: Path
) -> None:
    with pytest.raises(PartitionValidationError, match="Partition descriptor"):
        _plan(nwb_zarr[0], tmp_path / "not-a-partition.zarr", (object(),))


def test_reduction_lowering_preserves_exact_disjoint_coverage(
    nwb_zarr: tuple[Path, np.ndarray], tmp_path: Path
) -> None:
    reduction = FunctionAdapter(
        function=lambda values: np.mean(values, axis=0),
        input_kind="array",
        output=ArrayOutput(dtype="float32", reduced_axes=("time",)),
        splittable_axes=("y", "x"),
    )
    valid = (
        _region("upper", (slice(0, 10), slice(0, 2), slice(0, 4))),
        _region("lower", (slice(0, 10), slice(2, 3), slice(0, 4))),
    )
    plan = _plan(
        nwb_zarr[0], tmp_path / "valid-reduction.zarr", valid, adapter=reduction
    )
    assert getattr(plan, "output_shape") == (3, 4)

    # These source rectangles have equal total lowered volume but overlap at
    # y=1 and leave y=2 uncovered after the time axis is removed.
    invalid = (
        _region("upper", (slice(0, 10), slice(0, 2), slice(0, 4))),
        _region("middle", (slice(0, 10), slice(1, 2), slice(0, 4))),
    )
    with pytest.raises(PartitionValidationError, match="overlap"):
        _plan(
            nwb_zarr[0],
            tmp_path / "invalid-reduction.zarr",
            invalid,
            adapter=reduction,
        )
