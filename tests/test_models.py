from dataclasses import asdict
from typing import Any, cast

import pytest

import neuroflow
from neuroflow.adapters import ArrayOutput, SegmentationOutputSchema, TableOutput
from neuroflow.partition import SpatialTilePlan, TimeWindowPlan
from neuroflow.provenance import stable_hash
from neuroflow.source import SourceIdentity
from neuroflow.storage import ParquetOutput, SegmentationOutput, ZarrOutput


def test_import_and_stable_hash() -> None:
    assert neuroflow.__version__ == "0.1.0"
    assert stable_hash({"b": 2, "a": 1}) == stable_hash({"a": 1, "b": 2})
    identity = SourceIdentity("DANDI:000123", "0.240101.1234", "asset-1")
    assert stable_hash(asdict(identity)) == stable_hash(asdict(identity))


@pytest.mark.parametrize(
    "schema",
    [
        lambda: ArrayOutput("float32", name="../escape"),
        lambda: TableOutput(name="tables/result"),
        lambda: SegmentationOutputSchema(objects_name="https://example.test/out"),
    ],
)
def test_output_component_names_cannot_traverse_paths(schema: object) -> None:
    with pytest.raises(ValueError, match="component names"):
        schema()  # type: ignore[operator]


@pytest.mark.parametrize(
    "factory",
    [
        lambda: TimeWindowPlan(size=0),
        lambda: TimeWindowPlan(size=10, overlap=10),
        lambda: SpatialTilePlan((64, 64), (8,), ("y", "x")),
    ],
)
def test_invalid_partition_declarations_are_rejected(factory: object) -> None:
    with pytest.raises(ValueError):
        factory()  # type: ignore[operator]


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ZarrOutput("result.zarr", mode=cast(Any, "append")),
        lambda: ParquetOutput("result", mode=cast(Any, "append")),
        lambda: SegmentationOutput("result", mode=cast(Any, "append")),
    ],
)
def test_unsupported_append_mode_is_rejected(factory: object) -> None:
    with pytest.raises(ValueError, match="create.*overwrite"):
        factory()  # type: ignore[operator]
