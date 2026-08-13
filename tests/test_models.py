from dataclasses import asdict

import pytest

import neuroflow
from neuroflow.partition import SpatialTilePlan, TimeWindowPlan
from neuroflow.provenance import stable_hash
from neuroflow.source import SourceIdentity


def test_import_and_stable_hash() -> None:
    assert neuroflow.__version__ == "0.1.0"
    assert stable_hash({"b": 2, "a": 1}) == stable_hash({"a": 1, "b": 2})
    identity = SourceIdentity("DANDI:000123", "0.240101.1234", "asset-1")
    assert stable_hash(asdict(identity)) == stable_hash(asdict(identity))


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
