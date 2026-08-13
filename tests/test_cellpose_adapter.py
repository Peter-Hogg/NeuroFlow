import importlib
import importlib.metadata
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

import neuroflow
from neuroflow.adapters import LoadedPartition, TaskContext
from neuroflow.exceptions import AdapterCompatibilityError
from neuroflow.partition import SpatialTilePlan
from neuroflow.selection import NWBQuery
from neuroflow.storage import SegmentationOutput
from neuroflow_cellpose import CellposeAdapter
from neuroflow_cellpose import adapter as cellpose_adapter_module


def _install_fake_cellpose(monkeypatch: pytest.MonkeyPatch) -> type:
    class FakeCellposeModel:
        instances: list["FakeCellposeModel"] = []
        evaluations: list[dict[str, object]] = []

        def __init__(self, **kwargs: object) -> None:
            self.options = kwargs
            self.instances.append(self)

        def eval(self, value: np.ndarray, **kwargs: object) -> tuple[object, ...]:
            self.evaluations.append(kwargs)
            masks = np.ones(value.shape, dtype="uint32")
            masks[..., value.shape[-1] // 2 :] = 2
            return masks, object(), object()

    package = ModuleType("cellpose")
    package.__path__ = []  # type: ignore[attr-defined]
    models = ModuleType("cellpose.models")
    models.CellposeModel = FakeCellposeModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cellpose", package)
    monkeypatch.setitem(sys.modules, "cellpose.models", models)
    cellpose_adapter_module._THREAD_STATE.models = {}
    return FakeCellposeModel


def test_cellpose_is_deferred_and_reports_missing_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cellpose_adapter_module._THREAD_STATE.models = {}
    original = importlib.import_module

    def missing(name: str, package: str | None = None) -> ModuleType:
        if name == "cellpose.models":
            raise ImportError("not installed")
        return original(name, package)

    monkeypatch.setattr(cellpose_adapter_module.importlib, "import_module", missing)
    adapter = CellposeAdapter(pretrained_model="cpsam_v2")
    loaded = LoadedPartition(
        np.zeros((4, 4), dtype="float32"),
        (slice(0, 4), slice(0, 4)),
        (slice(0, 4), slice(0, 4)),
        (slice(0, 4), slice(0, 4)),
    )
    with pytest.raises(AdapterCompatibilityError, match="optional"):
        adapter.run(loaded, TaskContext("partition"))


def test_cellpose_adapter_runs_as_optional_spatial_integration(
    nwb_zarr: tuple[Path, np.ndarray],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_model = _install_fake_cellpose(monkeypatch)
    original_version = importlib.metadata.version

    def package_version(name: str) -> str:
        return "4.2.1" if name == "cellpose" else original_version(name)

    monkeypatch.setattr(importlib.metadata, "version", package_version)
    source = neuroflow.open_source(nwb_zarr[0])
    movie = source.select(NWBQuery(name="movie"))
    adapter = CellposeAdapter(
        pretrained_model="cpsam_v2",
        gpu=True,
        diameter=30.0,
        cellprob_threshold=0.25,
        halo={"y": 1, "x": 1},
    )
    assert adapter.requirements().resources.gpu == 1
    output = SegmentationOutput(str(tmp_path / "cellpose-result"))
    result = neuroflow.run(
        source=source,
        selection=movie,
        adapter=adapter,
        partition=SpatialTilePlan((2, 2), (1, 1), ("y", "x")),
        output=output,
        execute=True,
    )
    assert fake_model.instances
    assert fake_model.evaluations
    assert all(item["cellprob_threshold"] == 0.25 for item in fake_model.evaluations)
    labels = result.arrays["labels"].as_dask_array().compute()
    objects = result.tables["objects"].as_dask_dataframe().compute()
    assert labels.dtype == np.dtype("uint64")
    assert len(objects) >= 4
    assert {int(value) for value in np.unique(labels) if value} == set(
        objects["label_id"].astype(int)
    )
    assert result.provenance is not None
    assert result.provenance["external_libraries"]["cellpose"] == "4.2.1"  # type: ignore[index]
    assert result.verify().valid

    changed = CellposeAdapter(
        pretrained_model="cpsam_v2",
        cellprob_threshold=0.5,
        halo={"y": 1, "x": 1},
    )
    changed_plan = neuroflow.plan(
        source=source,
        selection=movie,
        adapter=changed,
        partition=SpatialTilePlan((2, 2), (1, 1), ("y", "x")),
        output=SegmentationOutput(str(tmp_path / "changed")),
    )
    assert changed_plan.workflow_id != result.plan.workflow_id
    source.close()
