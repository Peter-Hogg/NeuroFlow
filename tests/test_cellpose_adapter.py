import importlib
import importlib.metadata
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import zarr

import neuroflow
from neuroflow.adapters import LoadedPartition, TaskContext
from neuroflow.exceptions import AdapterCompatibilityError
from neuroflow.partition import SpatialTilePlan
from neuroflow.selection import NWBQuery
from neuroflow.source.array import ArraySource
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
    prepared = adapter.prepare(loaded, TaskContext("partition"))
    with pytest.raises(AdapterCompatibilityError, match="optional"):
        adapter.run(prepared, TaskContext("partition"))


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


def test_persisted_projection_chains_into_plane_segmentation(
    nwb_zarr: tuple[Path, np.ndarray],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_cellpose(monkeypatch)
    movie = neuroflow.load(nwb_zarr[0], name="movie")
    projection = movie.median(
        "time", output=tmp_path / "projection.zarr", chunks=(2, 2)
    )
    result = projection.segment(
        CellposeAdapter(pretrained_model="fake", memory="1 GiB"),
        output=tmp_path / "cells",
        tile_shape=projection.shape,
        axes=("y", "x"),
        max_workers=1,
    )

    assert result.verify().valid
    label_source, labels = neuroflow.open_array(
        result.output.uri, component="labels"
    )
    assert labels.metadata.axes == ("y", "x")
    assert labels.metadata.shape == projection.shape
    label_source.close()
    projection.close()
    movie.close()


def test_friendly_segmentation_rejects_unreconciled_spatial_tiles(
    nwb_zarr: tuple[Path, np.ndarray], tmp_path: Path
) -> None:
    movie = neuroflow.load(nwb_zarr[0], name="movie")
    with pytest.raises(ValueError, match="allow_unmerged"):
        movie.segment(
            CellposeAdapter(pretrained_model="fake"),
            output=tmp_path / "unsafe-cells",
            tile_shape=(2, 2),
            axes=("y", "x"),
        )
    movie.close()


def test_cellpose_can_process_a_rank_preserving_single_plane(
    nwb_zarr: tuple[Path, np.ndarray],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_cellpose(monkeypatch)
    movie = neuroflow.load(nwb_zarr[0], name="movie").median(
        "time", output=tmp_path / "projection.zarr", chunks=(3, 4)
    )
    # Add a rank-preserving z axis, matching the fish projection layout.
    path = tmp_path / "projection-3d.zarr"
    root = zarr.open_group(str(path), mode="w")
    root.create_dataset(
        "projection", data=movie.compute()[..., None], chunks=(3, 4, 1)
    )
    source = ArraySource(path, component="projection", axes=("y", "x", "z"))
    projection = neuroflow.NeuroArray(source, source.select())
    result = projection.segment(
        CellposeAdapter(
            pretrained_model="fake",
            squeeze_singleton_axis=2,
            memory="1 GiB",
        ),
        output=tmp_path / "plane-labels",
        tile_shape=(1,),
        axes=("z",),
        memory_limit="1 GiB",
    )

    assert result.arrays["labels"].as_dask_array().shape == projection.shape
    assert result.verify().valid
    projection.close()
    movie.close()


def test_cellpose_convenience_returns_composable_labels(
    nwb_zarr: tuple[Path, np.ndarray],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_cellpose(monkeypatch)
    movie = neuroflow.load(nwb_zarr[0], name="movie")
    projection = movie.median(
        "time", output=tmp_path / "projection-convenience.zarr"
    )

    labels = projection.cellpose(
        pretrained_model="fake",
        output=tmp_path / "cellpose-convenience.zarr",
        memory_limit="1 GiB",
    )

    assert labels.axes == ("y", "x")
    assert labels.shape == projection.shape
    assert labels.workflow is not None
    assert labels.workflow.verify().valid
    labels.close()
    projection.close()
    movie.close()
