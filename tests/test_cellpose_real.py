"""Opt-in equivalence test using the actual Cellpose model and weights."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr

import neuroflow
from benchmarks.cellpose_reference import load_reference
from neuroflow.source.array import ArraySource
from neuroflow_cellpose import CellposeAdapter

pytestmark = pytest.mark.cellpose_real


@pytest.mark.skipif(
    os.environ.get("NEUROFLOW_RUN_REAL_CELLPOSE") != "1",
    reason="set NEUROFLOW_RUN_REAL_CELLPOSE=1 for model-backed validation",
)
def test_real_cellpose_matches_direct_execution(tmp_path: Path) -> None:
    models = pytest.importorskip("cellpose.models")
    projection, _ = load_reference()
    model_name = os.environ.get("NEUROFLOW_CELLPOSE_MODEL", "cpsam")
    settings: dict[str, Any] = {
        "batch_size": 1,
        "channels": None,
        "channel_axis": None,
        "z_axis": None,
        "normalize": True,
        "diameter": None,
        "flow_threshold": 0.4,
        "cellprob_threshold": 0.0,
        "do_3D": False,
        "anisotropy": None,
        "min_size": 15,
        "tile_overlap": 0.1,
    }
    direct_model = models.CellposeModel(
        gpu=False,
        pretrained_model=model_name,
        use_bfloat16=False,
    )
    direct_result = direct_model.eval(projection, **settings)
    direct_labels = np.asarray(direct_result[0])
    assert np.any(direct_labels), "reference/model combination detected no objects"

    source_path = tmp_path / "cellpose-reference.zarr"
    root = zarr.open_group(str(source_path), mode="w")
    root.create_dataset("projection", data=projection, chunks=projection.shape)
    source = ArraySource(
        source_path,
        component="projection",
        axes=("y", "x"),
    )
    lazy_projection = neuroflow.NeuroArray(source, source.select())
    adapter = CellposeAdapter(
        pretrained_model=model_name,
        gpu=False,
        use_bfloat16=False,
        batch_size=1,
        memory="2 GiB",
    )
    mediated = lazy_projection.segment(
        adapter,
        output=tmp_path / "cellpose-mediated",
        tile_shape=projection.shape,
        axes=("y", "x"),
        max_workers=1,
        memory_limit="2 GiB",
    )
    mediated_labels = mediated.arrays["labels"].as_dask_array().compute()
    local_labels = np.where(
        mediated_labels,
        mediated_labels - np.uint64(1 << 32),
        0,
    ).astype(direct_labels.dtype)

    np.testing.assert_array_equal(local_labels, direct_labels)
    assert mediated.verify().valid
    assert mediated.provenance is not None
    assert mediated.provenance["external_libraries"]["cellpose"]  # type: ignore[index]
    lazy_projection.close()
