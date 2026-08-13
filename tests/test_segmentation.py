from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import neuroflow
from neuroflow.adapters import (
    SegmentationFunctionAdapter,
    SegmentationTaskOutput,
)
from neuroflow.exceptions import AdapterCompatibilityError
from neuroflow.partition import SpatialTilePlan
from neuroflow.selection import NWBQuery
from neuroflow.storage import SegmentationOutput


def test_spatial_segmentation_persists_labels_and_variable_objects(
    nwb_zarr: tuple[Path, np.ndarray], tmp_path: Path
) -> None:
    source = neuroflow.open_source(nwb_zarr[0])
    movie = source.select(NWBQuery(name="movie"))
    calls: list[tuple[int, ...]] = []

    def threshold(block: np.ndarray) -> SegmentationTaskOutput:
        calls.append(block.shape)
        labels = (block > 50).astype("uint32")
        objects = pd.DataFrame(
            {
                "label_id": [1, 99],
                "voxel_count_with_halo": [int(labels.sum()), 1],
            }
        )
        return SegmentationTaskOutput(labels, objects)

    adapter = SegmentationFunctionAdapter(
        function=threshold,
        name="threshold-stand-in",
        version="1",
        requires_overlap={"y": 1, "x": 1},
    )
    output = SegmentationOutput(str(tmp_path / "segmentation"))
    result = neuroflow.run(
        source=source,
        selection=movie,
        adapter=adapter,
        partition=SpatialTilePlan(
            tile_shape=(2, 2),
            halo=(1, 1),
            axes=("y", "x"),
        ),
        output=output,
        execute=True,
    )

    assert len(calls) == 4
    labels = result.arrays["labels"].as_dask_array().compute()
    objects = result.tables["objects"].as_dask_dataframe().compute()
    label_ids = {int(value) for value in np.unique(labels) if value != 0}
    assert label_ids == set(objects["label_id"].astype(int))
    assert len(label_ids) == 4
    assert set(objects["local_label_id"].astype(int)) == {1}
    assert len(set(objects["tile_id"])) == 4
    assert result.provenance is not None
    assert result.provenance["output"]["merge_status"] == "unmerged"  # type: ignore[index]
    assert result.verify().valid

    calls.clear()
    result.resume()
    assert calls == []
    reopened = neuroflow.open_result(output.uri)
    assert set(reopened.arrays) == {"labels"}
    assert set(reopened.tables) == {"objects"}
    assert len(reopened.tables["objects"].as_dask_dataframe().compute()) == 4
    assert reopened.verify().valid
    source.close()


def test_segmentation_adapter_rejects_insufficient_halo(
    nwb_zarr: tuple[Path, np.ndarray], tmp_path: Path
) -> None:
    source = neuroflow.open_source(nwb_zarr[0])
    movie = source.select(NWBQuery(name="movie"))
    adapter = SegmentationFunctionAdapter(
        function=lambda value: (
            np.zeros_like(value, dtype="uint32"),
            pd.DataFrame({"label_id": pd.Series(dtype="uint32")}),
        ),
        requires_overlap={"y": 2, "x": 2},
    )
    with pytest.raises(AdapterCompatibilityError, match="does not satisfy"):
        neuroflow.run(
            source=source,
            selection=movie,
            adapter=adapter,
            partition=SpatialTilePlan((2, 2), (1, 1), ("y", "x")),
            output=SegmentationOutput(str(tmp_path / "invalid")),
        )
    source.close()
