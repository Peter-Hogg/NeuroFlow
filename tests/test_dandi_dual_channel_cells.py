from pathlib import Path

import numpy as np

from examples.dandi_dual_channel_cells import (
    DEFAULT_OUTPUT,
    _reference_chunks,
    detect_blob_candidates,
    parse_args,
)


def test_dual_channel_default_output_is_versioned_for_numpy_engine() -> None:
    config = parse_args(["--neuron-name", "green", "--glia-name", "red"])

    assert config.output == DEFAULT_OUTPUT
    assert config.output.name == "dual-channel-cells-numpy"


def test_candidate_detector_preserves_named_native_coordinates() -> None:
    volume = np.zeros((9, 11, 5), dtype=np.float32)
    volume[3, 4, 2] = 20
    volume[7, 8, 3] = 15

    result = detect_blob_candidates(
        volume,
        axes=("y", "x", "z"),
        cell_class="neuron",
        sigma=0,
        percentile=98,
        minimum_distance=1,
    )

    assert set(result.columns) >= {"y_voxel", "x_voxel", "z_voxel", "cell_class"}
    coordinates = set(
        zip(result["y_voxel"], result["x_voxel"], result["z_voxel"], strict=True)
    )
    assert coordinates == {
        (3, 4, 2),
        (7, 8, 3),
    }
    assert set(result.cell_class) == {"neuron"}


def test_candidate_detector_collapses_flat_maxima_and_enforces_distance() -> None:
    volume = np.zeros((7, 7, 3), dtype=np.float32)
    volume[2:4, 2:4, 1] = 10
    volume[2, 4, 1] = 10
    result = detect_blob_candidates(
        volume,
        axes=("y", "x", "z"),
        cell_class="candidate",
        sigma=0,
        percentile=90,
        minimum_distance=2,
    )
    assert len(result) == 1
    assert tuple(result.loc[0, ["y_voxel", "x_voxel", "z_voxel"]]) == (2, 2, 1)


def test_dual_channel_arguments_keep_identity_and_assets_separate(
    tmp_path: Path,
) -> None:
    config = parse_args(
        [
            "--neuron-name",
            "green",
            "--glia-name",
            "red",
            "--neuron-asset",
            "asset-a",
            "--glia-asset",
            "asset-b",
            "--frames",
            "20",
            "--output",
            str(tmp_path),
            "--detect",
        ]
    )

    assert config.neuron.name == "green"
    assert config.neuron.asset == "asset-a"
    assert config.glia.name == "red"
    assert config.glia.asset == "asset-b"
    assert config.glia.cell_class == "radial_astrocyte"
    assert config.frames == 20
    assert config.detect


def test_reference_chunks_follow_named_axes() -> None:
    assert _reference_chunks(("time", "y", "x", "z"), (50, 888, 2048, 29)) == (
        256,
        256,
        1,
    )
