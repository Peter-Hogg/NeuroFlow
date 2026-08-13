from pathlib import Path

import numpy as np
import pytest

from examples.dandi_fish_projection import (
    FishProjectionConfig,
    build_adapter,
    parse_args,
    temporal_median,
)
from neuroflow.adapters import ArrayOutput


def test_fish_projection_defaults_cover_full_volume() -> None:
    config = parse_args([])
    adapter = build_adapter(config)
    assert isinstance(adapter.output, ArrayOutput)

    assert config.frames == 50
    assert (config.tile_y, config.tile_x) == (256, 256)
    assert adapter.output.reduced_axes == ("time",)
    assert adapter.output.chunks == (256, 256, 1)
    assert adapter.splittable_axes == ("z",)


def test_temporal_median_has_numpy_semantics() -> None:
    values = np.arange(2 * 3 * 4 * 2, dtype=np.uint16).reshape(2, 3, 4, 2)
    actual = temporal_median(values)

    assert actual.shape == (3, 4, 2)
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, np.median(values, axis=0))


def test_fish_projection_arguments_remain_bounded(tmp_path: Path) -> None:
    output = tmp_path / "fish.zarr"
    preview = tmp_path / "fish.png"
    config = parse_args(
        [
            "--frames",
            "5",
            "--tile-y",
            "128",
            "--tile-x",
            "512",
            "--output",
            str(output),
            "--preview",
            str(preview),
        ]
    )

    assert config == FishProjectionConfig(
        frames=5, tile_y=128, tile_x=512, output=output, preview=preview
    )


@pytest.mark.parametrize(
    "arguments",
    [
        ["--frames", "51"],
        ["--tile-y", "63"],
        ["--tile-y", "889"],
        ["--tile-x", "63"],
        ["--tile-x", "2049"],
        ["--block-size", "65535"],
        ["--cache-size-mib", "513"],
    ],
)
def test_fish_projection_rejects_work_outside_caps(arguments: list[str]) -> None:
    with pytest.raises(SystemExit):
        parse_args(arguments)
