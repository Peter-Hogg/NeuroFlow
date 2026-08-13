from pathlib import Path

import pytest

from examples.dandi_fish_projection import (
    FishProjectionConfig,
    parse_args,
    projection_slices,
)


def test_fish_projection_defaults_touch_fifty_bounded_chunks() -> None:
    config = parse_args([])

    assert config.frames == 50
    assert config.crop_size == 128
    assert config.crop_y == 380
    assert config.crop_x == 960
    assert config.z_plane == 14
    assert projection_slices(config) == (
        slice(0, 50),
        slice(380, 508),
        slice(960, 1088),
        14,
    )


def test_fish_projection_arguments_remain_bounded(tmp_path: Path) -> None:
    output = tmp_path / "fish.png"
    config = parse_args(
        [
            "--frames",
            "5",
            "--crop-size",
            "96",
            "--crop-y",
            "100",
            "--crop-x",
            "200",
            "--z-plane",
            "4",
            "--output",
            str(output),
        ]
    )

    assert config == FishProjectionConfig(
        frames=5, crop_size=96, crop_y=100, crop_x=200, z_plane=4, output=output
    )


@pytest.mark.parametrize(
    "arguments",
    [
        ["--frames", "51"],
        ["--crop-size", "129"],
        ["--crop-y", "761"],
        ["--crop-x", "1921"],
        ["--z-plane", "29"],
    ],
)
def test_fish_projection_rejects_work_outside_caps(arguments: list[str]) -> None:
    with pytest.raises(SystemExit):
        parse_args(arguments)
