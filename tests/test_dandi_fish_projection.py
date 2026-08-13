from pathlib import Path

from examples.dandi_fish_projection import (
    FishProjectionConfig,
    parse_args,
    projection_slices,
)


def test_fish_projection_defaults_touch_three_bounded_chunks() -> None:
    config = parse_args([])

    assert config.frames == 3
    assert config.crop_size == 64
    assert config.z_plane == 0
    assert projection_slices(config) == (
        slice(0, 3),
        slice(0, 64),
        slice(0, 64),
        0,
    )


def test_fish_projection_arguments_remain_bounded(tmp_path: Path) -> None:
    output = tmp_path / "fish.png"
    config = parse_args(
        [
            "--frames",
            "5",
            "--crop-size",
            "96",
            "--z-plane",
            "4",
            "--output",
            str(output),
        ]
    )

    assert config == FishProjectionConfig(
        frames=5, crop_size=96, z_plane=4, output=output
    )
