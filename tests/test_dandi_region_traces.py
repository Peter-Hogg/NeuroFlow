from pathlib import Path

import pytest

from examples.dandi_region_traces import (
    DEFAULT_OUTPUT_ROOT,
    MOVIE_SHAPE,
    RegionTracesConfig,
    parse_args,
)


def test_region_traces_defaults_match_the_measured_configuration() -> None:
    """Pin the defaults the README/docs transfer numbers were measured with."""
    config, record = parse_args([])

    assert config == RegionTracesConfig()
    assert record is None
    assert config.block_size == 32_768
    assert config.backend == "remfile"
    assert config.projection_frames == 96
    assert (config.height, config.width) == (256, 256)
    assert config.output_root == DEFAULT_OUTPUT_ROOT
    # The documented region must stay chunk-aligned to the 128-pixel chunks.
    assert config.y_start % 128 == 0
    assert config.x_start % 128 == 0
    assert config.y_start + config.height <= MOVIE_SHAPE[1]
    assert config.x_start + config.width <= MOVIE_SHAPE[2]


def test_region_traces_arguments_round_trip(tmp_path: Path) -> None:
    output_root = tmp_path / "region"
    record = tmp_path / "record.json"
    config, record_path = parse_args(
        [
            "--y-start",
            "256",
            "--x-start",
            "0",
            "--height",
            "128",
            "--width",
            "128",
            "--projection-frames",
            "32",
            "--backend",
            "lindi",
            "--cellpose-device",
            "cpu",
            "--block-size",
            "65536",
            "--output-root",
            str(output_root),
            "--record",
            str(record),
        ]
    )

    assert config == RegionTracesConfig(
        y_start=256,
        x_start=0,
        height=128,
        width=128,
        projection_frames=32,
        backend="lindi",
        cellpose_device="cpu",
        block_size=65536,
        output_root=output_root,
    )
    assert record_path == record


@pytest.mark.parametrize(
    "arguments",
    [
        ["--y-start", "100"],
        ["--x-start", "70"],
    ],
)
def test_region_traces_rejects_unaligned_regions(arguments: list[str]) -> None:
    with pytest.raises(SystemExit):
        parse_args(arguments)
