import struct
from pathlib import Path

import numpy as np

from examples.dandi_hdf5 import (
    ASSET_ID,
    DANDISET,
    OBJECT_NAME,
    build_query,
    parse_args,
    save_reference_png,
)


def test_remote_example_defaults_are_pinned_and_bounded() -> None:
    config = parse_args([])
    query = build_query(config)

    assert config.source == DANDISET
    assert query.asset == ASSET_ID
    assert query.name == OBJECT_NAME
    assert config.block_size == 262_144
    assert config.preview_size == 128


def test_remote_example_accepts_safe_scaling_parameters(tmp_path: Path) -> None:
    output = tmp_path / "result.zarr"
    config = parse_args(
        [
            "--factor",
            "2",
            "--block-size",
            "524288",
            "--preview-size",
            "64",
            "--output",
            str(output),
        ]
    )

    assert config.factor == 2.0
    assert config.block_size == 524_288
    assert config.preview_size == 64
    assert config.output == output


def test_reference_png_is_valid_and_has_expected_dimensions(tmp_path: Path) -> None:
    path = tmp_path / "preview.png"
    save_reference_png(np.arange(30, dtype=np.float32).reshape(5, 6), path)

    value = path.read_bytes()
    assert value.startswith(b"\x89PNG\r\n\x1a\n")
    assert struct.unpack(">II", value[16:24]) == (6, 5)
